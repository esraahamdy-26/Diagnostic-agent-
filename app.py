"""
FastAPI Backend for the Unified Diagnostic Agent
====================================================
Endpoints:
  POST /session/start          -> creates a new conversation, returns session_id
  POST /session/{id}/message   -> sends an answer, returns the agent's next response
  GET  /domains                -> lists supported domains
  GET  /health                 -> health check

Run:
    pip install fastapi uvicorn
    uvicorn app:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI, or use the
included chat.html as a simple frontend (see README).
"""

import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()  # بيقرا ملف .env تلقائياً لو موجود (فيه GEMINI_API_KEY/OPENAI_API_KEY)
except ImportError:
    pass

from src.conversational_agent import SAFETY_FIELD_SPECS, ConversationalDiagnosticAgent
from src.orchestrator import AgentAction, UnifiedDiagnosticAgent
from src.vision_extraction import VisionExtractionError, extract_attachment_content

app = FastAPI(title="Diagnostic Agent API", version="1.0.0")

# عشان الـ frontend (chat.html) يقدر يكلم الـ API من متصفح مختلف
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_DOMAINS = [
    "heart_disease", "cbc", "pathology",
    "diabetes_risk_screening", "hypertension_risk_screening", "radiology",
]
# الـ chat endpoints بتدعم domain إضافي (general_wellness / RAG) — مش متاح
# في الـ structured endpoints (/session/...) لأن UnifiedDiagnosticAgent
# مصمم بس للـ 6 domains اللي عندها موديل/rule فعلي
CHAT_SUPPORTED_DOMAINS = SUPPORTED_DOMAINS + ["general_wellness"]

# In-memory session store — كافي للتجربة/الديمو، في production استخدمي Redis/DB
SESSIONS: dict[str, UnifiedDiagnosticAgent] = {}
CHAT_SESSIONS: dict[str, ConversationalDiagnosticAgent] = {}


class StartSessionRequest(BaseModel):
    domain: str


class StartSessionResponse(BaseModel):
    session_id: str
    action: str
    message: str
    data: dict[str, Any] = {}


class MessageRequest(BaseModel):
    # المريض بيرد على آخر سؤال. المفتاح لازم يطابق الـ "feature" اللي
    # اتبعت في آخر رد. القيمة null معناها "رفض/معرفش" (للأسئلة الاختيارية).
    answers: dict[str, Any]


class MessageResponse(BaseModel):
    action: str
    message: str
    data: dict[str, Any] = {}


def _to_response(r) -> dict:
    return {"action": r.action.value, "message": r.message, "data": r.data}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/llm_status")
def llm_status():
    """
    بتختبر المفتاح فعلياً بنداء حقيقي على Gemini/OpenAI، مش بس تشوف هل
    الـ environment variable موجود ولا لأ. ده الفرق المهم: متغير موجود
    مش معناه إنه شغال (ممكن يكون غلط، منتهي، أو حتى اسم الموديل نفسه
    مش متاح بيه).
    """
    import os
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    result = {
        "gemini_key_detected": bool(gemini_key),
        "openai_key_detected": bool(openai_key),
        "gemini_live_test": None,
        "openai_live_test": None,
    }

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
            response = model.generate_content("Reply with exactly: OK")
            result["gemini_live_test"] = {"success": True, "response": response.text.strip()}
        except Exception as e:
            result["gemini_live_test"] = {"success": False, "error": str(e)}

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key, timeout=20.0)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            )
            result["openai_live_test"] = {"success": True, "response": resp.choices[0].message.content.strip()}
        except Exception as e:
            result["openai_live_test"] = {"success": False, "error": str(e)}

    working = (result["gemini_live_test"] or {}).get("success") or (result["openai_live_test"] or {}).get("success")
    result["nlu_mode"] = "llm (فهم لغوي حقيقي - المفتاح شغال فعلاً)" if working else \
        "fallback (المفتاح موجود لكن فشل الاختبار الفعلي - شوفي gemini_live_test/openai_live_test)" if (gemini_key or openai_key) else \
        "fallback (مفيش مفتاح خالص)"
    return result


@app.get("/domains")
def domains():
    return {"supported_domains": SUPPORTED_DOMAINS}


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    if req.domain not in SUPPORTED_DOMAINS:
        raise HTTPException(400, f"Unsupported domain '{req.domain}'. "
                                  f"Choose from: {SUPPORTED_DOMAINS}")

    agent = UnifiedDiagnosticAgent(req.domain)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = agent

    # أول turn بيسأل أسئلة الأمان الأساسية (مفيش بيانات لسه)
    response = agent.handle_turn({})
    return {"session_id": session_id, **_to_response(response)}


@app.post("/session/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, req: MessageRequest):
    agent = SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, "Session not found. Start a new one via /session/start.")

    response = agent.handle_turn(req.answers)
    return _to_response(response)


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"status": "ended"}


# ---------------------------------------------------------------------------
# Free-text chat endpoints (NLU layer) — محادثة حرة زي شات حقيقي
# ---------------------------------------------------------------------------

class ChatStartRequest(BaseModel):
    domain: str | None = None  # اختياري: لو عايزة تحددي الـ domain مقدماً


class ChatMessageRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    session_id: str | None = None
    action: str
    message: str
    data: dict[str, Any] = {}


@app.post("/chat/start", response_model=ChatResponse)
def chat_start(req: ChatStartRequest):
    if req.domain and req.domain not in CHAT_SUPPORTED_DOMAINS:
        raise HTTPException(400, f"Unsupported domain '{req.domain}'.")

    agent = ConversationalDiagnosticAgent(initial_domain=req.domain)
    session_id = str(uuid.uuid4())
    CHAT_SESSIONS[session_id] = agent

    if req.domain == "general_wellness":
        return {"session_id": session_id, "action": "ask_question",
                "message": "تمام، احكيلي إيه اللي حاسة بيه أو عايزة تعرفي إيه؟", "data": {}}
    if req.domain:
        return {"session_id": session_id, "action": "ask_question",
                "message": "تمام. احكيلي براحتك إيه اللي حابة تعرفيه أو ابعتي البيانات/النتيجة اللي عندك.", "data": {}}
    return {"session_id": session_id, "action": "ask_question",
            "message": "أهلاً! احكيلي حصل معاكي إيه؟", "data": {}}


@app.post("/chat/{session_id}/message", response_model=ChatResponse)
async def chat_message(session_id: str, request: Request):
    """Accept either the old JSON message or a multipart message + attachment.

    Multipart is what the chat UI uses now: the patient can attach a file/image,
    type a question about it, and send both together.
    """
    agent = CHAT_SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, "Session not found. Start a new one via /chat/start.")

    content_type = (request.headers.get("content-type") or "").lower()
    attachment_info = None

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        text = str(form.get("text") or "").strip()
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            content = await upload.read()
            try:
                attachment_info = extract_attachment_content(
                    content,
                    mime_type=upload.content_type or "application/octet-stream",
                    filename=upload.filename or "attachment",
                    api_key=agent.api_key,
                    provider=agent.provider,
                )
            except VisionExtractionError as exc:
                return {
                    "session_id": session_id,
                    "action": "ask_question",
                    "message": str(exc),
                    "data": {"attachment_error": True},
                }

            extracted = attachment_info.get("text", "")
            question = text or "اقرأي الملف/الصورة دي واشرحيلي ببساطة إيه اللي ظاهر فيها وإيه أهم حاجة المفروض أفهمها."
            # Keep the actual attachment content in the conversation context so a
            # follow-up such as "طيب وده معناه إيه؟" can refer to the same file.
            message_for_agent = (
                f"{question}\n\n"
                f"[ATTACHMENT: {attachment_info.get('filename')}]\n"
                f"[EXTRACTED ATTACHMENT CONTENT]\n{extracted[:20000]}"
            )
        else:
            message_for_agent = text
    else:
        body = await request.json()
        message_for_agent = str(body.get("text") or "").strip()

    if not message_for_agent.strip():
        raise HTTPException(400, "اكتبي رسالة أو ارفعي ملفاً.")

    result = agent.handle_message(message_for_agent)
    if attachment_info:
        result.setdefault("data", {})
        result["data"]["attachment"] = {
            "filename": attachment_info.get("filename"),
            "mime_type": attachment_info.get("mime_type"),
            "kind": attachment_info.get("kind"),
            "extracted_text_preview": attachment_info.get("text", "")[:600],
        }
    return {"session_id": session_id, **result}


@app.post("/chat/{session_id}/set_domain", response_model=ChatResponse)
def chat_set_domain(session_id: str, req: StartSessionRequest):
    agent = CHAT_SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, "Session not found.")
    if req.domain not in CHAT_SUPPORTED_DOMAINS:
        raise HTTPException(400, f"Unsupported domain '{req.domain}'.")
    result = agent.set_domain(req.domain)
    return {"session_id": session_id, **result}


@app.post("/chat/{session_id}/upload", response_model=ChatResponse)
async def chat_upload(session_id: str, file: UploadFile = File(...)):
    """Backward-compatible upload endpoint. New UI sends attachment + text together."""
    agent = CHAT_SESSIONS.get(session_id)
    if agent is None:
        raise HTTPException(404, "Session not found. Start a new one via /chat/start.")

    content = await file.read()
    try:
        attachment_info = extract_attachment_content(
            content,
            mime_type=file.content_type or "application/octet-stream",
            filename=file.filename or "attachment",
            api_key=agent.api_key,
            provider=agent.provider,
        )
    except VisionExtractionError as exc:
        return {"session_id": session_id, "action": "ask_question", "message": str(exc), "data": {"attachment_error": True}}

    question = "اقرأي الملف/الصورة دي واشرحيلي ببساطة إيه اللي ظاهر فيها وإيه أهم حاجة المفروض أفهمها."
    message_for_agent = (
        f"{question}\n\n[ATTACHMENT: {attachment_info.get('filename')}]\n"
        f"[EXTRACTED ATTACHMENT CONTENT]\n{attachment_info.get('text', '')[:20000]}"
    )
    result = agent.handle_message(message_for_agent)
    result.setdefault("data", {})
    result["data"]["attachment"] = {
        "filename": attachment_info.get("filename"),
        "mime_type": attachment_info.get("mime_type"),
        "kind": attachment_info.get("kind"),
        "extracted_text_preview": attachment_info.get("text", "")[:600],
    }
    return {"session_id": session_id, **result}
