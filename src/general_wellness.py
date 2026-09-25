"""General medical conversation backed by curated sources + MedlinePlus fallback.

The goal of this layer is different from the structured diagnostic models:
it should behave like a real patient-facing conversation. It can explain terms,
answer follow-up questions, translate the previous answer, and give practical
next steps while keeping the medical-safety boundary (education != diagnosis).
"""

from __future__ import annotations

import os
import re

from src import nlu
from src.knowledge_base import KnowledgeBase
from src.live_medical_search import query_medlineplus


class GeneralWellnessHandler:
    def __init__(self, api_key: str | None = None, provider: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider or (
            "gemini" if os.getenv("GEMINI_API_KEY")
            else "openai" if os.getenv("OPENAI_API_KEY")
            else None
        )
        self.kb = KnowledgeBase(api_key=self.api_key, provider=self.provider)

    def respond(self, patient_message: str, conversation_history: str = "") -> dict:
        patient_message = (patient_message or "").strip()

        # Language tasks should not go through medical retrieval.  In particular,
        # "ترجملي اللي قولته" refers to the previous assistant message.
        if nlu.classify_intent(patient_message, self.api_key, self.provider) == "translation":
            translated = self._translate_from_history(patient_message, conversation_history)
            if translated:
                return self._result(
                    translated,
                    sources=[],
                    engine="conversation_translation",
                )

        # For a short follow-up ("طيب أعمل إيه؟", "وده؟", "طب والعلاج؟"),
        # the current message alone is not a sufficient retrieval query.
        retrieval_query = self._build_retrieval_query(patient_message, conversation_history)

        result = self.kb.answer_with_context(
            retrieval_query,
            top_k=4,
            conversation_context=conversation_history,
        )
        used_live_source = False

        if not result["context_used"]:
            search_term = self._to_english_search_term(retrieval_query)
            live_results = query_medlineplus(search_term, max_results=3, timeout=8)
            if live_results:
                used_live_source = True
                live_context = "\n\n---\n\n".join(
                    f"[{r['title']}]\n{r['summary']}" for r in live_results
                )
                result = self._answer_from_live_context(
                    patient_message,
                    live_context,
                    live_results,
                    conversation_history,
                )

        if result.get("answer"):
            message = result["answer"]
        elif result.get("context_used"):
            # IMPORTANT: context availability does NOT mean the LLM is unavailable.
            # The previous message incorrectly told the patient that the LLM was off
            # whenever the context-to-answer generation call raised an exception.
            # Keep the user-facing fallback neutral and log the real error server-side.
            llm_error = result.get("llm_error")
            if llm_error:
                print(f"GeneralWellness LLM generation error: {llm_error}")
            message = self._fallback_context_message(result["context_used"], llm_error=llm_error)
        else:
            message = self._no_source_message(patient_message)

        engine = (
            "rag_live_medlineplus" if used_live_source
            else "rag_curated_kb" if result["context_used"]
            else "no_source_found"
        )
        return self._result(message, result["sources"], engine)

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def _build_retrieval_query(self, message: str, history: str) -> str:
        """Make short/deictic follow-ups searchable without polluting normal queries."""
        if not history or not self._is_follow_up(message):
            return message

        user_turns = [
            line.split(":", 1)[1].strip()
            for line in history.splitlines()
            if line.startswith("user:")
        ]
        if not user_turns:
            return message

        # Keep the current request first so it remains the main intent.
        previous = " ".join(user_turns[-2:])
        return f"{message}\nPrevious patient context: {previous}"

    @staticmethod
    def _is_follow_up(message: str) -> bool:
        t = message.lower().strip()
        follow_up_markers = [
            "طب", "طيب", "وده", "دي", "ده", "يعني ايه ده", "اعمل ايه",
            "أعمل إيه", "اعمل ايه", "ماذا افعل", "what should i do",
            "what about", "and then", "how about", "treatment", "العلاج",
            "الخطوة الجاية", "بعد كده", "وبعدين", "this", "that", "it",
        ]
        # Do not treat every short message as a follow-up: a single word such
        # as "diabetes" or "headache" is usually a NEW topic.
        if any(x in t for x in follow_up_markers):
            return True
        return len(t.split()) <= 5 and bool(
            re.search(r"\b(it|this|that)\b|\b(ده|دي|ده؟|دي؟)\b", t)
        )

    @staticmethod
    def _last_assistant_message(history: str) -> str | None:
        lines = [line for line in history.splitlines() if line.startswith("assistant:")]
        if not lines:
            return None
        return lines[-1].split(":", 1)[1].strip()

    def _translate_from_history(self, request: str, history: str) -> str | None:
        source = self._last_assistant_message(history)
        if not source:
            # If the user explicitly supplied text after "ترجم", use that text
            # instead of pretending there is a previous answer.
            cleaned = re.sub(
                r"^\s*(ترجملي|ترجمه|ترجم|translate|translation)\s*[:：]?\s*",
                "",
                request,
                flags=re.I,
            ).strip()
            source = cleaned or None

        if not source:
            return "أكيد، ابعتيلي الكلام اللي عايزة تترجميه وأنا أترجمهولك."

        target = self._translation_target(request, source)
        if not self.api_key or not self.provider:
            # Avoid the old misleading "no medical source" message.
            return (
                "أقدر أترجمهولك، لكن الترجمة الطبيعية داخل الشات محتاجة مفتاح "
                "LLM شغال (GEMINI_API_KEY أو OPENAI_API_KEY)."
            )

        system = (
            "You are translating a medical conversation for a patient. Translate the supplied "
            "text faithfully into the requested target language. Do not add medical advice, "
            "diagnosis, warnings, or facts that are not in the source. Keep medical terms clear "
            "and patient-friendly. If the target is Arabic, use natural Egyptian Arabic."
        )
        user = f"Target language: {target}\n\nText to translate:\n{source}"
        return self._call_llm_text(system, user)

    @staticmethod
    def _translation_target(request: str, source: str) -> str:
        t = request.lower()
        if any(x in t for x in ["بالانجليزي", "بالإنجليزي", "بالانجليزية", "بالإنجليزية", "in english"]):
            return "English"
        if any(x in t for x in ["بالعربي", "بالعربية", "in arabic"]):
            return "Egyptian Arabic"
        # Otherwise switch to the other language used by the source.
        return "English" if re.search(r"[\u0600-\u06ff]", source) else "Egyptian Arabic"

    # ------------------------------------------------------------------
    # LLM answer generation
    # ------------------------------------------------------------------

    def _patient_friendly_system_prompt(
        self,
        conversation_history: str,
        trusted_context: str,
    ) -> str:
        return (
            "You are a warm, patient-facing medical information assistant. "
            "Your job is to help a non-doctor understand health information, not to replace a doctor. "
            "Answer in the SAME language and style as the patient: natural Egyptian Arabic for Arabic, "
            "simple English for English, and natural mixed Arabic/English when the patient mixes them. "
            "Do not switch to English just because the medical source is English. Translate/explain "
            "source terminology into the patient's language.\\n\\n"
            "Use the trusted context as the factual basis. Do not invent facts that are not supported "
            "by it. However, do not merely paste the source: explain it in simple words. When useful, "
            "structure the answer as: 1) what it means, 2) common symptoms/signs, 3) how doctors "
            "usually confirm/evaluate it, 4) what the person can do now, and 5) when it needs urgent "
            "medical attention. Not every answer needs all five sections.\\n\\n"
            "For follow-up questions, use the conversation history to resolve words such as 'ده', 'دي', "
            "'طيب', 'what about it', or 'what should I do?'. Keep the answer focused on the current "
            "question instead of restarting the whole explanation.\\n\\n"
            "Do not diagnose the patient from symptoms alone. Do not prescribe or change medication "
            "doses. For treatment questions, explain general options and tell the patient when clinician "
            "review is needed. If emergency red flags are relevant, make them prominent and actionable.\\n\\n"
            "End with a short reminder that this is educational information, not a diagnosis, when "
            "the answer is medical.\\n\\n"
            f"Recent conversation:\\n{conversation_history or 'none'}\\n\\n"
            f"Trusted medical context:\\n{trusted_context}"
        )

    def _answer_from_live_context(
        self,
        query: str,
        live_context: str,
        live_results: list[dict],
        conversation_history: str = "",
    ) -> dict:
        sources = [r["source"] + " — " + r["title"] for r in live_results]
        if not self.api_key or not self.provider:
            return {
                "answer": None,
                "sources": sources,
                "context_used": [live_context],
            }

        system_prompt = self._patient_friendly_system_prompt(
            conversation_history,
            live_context,
        )
        try:
            answer = self._call_llm_text(system_prompt, query)
            return {
                "answer": answer,
                "sources": sources,
                "context_used": [live_context],
            }
        except Exception as exc:
            print(f"Live-source answer generation failed: {exc}")
            return {
                "answer": None,
                "sources": sources,
                "context_used": [live_context],
            }

    def _call_llm_text(self, system_prompt: str, user_content: str) -> str | None:
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                system_instruction=system_prompt,
            )
            response = model.generate_content(user_content)
            text = (getattr(response, "text", "") or "").strip()
            return text or None

        if self.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, timeout=20.0)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip() or None

        return None

    def _to_english_search_term(self, message: str) -> str:
        """Turn a patient sentence into a short MedlinePlus search term."""
        # Useful even without an LLM key.
        direct_terms = {
            "السكري": "diabetes",
            "السكر": "diabetes",
            "ما قبل السكري": "prediabetes",
            "صداع": "headache",
            "الصداع": "headache",
            "الشقيقة": "migraine",
            "ضغط الدم": "high blood pressure",
            "الضغط": "high blood pressure",
            "الهيموجلوبين": "hemoglobin",
            "فقر الدم": "anemia",
            "انيميا": "anemia",
            "أنيميا": "anemia",
            "الكوليسترول": "cholesterol",
            "الربو": "asthma",
        }
        for ar, en in direct_terms.items():
            if ar in message:
                return en

        if not self.api_key or not self.provider:
            return message

        prompt = (
            "Convert this patient health question into a short English medical search query "
            "(2-6 words). Return only the query, no explanation.\\n" + message
        )
        try:
            result = self._call_llm_text(
                "You create concise medical search queries for MedlinePlus.",
                prompt,
            )
            return result or message
        except Exception as exc:
            print(f"Medical search-term generation failed: {exc}")
            return message

    # ------------------------------------------------------------------
    # Fallbacks / response shape
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_context_message(context_used: list[str], llm_error: str | None = None) -> str:
        # Do not expose internal API/model errors to the patient.
        return (
            "لقيت معلومة مرتبطة بسؤالك في المصادر الموثوقة عندي، لكن حصلت مشكلة مؤقتة "
            "وأنا بصيغها في شكل شرح مبسط. دي المعلومة الأساسية المتاحة عندي حالياً:\n\n"
            + context_used[0][:1600]
        )

    @staticmethod
    def _no_source_message(message: str) -> str:
        return (
            "مش لاقية مصدر طبي موثوق داخل المكتبة الحالية عن النقطة دي، ومش عايزة "
            "أخمن عليكي. لو سؤالك عن مصطلح أو حالة معينة، اكتبي اسمها بوضوح، "
            "ولو الـ LLM/API شغال هقدر أبحث في MedlinePlus وأشرحها بلغة بسيطة."
        )

    @staticmethod
    def _result(message: str | None, sources: list, engine: str) -> dict:
        return {
            "action": "give_recommendation",
            "message": message or "ممكن توضحيلي السؤال أكتر؟",
            "data": {
                "sources": sources,
                "engine": engine,
                "safety_disclaimer": "معلومات تثقيفية عامة، مش تشخيص طبي.",
            },
        }
