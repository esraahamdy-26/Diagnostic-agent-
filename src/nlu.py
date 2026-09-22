"""
Natural-language understanding for the conversational medical assistant.

Design goals:
1. Fast local routing for common medical intents (no API call needed).
2. LLM is used only when the local router is genuinely unsure.
3. Field extraction is context-aware: a short "اه/لأ" is attached to the
   currently active question instead of being guessed against a long list.
4. No value is invented.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


DOMAIN_DESCRIPTIONS = {
    "heart_disease": "cardiovascular risk assessment or heart disease risk factors",
    "cbc": "interpretation of a complete blood count (CBC) result",
    "pathology": "interpretation of chemistry/lab results such as glucose, kidney, liver or cholesterol tests",
    "diabetes_risk_screening": "diabetes risk screening when the patient is asking whether they may be at risk",
    "hypertension_risk_screening": "hypertension risk screening when the patient is asking about risk of high blood pressure",
    "radiology": "interpretation of an imaging/radiology report such as X-ray, CT, MRI or ultrasound",
    "general_wellness": "general medical information, symptom discussion, medical terminology, prevention or health advice",
}

INTENT_DESCRIPTIONS = {
    "medical_information": "Explain a medical term, condition, medication concept, test meaning, or general health question.",
    "translation": "Translate text from the conversation or explain the same medical information in another language. This is a language task, not a request for a new medical answer.",
    "symptom_discussion": "Discuss symptoms in a conversational way without enough evidence that a specific diagnostic model is requested.",
    "lab_analysis": "Interpret actual laboratory values or a laboratory report uploaded/provided by the patient.",
    "radiology_analysis": "Interpret an actual imaging/radiology report or image.",
    "risk_screening": "The patient explicitly wants a risk assessment for diabetes, hypertension, or heart disease.",
    "other": "Other general health conversation.",
}


def _get_provider_and_key(api_key: str | None, provider: str | None) -> tuple[str | None, str | None]:
    if provider:
        return provider, api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        return ("openai" if api_key.startswith("sk-") else "gemini"), api_key
    if os.getenv("GEMINI_API_KEY"):
        return "gemini", os.getenv("GEMINI_API_KEY")
    if os.getenv("OPENAI_API_KEY"):
        return "openai", os.getenv("OPENAI_API_KEY")
    return None, None


def _call_llm_json(system_prompt: str, user_content: str, provider: str, api_key: str) -> dict | None:
    try:
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                system_instruction=system_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            text = model.generate_content(user_content).text.strip()
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=20.0)
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = (resp.choices[0].message.content or "{}").strip()
        else:
            return None
        return json.loads(text)
    except Exception as exc:
        print(f"NLU LLM call failed ({provider}): {exc}")
        return None


def classify_intent(message: str, api_key: str | None = None, provider: str | None = None) -> str:
    """Classify the *purpose* of a message, not merely its medical topic."""
    local = _classify_intent_fallback(message)
    if local != "other":
        return local

    provider, api_key = _get_provider_and_key(api_key, provider)
    if not provider or not api_key:
        return local

    prompt = (
        "Classify the patient's message into exactly one intent: "
        f"{json.dumps(INTENT_DESCRIPTIONS, ensure_ascii=False)}. "
        'Return JSON only: {"intent":"..."}. Do not infer a diagnosis.'
    )
    result = _call_llm_json(prompt, message, provider, api_key)
    intent = result.get("intent") if isinstance(result, dict) else None
    return intent if intent in INTENT_DESCRIPTIONS else local


def _classify_intent_fallback(message: str) -> str:
    t = _norm(message)
    if any(k in t for k in [
        "ترجم", "ترجمه", "ترجمة", "translate", "translation",
        "بالانجليزي", "بالإنجليزي", "بالانجليزية", "بالإنجليزية",
        "بالعربي", "بالعربية", "in english", "in arabic",
    ]):
        return "translation"
    if any(k in t for k in ["يعني ايه", "يعني إيه", "ايه معنى", "ما معنى", "اشرح", "شرح", "what is", "meaning of", "explain"]):
        return "medical_information"
    if any(k in t for k in ["تحليل", "cbc", "hemoglobin", "wbc", "platelet", "glucose", "hba1c", "creatinine", "alt", "ast", "cholesterol"]):
        if _looks_like_actual_result(t):
            return "lab_analysis"
    if any(k in t for k in ["اشعه", "الاشعة", "أشعة", "الأشعة", "x-ray", "xray", "radiology", "ct scan", "mri", "تقرير الاشعة", "تقرير الأشعة"]):
        return "radiology_analysis"
    if any(k in t for k in ["خطر السكر", "هل عندي سكر", "ممكن يكون عندي سكر", "خايفة يكون عندي السكر", "خايف يكون عندي السكر", "diabetes risk"]):
        return "risk_screening"
    if any(k in t for k in ["خطر الضغط", "هل عندي ضغط", "ممكن يكون عندي ضغط", "hypertension risk"]):
        return "risk_screening"
    if any(k in t for k in ["خطر القلب", "risk of heart disease", "heart disease risk"]):
        return "risk_screening"
    if any(k in t for k in ["عندي", "حاسس", "حاسة", "وجع", "ألم", "صداع", "دوخة", "تعب", "نهجان", "غثيان", "pain", "headache"]):
        return "symptom_discussion"

    # A user can simply type a condition/term ("diabetes", "headache", "hypertension").
    # Treat common medical topics as information requests instead of falling through
    # to the generic "other" path, which used to produce the "no source found" reply.
    medical_topic_terms = [
        "diabetes", "diabeties", "diabtes", "diabetes mellitus", "السكري", "السكر",
        "headache", "migraine", "صداع", "الشقيقة",
        "hypertension", "high blood pressure", "ضغط الدم", "الضغط",
        "anemia", "anaemia", "فقر الدم", "انيميا", "أنيميا",
        "asthma", "الربو", "cholesterol", "الكوليسترول",
        "heart attack", "heart disease", "ازمة قلبية", "أزمة قلبية", "امراض القلب", "أمراض القلب",
        "prediabetes", "ما قبل السكري",
        "cbc", "hemoglobin", "الهيموجلوبين",
    ]
    if any(term in t for term in medical_topic_terms):
        return "medical_information"

    return "other"


def classify_domain(message: str, api_key: str | None = None, provider: str | None = None) -> str | None:
    """Route the message to the diagnostic capability only when appropriate."""
    intent = classify_intent(message, api_key=api_key, provider=provider)
    if intent in {"medical_information", "translation", "symptom_discussion", "other"}:
        return "general_wellness"
    if intent == "radiology_analysis":
        return "radiology"
    if intent == "lab_analysis":
        t = _norm(message)
        if any(k in t for k in ["cbc", "hemoglobin", "wbc", "platelet", "mcv", "هيموجلوبين", "صفائح", "كرات الدم البيضاء"]):
            return "cbc"
        return "pathology"

    t = _norm(message)
    if any(k in t for k in ["سكر", "diabetes", "glucose", "hba1c"]):
        return "diabetes_risk_screening"
    if any(k in t for k in ["ضغط", "pressure", "hypertension"]):
        return "hypertension_risk_screening"
    if any(k in t for k in ["قلب", "heart", "chest pain", "ألم صدر"]):
        return "heart_disease"
    return "general_wellness"


def _looks_like_actual_result(t: str) -> bool:
    return bool(re.search(r"(?:^|\s)(?:\d+(?:[.,]\d+)?)(?:\s|$)", t)) or any(
        k in t for k in ["نتيجة", "طلع", "طلعلي", "التحليل عندي", "report", "my result", "my cbc"]
    )


def extract_fields(
    message: str,
    field_specs: list[dict[str, Any]],
    api_key: str | None = None,
    provider: str | None = None,
    active_field: str | None = None,
    active_prompt: str | None = None,
    conversation_context: str | None = None,
) -> dict[str, Any]:
    """Extract only fields supported by the current task.

    `active_field` is deliberately explicit. This fixes the classic chat bug
    where a reply such as "لا" was compared with the first item of a large
    safety-field list instead of the question the assistant actually asked.
    """
    if not field_specs:
        return {}

    provider, api_key = _get_provider_and_key(api_key, provider)
    if provider and api_key:
        fields_desc = [
            {"feature": f["feature"], "means": f.get("prompt", ""), "type": f.get("type", "text"), "valid_values": f.get("options")}
            for f in field_specs
        ]
        system_prompt = (
            "You extract structured values from a patient's message. The patient may use Egyptian Arabic, "
            "English, or mixed Franco-Arabic. Extract only values explicitly supported by the message. Never guess. "
            "A short yes/no answer MUST be assigned to active_field when active_field is provided. "
            "For choice fields, return one of the supplied exact values. For numbers, return numeric values. "
            "For text, preserve the patient's report text. Return JSON only.\n\n"
            f"Active question: {active_prompt or 'none'}\n"
            f"Active field: {active_field or 'none'}\n"
            f"Recent conversation: {conversation_context or 'none'}\n"
            f"Fields: {json.dumps(fields_desc, ensure_ascii=False)}"
        )
        result = _call_llm_json(system_prompt, message, provider, api_key)
        if isinstance(result, dict):
            valid = {f["feature"] for f in field_specs}
            cleaned = {k: v for k, v in result.items() if k in valid and v is not None}
            if cleaned:
                return cleaned

    return _extract_fields_fallback(message, field_specs, active_field=active_field)


_YES_WORDS = {"اه", "أه", "ايوه", "أيوه", "ايوة", "أيوة", "نعم", "yes", "true", "صح", "بالضبط"}
_NO_WORDS = {"لا", "لأ", "لأه", "لاا", "no", "false", "مش", "معنديش", "مفيش", "مش عندي", "كلا"}
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _normalize_arabic(text: str) -> str:
    return text.lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").strip()


def _is_yes(text: str) -> bool:
    t = _normalize_arabic(text)
    return t in {_normalize_arabic(x) for x in _YES_WORDS}


def _is_no(text: str) -> bool:
    t = _normalize_arabic(text)
    return t in {_normalize_arabic(x) for x in _NO_WORDS}


def _is_decline(text: str) -> bool:
    t = _normalize_arabic(text)
    return t in {
        "معرفش", "مش عارف", "مش عارفه", "مش عارفه", "مش عندي",
        "مفيش", "مش متوفر", "لا اعرف", "لا أعرف", "dont know", "don't know",
    }


def _extract_fields_fallback(message: str, field_specs: list[dict[str, Any]], active_field: str | None = None) -> dict[str, Any]:
    t = message.strip()
    if not field_specs:
        return {}

    by_feature = {f["feature"]: f for f in field_specs}
    if active_field and active_field in by_feature:
        f = by_feature[active_field]
        if f.get("type") == "bool":
            if _is_yes(t):
                return {active_field: True}
            if _is_no(t):
                return {active_field: False}
        if f.get("type") in {"number", "float", "int"}:
            m = _NUMBER_RE.search(t)
            if m:
                return {active_field: float(m.group().replace(",", "."))}
            if f.get("optional") and _is_decline(t):
                return {active_field: None}
        if f.get("type") == "text" and len(t) > 0:
            return {active_field: t}

    # Multiple explicit values in one message.
    out: dict[str, Any] = {}
    lower = _normalize_arabic(t)
    for f in field_specs:
        feature = f["feature"]
        if f.get("type") == "bool":
            # Only assign a bool when the field name/prompt is clearly mentioned.
            label = _normalize_arabic(f.get("prompt", ""))
            if label and any(token in lower for token in _important_tokens(label)):
                if _is_yes(t):
                    out[feature] = True
                elif _is_no(t):
                    out[feature] = False
        elif f.get("type") in {"number", "float", "int"}:
            m = _number_near_field(lower, feature)
            if m:
                out[feature] = float(m.replace(",", "."))
        elif f.get("type") == "choice" and f.get("options"):
            for label, value in f["options"].items():
                if _normalize_arabic(str(label)) in lower:
                    out[feature] = value
                    break
        elif f.get("type") == "text" and len(t) > 15:
            out[feature] = t
    if out:
        return out

    # If only one number field exists, it is safe to attach the number to it.
    numeric = [f for f in field_specs if f.get("type") in {"number", "float", "int"}]
    if len(numeric) == 1:
        m = _NUMBER_RE.search(t)
        if m:
            return {numeric[0]["feature"]: float(m.group().replace(",", "."))}

    return {}


def _important_tokens(prompt: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+|[\\u0600-\\u06ff]+", prompt)
    stop = {"هل", "لو", "عندك", "كام", "كان", "كانت", "في", "من", "ال", "او", "ولا", "ايه", "إيه"}
    return {w for w in words if len(w) >= 3 and w not in stop}


def _number_near_field(text: str, feature: str) -> str | None:
    aliases = {
        "age": ["عمري", "عمر", "age"],
        "height_cm": ["طولي", "طول", "height"],
        "weight_kg": ["وزني", "وزن", "weight"],
        "systolic_bp": ["الرقم الكبير", "انقباضي", "systolic", "ضغط"],
        "diastolic_bp": ["الرقم الصغير", "انبساطي", "diastolic"],
        "hemoglobin_g_dl": ["هيموجلوبين", "hemoglobin"],
        "wbc_10e3_ul": ["كرات الدم البيضاء", "wbc"],
        "platelets_10e3_ul": ["صفائح", "platelet"],
        "mcv_fl": ["mcv"],
        "neutrophils_pct": ["neutrophils", "العدلات"],
        "fasting_glucose": ["سكر صايم", "سكر الصيام", "fasting glucose", "glucose"],
        "hba1c": ["تراكمي", "hba1c"],
        "creatinine_mg_dl": ["كرياتينين", "creatinine"],
        "alt_u_l": ["alt"],
        "ast_u_l": ["ast"],
        "total_cholesterol": ["كوليسترول", "cholesterol"],
    }
    for alias in aliases.get(feature, [feature]):
        idx = text.find(_normalize_arabic(alias))
        if idx >= 0:
            m = _NUMBER_RE.search(text[idx:])
            if m:
                return m.group()
    return None


def _norm(text: str) -> str:
    return _normalize_arabic(str(text or "")).replace("ـ", "")
