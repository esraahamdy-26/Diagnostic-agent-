from __future__ import annotations

import json
import os
import re
from typing import Any


def label_radiology_fallback(text: str) -> dict[str, Any]:
    """Fallback keyword-based extraction if LLM is unavailable.

    Uses the negation-aware engine in src/radiology_triage.py — NOT the naive
    3-phrase negation list this function used to have. See CHANGELOG.md:
    the old version misclassified ~42% of reports as urgent_review, almost
    all due to missed negation phrasings (e.g. "no evidence of pneumothorax").
    """
    from src.radiology_triage import evaluate_report

    result = evaluate_report(text)
    urgency = result.urgency

    if urgency == "urgent_review":
        findings = [f"Positive finding (not negated): {t}" for t in result.matched_terms]
        justification = f"Non-negated urgent terms found: {result.matched_terms}"
        explanation_eg = "تقرير الأشعة بتاعك فيه علامات محتاجة مراجعة طبية فورا من الدكتور. يرجى التواصل مع الطبيب المعالج أو التوجه للطوارئ لو حاسس بتعب شديد."
    elif urgency == "follow_up":
        findings = [f"Non-negated follow-up finding: {t}" for t in result.matched_terms]
        justification = f"Non-negated follow-up terms found: {result.matched_terms}"
        explanation_eg = "تقرير الأشعة بتاعك فيه شوية ملاحظات محتاجة متابعة مع الدكتور قريب بس مفيش قلق فوري. احجز ميعاد قريب مع دكتورك يطمنك."
    else:
        findings = ["No non-negated abnormal findings detected."]
        justification = "No urgent or follow-up terms matched (after negation filtering)."
        explanation_eg = "الحمد لله، تقرير الأشعة بتاعك سليم ومفيهوش أي علامات خطر أو قلق واضحة."

    return {
        "urgency": urgency,
        "findings": findings,
        "justification": justification,
        "explanation_eg": explanation_eg,
        "negated_terms_found": result.negated_terms,
    }


def extract_radiology_triage(
    report_text: str,
    api_key: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """LLM-based structured clinical extractor for radiology reports.

    Extracts:
    - urgency: 'urgent_review', 'follow_up', or 'routine'
    - findings: list of key clinical findings
    - justification: clinical rationale for the triage urgency level
    - explanation_eg: patient-friendly summary in Egyptian Arabic (العامية المصرية)
    """
    if not provider:
        if api_key:
            provider = "openai" if api_key.startswith("sk-") else "gemini"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
            api_key = os.getenv("GEMINI_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            return label_radiology_fallback(report_text)

    system_prompt = (
        "You are an expert clinical radiology classification and triage assistant.\n"
        "Your task is to analyze the provided chest X-ray/radiology report text and extract a structured JSON response.\n"
        "The JSON object MUST contain exactly the following keys:\n"
        "1. \"urgency\": A string that is exactly one of: \"urgent_review\", \"follow_up\", or \"routine\".\n"
        "   - \"urgent_review\": for acute conditions requiring immediate attention (e.g., active pneumothorax, pulmonary edema, pleural effusion with mediastinal shift, large consolidation, acute collapse).\n"
        "   - \"follow_up\": for abnormal but non-acute findings requiring doctor follow-up (e.g., opacity, mass, nodule, cardiomegaly, emphysema, atelectasis, mild pleural effusion).\n"
        "   - \"routine\": for normal, clear, or stable reports with no acute or follow-up findings.\n"
        "2. \"findings\": A list of strings representing the key medical findings identified in the report.\n"
        "3. \"justification\": A brief string explaining the clinical rationale for your urgency classification.\n"
        "4. \"explanation_eg\": A patient-friendly explanation in friendly Egyptian Arabic (العامية المصرية) explaining what the report means and the next steps in a comforting but clear tone. Do not use Modern Standard Arabic (الفصحى), use conversational Egyptian Arabic.\n\n"
        "Format the output strictly as a JSON object, with no other text."
    )

    try:
        if provider == "gemini":
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-3.6-flash",
                system_instruction=system_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(report_text)
            output_text = response.text.strip()
        
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": report_text}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            output_text = response.choices[0].message.content.strip()
        else:
            return label_radiology_fallback(report_text)

        # Parse JSON
        data = json.loads(output_text)
        
        # Verify required keys
        required_keys = {"urgency", "findings", "justification", "explanation_eg"}
        if all(k in data for k in required_keys):
            # Validate urgency class
            if data["urgency"] not in {"urgent_review", "follow_up", "routine"}:
                data["urgency"] = "follow_up"
            return data
            
        return label_radiology_fallback(report_text)

    except Exception as e:
        print(f"Error calling LLM radiology extractor ({provider}): {e}. Falling back to rule-based.")
        return label_radiology_fallback(report_text)
