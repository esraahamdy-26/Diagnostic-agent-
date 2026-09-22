from __future__ import annotations

import json
import os
from typing import Any


def route_request_fallback(message: str, uploaded_payload: dict[str, Any]) -> str:
    """Keyword-based fallback router if LLM is unavailable or fails."""
    text = message.lower()
    if "cbc" in text or "hemoglobin" in text or "wbc" in text or "platelet" in text or "hemoglobin_g_dl" in uploaded_payload:
        return "cbc"
    if "xray" in text or "ct" in text or "mri" in text or "radiology" in text or "report_text" in uploaded_payload:
        return "radiology"
    if "heart" in text or "chest pain" in text or "cholesterol" in uploaded_payload or "trestbps" in uploaded_payload:
        return "heart_disease"
    if "diabetes" in text or "hba1c" in uploaded_payload or "glucose" in text or "fasting_glucose" in uploaded_payload:
        return "diabetes"
    if "pressure" in text or "hypertension" in text or "systolic_bp" in uploaded_payload or "diastolic_bp" in uploaded_payload:
        return "hypertension"
    return "pathology"


def route_request_llm(
    message: str,
    uploaded_payload: dict[str, Any],
    api_key: str | None = None,
    provider: str | None = None,
) -> str:
    """Smart semantic router using LLMs (Gemini or OpenAI).

    Routes the request to one of the 6 medical classification models:
    - 'cbc': Complete Blood Count values
    - 'pathology': Chemistry panel (creatinine, AST/ALT, fasting glucose, lipids)
    - 'radiology': Radiology imaging reports (Chest X-ray, CT, MRI)
    - 'heart_disease': Cardiovascular risk indicators
    - 'diabetes': Glucose and HbA1c metabolic risk
    - 'hypertension': Blood pressure risk

    Falls back to keyword-based routing if API keys are missing or calls fail.
    """
    # Auto-detect provider based on environment variables if not specified
    if not provider:
        if api_key:
            # If key is provided but not provider, try to detect shape
            provider = "openai" if api_key.startswith("sk-") else "gemini"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
            api_key = os.getenv("GEMINI_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            return route_request_fallback(message, uploaded_payload)

    # Prompt design for routing
    system_prompt = (
        "You are an expert clinical routing agent. Your job is to classify a patient's query and their uploaded data "
        "into the most appropriate clinical triage model category. The available categories are:\n"
        "1. 'cbc' - Select this if the query or data mentions blood counts (hemoglobin, platelets, red/white blood cells, MCV, hematocrit, CBC).\n"
        "2. 'radiology' - Select this if the query or data contains imaging report text (X-ray, CT scan, MRI, ultrasound, radiography).\n"
        "3. 'heart_disease' - Select this if the query or data refers to cardiovascular indicators, chest pain, ECG changes, or cholesterol panel (e.g. LDL, HDL, total cholesterol, trestbps).\n"
        "4. 'diabetes' - Select this if the query or data refers to blood sugar, HbA1c, fasting glucose, or diabetes risk.\n"
        "5. 'hypertension' - Select this if the query or data relates to blood pressure readings (systolic, diastolic) or hypertension risk.\n"
        "6. 'pathology' - Select this for other general chemistry panels, kidney/liver function tests (creatinine, AST, ALT, BMI, lipids) not specific to only blood pressure or glucose.\n\n"
        "Respond ONLY with the category name (one of: cbc, pathology, radiology, heart_disease, diabetes, hypertension) in lowercase, "
        "with absolutely no other words, punctuation, or formatting."
    )

    user_content = f"User Message: {message}\nUploaded Data JSON: {json.dumps(uploaded_payload)}"

    try:
        if provider == "gemini":
            import google.generativeai as genai
            if api_key:
                genai.configure(api_key=api_key)
            # Use gemini-3.6-flash as default fast and reliable model
            model = genai.GenerativeModel(
                model_name="gemini-3.6-flash",
                system_instruction=system_prompt
            )
            response = model.generate_content(user_content)
            category = response.text.strip().lower()
        
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Use gpt-4o-mini as default fast and cheap model
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                max_tokens=10
            )
            category = response.choices[0].message.content.strip().lower()
        else:
            return route_request_fallback(message, uploaded_payload)

        # Validate category matches one of the expected options
        valid_categories = {"cbc", "pathology", "radiology", "heart_disease", "diabetes", "hypertension"}
        if category in valid_categories:
            return category
        
        # If the model output doesn't match perfectly, check if it's a substring
        for cat in valid_categories:
            if cat in category:
                return cat
                
        return route_request_fallback(message, uploaded_payload)

    except Exception as e:
        print(f"Error calling LLM router ({provider}): {e}. Falling back to rule-based routing.")
        return route_request_fallback(message, uploaded_payload)
