"""
Vision Extraction — قراءة صور التحاليل والتقارير
====================================================
بتحوّل صورة (تحليل دم متصور بالموبايل، تقرير أشعة PDF/صورة) لنص خام، وبعدين
نفس نص ده بيتغذّى لنفس `src/nlu.py` اللي بنيناها قبل كده — يعني مفيش pipeline
تاني منفصل، الصورة بس "مصدر تاني للنص" بيدخل على نفس الطريق.

⚠️ محتاجة API key حقيقي (Gemini بالذات بيدعم الصور مجاناً في الـ free tier) —
مفيش fallback هنا، لأن استخراج نص من صورة من غير vision model مستحيل عملياً.
"""

from __future__ import annotations

import base64
import os


class VisionExtractionError(Exception):
    pass


def extract_text_from_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    api_key: str | None = None,
    provider: str | None = None,
) -> str:
    """
    بترجع النص/القيم الظاهرة في الصورة كنص عادي (مش JSON) — بعد كده بيتبعت
    لـ `nlu.extract_fields()` بالظبط زي أي رسالة حرة تانية.
    """
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") or (api_key and not api_key.startswith("sk-")) else "openai"
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise VisionExtractionError(
            "محتاجة API key (GEMINI_API_KEY أو OPENAI_API_KEY) عشان تقدري تقري صور. "
            "من غيره مفيش طريقة تانية تستخرج بيها نص من صورة."
        )

    prompt = (
        "This is a photo or scan of a medical document (lab report, prescription, or radiology "
        "report). Transcribe ALL visible text and numeric values exactly as written, including "
        "units. If it's a lab report, list each test name with its value and reference range if "
        "shown. Output plain text only, no commentary, no markdown formatting."
    )

    try:
        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3.6-flash")
            image_part = {"mime_type": mime_type, "data": image_bytes}
            response = model.generate_content([prompt, image_part])
            return response.text.strip()

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ],
                }],
            )
            return response.choices[0].message.content.strip()

        raise VisionExtractionError(f"Unknown provider: {provider}")

    except VisionExtractionError:
        raise
    except Exception as e:
        raise VisionExtractionError(f"Vision extraction failed: {e}") from e
