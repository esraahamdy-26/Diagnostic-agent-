"""
Knowledge Base — RAG على مصادر مُنتقاة يدوياً (مش الإنترنت المفتوح)
========================================================================
ليه curated مش open-web crawler؟

1. **جودة وموثوقية**: أي حد يقدر ينشر موقع "طبي" بمعلومات غلط. مفيش ضمان
   جودة تلقائي على الإنترنت المفتوح، وفي سياق طبي ده خطر حقيقي.
2. **قابلية التدقيق**: هنا نعرف بالظبط أي مستند موجود في المكتبة ومين كتبه
   ومصدره (شوفي `data/knowledge_base/*.md`، كل ملف في الآخر مكتوب مصدره).
   تقدري تراجعيهم مع طبيب وتضيفي/تشيلي بسهولة.
3. **ثبات**: الإجابة معتمدة على نفس المصادر كل مرة، مش عرضة لتغيّر نتائج
   بحث جوجل من يوم لآخر.

**كيف يشتغل:**
    1. كل ملف .md في `data/knowledge_base/` بيتقسم لـ chunks (فقرات)
    2. كل chunk بياخد embedding (تمثيل رقمي للمعنى) عن طريق Gemini
       Embedding API (أو TF-IDF كـ fallback من غير API key)
    3. سؤال المريض بياخد embedding برضه، ونقارنه بكل الـ chunks (cosine
       similarity) ونرجع أقرب كام chunk (الأكثر صلة بالسؤال)
    4. الـ chunks دي بتتحط كـ context للـ LLM عشان يجاوب المريض **مبني على
       المصدر الفعلي**، مش من معرفته العامة وحدها

⚠️ لتوسيع المكتبة: زودي ملف .md جديد في `data/knowledge_base/` بنفس الأسلوب
(محتوى + قسم "المصادر" في الآخر)، والنظام هياخده تلقائياً من غير أي كود إضافي.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


# Small bilingual glossary used only for retrieval.  The answer itself is still
# generated from the retrieved trusted context.  This fixes a common failure
# mode where an English term such as "diabetes" cannot retrieve an Arabic KB
# article titled "السكري".
BILINGUAL_RETRIEVAL_ALIASES = {
    "diabetes": "السكري السكر diabetes diabeties diabtes mellitus prediabetes ما قبل السكري",
    "diabetes mellitus": "السكري السكر diabetes diabeties diabtes prediabetes ما قبل السكري",
    "diabeties": "السكري السكر diabetes diabeties",
    "diabtes": "السكري السكر diabetes diabtes",
    "prediabetes": "ما قبل السكري prediabetes diabetes",
    "headache": "الصداع صداع headache migraine الشقيقة",
    "migraine": "الشقيقة الصداع migraine headache",
    "hypertension": "ضغط الدم الضغط hypertension high blood pressure",
    "high blood pressure": "ضغط الدم الضغط hypertension",
    "anemia": "فقر الدم انيميا أنيميا anemia hemoglobin الهيموجلوبين",
    "anaemia": "فقر الدم انيميا أنيميا anemia hemoglobin الهيموجلوبين",
    "cbc": "تحليل صورة الدم الكاملة CBC hemoglobin الهيموجلوبين WBC الصفائح platelets",
    "hemoglobin": "الهيموجلوبين hemoglobin فقر الدم anemia",
    "cholesterol": "الكوليسترول cholesterol",
    "heart attack": "أزمة قلبية ازمة قلبية heart attack ألم الصدر",
    "heart disease": "أمراض القلب امراض القلب heart disease",
    "asthma": "الربو asthma",
    "stress": "التوتر stress الارهاق التعب",
}

def expand_retrieval_query(query: str) -> str:
    """Add bilingual aliases for common medical topics without changing the user text."""
    q = str(query or "").strip()
    lowered = q.lower()
    aliases = []
    for key, expansion in BILINGUAL_RETRIEVAL_ALIASES.items():
        if key in lowered:
            aliases.append(expansion)
    normalized = _normalize_for_lookup(q)
    for key, expansion in BILINGUAL_RETRIEVAL_ALIASES.items():
        if _normalize_for_lookup(key) in normalized:
            aliases.append(expansion)
    return q + (" " + " ".join(dict.fromkeys(aliases)) if aliases else "")

def _normalize_for_lookup(text: str) -> str:
    return (
        str(text or "").lower()
        .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        .replace("ى", "ي").replace("ـ", "")
    )


@dataclass
class Chunk:
    text: str
    source_file: str


def _load_and_chunk_documents() -> list[Chunk]:
    chunks = []
    if not KB_DIR.exists():
        return chunks
    for md_file in sorted(KB_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # تقسيم على العناوين الفرعية (##) — كل قسم chunk مستقل بمعنى متكامل
        sections = re.split(r"\n(?=## )", text)
        for section in sections:
            section = section.strip()
            if len(section) > 30:  # تجاهل فقرات فاضية/قصيرة جداً
                chunks.append(Chunk(text=section, source_file=md_file.stem))
    return chunks


class KnowledgeBase:
    def __init__(self, api_key: str | None = None, provider: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider or ("gemini" if os.getenv("GEMINI_API_KEY") else
                                      "openai" if os.getenv("OPENAI_API_KEY") else None)
        self.chunks = _load_and_chunk_documents()
        self._embeddings_cache: list | None = None
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

    def _embed_gemini(self, texts: list[str]) -> list:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        result = genai.embed_content(model="models/text-embedding-004", content=texts,
                                       task_type="retrieval_document")
        return result["embedding"]

    def _embed_query_gemini(self, query: str) -> list:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        result = genai.embed_content(model="models/text-embedding-004", content=query,
                                       task_type="retrieval_query")
        return result["embedding"]

    def _ensure_embeddings(self):
        if self._embeddings_cache is not None or not self.chunks:
            return
        if self.provider == "gemini":
            try:
                self._embeddings_cache = self._embed_gemini([c.text for c in self.chunks])
                return
            except Exception as e:
                print(f"KB embedding failed, falling back to TF-IDF: {e}")
        self._ensure_tfidf()

    def _ensure_tfidf(self):
        if self._tfidf_vectorizer is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf_vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """بترجع [(chunk, score), ...] — الـ score مهم عشان نعرف نفرّق بين
        'لقينا حاجة فعلاً مرتبطة' و'تطابق ضعيف جداً في كلمات ربط عامة'."""
        if not self.chunks:
            return []

        if self.provider == "gemini" and self.api_key:
            self._ensure_embeddings()
            if self._embeddings_cache is not None:
                import numpy as np
                query_emb = np.array(self._embed_query_gemini(query))
                doc_embs = np.array(self._embeddings_cache)
                sims = doc_embs @ query_emb / (
                    np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-9
                )
                top_idx = sims.argsort()[::-1][:top_k]
                return [(self.chunks[i], float(sims[i])) for i in top_idx]

        # Fallback: TF-IDF keyword-based similarity (من غير API key)
        self._ensure_tfidf()
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    # عتبات الثقة (calibrated تقريبياً — starter values محتاجة ضبط أدق مع
    # بيانات استخدام حقيقية). TF-IDF بيدّي درجات صغيرة حتى لتشابه ضعيف
    # (كلمات ربط عامة)، فمحتاج threshold أعلى نسبياً من الـ embeddings.
    MIN_SCORE_TFIDF = 0.12
    MIN_SCORE_EMBEDDING = 0.45

    def answer_with_context(self, query: str, top_k: int = 3, conversation_context: str = "") -> dict:
        """
        بترجع إجابة مبنية على الـ context المسترجع، مع ذكر المصادر بوضوح.
        لو أعلى score تحت العتبة، بنعتبر المكتبة المنقّاة "معندهاش حاجة
        فعلية مرتبطة" حتى لو رجّعت نتائج شكلياً — عشان نسيب المجال للمصدر
        الحي (MedlinePlus) بدل إجابة ضعيفة الصلة.
        """
        ranked = self.retrieve(expand_retrieval_query(query), top_k=top_k)
        threshold = self.MIN_SCORE_EMBEDDING if (self.provider == "gemini" and self.api_key) \
            else self.MIN_SCORE_TFIDF
        relevant = [chunk for chunk, score in ranked if score >= threshold]

        if not relevant:
            return {"answer": None, "sources": [], "context_used": []}

        context_text = "\n\n---\n\n".join(c.text for c in relevant)
        sources = sorted(set(c.source_file for c in relevant))

        if not self.provider or not self.api_key:
            # من غير LLM، نرجع الـ context الخام (المريض/الـ agent يقدر يستخدمه كـ reference)
            return {"answer": None, "sources": sources, "context_used": [c.text for c in relevant]}

        system_prompt = (
            "You are a warm, patient-friendly medical information assistant. "
            "Use the trusted context below as the factual source of your answer. "
            "Explain the topic in the SAME language style as the patient: if they write Arabic, "
            "answer in clear natural Egyptian Arabic; if English, answer in clear simple English; "
            "if they mix Arabic and English, you may naturally mix both. Keep important medical "
            "terms in English in parentheses when that helps understanding.\n\n"
            "The patient may be a non-doctor. Do not just copy the source: explain it simply and "
            "organize it into useful sections when appropriate, such as: what it means, common "
            "symptoms/signs, how doctors confirm it, what the person can do now, and when to seek "
            "urgent care. Do not invent facts that are not supported by the context. Do not claim "
            "that the patient has a disease based on symptoms alone. For treatment or medication "
            "questions, give general educational information and recommend clinician/pharmacist "
            "review rather than prescribing. If emergency warning signs are present in the context, "
            "state them clearly. If the user asks a follow-up like 'طيب أعمل إيه؟' use the recent "
            "conversation to understand what 'it' refers to.\n\n"
            "Always make clear, briefly, that this is educational information and not a diagnosis.\n\n"
            f"Recent conversation:\n{conversation_context or 'none'}\n\n"
            f"Trusted context:\n{context_text}"
        )
        try:
            if self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(model_name=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
                                               system_instruction=system_prompt)
                answer = model.generate_content(query).text.strip()
            else:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, timeout=20.0)
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": query}],
                )
                answer = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"KB answer generation failed: {e}")
            return {"answer": None, "sources": sources, "context_used": [c.text for c in relevant]}

        return {"answer": answer, "sources": sources, "context_used": [c.text for c in relevant]}


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"Loaded {len(kb.chunks)} chunks from {KB_DIR}")
    for c in kb.chunks[:3]:
        print(f"- [{c.source_file}] {c.text[:60]}...")

    print("\n=== اختبار retrieval (TF-IDF fallback من غير API key) ===")
    result = kb.answer_with_context("عندي صداع شديد جه فجأة زي ضربة، أعمل إيه؟")
    print("Sources:", result["sources"])
    for ctx in result["context_used"]:
        print("-", ctx[:100].replace("\n", " "))
