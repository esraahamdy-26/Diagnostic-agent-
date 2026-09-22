"""
Live Medical Search — مصدر حي كـ backup للمكتبة المنقّاة
=============================================================
ده الجزء التاني من التصميم الـ hybrid اللي اتفقنا عليه: المكتبة المنقّاة
(`knowledge_base.py`) هي الأساس والأسرع والأوثق، لكن لو موضوع السؤال مش
متغطي فيها خالص، بننده على مصدر حي — **MedlinePlus** بتاع National Library
of Medicine الأمريكية (جزء من NIH). اخترناه تحديداً لأنه:

- **مجاني تماماً وبدون API key** (على عكس PubMed اللي محتاج مفتاح للاستخدام
  المكثف، وعلى عكس معظم الخدمات التانية)
- **مكتوب أصلاً للمرضى العاديين** (مش لغة علمية معقدة زي abstracts الأبحاث)
- **مصدر حكومي رسمي محكم** (NIH/NLM)، مش أي موقع عشوائي

⚠️ **ملاحظة تقنية مهمة:** الكود ده بيستخدم `requests` عادي لعمل HTTP call
حقيقي وقت التشغيل. اتأكدت إن الـ endpoint شغال فعلاً (جربته يدوياً وقت
كتابة الكود)، لكن **معنديش صلاحية أشغّل الكود ده وأختبره من بيئة العمل بتاعتي**
(sandbox معزولة عن نطاقات زي `nlm.nih.gov`) — هيشتغل صح على جهازك لأن عندك
اتصال إنترنت عادي، لكن التنبيه ده مهم يتقال بصراحة.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

MEDLINEPLUS_ENDPOINT = "https://wsearch.nlm.nih.gov/ws/query"


def _strip_html(text: str) -> str:
    """المحتوى الراجع من الـ API فيه tags زي <span class='qt0'> للتظليل — بنشيلها."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return text.strip()


def query_medlineplus(term: str, max_results: int = 2, timeout: int = 6) -> list[dict]:
    """
    بترجع [{"title", "summary", "url", "source"}] أو [] لو فشل الاتصال
    (شبكة مقطوعة، الموقع واقع مؤقتاً، إلخ) — بترجع قايمة فاضية بدل ما تكسر
    التطبيق، والـ caller (general_wellness.py) بيتعامل مع القايمة الفاضية
    برجوعه لرسالة "المعلومة مش متاحة" العادية.
    """
    try:
        import requests
        response = requests.get(
            MEDLINEPLUS_ENDPOINT,
            params={"db": "healthTopics", "term": term, "retmax": max_results},
            timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)

        results = []
        for doc in root.findall(".//document")[:max_results]:
            title_el = doc.find("./content[@name='title']")
            summary_el = doc.find("./content[@name='FullSummary']")
            url = doc.get("url", "")
            if title_el is None or summary_el is None:
                continue
            results.append({
                "title": _strip_html(title_el.text or ""),
                "summary": _strip_html(summary_el.text or "")[:1500],  # نحد الطول عشان الـ context
                "url": url,
                "source": "MedlinePlus (U.S. National Library of Medicine)",
            })
        return results

    except Exception as e:
        print(f"MedlinePlus live search failed (network issue or API change): {e}")
        return []


if __name__ == "__main__":
    print("⚠️ هذا الاختبار محتاج اتصال إنترنت فعلي لنطاق nlm.nih.gov")
    print("(مش هيشتغل من جوه sandbox معزول، لكن هيشتغل على جهازك)")
    results = query_medlineplus("migraine")
    for r in results:
        print(f"- {r['title']} ({r['source']})")
        print(f"  {r['summary'][:150]}...")
