"""
Radiology Report Triage — Negation-Aware Rule Engine
========================================================
ليه مش DL هنا برضه؟

نفس مشكلة CBC/pathology بالظبط: الـ label الأصلي (`label_radiology`) اتحسب
بـ keyword matching على نفس النص اللي كان بيتغذّى للموديل (TF-IDF + Random
Forest). يعني الموديل مكنش بيتعلم طب، كان بيحاول يعيد اكتشاف نفس الـ keyword
list. تدريب DL أعمق (LSTM/Transformer) على نفس الـ labels **مش هيصلّح** المشكلة
— هيبقى بس نسخة أعقد من نفس الخطأ (وأصعب تفسيراً للطبيب كمان).

المشكلة الحقيقية اللي محتاجة حل: الـ negation detection في النظام الحالي
(`radiology_extractor.py::label_radiology_fallback`) كانت بدائية جداً —
قائمة يدوية بـ 3 صيغ نفي بس (`no X`, `without X`, `no acute`). فحص فعلي على
الداتا (data/real_processed/iu_xray_reports_real.csv) لقى **703 من أصل 2955
تقرير (24%)** فيهم صيغة نفي مختلفة (`no evidence of X`, `negative for X`,
`no definitive X`...) هتتفوت وتتصنف Urgent غلط.

الحل هنا: NegEx-lite — negation scope detection حوالين كل مصطلح طبي، بدل
substring matching بدائي. ده تحسين حقيقي وقابل للقياس، وبيفضل شفاف 100%
(قابل للمراجعة سطر بسطر من طبيب أشعة) — عكس أي "تعلّم" غامض جوه شبكة عصبية.

⚠️ starter template — قائمة المصطلحات وصيغ النفي لازم تتراجع من طبيب أشعة.
"""

import re
from dataclasses import dataclass


# صيغ النفي (negation triggers) — قبل المصطلح الطبي
PRE_NEGATION_TRIGGERS = [
    "no evidence of", "no evidence for", "no signs of", "no findings of",
    "no definitive", "no significant", "no acute", "negative for",
    "free of", "free from", "absence of", "absent", "without evidence of",
    "without", "no ", "not ",
]

# صيغ النفي بعد المصطلح الطبي (زي "pneumothorax is not seen")
POST_NEGATION_TRIGGERS = [
    "is not seen", "is not identified", "is not present", "not appreciated",
    "was not identified", "resolved", "resolution of",
]

# صيغ "شك/عدم يقين" — دي المهم إننا متعتبرهاش نفي، لأنها لسه محتاجة مراجعة
# طبيب (عكس negation الحقيقي)
UNCERTAINTY_MARKERS = ["cannot exclude", "cannot rule out", "cannot be excluded",
                       "concerning for", "suspicious for", "possible", "questionable"]

URGENT_TERMS = ["pneumothorax", "pulmonary edema", "acute", "collapse",
                "mediastinal shift", "large pleural effusion", "consolidation"]
FOLLOW_UP_TERMS = ["opacity", "nodule", "mass", "effusion", "atelectasis",
                    "cardiomegaly", "emphysema", "fibrosis", "granuloma",
                    "hernia", "infiltrate"]

WINDOW_CHARS = 45  # نافذة البحث عن سياق النفي حوالين المصطلح (تقريباً 6-7 كلمات)


@dataclass
class RadiologyTriageResult:
    urgency: str  # urgent_review | follow_up | routine
    matched_terms: list[str]
    negated_terms: list[str]

    @property
    def triage_action(self) -> str:
        return {
            "urgent_review": "recommend_doctor_booking",
            "follow_up": "ask_follow_up_questions",
            "routine": "reassure_and_explain",
        }[self.urgency]


def _split_sentences(text: str) -> list[str]:
    # تقسيم بسيط على علامات الوقف، عشان نطاق النفي (negation scope) يشمل
    # الجملة كلها -- مهم جداً لصيغ زي "no evidence of A, B, or C" حيث A/B/C
    # ممكن يكونوا بعيدين بالحروف عن "no evidence of" في حالة تعداد طويل.
    return re.split(r"[.;]", text)


def _term_is_negated(text: str, term: str) -> bool:
    """
    بتفحص كل ظهور للمصطلح جوه النص، وتشوف لو محاط بصيغة نفي (قبله أو بعده)
    *في نفس الجملة*، من غير ما تقع في فخ صيغ الشك (uncertainty) اللي مش نفي
    (زي "cannot exclude" -- ده لسه لازم يتعتبر إيجابي/محتاج مراجعة).
    """
    sentences = _split_sentences(text)
    any_occurrence = False

    for sentence in sentences:
        if term not in sentence:
            continue
        for match in re.finditer(re.escape(term), sentence):
            any_occurrence = True
            start, end = match.span()
            before = sentence[:start]                       # الجملة كلها قبل المصطلح
            after = sentence[end:end + WINDOW_CHARS]         # نافذة محدودة بعده

            if any(marker in before for marker in UNCERTAINTY_MARKERS):
                return False  # شك = لسه إيجابي (محتاج مراجعة)، مش نفي

            if any(trigger in before for trigger in PRE_NEGATION_TRIGGERS):
                continue  # negated في الجملة دي -> نفحص ظهورات تانية

            if any(trigger in after for trigger in POST_NEGATION_TRIGGERS):
                continue

            return False  # ظهور حقيقي غير منفي -> المصطلح موجود فعلاً

    return any_occurrence or False  # لو مفيش ظهور خالص، برضه مش negated (مش موجود أصلاً)


def evaluate_report(report_text: str) -> RadiologyTriageResult:
    text = str(report_text).lower()

    matched_urgent = [t for t in URGENT_TERMS if t in text and not _term_is_negated(text, t)]
    if matched_urgent:
        negated = [t for t in URGENT_TERMS if t in text and _term_is_negated(text, t)]
        return RadiologyTriageResult("urgent_review", matched_urgent, negated)

    matched_follow = [t for t in FOLLOW_UP_TERMS if t in text and not _term_is_negated(text, t)]
    if matched_follow:
        negated = [t for t in FOLLOW_UP_TERMS if t in text and _term_is_negated(text, t)]
        return RadiologyTriageResult("follow_up", matched_follow, negated)

    return RadiologyTriageResult("routine", [], [])


if __name__ == "__main__":
    test_cases = [
        ("Large pneumothorax with mediastinal shift, needs urgent evaluation.", "urgent_review"),
        ("The lungs are clear bilaterally. No evidence of focal consolidation, "
         "pneumothorax, or pleural effusion.", "routine"),
        ("Negative for pneumothorax. Trachea is midline.", "routine"),
        ("Cannot exclude small pneumothorax at the apex, recommend follow-up CT.", "urgent_review"),
        ("Mild cardiomegaly, otherwise unremarkable.", "follow_up"),
    ]
    for text, expected in test_cases:
        result = evaluate_report(text)
        status = "✅" if result.urgency == expected else "❌"
        print(f"{status} expected={expected:15s} got={result.urgency:15s} | {text[:70]}")
