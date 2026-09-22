"""
Lab Triage Rules — CBC & Pathology
=====================================
ليه مفيش ML هنا خالص؟

الموديل الأصلي كان بيتدرب على labels اتحسبت بنفس القواعد دي بالظبط من نفس القيم
اللي بتتغذّى للموديل — يعني عمرها ما كانت مشكلة "تحتاج تعلّم"، كانت أصلاً محلولة.

لما تكون القاعدة الطبية معروفة وموثقة (critical lab values هي حاجة معرّفة في
الطب من عقود، مش حاجة "نتعلمها" من بيانات)، والمدخلات كلها متاحة وقت القرار
(التحليل بيرجع كل القيم مع بعض)، فالحل الصح هو **rule engine شفاف 100%**:
- مفيش احتمال false negative بسبب "الموديل ما اتدربش كويس على أمثلة قليلة"
- كل قرار قابل للتفسير والمراجعة سطر بسطر
- مفيش "دقة 100% مزيفة" — القواعد دقيقة لأنها مطبّقة مباشرة، مش متعلّمة

⚠️ العتبات هنا starter template من الكود الأصلي، ولازم تتراجع من طبيب باثولوجي/
معملي قبل أي استخدام حقيقي — تماماً زي تحذير `safety_layer.py`.
"""

from dataclasses import dataclass
from enum import Enum


class TriageLevel(Enum):
    URGENT_REVIEW = "urgent_review"
    FOLLOW_UP = "follow_up"
    WITHIN_REFERENCE = "within_reference"


@dataclass
class TriageResult:
    level: TriageLevel
    triggered_rules: list[str]

    @property
    def triage_action(self) -> str:
        return {
            TriageLevel.URGENT_REVIEW: "recommend_doctor_booking",
            TriageLevel.FOLLOW_UP: "ask_follow_up_questions",
            TriageLevel.WITHIN_REFERENCE: "reassure_and_explain",
        }[self.level]


class CBCTriage:
    """تصنيف نتيجة تحليل صورة الدم الكاملة (CBC)."""

    def evaluate(self, labs: dict) -> TriageResult:
        urgent_checks = [
            (labs.get("hemoglobin_g_dl", 999) < 8, "Hemoglobin < 8 g/dL (severe anemia)"),
            (labs.get("wbc_10e3_ul", 5) < 2, "WBC < 2 (severe leukopenia)"),
            (labs.get("wbc_10e3_ul", 5) > 20, "WBC > 20 (severe leukocytosis)"),
            (labs.get("platelets_10e3_ul", 250) < 50, "Platelets < 50 (severe thrombocytopenia)"),
            (labs.get("platelets_10e3_ul", 250) > 800, "Platelets > 800 (severe thrombocytosis)"),
        ]
        follow_up_checks = [
            (labs.get("hemoglobin_g_dl", 999) < 11, "Hemoglobin < 11 g/dL (mild-moderate anemia)"),
            (labs.get("wbc_10e3_ul", 5) < 3.5, "WBC < 3.5 (mild leukopenia)"),
            (labs.get("wbc_10e3_ul", 5) > 12, "WBC > 12 (mild leukocytosis)"),
            (labs.get("platelets_10e3_ul", 250) < 150, "Platelets < 150 (mild thrombocytopenia)"),
            (labs.get("platelets_10e3_ul", 250) > 450, "Platelets > 450 (mild thrombocytosis)"),
            (labs.get("mcv_fl", 90) < 78, "MCV < 78 (microcytosis)"),
            (labs.get("neutrophils_pct", 50) > 80, "Neutrophils > 80% (neutrophilia)"),
        ]
        return self._resolve(urgent_checks, follow_up_checks)

    @staticmethod
    def _resolve(urgent_checks, follow_up_checks) -> TriageResult:
        urgent_hits = [msg for hit, msg in urgent_checks if hit]
        if urgent_hits:
            return TriageResult(TriageLevel.URGENT_REVIEW, urgent_hits)
        follow_hits = [msg for hit, msg in follow_up_checks if hit]
        if follow_hits:
            return TriageResult(TriageLevel.FOLLOW_UP, follow_hits)
        return TriageResult(TriageLevel.WITHIN_REFERENCE, [])


class PathologyTriage:
    """تصنيف نتائج تحاليل الكيمياء الحيوية (سكر، كوليسترول، وظائف كبد/كلى)."""

    def evaluate(self, labs: dict) -> TriageResult:
        urgent_checks = [
            (labs.get("fasting_glucose", 0) >= 250, "Fasting glucose >= 250 (severe hyperglycemia)"),
            (labs.get("hba1c", 0) >= 10, "HbA1c >= 10 (very poor glycemic control)"),
            (labs.get("creatinine_mg_dl", 0) >= 2, "Creatinine >= 2 (significant renal impairment)"),
            (labs.get("alt_u_l", 0) >= 200, "ALT >= 200 (marked liver enzyme elevation)"),
            (labs.get("ast_u_l", 0) >= 200, "AST >= 200 (marked liver enzyme elevation)"),
            (labs.get("total_cholesterol", 0) >= 320, "Total cholesterol >= 320 (severe hypercholesterolemia)"),
        ]
        follow_up_checks = [
            (labs.get("fasting_glucose", 0) >= 126, "Fasting glucose >= 126 (diabetes range)"),
            (labs.get("hba1c", 0) >= 6.5, "HbA1c >= 6.5 (diabetes range)"),
            (labs.get("creatinine_mg_dl", 0) >= 1.3, "Creatinine >= 1.3 (mild renal impairment)"),
            (labs.get("alt_u_l", 0) >= 45, "ALT >= 45 (mild liver enzyme elevation)"),
            (labs.get("ast_u_l", 0) >= 45, "AST >= 45 (mild liver enzyme elevation)"),
            (labs.get("total_cholesterol", 0) >= 240, "Total cholesterol >= 240 (high)"),
        ]
        return CBCTriage._resolve(urgent_checks, follow_up_checks)


if __name__ == "__main__":
    cbc = CBCTriage()
    path = PathologyTriage()

    print("=== CBC: حالة طبيعية ===")
    r = cbc.evaluate({"hemoglobin_g_dl": 13.5, "wbc_10e3_ul": 6.0,
                       "platelets_10e3_ul": 260, "mcv_fl": 88, "neutrophils_pct": 55})
    print(r.level.value, "-", r.triage_action)

    print("\n=== CBC: حالة حرجة (فقر دم شديد) ===")
    r = cbc.evaluate({"hemoglobin_g_dl": 6.5, "wbc_10e3_ul": 6.0, "platelets_10e3_ul": 260})
    print(r.level.value, "-", r.triage_action, "-", r.triggered_rules)

    print("\n=== Pathology: حالة سكر غير منضبط بشدة ===")
    r = path.evaluate({"fasting_glucose": 280, "hba1c": 8.0, "creatinine_mg_dl": 0.9,
                        "alt_u_l": 30, "ast_u_l": 28, "total_cholesterol": 210})
    print(r.level.value, "-", r.triage_action, "-", r.triggered_rules)
