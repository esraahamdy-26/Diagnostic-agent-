"""
Adaptive Questioners
======================
كلاستين مختلفتين لنوعين مختلفين من الأسئلة (شوفي docstring بتاع question_banks.py):

- RiskFactorQuestioner: للموديلات اللي بتتنبأ من عوامل خطر (heart_disease,
  diabetes_risk_screening, hypertension_risk_screening). الترتيب ديناميكي
  حسب أهمية الـ feature في الموديل المتدرب فعلياً.

- LabPanelQuestioner: للموديلات اللي بتصنّف نتيجة تحليل فعلي (cbc, pathology).
  الترتيب ثابت، القيم المحدِّدة لـ "urgent" الأول.
"""

import joblib
import numpy as np

from src.question_banks import BMI_PAIR_QUESTIONS, LAB_PANEL_BANKS, RISK_FACTOR_BANKS


class RiskFactorQuestioner:
    def __init__(self, domain: str, model_path: str, always_ask: set[str] | None = None,
                 min_features: int | None = None):
        self.domain = domain
        self.bank = RISK_FACTOR_BANKS[domain]
        self.model = joblib.load(model_path)
        self.always_ask = always_ask or set()
        self.min_features = min_features if min_features is not None else max(1, len(self.bank) - 2)
        self.importance = self._get_importance()
        self.declined: set[str] = set()

    def decline(self, feature: str) -> None:
        self.declined.add(feature)

    def _get_importance(self) -> dict:
        """
        بتشتغل مع أي نوع موديل: tree-based (feature_importances_) أو
        linear (coef_, بناخد القيمة المطلقة). لو الموديل جوه Pipeline
        (زي عندنا)، بنستخرج الخطوة الأخيرة.
        """
        estimator = self.model
        if hasattr(estimator, "named_steps"):
            estimator = estimator.named_steps.get("model", estimator)

        feature_names = list(self.bank.keys())
        try:
            if hasattr(estimator, "feature_importances_"):
                raw = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                raw = np.abs(estimator.coef_[0])
            else:
                raw = [1.0] * len(feature_names)
            # لو عدد الـ features في الموديل (بعد الـ preprocessing) مختلف عن
            # طول القائمة هنا (زي one-hot)، منقدرش نطابقهم بأمان -> أهمية متساوية
            if len(raw) != len(feature_names):
                raw = [1.0] * len(feature_names)
        except Exception:
            raw = [1.0] * len(feature_names)

        return dict(zip(feature_names, raw))

    def enrich_derived(self, known: dict) -> dict:
        if "bmi" not in known and "height_cm" in known and "weight_kg" in known:
            h, w = known["height_cm"], known["weight_kg"]
            known["bmi"] = round(w / ((h / 100) ** 2), 1)
        return known

    def next_question(self, known: dict):
        known = self.enrich_derived(known)

        # لو bmi لسه ناقصة، نسأل عن الطول/الوزن الأول (بترتيب الأهمية لو فيهم فرق)
        if "bmi" in self.bank and "bmi" not in known:
            for sub_field in ("height_cm", "weight_kg"):
                if sub_field not in known:
                    q = BMI_PAIR_QUESTIONS[sub_field]
                    return {"feature": sub_field, "prompt": q["prompt"], "type": q["type"], "options": None}

        missing = [f for f in self.bank if f not in known and f != "bmi" and f not in self.declined]
        if not missing:
            return None

        missing.sort(key=lambda f: (f not in self.always_ask, -self.importance.get(f, 0)))
        top = missing[0]
        q = self.bank[top]
        return {"feature": top, "prompt": q["prompt"], "type": q["type"], "options": q.get("options"), "optional": q.get("optional", False)}

    def enough_info(self, known: dict) -> bool:
        known = self.enrich_derived(known)
        required_ok = all(f in known or f == "bmi" for f in self.always_ask)
        # الأسئلة الاختيارية (زي optional creatinine) متتحسبش ضد الـ threshold
        optional = {f for f, spec in self.bank.items() if isinstance(spec, dict) and spec.get("optional")}
        countable = [f for f in self.bank if f not in optional]
        available = [f for f in countable if f in known or (f == "bmi" and "bmi" in known)]
        return required_ok and len(available) >= min(self.min_features, len(countable))

    def remaining_questions(self, known: dict, limit: int = 6) -> list[dict]:
        """كل الأسئلة الناقصة (مش بس الأهم) — تُستخدم في الاستخراج من نص حر
        عشان الـ LLM يقدر يلقط كذا إجابة من رسالة واحدة."""
        known = self.enrich_derived(known)
        out = []
        if "bmi" in self.bank and "bmi" not in known:
            for sub_field in ("height_cm", "weight_kg"):
                if sub_field not in known:
                    q = BMI_PAIR_QUESTIONS[sub_field]
                    out.append({"feature": sub_field, "prompt": q["prompt"], "type": q["type"], "options": None})
        for f in self.bank:
            if f == "bmi" or f in known or f in self.declined:
                continue
            q = self.bank[f]
            out.append({"feature": f, "prompt": q["prompt"], "type": q["type"], "options": q.get("options"), "optional": q.get("optional", False)})
        return out[:limit]


class LabPanelQuestioner:
    def __init__(self, domain: str):
        self.domain = domain
        self.fields = LAB_PANEL_BANKS[domain]
        # لازم نفرّق بين "لسه ما سألناش" و"سألنا والمريض رفض/معندوش قيمة"،
        # وإلا لو سؤال اختياري اتسأل ومفيش رد، النظام هيفضل يسأله لحد ما الآخر
        # (infinite loop) — declined بيمنع ده.
        self.declined: set[str] = set()

    def decline(self, feature: str) -> None:
        self.declined.add(feature)

    def next_question(self, known: dict):
        for spec in self.fields:
            if spec["field"] not in known and spec["field"] not in self.declined:
                prompt = spec["prompt"]
                return {"feature": spec["field"], "prompt": prompt, "type": "number", "options": None}
        return None

    def enough_info(self, known: dict) -> bool:
        required = [spec["field"] for spec in self.fields if not spec.get("optional")]
        return all(f in known for f in required)

    def remaining_questions(self, known: dict, limit: int = 6) -> list[dict]:
        out = []
        for spec in self.fields:
            if spec["field"] not in known and spec["field"] not in self.declined:
                out.append({"feature": spec["field"], "prompt": spec["prompt"], "type": "number", "options": None})
        return out[:limit]


class RadiologyQuestioner:
    """أبسط أنواع الأسئلة: سؤال واحد بس — نص تقرير الأشعة كامل."""

    def next_question(self, known: dict):
        if "report_text" not in known:
            return {"feature": "report_text",
                    "prompt": "ابعتيلي نص تقرير الأشعة (X-ray report) كامل من فضلك.",
                    "type": "text", "options": None}
        return None

    def enough_info(self, known: dict) -> bool:
        return "report_text" in known and bool(str(known["report_text"]).strip())

    def remaining_questions(self, known: dict, limit: int = 6) -> list[dict]:
        q = self.next_question(known)
        return [q] if q else []
