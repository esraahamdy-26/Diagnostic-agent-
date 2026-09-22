import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.radiology_triage import evaluate_report


class TestNegationHandling:
    """Regression tests للـ bug اللي اكتشفناه: 24% من الداتا فيها صيغ نفي
    النظام القديم مكنش بيعرفها، وده كان بيسبب False Positive rate ~42%."""

    def test_no_evidence_of_phrasing(self):
        r = evaluate_report("No evidence of pneumothorax or pleural effusion.")
        assert r.urgency == "routine"

    def test_negative_for_phrasing(self):
        r = evaluate_report("Negative for pneumothorax. Trachea is midline.")
        assert r.urgency == "routine"

    def test_no_definitive_phrasing(self):
        r = evaluate_report("No definitive pneumothorax identified.")
        assert r.urgency == "routine"

    def test_enumerated_negation_list(self):
        """الحالة اللي كسرت أول نسخة: نفي بيغطي قائمة معدودة بالفاصلة."""
        r = evaluate_report(
            "The lungs are clear bilaterally. Specifically, no evidence of focal "
            "consolidation, pneumothorax, or pleural effusion."
        )
        assert r.urgency == "routine"

    def test_uncertainty_is_not_treated_as_negation(self):
        """'cannot exclude' شك، مش نفي - لازم يفضل يتعتبر يستاهل مراجعة."""
        r = evaluate_report("Cannot exclude small pneumothorax, recommend follow-up CT.")
        assert r.urgency == "urgent_review"


class TestGenuinePositiveFindings:
    def test_real_pneumothorax_still_caught(self):
        r = evaluate_report("Large pneumothorax with mediastinal shift.")
        assert r.urgency == "urgent_review"

    def test_real_follow_up_finding_still_caught(self):
        r = evaluate_report("Mild cardiomegaly, otherwise unremarkable.")
        assert r.urgency == "follow_up"

    def test_fully_normal_report(self):
        r = evaluate_report("The heart size and pulmonary vascularity are within normal limits.")
        assert r.urgency == "routine"


class TestFullDatasetRegression:
    """اختبار على الداتا الحقيقية كلها - بيتأكد إن النسبة العامة معقولة طبياً
    (مش 42% urgent زي النظام القديم)."""

    def test_urgent_rate_is_clinically_plausible(self):
        import pandas as pd
        path = os.path.join(os.path.dirname(__file__), "..", "data", "real_processed",
                             "iu_xray_reports_real.csv")
        if not os.path.exists(path):
            pytest.skip("dataset not present in this environment")
        df = pd.read_csv(path).dropna(subset=["report_text"])
        urgent_rate = df["report_text"].map(lambda t: evaluate_report(t).urgency == "urgent_review").mean()
        # النظام القديم كان 41.7% - ده غير معقول طبياً لمجموعة تقارير أشعة عامة.
        # بعد التصحيح المتوقع أقل من 10%.
        assert urgent_rate < 0.10, f"Urgent rate {urgent_rate:.1%} looks too high — check negation rules"
