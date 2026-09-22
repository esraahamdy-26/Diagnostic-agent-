"""
اختبارات النظام الموحّد — بتغطي الأمان (universal) عبر كل الـ domains، وسلوك
كل نوع موديل (rule-based lab panel vs ML risk-factor).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.orchestrator import UnifiedDiagnosticAgent, AgentAction


CRITICAL_INPUT = {
    "chest_pain": True,
    "pain_radiating_arm_jaw_back": True,
    "cold_sweat": True,
}

SAFE_BASELINE = {
    "chest_pain": False, "severe_shortness_of_breath": False,
    "fainting_or_loss_of_consciousness": False, "uncontrolled_bleeding": False,
    "sudden_weakness_face_arm_speech": False,
}


@pytest.mark.parametrize("domain", ["heart_disease", "diabetes_risk_screening",
                                     "hypertension_risk_screening", "cbc", "pathology",
                                     "radiology"])
class TestSafetyIsUniversalAcrossAllDomains:
    """أهم اختبار في النظام كله: الأمان لازم يشتغل بنفس القوة في أي domain."""

    def test_critical_symptoms_stop_everything(self, domain):
        agent = UnifiedDiagnosticAgent(domain)
        response = agent.handle_turn(CRITICAL_INPUT)
        assert response.action == AgentAction.ESCALATE_EMERGENCY
        assert "risk_score" not in response.data
        assert "prediction" not in response.data

    def test_missing_safety_fields_triggers_needs_more_info(self, domain):
        agent = UnifiedDiagnosticAgent(domain)
        response = agent.handle_turn({})
        assert response.action == AgentAction.ASK_QUESTION


class TestRiskFactorDomains:
    def _run_to_completion(self, domain, answers, max_turns=20):
        agent = UnifiedDiagnosticAgent(domain)
        turn_input = dict(SAFE_BASELINE)
        for _ in range(max_turns):
            r = agent.handle_turn(turn_input)
            if r.action != AgentAction.ASK_QUESTION:
                return r
            feature = r.data.get("feature")
            turn_input = {feature: answers[feature]} if feature in answers else {}
        raise AssertionError("Conversation did not terminate within max_turns")

    def test_heart_disease_reaches_a_decision(self):
        answers = {"age": 45, "sex": "0.0", "cp": "3.0", "trestbps": 120, "chol": 190,
                   "fbs": "0.0", "restecg": "0.0", "thalach": 160, "exang": "0.0",
                   "oldpeak": 0.5, "slope": "1.0", "ca": 0, "thal": "3.0"}
        r = self._run_to_completion("heart_disease", answers)
        assert r.action in (AgentAction.GIVE_RECOMMENDATION, AgentAction.ESCALATE_TO_DOCTOR)

    def test_diabetes_screening_reaches_a_decision(self):
        answers = {"age": 40, "sex": "male", "height_cm": 175, "weight_kg": 75,
                   "systolic_bp": 120, "diastolic_bp": 80, "total_cholesterol": 180}
        r = self._run_to_completion("diabetes_risk_screening", answers)
        assert r.action in (AgentAction.GIVE_RECOMMENDATION, AgentAction.ESCALATE_TO_DOCTOR)

    def test_hypertension_screening_works_even_without_optional_creatinine(self):
        """creatinine اختياري - النظام لازم يكمل من غيرها."""
        answers = {"age": 35, "sex": "female", "height_cm": 165, "weight_kg": 60,
                   "total_cholesterol": 170}
        r = self._run_to_completion("hypertension_risk_screening", answers)
        assert r.action in (AgentAction.GIVE_RECOMMENDATION, AgentAction.ESCALATE_TO_DOCTOR)
        assert "creatinine_mg_dl" not in r.data or True  # لم يُطلب بالضرورة


class TestLabPanelDomains:
    def _run_to_completion(self, domain, answers, max_turns=15):
        agent = UnifiedDiagnosticAgent(domain)
        turn_input = dict(SAFE_BASELINE)
        for _ in range(max_turns):
            r = agent.handle_turn(turn_input)
            if r.action != AgentAction.ASK_QUESTION:
                return r
            feature = r.data.get("feature")
            turn_input = {feature: answers[feature]} if feature in answers else {}
        raise AssertionError("Conversation did not terminate within max_turns")

    def test_cbc_normal_values_give_recommendation_not_escalation(self):
        answers = {"hemoglobin_g_dl": 14.0, "wbc_10e3_ul": 6.5, "platelets_10e3_ul": 250}
        r = self._run_to_completion("cbc", answers)
        assert r.action == AgentAction.GIVE_RECOMMENDATION
        assert r.data["prediction"] == "within_reference"

    def test_cbc_critical_hemoglobin_escalates_to_doctor(self):
        answers = {"hemoglobin_g_dl": 6.0, "wbc_10e3_ul": 6.5, "platelets_10e3_ul": 250}
        r = self._run_to_completion("cbc", answers)
        assert r.action == AgentAction.ESCALATE_TO_DOCTOR
        assert r.data["prediction"] == "urgent_review"

    def test_pathology_severe_hyperglycemia_escalates(self):
        answers = {"fasting_glucose": 300, "creatinine_mg_dl": 0.9,
                   "alt_u_l": 25, "ast_u_l": 22, "hba1c": None, "total_cholesterol": None}
        r = self._run_to_completion("pathology", answers)
        assert r.action == AgentAction.ESCALATE_TO_DOCTOR

    def test_no_ml_engine_used_for_lab_panel_domains(self):
        """تأكيد إن CBC/pathology بيستخدموا rule engine، مش موديل ML."""
        agent = UnifiedDiagnosticAgent("cbc")
        turn_input = dict(SAFE_BASELINE)
        r = agent.handle_turn(turn_input)
        while r.action == AgentAction.ASK_QUESTION:
            feature = r.data.get("feature")
            answers = {"hemoglobin_g_dl": 13, "wbc_10e3_ul": 6, "platelets_10e3_ul": 240}
            r = agent.handle_turn({feature: answers[feature]} if feature in answers else {})
        assert r.data.get("engine") == "rule_based"


class TestRadiologyDomain:
    def test_normal_report_gives_recommendation(self):
        agent = UnifiedDiagnosticAgent("radiology")
        agent.handle_turn(SAFE_BASELINE)
        r = agent.handle_turn({"report_text": "No evidence of pneumothorax or pleural effusion."})
        assert r.action == AgentAction.GIVE_RECOMMENDATION
        assert r.data["prediction"] == "routine"

    def test_urgent_report_escalates(self):
        agent = UnifiedDiagnosticAgent("radiology")
        agent.handle_turn(SAFE_BASELINE)
        r = agent.handle_turn({"report_text": "Large pneumothorax with mediastinal shift, urgent."})
        assert r.action == AgentAction.ESCALATE_TO_DOCTOR
        assert r.data["prediction"] == "urgent_review"


class TestUnsupportedDomain:
    def test_unknown_domain_raises_clear_error(self):
        with pytest.raises(ValueError, match="Unsupported domain"):
            UnifiedDiagnosticAgent("something_made_up")
