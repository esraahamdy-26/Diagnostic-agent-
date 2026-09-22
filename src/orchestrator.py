"""
Unified Diagnostic Agent Orchestrator
========================================
المخ اللي بيربط كل حاجة بنيناها:

    1. Safety Layer (rule-based, universal)  → لو CRITICAL: توقف فوراً
    2. Router (LLM + keyword fallback، موجود أصلاً في src/router.py) → يحدد
       أي domain من الستة المطلوب (cbc/pathology/heart_disease/
       diabetes_risk_screening/hypertension_risk_screening/radiology)
    3. Adaptive Questioner المناسب للـ domain (risk-factor أو lab-panel)
    4. predict_for_backend() من src/diagnostic_models.py
    5. قرار: توصية مباشرة / تحويل لطبيب (borderline أو rule-based urgent)

القاعدة الذهبية اللي بتحكم كل حاجة: الموديل بيقترح، مش بيقرر. أي شك → إنسان.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.adaptive_questioner import LabPanelQuestioner, RadiologyQuestioner, RiskFactorQuestioner
from src.diagnostic_models import predict_for_backend
from src.safety_layer import RedFlagChecker, Urgency

BASE_DIR = Path(__file__).resolve().parents[1]

RISK_FACTOR_DOMAINS = {
    "heart_disease": {
        "model_path": BASE_DIR / "models" / "heart_disease_pipeline.joblib",
        "always_ask": {"age", "sex", "cp", "trestbps", "chol"},
        "min_features": 8,
    },
    "diabetes_risk_screening": {
        "model_path": BASE_DIR / "models" / "diabetes_risk_screening.joblib",
        "always_ask": {"age", "sex", "bmi", "systolic_bp"},
        "min_features": 5,
    },
    "hypertension_risk_screening": {
        "model_path": BASE_DIR / "models" / "hypertension_risk_screening.joblib",
        "always_ask": {"age", "sex", "bmi"},
        "min_features": 3,
    },
}
LAB_PANEL_DOMAINS = {"cbc", "pathology"}
TEXT_REPORT_DOMAINS = {"radiology"}


class AgentAction(Enum):
    ASK_QUESTION = "ask_question"
    ESCALATE_EMERGENCY = "escalate_emergency"
    ESCALATE_TO_DOCTOR = "escalate_to_doctor"
    GIVE_RECOMMENDATION = "give_recommendation"


@dataclass
class AgentResponse:
    action: AgentAction
    message: str
    data: dict = field(default_factory=dict)


class UnifiedDiagnosticAgent:
    BORDERLINE_LOW = 0.35
    BORDERLINE_HIGH = 0.65

    def __init__(self, domain: str):
        if domain not in RISK_FACTOR_DOMAINS and domain not in LAB_PANEL_DOMAINS \
                and domain not in TEXT_REPORT_DOMAINS:
            raise ValueError(f"Unsupported domain '{domain}'.")
        self.domain = domain
        self.safety_checker = RedFlagChecker()
        if domain in RISK_FACTOR_DOMAINS:
            cfg = RISK_FACTOR_DOMAINS[domain]
            self.questioner = RiskFactorQuestioner(
                domain, str(cfg["model_path"]), always_ask=cfg["always_ask"],
                min_features=cfg["min_features"],
            )
        elif domain in LAB_PANEL_DOMAINS:
            self.questioner = LabPanelQuestioner(domain)
        else:
            self.questioner = RadiologyQuestioner()
        self.collected_data: dict = {}

    def handle_turn(self, new_input: dict) -> AgentResponse:
        # لو المريض رد بـ None على سؤال، ده معناه "معرفش/مش عايز يجاوب" —
        # مش "لسه ما جاوبش"، فبنسجلها كـ decline بدل ما نضيفها كقيمة حقيقية
        for feature, value in list(new_input.items()):
            if value is None:
                self.questioner.decline(feature)
                new_input.pop(feature)

        self.collected_data.update(new_input)
        if hasattr(self.questioner, "enrich_derived"):
            self.collected_data = self.questioner.enrich_derived(self.collected_data)

        # 1) الأمان أولاً، دايماً، بغض النظر عن الـ domain
        safety = self.safety_checker.check(self.collected_data)
        if safety.urgency == Urgency.CRITICAL:
            return AgentResponse(
                action=AgentAction.ESCALATE_EMERGENCY,
                message=safety.message_to_patient,
                data={"triggered_rules": safety.triggered_rules},
            )
        if safety.urgency == Urgency.NEEDS_MORE_INFO:
            return AgentResponse(
                action=AgentAction.ASK_QUESTION,
                message="محتاجة أسألك كذا سؤال بسيط الأول للتأكد إن مفيش حاجة عاجلة.",
                data={"next_fields_needed": ["chest_pain", "severe_shortness_of_breath",
                                              "fainting_or_loss_of_consciousness",
                                              "uncontrolled_bleeding",
                                              "sudden_weakness_face_arm_speech"]},
            )
        if safety.urgency == Urgency.NEEDS_CONFIRMATION:
            return AgentResponse(
                action=AgentAction.ASK_QUESTION,
                message=safety.message_to_patient,
                data={"feature": safety.confirmation_field},
            )

        # 2) أسئلة الـ domain نفسه
        if not self.questioner.enough_info(self.collected_data):
            next_q = self.questioner.next_question(self.collected_data)
            if next_q:
                return AgentResponse(
                    action=AgentAction.ASK_QUESTION,
                    message=next_q["prompt"],
                    data={"feature": next_q["feature"], "options": next_q.get("options")},
                )

        # 3) القرار: ننادي الموديل/القاعدة المناسبة
        result = predict_for_backend(self.domain, self.collected_data, str(BASE_DIR))

        # Rule-based domains (cbc/pathology/radiology) بترجع triage tier مباشر —
        # الـ tier نفسه هو الأمانة، مفيش borderline probability نتعامل معاها
        if self.domain in LAB_PANEL_DOMAINS or self.domain in TEXT_REPORT_DOMAINS:
            if result["prediction"] == "urgent_review":
                return AgentResponse(
                    action=AgentAction.ESCALATE_TO_DOCTOR,
                    message="نتيجة التحليل فيها قيم محتاجة مراجعة طبيب فوراً، مش مجرد متابعة عادية.",
                    data=result,
                )
            return AgentResponse(
                action=AgentAction.GIVE_RECOMMENDATION,
                message=result["explanation"],
                data=result,
            )

        # ML domains: منطق borderline زي ما اتفقنا
        risk_score = result.get("risk_score", 0.5)
        if self.BORDERLINE_LOW <= risk_score <= self.BORDERLINE_HIGH:
            return AgentResponse(
                action=AgentAction.ESCALATE_TO_DOCTOR,
                message="النتيجة مش قاطعة بشكل واضح، فالأصح إننا نحولك لمراجعة طبيب بشري.",
                data=result,
            )
        return AgentResponse(
            action=AgentAction.GIVE_RECOMMENDATION,
            message=result["explanation"],
            data=result,
        )
