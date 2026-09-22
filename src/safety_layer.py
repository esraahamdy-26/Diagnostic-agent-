"""
Universal medical safety guardrails.

Important design rule:
- The conversational agent does NOT run a universal questionnaire.
- Safety is checked from what the patient actually says.
- Only when a potentially dangerous symptom is mentioned do we ask a
  targeted confirmation question.
- The deterministic rules in this file remain the source of truth for
  structured diagnostic sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


class Urgency(Enum):
    CRITICAL = "critical"
    ROUTINE = "routine"
    NEEDS_MORE_INFO = "unclear"
    NEEDS_CONFIRMATION = "needs_confirmation"


@dataclass
class SafetyCheckResult:
    urgency: Urgency
    triggered_rules: list[str] = field(default_factory=list)
    message_to_patient: str = ""
    confirmation_field: str | None = None


CONFIRMATION_MAP = {
    "severe_shortness_of_breath": {
        "confirm_field": "breathlessness_persists_at_rest_unrelated_to_stress",
        "prompt": (
            "علشان أتأكد من سلامتك: ضيق النفس ده مستمر دلوقتي وإنتِ قاعدة ومرتاحة، "
            "ولا بيظهر مع التوتر أو المجهود وبيختفي بسرعة؟"
        ),
    },
    "uncontrolled_bleeding": {
        "confirm_field": "bleeding_still_active_now",
        "prompt": "النزيف لسه مستمر دلوقتي ومش بيقف رغم الضغط المباشر؟",
    },
    "sudden_weakness_face_arm_speech": {
        "confirm_field": "weakness_started_suddenly_and_ongoing",
        "prompt": "الضعف أو صعوبة الكلام بدأت فجأة ولسه موجودة دلوقتي؟",
    },
}

# These are intentionally conservative phrase groups. They are only a
# conversation guardrail; they do not diagnose a condition.
_CRITICAL_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Chest pain with radiation and/or cold sweat",
        (
            "ألم صدر.*عرق بارد", "ألم في الصدر.*عرق بارد", "chest pain.*cold sweat",
            "chest pain.*sweating", "ألم صدر.*يمتد", "ألم صدر.*للذراع", "ألم صدر.*للفك",
        ),
    ),
    (
        "Thunderclap headache",
        (
            "أسوأ صداع.*فجأة", "أسوأ صداع.*في حيات", "صداع.*زي ضربة.*فجأة",
            "صداع.*وصل لأقصى.*دقيقة", "thunderclap headache", "worst headache.*sudden",
        ),
    ),
    (
        "Chest pain with loss of consciousness",
        (
            "ألم صدر.*إغماء", "ألم في الصدر.*فقدان وعي", "chest pain.*faint",
        ),
    ),
    (
        "Confirmed severe breathlessness at rest",
        (
            "ضيق نفس شديد.*مش قادر", "مش قادرة.*أتنفس", "مش قادر.*أتنفس",
            "صعوبة تنفس شديدة", "severe shortness of breath.*at rest",
        ),
    ),
    (
        "Active uncontrolled bleeding",
        (
            "نزيف.*مش بيقف", "نزيف.*مستمر", "نزيف شديد.*مش بيقف", "bleeding.*won't stop",
            "uncontrolled bleeding",
        ),
    ),
    (
        "Sudden stroke-like symptoms",
        (
            "ضعف مفاجئ.*وش", "ضعف مفاجئ.*ذراع", "صعوبة كلام.*فجأة",
            "مش قادر.*أحرك.*ذراع", "مش قادرة.*أحرك.*ذراع", "sudden weakness.*speech",
        ),
    ),
    (
        "Headache with fever and neck stiffness",
        (
            "صداع.*حرارة.*تيبس الرقبة", "صداع.*سخونية.*رقبة", "headache.*fever.*stiff neck",
        ),
    ),
]

_SOFT_PATTERNS: list[tuple[str, tuple[str, ...], str]] = [
    (
        "severe_shortness_of_breath",
        ("ضيق نفس", "نهجان شديد", "مش عارفة أتنفس", "مش قادر أتنفس", "shortness of breath", "breathless"),
        "ضيق النفس",
    ),
    (
        "uncontrolled_bleeding",
        ("عندي نزيف", "فيه نزيف", "نزيف", "bleeding"),
        "النزيف",
    ),
    (
        "sudden_weakness_face_arm_speech",
        ("ضعف مفاجئ", "صعوبة كلام", "كلامي تقيل", "وشي واجعني من ناحية", "sudden weakness"),
        "أعراض عصبية مفاجئة",
    ),
]


def _norm(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def detect_text_safety(message: str) -> SafetyCheckResult:
    """Fast deterministic safety scan for a free-text chat message."""
    t = _norm(message)
    triggered: list[str] = []
    for reason, patterns in _CRITICAL_PATTERNS:
        if any(re.search(_norm(pattern), t, flags=re.I) for pattern in patterns):
            triggered.append(reason)

    if triggered:
        return SafetyCheckResult(
            urgency=Urgency.CRITICAL,
            triggered_rules=triggered,
            message_to_patient=(
                "الأعراض اللي وصفتيها ممكن تكون علامة على حالة طارئة. "
                "من فضلك اتصلي بالإسعاف أو روحي لأقرب طوارئ فوراً. "
                "المساعد ده مش بديل عن التقييم الطبي الفوري."
            ),
        )

    for feature, patterns, _label in _SOFT_PATTERNS:
        if any(p in t for p in patterns):
            spec = CONFIRMATION_MAP.get(feature)
            if spec:
                return SafetyCheckResult(
                    urgency=Urgency.NEEDS_CONFIRMATION,
                    message_to_patient=spec["prompt"],
                    confirmation_field=spec["confirm_field"],
                )

    return SafetyCheckResult(urgency=Urgency.ROUTINE)


class RedFlagChecker:
    """Deterministic rules used by structured diagnostic sessions."""

    def __init__(self):
        self.rules = [
            self._rule_classic_acs_pattern,
            self._rule_severe_chest_pain,
            self._rule_severe_breathlessness,
            self._rule_syncope_with_chest_symptoms,
            self._rule_severe_bleeding,
            self._rule_signs_of_stroke,
            self._rule_thunderclap_headache,
            self._rule_headache_with_neuro_or_infection_signs,
        ]

    @staticmethod
    def _rule_classic_acs_pattern(data: dict) -> tuple[bool, str]:
        chest_pain = data.get("chest_pain", False)
        radiating = data.get("pain_radiating_arm_jaw_back", False)
        sweating = data.get("cold_sweat", False)
        if chest_pain and (radiating or sweating):
            return True, "Chest pain with radiation and/or cold sweat (classic ACS pattern)"
        return False, ""

    @staticmethod
    def _rule_severe_chest_pain(data: dict) -> tuple[bool, str]:
        severity = data.get("chest_pain_severity", 0)
        duration_min = data.get("chest_pain_duration_minutes", 0)
        if data.get("chest_pain", False) and severity >= 7 and duration_min >= 15:
            return True, f"Severe chest pain (severity={severity}/10) lasting >15 min"
        return False, ""

    @staticmethod
    def _rule_severe_breathlessness(data: dict) -> tuple[bool, str]:
        if data.get("severe_shortness_of_breath", False) and data.get("breathlessness_persists_at_rest_unrelated_to_stress", False):
            return True, "Confirmed severe breathlessness at rest, unrelated to stress/exertion"
        return False, ""

    @staticmethod
    def _rule_syncope_with_chest_symptoms(data: dict) -> tuple[bool, str]:
        if data.get("fainting_or_loss_of_consciousness", False) and data.get("chest_pain", False):
            return True, "Loss of consciousness combined with chest pain"
        return False, ""

    @staticmethod
    def _rule_severe_bleeding(data: dict) -> tuple[bool, str]:
        if data.get("uncontrolled_bleeding", False) and data.get("bleeding_still_active_now", False):
            return True, "Confirmed active uncontrolled bleeding"
        return False, ""

    @staticmethod
    def _rule_signs_of_stroke(data: dict) -> tuple[bool, str]:
        if data.get("sudden_weakness_face_arm_speech", False) and data.get("weakness_started_suddenly_and_ongoing", False):
            return True, "Confirmed sudden, ongoing weakness/speech difficulty (stroke signs)"
        return False, ""

    @staticmethod
    def _rule_thunderclap_headache(data: dict) -> tuple[bool, str]:
        if data.get("headache_worst_ever_sudden_onset", False):
            return True, "Thunderclap headache (worst-ever, sudden onset <1 min)"
        return False, ""

    @staticmethod
    def _rule_headache_with_neuro_or_infection_signs(data: dict) -> tuple[bool, str]:
        if data.get("headache_present", False) and (
            data.get("headache_with_fever_and_neck_stiffness", False)
            or data.get("headache_with_neuro_symptoms", False)
        ):
            return True, "Headache with fever+neck stiffness or neurological symptoms"
        return False, ""

    def check(self, patient_data: dict) -> SafetyCheckResult:
        triggered = []
        for rule in self.rules:
            hit, reason = rule(patient_data)
            if hit:
                triggered.append(reason)
        if triggered:
            return SafetyCheckResult(
                urgency=Urgency.CRITICAL,
                triggered_rules=triggered,
                message_to_patient=(
                    "الأعراض اللي وصفتها ممكن تكون علامة على حالة طارئة. من فضلك "
                    "اتصل بالإسعاف أو روح لأقرب طوارئ فوراً — الأداة دي مش بديل عن تقييم طبي فوري."
                ),
            )

        for raw_field, spec in CONFIRMATION_MAP.items():
            if patient_data.get(raw_field, False) and spec["confirm_field"] not in patient_data:
                return SafetyCheckResult(
                    urgency=Urgency.NEEDS_CONFIRMATION,
                    message_to_patient=spec["prompt"],
                    confirmation_field=spec["confirm_field"],
                )

        required_fields = [
            "chest_pain", "severe_shortness_of_breath",
            "fainting_or_loss_of_consciousness", "uncontrolled_bleeding",
            "sudden_weakness_face_arm_speech",
        ]
        if any(f not in patient_data for f in required_fields):
            return SafetyCheckResult(
                urgency=Urgency.NEEDS_MORE_INFO,
                message_to_patient="محتاجين نسألك كذا سؤال أساسي الأول للتأكد من سلامتك.",
            )
        return SafetyCheckResult(urgency=Urgency.ROUTINE)
