"""
Natural conversational layer for the Medical Diagnostic Agent.

The chat UI is intentionally conversational:
- We do NOT ask a universal list of emergency questions at the beginning.
- We inspect what the patient actually says for safety signals.
- We route the message to general medical conversation or to one of the
  existing diagnostic capabilities.
- Diagnostic capabilities may ask only the fields they actually need.
- The active question is remembered, so "اه/لا" is attached to the correct
  question instead of being compared with an arbitrary field.
"""

from __future__ import annotations

from typing import Any

from src import nlu
from src.general_wellness import GeneralWellnessHandler
from src.orchestrator import AgentAction, AgentResponse, UnifiedDiagnosticAgent
from src.safety_layer import CONFIRMATION_MAP, RedFlagChecker, Urgency, detect_text_safety


SAFETY_FIELD_SPECS: list[dict[str, Any]] = [
    {"feature": "chest_pain", "prompt": "هل حاسة بألم في الصدر دلوقتي؟", "type": "bool", "options": None},
    {"feature": "severe_shortness_of_breath", "prompt": "هل حاسة بضيق نفس شديد وانتي مرتاحة؟", "type": "bool", "options": None},
    {"feature": "fainting_or_loss_of_consciousness", "prompt": "هل حصلك إغماء أو فقدان وعي؟", "type": "bool", "options": None},
    {"feature": "uncontrolled_bleeding", "prompt": "هل فيه نزيف مش بيقف؟", "type": "bool", "options": None},
    {"feature": "sudden_weakness_face_arm_speech", "prompt": "هل حسيتي بضعف مفاجئ في الوش أو الذراع أو صعوبة كلام؟", "type": "bool", "options": None},
    {"feature": "headache_worst_ever_sudden_onset", "prompt": "لو عندك صداع، هل هو أسوأ صداع حصلك في حياتك ووجه فجأة في أقل من دقيقة؟", "type": "bool", "options": None},
    {"feature": "headache_with_fever_and_neck_stiffness", "prompt": "لو عندك صداع، هل مصحوب بحرارة وتصلب في الرقبة؟", "type": "bool", "options": None},
]

_SAFE_BASELINE = {
    "chest_pain": False,
    "severe_shortness_of_breath": False,
    "fainting_or_loss_of_consciousness": False,
    "uncontrolled_bleeding": False,
    "sudden_weakness_face_arm_speech": False,
}


def _response_dict(action: str, message: str, data: dict | None = None) -> dict:
    return {"action": action, "message": message, "data": data or {}}


def _agent_response_to_dict(r: AgentResponse) -> dict:
    return {"action": r.action.value, "message": r.message, "data": r.data}


class ConversationalDiagnosticAgent:
    def __init__(self, api_key: str | None = None, provider: str | None = None, initial_domain: str | None = None):
        self.api_key = api_key
        self.provider = provider
        self.domain = initial_domain
        self.agent: UnifiedDiagnosticAgent | None = None
        self.wellness_handler: GeneralWellnessHandler | None = None
        self.pending_safety_data: dict[str, Any] = dict(_SAFE_BASELINE)
        self.pending_domain_data: dict[str, Any] = {}
        self.first_message: str | None = None
        self.history: list[dict[str, str]] = []
        self.active_field: str | None = None
        self.active_prompt: str | None = None
        self.completed_domain: bool = False
        self.awaiting_safety_confirmation: str | None = None

        if initial_domain == "general_wellness":
            self._ensure_wellness_handler()
        elif initial_domain:
            self._ensure_diagnostic_agent(initial_domain)

    def _ensure_wellness_handler(self) -> None:
        if self.wellness_handler is None:
            self.wellness_handler = GeneralWellnessHandler(api_key=self.api_key, provider=self.provider)

    def _ensure_diagnostic_agent(self, domain: str) -> None:
        if self.agent is None or self.agent.domain != domain:
            self.agent = UnifiedDiagnosticAgent(domain)
            self.agent.collected_data.update(self.pending_safety_data)
            self.agent.collected_data.update(self.pending_domain_data)
        self.domain = domain

    def _history_text(self, limit: int = 8) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in self.history[-limit:])

    def _remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.history = self.history[-16:]

    @staticmethod
    def _safety_confirmation_specs(known: dict) -> list[dict]:
        out = []
        for raw_field, spec in CONFIRMATION_MAP.items():
            if known.get(raw_field, False) and spec["confirm_field"] not in known:
                out.append({"feature": spec["confirm_field"], "prompt": spec["prompt"], "type": "bool", "options": None})
        return out

    def _set_active_question(self, response: dict) -> None:
        if response.get("action") == AgentAction.ASK_QUESTION.value:
            data = response.get("data") or {}
            self.active_field = data.get("feature")
            self.active_prompt = response.get("message")
        else:
            self.active_field = None
            self.active_prompt = None

    def _run_text_safety(self, message: str, intent: str) -> dict | None:
        safety = detect_text_safety(message)
        if safety.urgency == Urgency.CRITICAL:
            return _response_dict("escalate_emergency", safety.message_to_patient, {"triggered_rules": safety.triggered_rules})

        # Educational questions such as "يعني إيه ضيق النفس؟" should not
        # trigger a confirmation question. First-person symptom statements do.
        first_person = any(token in message.lower() for token in ["عندي", "حاسه", "حاسة", "حاسس", "عندي", "i have", "i'm"])
        if safety.urgency == Urgency.NEEDS_CONFIRMATION and (intent != "medical_information" or first_person):
            self.awaiting_safety_confirmation = safety.confirmation_field
            self.active_field = safety.confirmation_field
            self.active_prompt = safety.message_to_patient
            return _response_dict("ask_question", safety.message_to_patient, {"feature": safety.confirmation_field})
        return None

    def _handle_confirmation_answer(self, message: str) -> dict | None:
        if not self.awaiting_safety_confirmation:
            return None
        feature = self.awaiting_safety_confirmation
        value = nlu.extract_fields(
            message,
            [{"feature": feature, "prompt": self.active_prompt or "", "type": "bool", "options": None}],
            api_key=self.api_key,
            provider=self.provider,
            active_field=feature,
            active_prompt=self.active_prompt,
            conversation_context=self._history_text(),
        ).get(feature)
        if value is None:
            return None

        self.pending_safety_data[feature] = bool(value)
        self.awaiting_safety_confirmation = None
        self.active_field = None
        self.active_prompt = None

        safety = RedFlagChecker().check(self.pending_safety_data)
        if safety.urgency == Urgency.CRITICAL:
            return _response_dict("escalate_emergency", safety.message_to_patient, {"triggered_rules": safety.triggered_rules})
        return None

    def handle_message(self, message: str) -> dict:
        message = (message or "").strip()
        if not message:
            return _response_dict("ask_question", "قوليلي إيه اللي حابة تعرفيه أو إيه اللي حاسة بيه؟")

        if self.first_message is None:
            self.first_message = message
        self._remember("user", message)

        # 1) Resolve an answer to a targeted safety confirmation first.
        confirmation_result = self._handle_confirmation_answer(message)
        if confirmation_result:
            self._remember("assistant", confirmation_result["message"])
            return confirmation_result
        if self.awaiting_safety_confirmation:
            return _response_dict("ask_question", self.active_prompt or "ممكن توضحيلي إجابتك بـ آه أو لأ؟", {"feature": self.active_field})

        # 2) If the assistant asked a diagnostic question, try to answer that
        # question FIRST. This is what makes short replies such as "50", "اه"
        # and "لأ" work naturally. We never compare them with a random field.
        if self.active_field and self.agent is not None:
            active_specs = self.agent.questioner.remaining_questions(
                self.agent.collected_data, limit=50
            )
            active_specs = [
                q for q in active_specs if q.get("feature") == self.active_field
            ]
            if active_specs:
                extracted_active = nlu.extract_fields(
                    message,
                    active_specs,
                    api_key=self.api_key,
                    provider=self.provider,
                    active_field=self.active_field,
                    active_prompt=self.active_prompt,
                    conversation_context=self._history_text(),
                )
                if extracted_active:
                    self.pending_domain_data.update(extracted_active)
                    response = self.agent.handle_turn(
                        {**_SAFE_BASELINE, **self.pending_safety_data, **extracted_active}
                    )
                    result = _agent_response_to_dict(response)
                    self._set_active_question(result)
                    self.completed_domain = response.action != AgentAction.ASK_QUESTION
                    self._remember("assistant", result["message"])
                    return result

                # If the user did not understand an optional numeric question,
                # do not route "معرفش" as a brand-new medical topic.
                active_meta = active_specs[0]
                if active_meta.get("optional") and nlu._is_decline(message):
                    self.pending_domain_data[self.active_field] = None
                    response = self.agent.handle_turn(
                        {**_SAFE_BASELINE, **self.pending_safety_data, self.active_field: None}
                    )
                    result = _agent_response_to_dict(response)
                    self._set_active_question(result)
                    self._remember("assistant", result["message"])
                    return result

        # 3) Fast safety scan over what the patient actually wrote.
        intent = nlu.classify_intent(message, api_key=self.api_key, provider=self.provider)
        safety_result = self._run_text_safety(message, intent)
        if safety_result:
            self._remember("assistant", safety_result["message"])
            return safety_result

        # 4) General medical information / normal symptom conversation.
        # If a diagnostic domain is already active, a clear new diagnostic
        # request may switch domains; a normal follow-up stays conversational.
        target_domain = nlu.classify_domain(message, api_key=self.api_key, provider=self.provider)
        diagnostic_intents = {"lab_analysis", "radiology_analysis", "risk_screening"}

        if intent not in diagnostic_intents:
            self._ensure_wellness_handler()
            result = self.wellness_handler.respond(message, conversation_history=self._history_text())
            self._remember("assistant", result["message"])
            self.active_field = None
            self.active_prompt = None
            return result

        # 5) Diagnostic request: create the corresponding existing agent.
        if target_domain == "general_wellness":
            self._ensure_wellness_handler()
            result = self.wellness_handler.respond(message, conversation_history=self._history_text())
            self._remember("assistant", result["message"])
            return result

        self._ensure_diagnostic_agent(target_domain)

        specs = self.agent.questioner.remaining_questions(self.agent.collected_data, limit=50)
        extracted = nlu.extract_fields(
            message,
            specs,
            api_key=self.api_key,
            provider=self.provider,
            active_field=self.active_field,
            active_prompt=self.active_prompt,
            conversation_context=self._history_text(),
        )
        self.pending_domain_data.update(extracted)

        response = self.agent.handle_turn({**_SAFE_BASELINE, **self.pending_safety_data, **extracted})
        result = _agent_response_to_dict(response)
        self._set_active_question(result)
        self.completed_domain = response.action != AgentAction.ASK_QUESTION
        self._remember("assistant", result["message"])
        return result

    def set_domain(self, domain: str) -> dict:
        if domain == "general_wellness":
            self.domain = domain
            self.agent = None
            self._ensure_wellness_handler()
            result = _response_dict("ask_question", "تمام ❤️ احكيلي براحتك إيه اللي حابة تعرفيه أو إيه اللي حاسة بيه؟")
            self._set_active_question(result)
            return result

        self._ensure_diagnostic_agent(domain)
        specs = self.agent.questioner.remaining_questions(self.agent.collected_data, limit=50)
        if not specs:
            return _agent_response_to_dict(self.agent.handle_turn(self.pending_safety_data))
        result = _response_dict("ask_question", specs[0]["prompt"], {"feature": specs[0]["feature"], "options": specs[0].get("options")})
        self._set_active_question(result)
        return result
