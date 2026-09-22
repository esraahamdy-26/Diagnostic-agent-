import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import nlu
from src.conversational_agent import ConversationalDiagnosticAgent
from src.safety_layer import Urgency, detect_text_safety


def fake_wellness_response(self, message, conversation_history=""):
    return {
        "action": "give_recommendation",
        "message": f"تم استلام سؤالك: {message}",
        "data": {"engine": "test"},
    }


class TestNLU:
    def test_medical_term_explanation_is_general_wellness(self):
        assert nlu.classify_intent("يعني ايه hypertension؟") == "medical_information"
        assert nlu.classify_domain("يعني ايه hypertension؟") == "general_wellness"


    def test_english_medical_term_is_information(self):
        assert nlu.classify_intent("diabetes") == "medical_information"
        assert nlu.classify_domain("diabetes") == "general_wellness"
        assert nlu.classify_intent("diabeties") == "medical_information"

    def test_translation_request_is_not_treated_as_missing_medical_source(self):
        assert nlu.classify_intent("ترجملي اللي قولته") == "translation"

    def test_diabetes_risk_is_diagnostic(self):
        assert nlu.classify_intent("خايفة يكون عندي السكر") == "risk_screening"
        assert nlu.classify_domain("خايفة يكون عندي السكر") == "diabetes_risk_screening"

    def test_cbc_result_is_diagnostic(self):
        assert nlu.classify_domain("نتيجة CBC عندي والهيموجلوبين 10") == "cbc"

    def test_radiology_report_is_diagnostic(self):
        assert nlu.classify_domain("ممكن تقري تقرير الأشعة ده؟") == "radiology"

    def test_active_question_controls_short_answer(self):
        specs = [{"feature": "age", "prompt": "كام عمرك؟", "type": "number", "options": None}]
        assert nlu.extract_fields("50", specs, active_field="age") == {"age": 50.0}
        bool_specs = [{"feature": "chest_pain", "prompt": "هل عندك ألم صدر؟", "type": "bool", "options": None}]
        assert nlu.extract_fields("لأ", bool_specs, active_field="chest_pain") == {"chest_pain": False}


class TestConversationalFlow:
    def test_normal_chat_does_not_start_safety_questionnaire(self, monkeypatch):
        monkeypatch.setattr("src.conversational_agent.GeneralWellnessHandler.respond", fake_wellness_response)
        agent = ConversationalDiagnosticAgent()
        r = agent.handle_message("يعني ايه hypertension؟")
        assert r["action"] == "give_recommendation"
        assert agent.active_field is None

    def test_short_answer_is_attached_to_active_diagnostic_question(self, monkeypatch):
        monkeypatch.setattr("src.conversational_agent.GeneralWellnessHandler.respond", fake_wellness_response)
        agent = ConversationalDiagnosticAgent()
        first = agent.handle_message("خايفة يكون عندي السكر")
        assert first["action"] == "ask_question"
        assert first["data"]["feature"]

        # Answer the exact question that the agent just asked.
        field = first["data"]["feature"]
        answer = "50" if field == "age" else "ست"
        second = agent.handle_message(answer)
        assert "مش متأكدة" not in second["message"]

    def test_critical_free_text_is_caught_without_questionnaire(self):
        agent = ConversationalDiagnosticAgent()
        r = agent.handle_message("عندي ألم صدر شديد وتعرق بارد")
        assert r["action"] == "escalate_emergency"

    def test_breathlessness_gets_one_targeted_confirmation(self):
        agent = ConversationalDiagnosticAgent()
        r = agent.handle_message("عندي ضيق نفس")
        assert r["action"] == "ask_question"
        assert "ضيق النفس" in r["message"]
        assert agent.active_field == "breathlessness_persists_at_rest_unrelated_to_stress"

    def test_denied_confirmation_returns_to_conversation(self, monkeypatch):
        monkeypatch.setattr("src.conversational_agent.GeneralWellnessHandler.respond", fake_wellness_response)
        agent = ConversationalDiagnosticAgent()
        agent.handle_message("عندي ضيق نفس")
        r = agent.handle_message("لأ")
        assert r["action"] == "give_recommendation"
        assert agent.awaiting_safety_confirmation is None
