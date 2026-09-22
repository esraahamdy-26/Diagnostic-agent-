import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safety_layer import RedFlagChecker, Urgency

checker = RedFlagChecker()


class TestCriticalCasesAreCaught:
    def test_classic_acs_pattern(self):
        r = checker.check({"chest_pain": True, "pain_radiating_arm_jaw_back": True, "cold_sweat": True})
        assert r.urgency == Urgency.CRITICAL

    def test_severe_breathlessness_needs_confirmation_first(self):
        r = checker.check({"severe_shortness_of_breath": True})
        assert r.urgency == Urgency.NEEDS_CONFIRMATION

    def test_severe_breathlessness_confirmed_escalates(self):
        r = checker.check({"severe_shortness_of_breath": True,
                            "breathlessness_persists_at_rest_unrelated_to_stress": True})
        assert r.urgency == Urgency.CRITICAL

    def test_severe_breathlessness_denied_on_confirmation_is_routine(self):
        r = checker.check({
            "chest_pain": False, "severe_shortness_of_breath": True,
            "fainting_or_loss_of_consciousness": False, "uncontrolled_bleeding": False,
            "sudden_weakness_face_arm_speech": False,
            "breathlessness_persists_at_rest_unrelated_to_stress": False,
        })
        assert r.urgency == Urgency.ROUTINE

    def test_stroke_signs_needs_confirmation_first(self):
        r = checker.check({"sudden_weakness_face_arm_speech": True})
        assert r.urgency == Urgency.NEEDS_CONFIRMATION

    def test_stroke_signs_confirmed_escalates(self):
        r = checker.check({"sudden_weakness_face_arm_speech": True,
                            "weakness_started_suddenly_and_ongoing": True})
        assert r.urgency == Urgency.CRITICAL

    def test_uncontrolled_bleeding_needs_confirmation_first(self):
        r = checker.check({"uncontrolled_bleeding": True})
        assert r.urgency == Urgency.NEEDS_CONFIRMATION

    def test_uncontrolled_bleeding_confirmed_escalates(self):
        r = checker.check({"uncontrolled_bleeding": True, "bleeding_still_active_now": True})
        assert r.urgency == Urgency.CRITICAL

    def test_syncope_with_chest_pain(self):
        r = checker.check({"chest_pain": True, "fainting_or_loss_of_consciousness": True})
        assert r.urgency == Urgency.CRITICAL


class TestRoutineAndMissingInfo:
    def test_no_symptoms(self):
        r = checker.check({
            "chest_pain": False, "severe_shortness_of_breath": False,
            "fainting_or_loss_of_consciousness": False, "uncontrolled_bleeding": False,
            "sudden_weakness_face_arm_speech": False,
        })
        assert r.urgency == Urgency.ROUTINE

    def test_empty_input_needs_more_info(self):
        r = checker.check({})
        assert r.urgency == Urgency.NEEDS_MORE_INFO
