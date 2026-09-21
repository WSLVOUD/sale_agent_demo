"""v2.6 §21/§22：问过并答过 / 已有结论的字段，不再重复问。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWERED,
    ASKED,
    UNKNOWN,
    AskedQuestionRegistry,
    get_conversation_state,
    next_question_plan,
    reset_conversation_state,
)
from src.models.requirement import RequirementProfile  # noqa: E402


class TestQuestionRegistryStateMachine:

    def test_state_machine_unknown_asked_answered(self):
        registry = AskedQuestionRegistry()
        assert registry.get("pixel_pitch") is None
        registry.note_asked("pixel_pitch", turn_index=5)
        record = registry.get("pixel_pitch")
        assert record.state == ASKED
        assert record.asked_at_turn == 5
        registry.note_answered("pixel_pitch", turn_index=6)
        assert record.state == ANSWERED
        assert record.answered is True
        assert record.answer_turn == 6
        assert UNKNOWN != record.state

    def test_answered_slot_is_skipped(self):
        registry = AskedQuestionRegistry()
        registry.note_answered("environment")
        assert registry.should_skip("environment") is True
        assert registry.should_skip("pixel_pitch") is False

    def test_registry_blocks_reasking_even_without_profile(self):
        """§22：档案被重建也不会重复问（登记簿是第二道保险）。"""
        from src.rag.query_understanding import extract_slots

        session_id = "no-dup-1"
        reset_conversation_state(session_id)
        state = get_conversation_state(session_id)
        state.registry.note_answered("pixel_pitch")
        state.registry.note_answered("installation")

        slots = {
            key: value
            for key, value in (extract_slots("i need a led display") or {}).items()
            if not str(key).startswith("_")
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        plan = next_question_plan(profile, session_id=session_id, conversation=state)
        assert getattr(plan, "slot", "") not in ("pixel_pitch", "installation")


class TestPitchAndDistanceAreNotReasked:

    def test_case5_pitch_known_means_no_distance_question(self):
        """§26 Case 5：客户已经给了能推导的 P 值 → 不再问观看距离。"""
        slots = {"environment": "indoor", "pixel_pitch_mm": 3.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        session_id = "no-dup-2"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", "") != "viewing_distance"

    def test_distance_known_means_no_pitch_question(self):
        slots = {"environment": "indoor", "viewing_distance_m": 5.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        session_id = "no-dup-3"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", "") != "pixel_pitch"
