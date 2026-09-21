"""v2.6 §10：答非所问 —— 信息不能丢，原问题保持未知。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_WRONG_SLOT,
    get_conversation_state,
    match_answer_to_question,
    next_question_plan,
    reset_conversation_state,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402


def _profile_from_message(message: str) -> RequirementProfile:
    slots = {
        key: value
        for key, value in (extract_slots(message) or {}).items()
        if not str(key).startswith("_")
    }
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestAnswerWrongSlot:

    def test_size_answer_to_environment_question_is_wrong_slot(self):
        match = match_answer_to_question("3*5", last_question_slot="environment")
        assert match.kind == ANSWER_WRONG_SLOT
        assert match.slot == "size"
        assert match.expected_slot == "environment"
        assert "environment" not in match.slots, "不能把尺寸当成室内外"

    def test_information_is_not_lost(self):
        """§10：AI 问 indoor/outdoor、客户答 3×5 → 尺寸照样要记下来。"""
        reset_conversation_state("wrong-slot-1")
        state = get_conversation_state("wrong-slot-1")
        state.note_ai_turn(question="Indoor or outdoor?", slot="environment")

        match = state.answer_to("3*5")
        state.note_customer_turn(text="3*5", answer_slot="")
        for slot in match.covered_slots:
            state.note_answered(slot)

        assert state.registry.should_skip("size") is True, "尺寸已经知道，不该再问"
        assert state.registry.should_skip("environment") is False, "环境仍然未知"

    def test_environment_is_still_asked_next_turn(self):
        """§26 Case 8：3×5 不会让系统丢掉尺寸，也不会让环境变成已知。"""
        profile = _profile_from_message("3*5")
        session_id = "wrong-slot-2"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert plan is not None
        assert plan.slot == "environment"
        assert (
            getattr(profile, "target_width_mm", None)
            or getattr(profile, "target_height_mm", None)
        )
