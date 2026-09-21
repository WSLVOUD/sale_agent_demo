"""v2.6 §26：计划里点名的 10 条核心回归用例（Case 1~10）。

这组用例对应计划里的每一个 Case，跑通它们就说明 v2.6 的两条硬约束站得住：

    ① ONE TURN → ONE ACTION → ONE RESPONSE
    ② RequirementProfile + ConversationState 双状态架构
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_PREVIOUS_QUESTION,
    ANSWER_WRONG_SLOT,
    ASK,
    MAX_QUESTIONS_PER_TURN,
    QUESTION_PRIORITY_ENGINEERING,
    QUESTION_PRIORITY_SALES_PREFERENCE,
    ConversationState,
    DialogueDecision,
    FinalResponseCoordinator,
    canonical_slots,
    environment_needs_asking,
    explicit_slots,
    get_conversation_state,
    match_answer_to_question,
    next_question_plan,
    reset_conversation_state,
    select_single_action,
    should_append_requirement_question,
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


def _next_slot(message: str, session_id: str):
    reset_conversation_state(session_id)
    profile = _profile_from_message(message)
    plan = next_question_plan(
        profile, session_id=session_id, conversation=get_conversation_state(session_id)
    )
    return getattr(plan, "slot", None)


class TestCase1GenericRequest:

    def test_only_environment_is_asked(self):
        assert _next_slot("i need a display", "case-1") == "environment"


class TestCase2SizeOnly:

    def test_size_is_kept_and_environment_still_asked(self):
        profile = _profile_from_message("3*5")
        assert (
            getattr(profile, "target_width_mm", None)
            or getattr(profile, "target_height_mm", None)
        )
        assert environment_needs_asking(profile) is True
        assert _next_slot("3*5", "case-2") == "environment"


class TestCase3Indoor:

    def test_at_most_one_more_question(self):
        profile = _profile_from_message("indoor")
        assert getattr(profile, "environment", "") == "indoor"
        slot = _next_slot("indoor", "case-3")
        assert slot and slot != "environment"


class TestCase4Pitch:

    def test_pitch_recorded_and_at_most_one_question(self):
        profile = _profile_from_message("P3")
        assert getattr(profile, "pixel_pitch_mm", None) == 3.0
        slot = _next_slot("P3", "case-4")
        assert slot is not None


class TestCase5NoDuplicateDistanceQuestion:

    def test_distance_is_not_reasked(self):
        slots = {"environment": "indoor", "pixel_pitch_mm": 3.0}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        session_id = "case-5"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", "") != "viewing_distance"


class TestCase6MultiRequirement:

    def test_one_update_not_four_questions(self):
        message = "Indoor, 3x5m, P3, fixed"
        raw = {
            key: value
            for key, value in (extract_slots(message) or {}).items()
            if not str(key).startswith("_")
        }
        covered = canonical_slots(explicit_slots(message))
        assert len(covered) >= 3
        profile = RequirementProfile.from_slots(raw, explicit_keys=set(raw))
        session_id = "case-6"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", None) not in covered


class TestCase7CustomerQuestionFirst:

    def test_delivery_question_has_no_requirement_question(self):
        assert should_append_requirement_question("How long is delivery?", None) is False


class TestCase8WrongSlotKeepsInformation:

    def test_size_answer_to_environment_question(self):
        match = match_answer_to_question("3x5m", last_question_slot="environment")
        assert match.kind == ANSWER_WRONG_SLOT
        assert match.slot == "size"
        assert "size" in match.covered_slots
        assert match.slots, "原始槽位（宽 / 高）同样要保留"


class TestCase9SingleVisibleMessage:

    def test_extras_never_make_a_second_message(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Screen 1: TW11-3216-P3.0.",
            extras=["Would you like to add another screen?"],
            action="recommend_only",
        )
        assert final.response_count == 1
        assert final.question_count <= MAX_QUESTIONS_PER_TURN


class TestCase10SingleAction:

    def test_two_asks_leave_exactly_one(self):
        candidates = [
            DialogueDecision(
                action=ASK,
                question_slot="pixel_pitch",
                question_priority=QUESTION_PRIORITY_ENGINEERING,
                question_count=1,
            ),
            DialogueDecision(
                action=ASK,
                question_slot="price_preference",
                question_priority=QUESTION_PRIORITY_SALES_PREFERENCE,
                question_count=1,
            ),
        ]
        selected, discarded = select_single_action(candidates)
        assert selected.question_slot == "pixel_pitch"
        assert len(discarded) == 1


class TestConversationStateRoundTrip:

    def test_ai_question_then_customer_answer(self):
        """双状态架构的最小闭环：AI 问 P 值 → 客户答 P3 → 系统知道答的是哪一项。"""
        state = ConversationState(session_id="round-trip")
        state.note_ai_turn(
            action="ask_only", question="What pixel pitch?", slot="pixel_pitch", turn_id="t1"
        )
        match = state.answer_to("P3")
        state.note_customer_turn(
            text="P3", answer_slot=match.slot if match.answers_previous_question else ""
        )
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert state.registry.is_answered("pixel_pitch") is True
        assert state.current_answer_slot == "pixel_pitch"
