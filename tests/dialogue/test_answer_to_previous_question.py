"""v2.6 §9：客户回答必须优先匹配上一轮 AI 问的那一项。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_PREVIOUS_QUESTION,
    ANSWER_WRONG_SLOT,
    NEW_REQUIREMENT,
    NO_REQUIREMENT,
    ConversationState,
    match_answer_to_question,
)


class TestAnswerToPreviousQuestion:

    def test_bare_pitch_answers_pitch_question_merged(self):
        """合并自 8 条同类测试（瘦身；断言全部保留）。"""

        # ── test_bare_pitch_answers_pitch_question ──
        match = match_answer_to_question("P3", last_question_slot="pixel_pitch")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "pixel_pitch"
        assert match.expected_slot == "pixel_pitch"

        # ── test_distance_answers_distance_question ──
        match = match_answer_to_question("about 5m", last_question_slot="viewing_distance")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "viewing_distance"

        # ── test_environment_answer_matches_environment_question ──
        match = match_answer_to_question("indoor", last_question_slot="environment")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "environment"

        # ── test_bare_both_answers_the_two_option_question ──
        """§9：both / 都行 本身没信息量，只能靠上一轮问的是哪一项落地。"""
        match = match_answer_to_question("both", last_question_slot="price_preference")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "price_preference"

        # ── test_answer_still_wins_when_more_info_is_given ──
        match = match_answer_to_question("indoor, 3x5m", last_question_slot="environment")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "environment"
        assert "size" in match.covered_slots, "同一句里的其它信息也要一起带上"

        # ── test_new_requirement_without_a_previous_question ──
        match = match_answer_to_question("P3", last_question_slot="")
        assert match.kind == NEW_REQUIREMENT
        assert match.slot == "pixel_pitch"

        # ── test_no_requirement_when_message_has_no_facts ──
        match = match_answer_to_question("how long is delivery?", last_question_slot="size")
        assert match.kind == NO_REQUIREMENT
        assert match.slots == {}

        # ── test_state_helper_uses_last_question_slot ──
        state = ConversationState(session_id="ans-1")
        state.note_ai_turn(question="Which pitch?", slot="pixel_pitch")
        match = state.answer_to("P3")
        assert match.kind == ANSWER_PREVIOUS_QUESTION
        assert match.slot == "pixel_pitch"
        assert ANSWER_WRONG_SLOT != match.kind
