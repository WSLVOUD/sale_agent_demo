"""v2.7 §9/§10（Phase 9）：一个 Turn 只有一个 Action + "缺失 ≠ 现在必须问"。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER,
    ANSWER_AND_ASK,
    ASK,
    CLARIFY,
    P0_CUSTOMER_QUESTION,
    P3_NATURAL_CONTINUATION,
    P4_REQUIRED_FOR_RECOMMENDATION,
    P5_AUXILIARY,
    RECOMMEND,
    WAIT,
    decide_turn_action,
    next_candidate_slot,
)


class TestSingleTurnAction:

    def test_customer_question_is_answered_first(self):
        """§10：即使还缺 viewing_distance，也先答 delivery。"""
        action = decide_turn_action(
            customer_question=True,
            question_kind="DELIVERY_QUESTION",
            missing_slots=["viewing_distance", "size"],
            question_candidates=["viewing_distance", "size"],
        )
        assert action.action == ANSWER
        assert action.priority == P0_CUSTOMER_QUESTION
        assert action.asks_question is False

    def test_product_question_asks_only_when_really_needed(self):
        needed = decide_turn_action(
            customer_question=True,
            question_kind="PRODUCT_QUESTION",
            missing_slots=["size", "installation"],
            question_candidates=["size", "installation"],
        )
        assert needed.action == ANSWER_AND_ASK
        assert needed.target_slot == "size"

        not_needed = decide_turn_action(
            customer_question=True,
            question_kind="PRODUCT_QUESTION",
            missing_slots=["installation", "purpose"],
            question_candidates=["installation", "purpose"],
        )
        assert not_needed.action == ANSWER
        assert not_needed.asks_question is False

    def test_momentum_wins_over_generic_missing_slot(self):
        """§11：顺着客户刚聊的话题问，而不是重新扫描所有 missing。"""
        action = decide_turn_action(
            newly_filled_slots=["size"],
            missing_slots=["installation", "pixel_pitch", "viewing_distance"],
            question_candidates=["pixel_pitch", "installation", "viewing_distance"],
            momentum_slot="pixel_pitch",
        )
        assert action.action == ASK
        assert action.target_slot == "pixel_pitch"
        assert action.priority == P3_NATURAL_CONTINUATION

    def test_required_for_recommendation_is_asked(self):
        action = decide_turn_action(
            missing_slots=["installation", "size"],
            question_candidates=["installation", "size"],
            hard_gate_slot="size",
        )
        assert action.action == ASK
        assert action.target_slot == "size"
        assert action.priority == P4_REQUIRED_FOR_RECOMMENDATION

    def test_conflict_clarifies_first(self):
        action = decide_turn_action(conflicts=["size_conflict"])
        assert action.action == CLARIFY

    def test_ready_to_recommend(self):
        action = decide_turn_action(ready_to_recommend=True, missing_slots=[])
        assert action.action == RECOMMEND

    def test_nothing_required_uses_wait_not_a_made_up_question(self):
        action = decide_turn_action(
            missing_slots=[],
            question_candidates=[],
            newly_filled_slots=[],
        )
        assert action.action == WAIT
        assert action.priority == P5_AUXILIARY
        assert action.asks_question is False

    def test_blocked_slot_is_not_reasked(self):
        action = decide_turn_action(
            missing_slots=["environment", "size"],
            question_candidates=["environment", "size"],
            blocked_slot="environment",
        )
        assert action.target_slot == "size"

    def test_only_one_action_field_is_returned(self):
        action = decide_turn_action(
            customer_question=True,
            question_kind="PRODUCT_QUESTION",
            missing_slots=["size"],
            question_candidates=["size"],
            momentum_slot="size",
        )
        assert action.action in ("ANSWER", "ANSWER_AND_ASK", "ASK", "WAIT", "RECOMMEND", "CLARIFY")
        assert isinstance(action.to_dict()["action"], str)


class TestNextCandidateSlot:

    def test_skips_blocked_and_answered(self):
        assert next_candidate_slot(
            ["environment", "size", "pixel_pitch"],
            blocked_slot="environment",
            answered_slots=["size"],
        ) == "pixel_pitch"

    def test_returns_empty_when_nothing_left(self):
        assert next_candidate_slot(["environment"], blocked_slot="environment") == ""
