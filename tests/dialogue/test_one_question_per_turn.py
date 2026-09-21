"""v2.6 §4.5/§4.6/§4.7：一个 turn 最多一个问题。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    MAX_QUESTIONS_PER_TURN,
    FinalResponseCoordinator,
    select_single_action,
)
from src.dialogue.action import (  # noqa: E402
    ASK,
    QUESTION_PRIORITY_ENGINEERING,
    QUESTION_PRIORITY_SALES_PREFERENCE,
    DialogueDecision,
)


class TestOneQuestionPerTurn:

    def test_two_questions_in_one_text_collapse_to_one(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Got it — P3. What is the viewing distance? Is it fixed or rental?",
            questions=[
                {"text": "What is the viewing distance?", "slot": "viewing_distance"},
                {"text": "Is it fixed or rental?", "slot": "installation"},
            ],
        )
        assert final.question_count <= MAX_QUESTIONS_PER_TURN
        assert final.dropped_questions, "被丢掉的问题要能追溯"

    def test_extras_never_become_a_second_bubble_with_a_question(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="P3 works well indoors.",
            extras=["Any other screens in this project?"],
        )
        assert final.response_count == 1
        assert final.question_count <= 1
        assert "other screens" in final.text, "内容不能丢，只是收进同一条回复"

    def test_answer_then_ask_has_at_most_one_question(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text=(
                "Got it — P3. What is the typical viewing distance, "
                "and is it fixed or rental?"
            ),
            questions=[
                {"text": "What is the typical viewing distance?", "slot": "viewing_distance"}
            ],
        )
        assert final.question_count <= 1

    def test_validation_marks_repair(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Indoor or outdoor? What size is the screen?",
            questions=[
                {"text": "Indoor or outdoor?", "slot": "environment"},
                {"text": "What size is the screen?", "slot": "size"},
            ],
        )
        assert final.validation_result in ("REPAIRED", "PASS")
        assert final.question_count <= 1


class TestSingleActionSelection:

    def test_policy_picks_one_action_and_discards_the_rest(self):
        """§26 Case 10：ask(pixel_pitch) + ask(viewing_distance) → 只留一个。"""
        candidates = [
            DialogueDecision(
                action=ASK,
                question_slot="viewing_distance",
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
        assert selected.question_slot == "viewing_distance"
        assert len(discarded) == 1
        assert discarded[0].question_slot == "price_preference"

    def test_no_candidates_means_no_question(self):
        selected, discarded = select_single_action([])
        assert discarded == []
        assert selected.question_count == 0
