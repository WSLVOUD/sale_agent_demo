"""v2.7 §18（Phase 7）：Answer Coverage —— 答非所问也不能立刻重问同一项。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import compute_answer_coverage, match_answer_to_question  # noqa: E402


class TestAnswerCoverage:

    def test_asked_size_answered_distance(self):
        """计划 §18 的原文例子。"""
        match = match_answer_to_question("Viewing distance is 8 meters", last_question_slot="size")
        coverage = compute_answer_coverage(
            asked_slot="size", match=match, newly_filled_slots=["viewing_distance"]
        )
        assert coverage.asked_slot == "size"
        assert "viewing_distance" in coverage.answered_slots
        assert coverage.answered_asked_slot is False
        assert coverage.is_wrong_slot is True
        assert "size" in coverage.unresolved_slots

    def test_answered_the_asked_slot(self):
        match = match_answer_to_question("P3", last_question_slot="pixel_pitch")
        coverage = compute_answer_coverage(
            asked_slot="pixel_pitch", match=match, newly_filled_slots=["pixel_pitch"]
        )
        assert coverage.answered_asked_slot is True
        assert coverage.is_wrong_slot is False
        assert coverage.unresolved_slots == []

    def test_multi_information_turn(self):
        match = match_answer_to_question(
            "Indoor, 3x5m, P3", last_question_slot="environment"
        )
        coverage = compute_answer_coverage(
            asked_slot="environment",
            match=match,
            newly_filled_slots=["environment", "size", "pixel_pitch"],
        )
        assert coverage.answered_asked_slot is True
        assert set(coverage.newly_filled_slots) >= {"size", "pixel_pitch"}

    def test_no_answer_at_all(self):
        coverage = compute_answer_coverage(asked_slot="size")
        assert coverage.answered_slots == []
        assert "size" in coverage.unresolved_slots

    def test_to_dict_is_serialisable(self):
        payload = compute_answer_coverage(asked_slot="size").to_dict()
        assert payload["asked_slot"] == "size"
        assert isinstance(payload["unresolved_slots"], list)
