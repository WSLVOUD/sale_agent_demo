"""v2.7 §19/§20（Phase 8）：重复提问闸门 + 回复条数护栏。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    DuplicateQuestionFirewall,
    ResponseCountGuard,
)


class TestDuplicateQuestionFirewall:

    def setup_method(self):
        self.firewall = DuplicateQuestionFirewall()

    def test_allows_a_new_slot(self):
        decision = self.firewall.check(current_slot="size", previous_slot="environment")
        assert decision.allowed is True

    def test_blocks_the_same_slot_without_new_information(self):
        """§19.1：上一轮问的就是它、客户又没给新信息 → 绝对禁止再问。"""
        decision = self.firewall.check(
            current_slot="environment",
            previous_slot="environment",
            answered_slots=["size"],
            newly_filled_slots=["size"],
            question_state="ASKED",
        )
        assert decision.allowed is False
        assert decision.reason == "duplicate_question_without_new_info"
        assert decision.blocked_slot == "environment"

    def test_allows_repeat_when_customer_gave_new_information(self):
        decision = self.firewall.check(
            current_slot="size",
            previous_slot="size",
            newly_filled_slots=["size"],
        )
        assert decision.allowed is True
        assert decision.reason == "new_information_about_slot"

    def test_blocks_regeneration_within_the_same_turn(self):
        decision = self.firewall.check(
            current_slot="size",
            previous_slot="pixel_pitch",
            current_turn_id="T1",
            previous_turn_id="T1",
        )
        assert decision.allowed is False
        assert decision.reason == "same_turn_regeneration"

    def test_blocks_more_than_one_question(self):
        decision = self.firewall.check(current_slot="size", response_count=2)
        assert decision.allowed is False
        assert decision.reason == "response_count_exceeded"

    def test_no_question_is_always_allowed(self):
        assert self.firewall.check(current_slot="").allowed is True


class TestResponseCountGuard:

    def test_single_response_budget(self):
        guard = ResponseCountGuard()
        assert guard.check(response_count=1).allowed is True
        guard.spend()
        assert guard.check(response_count=1).allowed is False
        assert guard.check(response_count=2).allowed is False
        guard.reset()
        assert guard.check(response_count=1).allowed is True
