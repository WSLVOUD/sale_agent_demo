"""v2.6 §17：客户主动提问优先级最高（先回答，不硬塞需求问题）。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ANSWER_ONLY,
    P0_CUSTOMER_QUESTION,
    decide_action,
    decide_speech_policy,
    detect_speech_act,
    should_append_requirement_question,
)
from src.models.requirement import RequirementProfile  # noqa: E402


class TestCustomerQuestionPriority:

    def test_case7_delivery_question_answers_only(self):
        """§26 Case 7：How long is delivery? → 先回答，不跳去问题采集。"""
        speech = detect_speech_act("How long is delivery?")
        action = decide_action(speech, profile=None)
        assert action.action == ANSWER_ONLY
        assert action.priority == P0_CUSTOMER_QUESTION

    def test_delivery_question_does_not_append_requirement_question(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor"}, explicit_keys={"environment"}
        )
        assert should_append_requirement_question("How long is delivery?", profile) is False

    def test_price_question_does_not_append_requirement_question(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor"}, explicit_keys={"environment"}
        )
        assert should_append_requirement_question("what's the price?", profile) is False

    def test_policy_bundle_reports_customer_question_priority(self):
        bundle = decide_speech_policy("How long is delivery?")
        assert bundle["dialogue_action"]["priority"] == P0_CUSTOMER_QUESTION
        assert bundle["dialogue_action"]["priority_label"] == "customer_question"
