"""v2.6 §4.3/§4.7/§24：最终返回对象唯一，且只有一个客户可见回复。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ONE_TURN_ONE_ACTION,
    ONE_TURN_ONE_RESPONSE,
    FinalResponse,
    FinalResponseCoordinator,
)


class TestFinalResponse:

    def test_structure_has_planned_fields(self):
        final = FinalResponse(text="Hello", action="acknowledge_only", turn_id="t-1")
        payload = final.to_dict()
        for key in (
            "text",
            "action",
            "question_count",
            "question_slot",
            "facts_used",
            "warnings",
            "turn_id",
            "validation_result",
        ):
            assert key in payload, key
        assert payload["response_count"] == 1

    def test_only_one_customer_visible_response(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Indoor or outdoor?",
            extras=["What pixel pitch do you need?"],
            questions=[{"text": "Indoor or outdoor?", "slot": "environment"}],
            action="ask_only",
            question_slot="environment",
            turn_id="t-2",
        )
        assert final.response_count == 1
        assert final.turn_id == "t-2"
        assert final.action == "ask_only"
        assert final.question_count <= 1

    def test_first_contact_assets_are_a_separate_channel(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Hello, I'm Mike from iSEMC.",
            first_contact_messages=[{"role": "assistant", "content": "intro"}],
        )
        assert final.extra_channels == ["first_contact"]
        assert ONE_TURN_ONE_RESPONSE and ONE_TURN_ONE_ACTION

    def test_empty_text_is_still_a_single_response(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(text="")
        assert final.response_count == 1
        assert final.question_count == 0

    def test_facts_are_carried_with_the_response(self):
        coordinator = FinalResponseCoordinator()
        final = coordinator.build(
            text="Indoor it is.",
            facts=[{"field": "environment", "value": "indoor", "source": "customer"}],
        )
        assert final.facts_used == [
            {"field": "environment", "value": "indoor", "source": "customer"}
        ]
