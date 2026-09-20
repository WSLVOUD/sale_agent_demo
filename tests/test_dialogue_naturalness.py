"""v2.5 Phase 2：DialogueAction + ResponseContext + 自然表达 + 话术质量指标。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ACK_ONLY,
    ANSWER_AND_ASK,
    CLARIFY,
    CONFIRM,
    DIRECT_ANSWER,
    RECOMMEND,
    ResponseContext,
    build_context,
    compose_from_context,
    compute_metrics,
    decide_dialogue_action,
    generate_response,
    validate_response,
)
from src.dialogue.action import ASK  # noqa: E402


class TestDialogueAction:

    def test_customer_question_is_answered_first(self):
        decision = decide_dialogue_action(customer_question=True)
        assert decision.action == DIRECT_ANSWER

    def test_customer_question_plus_missing_info_answers_then_asks(self):
        decision = decide_dialogue_action(customer_question=True, has_pending_question=True)
        assert decision.action == ANSWER_AND_ASK

    def test_recommend_when_requested(self):
        assert decide_dialogue_action(recommend_requested=True).action == RECOMMEND
        assert decide_dialogue_action(ready_to_recommend=True).action == RECOMMEND

    def test_clarify_on_conflict(self):
        assert decide_dialogue_action(conflicts=True).action == CLARIFY

    def test_confirm_for_vision(self):
        assert decide_dialogue_action(needs_confirmation=True).action == CONFIRM

    def test_ask_when_missing_hard_condition(self):
        decision = decide_dialogue_action(has_pending_question=True, question_slot="size")
        assert decision.action == ASK and decision.question_slot == "size"

    def test_ack_only_when_nothing_to_ask(self):
        assert decide_dialogue_action(facts_added=True).action == ACK_ONLY
        assert decide_dialogue_action().action == ACK_ONLY

    def test_never_defaults_to_ack_plus_ask(self):
        """不能每轮都是 ACK → Connector → ASK：行为必须随输入变化。"""
        actions = {
            decide_dialogue_action(customer_question=True).action,
            decide_dialogue_action(has_pending_question=True).action,
            decide_dialogue_action(ready_to_recommend=True).action,
            decide_dialogue_action(conflicts=True).action,
            decide_dialogue_action(facts_added=True).action,
        }
        assert len(actions) >= 4


class TestResponseContextAndGeneration:

    def _context(self, **overrides):
        payload = dict(
            action=RECOMMEND,
            customer_message="we need an indoor church screen",
            recommendation={"model": "TW11-3216-P3.0", "reasons": ["pitch fits the distance"]},
        )
        payload.update(overrides)
        return build_context(**payload)

    def test_context_carries_decisions_not_judgement(self):
        context = self._context(
            engineering_result={"resolution_fit": {"target": [3840, 2160], "actual": [3780, 2160], "fit_level": "NEAR_MATCH"}}
        )
        block = context.prompt_block()
        assert "already decided, do not change" in block
        assert "3780" in block and "NEAR_MATCH" in block

    def test_structured_compose_mentions_model_and_one_question(self):
        context = self._context(
            question="Shall I prepare the quotation?", why="",
        )
        text = compose_from_context(context)
        assert "TW11-3216-P3.0" in text
        assert text.count("?") == 1

    def test_answer_and_ask_keeps_both(self):
        context = build_context(
            action=ANSWER_AND_ASK,
            answer="Delivery is usually 15 days.",
            question="What screen size do you need?",
        )
        text = compose_from_context(context)
        assert text.startswith("Delivery is usually 15 days.")
        assert text.rstrip().endswith("What screen size do you need?")

    def test_generator_falls_back_without_llm(self):
        context = self._context(question="What width and height do you need?")
        assert "TW11-3216-P3.0" in generate_response(context)

    def test_generator_rejects_bad_polish(self):
        class _BadLLM:
            def invoke(self, _prompt):
                class _R:
                    content = "Got it. What's the size? And the pitch?"
                return _R()

        context = self._context(question="What width and height do you need?")
        text = generate_response(context, llm=_BadLLM())
        # 润色结果不合格（泛 ACK + 两个问句）→ 退回结构化拼装
        assert text.count("?") == 1
        assert not text.lower().startswith("got it")

    def test_context_to_dict(self):
        assert self._context().to_dict()["action"] == RECOMMEND


class TestResponseValidation:

    def test_generic_ack_detected(self):
        result = validate_response(
            "Thanks for the information. What size do you need?", allow_ack=False
        )
        assert result.has_generic_ack is True and "generic_ack" in result.issues

    def test_connector_detected(self):
        result = validate_response("Based on that, what is the viewing distance?")
        assert result.has_connector is True

    def test_too_many_questions(self):
        result = validate_response("What is the size? And the pitch?")
        assert result.question_count == 2 and "too_many_questions" in result.issues

    def test_internal_term_leak(self):
        result = validate_response("Your RequirementProfile says size is missing.")
        assert result.internal_terms and "internal_term_leak" in result.issues

    def test_price_is_blocked(self):
        assert "mentions_price" in validate_response("It costs about $5000.").issues

    def test_unsupported_parameter(self):
        result = validate_response(
            "This P5 model fits.", supported_parameters=["P3"]
        )
        assert result.unsupported_parameters == ["P5"]

    def test_customer_question_answered(self):
        result = validate_response(
            "Delivery usually takes 15 days.",
            customer_question=True, answer="Delivery usually takes 15 days.",
        )
        assert result.customer_question_answered is True
        missed = validate_response(
            "What size do you need?", customer_question=True,
            answer="Delivery usually takes 15 days.",
        )
        assert missed.customer_question_answered is False

    def test_metrics_aggregate(self):
        samples = [
            {"is_question": True, "validation": validate_response("Got it. Size?", allow_ack=False)},
            {"is_question": False, "validation": validate_response("Sure, what size do you need?")},
            {"is_question": True, "validation": validate_response("Delivery is 15 days.")},
        ]
        metrics = compute_metrics(samples)
        assert set(metrics) == {
            "generic_ack_rate", "connector_repeat_rate", "question_repeat_rate",
            "one_question_compliance", "internal_term_leak_rate",
            "unsupported_fact_rate", "customer_question_answer_rate",
        }
        assert metrics["generic_ack_rate"] > 0
