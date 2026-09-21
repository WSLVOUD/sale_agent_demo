"""客户实测反馈（2026-09-21）的两个回归：

    ① 客户回答"更看重价格还是质量"时说的 "cost down the priority"，
       被旧实现当成"问价格" → AI 回了报价口径，偏好没被记下来。
    ② 客户明确说"给我推荐"时，如果条件不全，系统既不推荐也不问缺什么，
       只回一句"我这就给你准备"。

客户口径：客户要推荐时 —— 需求够了就推荐；不够就**只问缺的那一项**，
不要再多说无关的话。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.speech_act import detect_speech_act  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402
from src.rag.reply_composer import (  # noqa: E402
    is_price_question,
    is_price_question_with_context,
)


class TestPricePreferenceIsAnAnswerNotAQuestion:

    def test_preference_phrases_are_parsed(self):
        for message in (
            "cost down the priority",
            "keeping the cost down",
            "cheaper is better",
            "best quality",
        ):
            slots = {
                key: value
                for key, value in (extract_slots(message) or {}).items()
                if not str(key).startswith("_")
            }
            assert slots.get("price_preference"), (message, slots)

    def test_answering_the_preference_question_is_not_a_price_question(self):
        assert (
            is_price_question_with_context(
                "cost down the priority", last_asked_slot="price_preference"
            )
            is False
        )
        assert is_price_question("cost down the priority") is False

    def test_real_price_questions_still_detected(self):
        assert (
            is_price_question_with_context(
                "what is the price?", last_asked_slot="price_preference"
            )
            is True
        )
        assert is_price_question("please send me a quotation") is True
        assert is_price_question("多少钱") is True

    def test_speech_act_treats_it_as_a_requirement_answer(self):
        result = detect_speech_act("cost down the priority", last_asked_slot="price_preference")
        assert result.speech_act == "ANSWER_REQUIREMENT"
        assert "PRICE_QUESTION" not in (result.customer_questions or [])

    def test_speech_act_still_flags_a_real_price_question(self):
        result = detect_speech_act("what about the price?", last_asked_slot="price_preference")
        assert "PRICE_QUESTION" in (result.customer_questions or [])


class TestExplicitRecommendationRouting:
    """需求够 → 推荐；不够 → 只问缺的那一项。"""

    def _fake_llm(self, monkeypatch):
        import src.agents.sales.nodes.requirement as sales_req

        class _Response:
            content = '{"usage": "", "additional_requirements": [], "ack": ""}'

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return _Response()

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        return sales_req

    def _turn(self, module, message, profile, *, intent="product_question"):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "reco-request-test",
            "requirements": {},
            "additional_requirements": [],
            "intent": intent,
            "next_action": intent,
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "requirement_profile": profile,
        }
        return module.requirement_mining(state)

    def _ready_profile(self):
        slots = {
            "display_type": "LED",
            "environment": "indoor",
            "purpose": "church",
            "installation": "fixed",
            "target_width_mm": 5000,
            "target_height_mm": 10000,
            "viewing_distance_m": 5.0,
            "price_preference": "price",
        }
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def _incomplete_profile(self):
        slots = {"display_type": "LED", "target_width_mm": 5000, "target_height_mm": 10000}
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_ready_requirements_recommend_immediately(self, monkeypatch):
        module = self._fake_llm(monkeypatch)
        result = self._turn(module, "给我推荐", self._ready_profile())
        assert result["should_generate_solution"] is True
        assert result["next_action"] == "router", result.get("next_action")

    def test_incomplete_requirements_ask_instead_of_talking(self, monkeypatch):
        module = self._fake_llm(monkeypatch)
        result = self._turn(module, "给我推荐", self._incomplete_profile())
        assert result["should_generate_solution"] is False
        assert result["next_action"] == "ask", result.get("next_action")
        assert result["pending_question"], "必须问缺的那一项，而不是说空话"

    def test_english_request_also_routes(self, monkeypatch):
        module = self._fake_llm(monkeypatch)
        result = self._turn(module, "recommend one for me", self._incomplete_profile())
        assert result["next_action"] == "ask"
        assert result["pending_question"]

    def test_price_request_is_not_treated_as_a_recommendation(self, monkeypatch):
        """单纯要报价 → 仍然走"先说明报价口径"的路径，不被强行改成推荐。"""
        module = self._fake_llm(monkeypatch)
        result = self._turn(module, "请给我报价", self._incomplete_profile(), intent="objection")
        assert result["next_action"] != "router"
