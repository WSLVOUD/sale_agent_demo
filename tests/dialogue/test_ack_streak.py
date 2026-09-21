"""客户口径（2026-09-21）：不要连续提问。

    客户没回答 AI 的上一问 → 这一轮只"承接/闲谈"，不问问题；
    连续承接最多 3 条，第 4 条必须拉回需求问题；
    客户回答了上一问（回到需求话题）→ 立刻跟着客户话题走，承接计数清零；
    客户主动问业务问题 → 正常回答，不占承接额度。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import MAX_ACK_STREAK, decide_continuation  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402


class TestContinuationBudget:

    def test_answered_pending_question_follows_customer_topic(self):
        decision = decide_continuation(has_question=True, answered_pending=True, ack_streak=2)
        assert decision.suppress_question is False
        assert decision.reason == "answered_pending_question"
        assert decision.next_streak == 0

    def test_unanswered_question_chats_first(self):
        decision = decide_continuation(
            has_question=True, answered_pending=False, ack_streak=0
        )
        assert decision.suppress_question is True
        assert decision.next_streak == 1

    def test_fourth_turn_must_ask_about_requirements(self):
        decision = decide_continuation(
            has_question=True,
            answered_pending=False,
            ack_streak=MAX_ACK_STREAK,
        )
        assert decision.suppress_question is False
        assert decision.reason == "ack_budget_exhausted"
        assert decision.next_streak == 0

    def test_first_turn_has_no_pending_question(self):
        decision = decide_continuation(
            has_question=True, answered_pending=False, had_pending_question=False
        )
        assert decision.suppress_question is False
        assert decision.next_streak == 0

    def test_customer_question_does_not_consume_the_chat_budget(self):
        decision = decide_continuation(
            has_question=False,
            answered_pending=False,
            ack_streak=2,
            customer_question=True,
        )
        assert decision.suppress_question is False
        assert decision.next_streak == 2, "回答客户问题不算闲聊额度"

    def test_plain_chat_turn_consumes_the_budget(self):
        decision = decide_continuation(
            has_question=False, answered_pending=False, ack_streak=1
        )
        assert decision.next_streak == 2


class _StubSales:
    """每轮都给出"要问的那一项"（模拟真实链路的 pending_question）。"""

    def __init__(self):
        self.pending_slot = "environment"

    def run(self, session_id, message, has_vision=False, **_kwargs):
        question = {
            "environment": "Will the screen be installed indoors or outdoors?",
            "pixel_pitch": "What pixel pitch do you have in mind?",
            "size": "What screen size do you have in mind (width x height)?",
        }.get(self.pending_slot, "Anything else I should know?")
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": f"{message.strip()}. {question}",
            "requirements": {},
            "products": [],
            "pending_question": question,
            "pending_slot": self.pending_slot,
            "acknowledgement": f"{message.strip()} — noted.",
            "speech_act": {"speech_act": "NEW_REQUIREMENT"},
            "dialogue_action": {"action": "ask_only", "priority": 3},
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestNoConsecutiveQuestions:

    def _run(self, session_id, messages, pending_slot="environment"):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        sales = _StubSales()
        sales.pending_slot = pending_slot
        orch = DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())
        try:
            return [orch.process_message(message, session_id) for message in messages]
        finally:
            memory.clear(session_id)

    def test_first_turn_asks_and_next_three_turns_chat_only(self):
        results = self._run(
            "ack-streak-1",
            ["i need a display", "3*5", "3*5", "3*5", "3*5"],
        )
        # 第 1 轮：没有上一问 → 正常问
        assert results[0]["question_count"] == 1
        assert "?" in results[0]["response"]
        # 第 2~4 轮：客户没回答 → 只承接，不提问（最多 3 条）
        for index in (1, 2, 3):
            assert results[index]["question_count"] == 0, results[index]["response"]
            assert "?" not in results[index]["response"]
            assert results[index]["suppressed_question"]["slot"] == "environment"
            assert results[index]["ack_streak"] == index
        # 第 5 轮：承接额度用完 → 必须拉回需求
        assert results[4]["question_count"] == 1
        assert results[4]["ack_streak"] == 0
        assert results[4]["continuation"]["reason"] == "ack_budget_exhausted"

    def test_customer_answering_resumes_the_normal_rhythm(self):
        """客户回答上一问（回到需求话题）→ 立刻接住 + 问下一项。"""
        memory.clear("ack-streak-2")
        memory.mark_first_contact_done("ack-streak-2")
        sales = _StubSales()
        sales.pending_slot = "environment"
        orch = DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())
        try:
            first = orch.process_message("i need a display", "ack-streak-2")
            assert first["question_count"] == 1
            # 客户这次正面回答室内外 → 不再承接
            sales.pending_slot = "size"
            second = orch.process_message("indoor", "ack-streak-2")
            assert second["question_count"] == 1, second["response"]
            assert second["continuation"]["suppress_question"] is False
            assert second["ack_streak"] == 0
        finally:
            memory.clear("ack-streak-2")
