"""客户口径（2026-09-21 → 2026-09-22）：不要连续提问，判定看"意图"不看"答没答"。

判定依据是**客户这句话跟需求有没有关系**，由销售节点在语境里判断
（LLM 语义理解 + 规则，见 `offtopic_turn`），**不是**关键词、也不是
"有没有回答上一问"：

    与需求有关（给参数 / 答了别的一项 / 主动聊需求 / 问业务问题）
        → 直接"接住 + 追问缺的那一项"，不做"只承接"；
          实测 bug：客户答 "maybe 5m"（观看距离）却只收到一句寒暄，对话停住。
    与需求无关（闲聊 / 寒暄 / 题外话）
        → 第一条只承接（正文里不留问句）；
        → 第二条必须"接住 + 提问"写在**同一条**消息里；
    客户回到需求话题 → 计数清零。
    同一轮里客户连发多条消息（前端聚合）只算**一次**。
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
    """纯函数：这一轮"只承接"还是"可以问需求"。"""

    def test_requirement_related_turn_asks_next_question(self):
        """与需求有关 → 接住 + 追问，不是"只承接"（哪怕没回答上一问）。"""
        decision = decide_continuation(has_question=True, off_topic=False, ack_streak=0)
        assert decision.suppress_question is False
        assert decision.reason == "requirement_related"
        assert decision.next_streak == 0

    def test_requirement_related_turn_resets_the_counter(self):
        decision = decide_continuation(has_question=True, off_topic=False, ack_streak=3)
        assert decision.suppress_question is False
        assert decision.next_streak == 0

    def test_first_off_topic_turn_only_acks(self):
        decision = decide_continuation(has_question=True, off_topic=True, ack_streak=0)
        assert decision.suppress_question is True
        assert decision.reason == "customer_off_topic_chat_first"
        assert decision.next_streak == 1

    def test_second_off_topic_turn_acks_and_asks(self):
        """第二条闲聊 → 承接额度用完 → 接住 + 提问（同一条消息）。"""
        decision = decide_continuation(
            has_question=True, off_topic=True, ack_streak=MAX_ACK_STREAK
        )
        assert decision.suppress_question is False
        assert decision.reason == "ack_budget_exhausted"
        assert decision.next_streak == 0

    def test_budget_is_one_chat_then_the_question(self):
        assert MAX_ACK_STREAK == 1
        first = decide_continuation(has_question=True, off_topic=True, ack_streak=0)
        second = decide_continuation(
            has_question=True, off_topic=True, ack_streak=first.next_streak
        )
        assert first.suppress_question is True, "第一条闲聊：只承接"
        assert second.suppress_question is False, "第二条闲聊：接住 + 提问"

    def test_off_topic_without_question_still_counts(self):
        decision = decide_continuation(has_question=False, off_topic=True, ack_streak=0)
        assert decision.suppress_question is False
        assert decision.reason == "no_question_this_turn"
        assert decision.next_streak == 1


class _StubSales:
    """每轮都给出"要问的那一项"（模拟真实链路的 pending_question）。

    `offtopic_messages` 里的句子模拟"销售节点在语境里判定这是闲聊"
    （真实链路由 LLM 语义理解 + 规则给出，见 requirement.py 的 offtopic_turn）。
    """

    def __init__(self, offtopic_messages=()):
        self.pending_slot = "environment"
        self.offtopic_messages = {item.strip().lower() for item in offtopic_messages}
        self.seen = []

    def _is_offtopic(self, message: str) -> bool:
        return message.strip().lower() in self.offtopic_messages

    def run(self, session_id, message, has_vision=False, **_kwargs):
        self.seen.append(message)
        question = {
            "environment": "Will the screen be installed indoors or outdoors?",
            "pixel_pitch": "What pixel pitch do you have in mind?",
            "size": "What screen size do you have in mind (width x height)?",
        }.get(self.pending_slot, "Anything else I should know?")
        offtopic = self._is_offtopic(message)
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": f"{message.strip()}. {question}",
            "requirements": {},
            "products": [],
            "pending_question": question,
            "pending_slot": self.pending_slot,
            "acknowledgement": f"{message.strip()} — noted.",
            "offtopic_turn": offtopic,
            "speech_act": {"speech_act": "CASUAL" if offtopic else "NEW_REQUIREMENT"},
            "dialogue_action": {"action": "ask_only", "priority": 3},
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestNoConsecutiveQuestions:

    def _run(self, session_id, messages, *, offtopic, pending_slot="environment"):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        sales = _StubSales(offtopic_messages=offtopic)
        sales.pending_slot = pending_slot
        orch = DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())
        try:
            return [orch.process_message(message, session_id) for message in messages]
        finally:
            memory.clear(session_id)

    def test_requirement_messages_always_get_a_question(self):
        """客户说的都是需求相关的话 → 每轮"接住 + 追问"，绝不只寒暄。"""
        results = self._run(
            "ack-req",
            ["i need a led display", "3*5", "maybe 5m"],
            offtopic=["haha nice weather"],
        )
        for index, result in enumerate(results):
            assert result["question_count"] == 1, (index, result["response"])
            assert "?" in result["response"], (index, result["response"])
            assert result["continuation"]["reason"] == "requirement_related"
            assert "suppressed_question" not in result
            assert result["ack_streak"] == 0

    def test_second_off_topic_turn_acks_and_asks_in_one_message(self):
        results = self._run(
            "ack-offtopic",
            ["i need a led display", "haha i am in nairobi", "nice weather today"],
            offtopic=["haha i am in nairobi", "nice weather today"],
        )
        # 第 1 轮：需求相关 → 正常问
        assert results[0]["question_count"] == 1
        # 第 2 轮：第一条闲聊 → 只承接，正文里不留问句
        assert results[1]["question_count"] == 0, results[1]["response"]
        assert "?" not in results[1]["response"]
        assert results[1]["suppressed_question"]["slot"] == "environment"
        assert results[1]["continuation"]["reason"] == "customer_off_topic_chat_first"
        assert results[1]["ack_streak"] == 1
        # 第 3 轮：第二条闲聊 → 接住这句话 + 同一条消息里提问
        assert results[2]["question_count"] == 1, results[2]["response"]
        assert "nice weather today" in results[2]["response"], "必须先接住客户这句话"
        assert results[2]["continuation"]["reason"] == "ack_budget_exhausted"
        assert results[2]["ack_streak"] == 0

    def test_customer_returning_to_requirements_resets_the_counter(self):
        """闲聊一句之后客户又聊需求 → 立刻恢复正常节奏，计数清零。"""
        memory.clear("ack-mixed")
        memory.mark_first_contact_done("ack-mixed")
        sales = _StubSales(offtopic_messages=["nice weather today"])
        sales.pending_slot = "environment"
        orch = DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())
        try:
            first = orch.process_message("i need a led display", "ack-mixed")
            assert first["question_count"] == 1
            small_talk = orch.process_message("nice weather today", "ack-mixed")
            assert small_talk["question_count"] == 0
            assert small_talk["ack_streak"] == 1
            # 客户回到需求话题（换了另一项信息）→ 直接追问，不再承接
            sales.pending_slot = "size"
            back = orch.process_message("3*5 indoor", "ack-mixed")
            assert back["question_count"] == 1, back["response"]
            assert back["continuation"]["reason"] == "requirement_related"
            assert back["ack_streak"] == 0
        finally:
            memory.clear("ack-mixed")
