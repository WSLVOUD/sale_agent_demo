"""客户口径（2026-09-22）：闲聊才承接，需求相关就追问 —— 判定看语境不看关键词。

实测日志（客户原话）：

    i need a led display → 3*5 → indoor
    🤖 …Is this going indoors or outdoors?          ← 已经说过 indoor 还在问
    maybe 5m
    🤖 Five metres of viewing distance tells me a lot…（只寒暄，没有下一问）

第二条回复的问题：`maybe 5m` 是**需求信息**（观看距离），却被当成
"没回答上一问" → 只承接、不追问，对话就停住了。

这里锁两件事：

1. `is_offtopic_message` 的判定 —— 需求相关（含 LLM 语义层给的结果）不算闲聊；
2. 端到端：需求相关的每一轮都必须"接住 + 追问"，只有连续闲聊才出现"只承接"。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.sales.nodes.requirement import is_offtopic_message  # noqa: E402
from src.memory.store import memory  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402


def _slots(message: str):
    return {
        key: value
        for key, value in (extract_slots(message) or {}).items()
        if not str(key).startswith("_")
    }


class TestOffTopicJudgement:

    def test_requirement_messages_are_not_offtopic(self):
        for message in (
            "i need a led display",
            "3*5",
            "indoor",
            "maybe 5m",
            "P3",
            "we are putting it up in our church hall",
        ):
            slots = _slots(message)
            if message == "we are putting it up in our church hall":
                # 规则解析不出来也没关系 —— LLM 语义层（usage）给到就算"在聊需求"
                assert is_offtopic_message(message, usage="church") is False, message
                continue
            assert is_offtopic_message(message, rule_slots=slots) is False, message

    def test_semantic_layer_counts_as_requirement(self):
        """LLM 语义层抽到的字段（本例：size）也算"在聊需求"，不看关键词。"""
        assert is_offtopic_message(
            "the whole front wall, about 8 metres across",
            semantic_payload={"size": "8m"},
        ) is False

    def test_business_questions_are_not_offtopic(self):
        for message in ("delivery time?", "how much does it cost?", "你们公司在哪里"):
            assert is_offtopic_message(message, rule_slots=_slots(message)) is False, message

    def test_small_talk_is_offtopic(self):
        for message in (
            "haha i am in nairobi",
            "nice weather today",
            "i am jack",
        ):
            assert is_offtopic_message(message, rule_slots=_slots(message)) is True, message


class _IntentStubSales:
    """按**真实判定**给 offtopic_turn：规则解析 + 语义层（这里用规则模拟）。"""

    def __init__(self):
        self.pending_slot = "environment"

    def run(self, session_id, message, has_vision=False, **_kwargs):
        slots = _slots(message)
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
            "offtopic_turn": is_offtopic_message(message, rule_slots=slots),
            "speech_act": {"speech_act": "NEW_REQUIREMENT"},
            "dialogue_action": {"action": "ask_only", "priority": 3},
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestRequirementTurnsAlwaysAsk:

    def _run(self, session_id, messages):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        orch = DualAgentOrchestrator(
            sales_agent=_IntentStubSales(), solution_agent=_StubSolution()
        )
        try:
            return [orch.process_message(message, session_id) for message in messages]
        finally:
            memory.clear(session_id)

    def test_logged_sequence_keeps_asking(self):
        """实测日志那串需求消息：每一轮都要"接住 + 追问"，不能只寒暄。"""
        results = self._run(
            "offtopic-req", ["i need a led display", "3*5", "indoor", "maybe 5m"]
        )
        for index, result in enumerate(results):
            assert result["question_count"] == 1, (index, result["response"])
            assert result["continuation"]["reason"] == "requirement_related", index
            assert "suppressed_question" not in result, index
            # 同一个问题不能连着问两遍（重复提问闸门）
            assert result["response"].count("?") == 1, result["response"]

    def test_only_consecutive_small_talk_is_acknowledged_alone(self):
        results = self._run(
            "offtopic-chat",
            ["i need a led display", "haha i am in nairobi", "nice weather today"],
        )
        assert results[0]["question_count"] == 1
        # 第一条闲聊：只承接
        assert results[1]["question_count"] == 0, results[1]["response"]
        assert results[1]["continuation"]["reason"] == "customer_off_topic_chat_first"
        # 第二条闲聊：接住 + 提问写在同一条消息里
        assert results[2]["question_count"] == 1, results[2]["response"]
        assert "nice weather today" in results[2]["response"]
        assert results[2]["continuation"]["reason"] == "ack_budget_exhausted"
