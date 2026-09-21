"""v2.6 §4（端到端）：编排器收口之后，一个 turn 只有一条客户可见回复。

用桩 Agent 跑真实编排链路，验证：

    · 追加气泡（extra_messages）不再单独发给客户；
    · 主回复 + 追加气泡合起来最多一个问题；
    · 结果里带着唯一的 FinalResponse / turn_id / action。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.store import memory  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402


class _StubSales:
    """按剧本返回：给出一个待问项 + 一句回答。"""

    def __init__(self, response="Got it.", pending_question="", pending_slot=""):
        self.response = response
        self.pending_question = pending_question
        self.pending_slot = pending_slot

    def run(self, session_id, message, has_vision=False, **_kwargs):
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": self.response,
            "requirements": {},
            "products": [],
            "pending_question": self.pending_question,
            "pending_slot": self.pending_slot,
            "speech_act": {"speech_act": "NEW_REQUIREMENT"},
            "dialogue_action": {"action": "ask_only", "priority": 3},
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestOrchestratorSingleOutput:

    def _orchestrator(self, sales):
        return DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())

    def test_extras_are_merged_and_not_returned_separately(self):
        session_id = "v26-single-output-1"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = self._orchestrator(_StubSales())
            result = {
                "response": "Screen 1: TW11-3216-P3.0.",
                "extra_messages": ["Any other screens in this project?"],
                "pending_question": "What pixel pitch do you need?",
                "pending_slot": "pixel_pitch",
                "next_action": "ask",
            }
            orch._finalize_turn_response(result, session_id, "i need a display")

            assert result["response_count"] == 1
            assert not result.get("extra_messages")
            assert result["question_count"] <= 1
            assert "TW11-3216-P3.0" in result["response"]
            assert isinstance(result.get("final_response"), dict)
            assert result["final_response"]["response_count"] == 1
            assert result.get("turn_id")
        finally:
            memory.clear(session_id)

    def test_process_message_returns_one_question_at_most(self):
        session_id = "v26-single-output-2"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            sales = _StubSales(
                response="P3 works well indoors.",
                pending_question="Indoor or outdoor?",
                pending_slot="environment",
            )
            orch = self._orchestrator(sales)
            result = orch.process_message("p3", session_id)

            assert result["response_count"] == 1
            assert result["question_count"] <= 1
            assert not result.get("extra_messages")
            # 计划 §24/§27：一个 turn 一个 turn_id / action
            assert result.get("turn_id")
            assert result.get("action")
            assert result["_turn"]["turn_id"]
            assert result["_turn"]["question_count"] <= 1
        finally:
            memory.clear(session_id)

    def test_turn_log_fields_are_present(self):
        """计划 §27：日志口径里要有 action / speech_act / question_count。"""
        session_id = "v26-single-output-3"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = self._orchestrator(_StubSales(response="Sure."))
            result = orch.process_message("hello", session_id)
            turn = result["_turn"]
            for key in (
                "turn_id",
                "action",
                "speech_act",
                "question_slot",
                "response_count",
                "question_count",
                "llm_calls",
                "llm_latency_ms",
                "total_latency_ms",
            ):
                assert key in turn, key
        finally:
            memory.clear(session_id)
