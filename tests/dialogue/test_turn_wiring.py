"""v2.6 §27/§28：一轮到底"做了什么"必须能从结果与日志里看出来。

真实日志（2026-09-21 14:50）暴露的接线问题：

    [Turn] ... action=ask speech_act=- question_slot=-      ← 全是空的
    [DecisionAudit] "speech_act": ""                        ← 同上

根因：SalesState 没有声明 speech_act / dialogue_action（LangGraph 不带出图），
而且 orchestrator 的结果字典没有透传 pending_question / pending_slot。
这组用例把这条接线钉住。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.store import memory  # noqa: E402
from src.orchestrator import DualAgentOrchestrator  # noqa: E402


class _StubSales:
    def __init__(self, response=None, pending_question="", pending_slot="",
                 speech_act=None, dialogue_action=None):
        self.pending_question = pending_question
        self.pending_slot = pending_slot
        # 真实链路里"待问的那句话"就是这一轮的回复文本（含没有别的寒暄的情况）
        self.response = response if response is not None else (pending_question or "Got it.")
        self.speech_act = speech_act if speech_act is not None else {
            "speech_act": "NEW_REQUIREMENT",
            "customer_questions": [],
        }
        self.dialogue_action = dialogue_action if dialogue_action is not None else {
            "action": "ask_only",
            "target_slot": pending_slot,
            "priority": 3,
            "reason": "requirement_answered_ask_next",
            "question": pending_question,
        }

    def run(self, session_id, message, has_vision=False, **_kwargs):
        return {
            "intent": "need_query",
            "next_action": "ask",
            "response": self.response,
            "requirements": {},
            "products": [],
            "pending_question": self.pending_question,
            "pending_slot": self.pending_slot,
            "speech_act": dict(self.speech_act),
            "dialogue_action": dict(self.dialogue_action),
        }


class _StubSolution:
    def run(self, *args, **kwargs):
        return {"answer": "Sure.", "products": []}


class TestTurnWiring:

    def _run(self, session_id, sales, message="i need a led display"):
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        orch = DualAgentOrchestrator(sales_agent=sales, solution_agent=_StubSolution())
        try:
            return orch.process_message(message, session_id)
        finally:
            memory.clear(session_id)

    def test_pending_question_reaches_final_response(self):
        result = self._run(
            "wire-1",
            _StubSales(
                pending_question="Will the screen be installed indoors or outdoors?",
                pending_slot="environment",
            ),
        )
        assert result["question_slot"] == "environment"
        assert result["question_count"] == 1
        assert result["final_response"]["question_slot"] == "environment"

    def test_policy_action_is_reported_not_the_route_name(self):
        result = self._run(
            "wire-2",
            _StubSales(
                response="Sure.",
                pending_question="Which pitch do you need?",
                pending_slot="pixel_pitch",
            ),
        )
        assert result["action"] == "ask_only", "应报 Dialogue Policy 的动作，而不是 next_action=ask"
        assert result["_turn"]["action"] == "ask_only"

    def test_speech_act_is_recorded(self):
        result = self._run(
            "wire-3",
            _StubSales(response="Sure.", pending_question="Q?", pending_slot="size"),
        )
        assert result["_turn"]["speech_act"] == "NEW_REQUIREMENT"
        assert result["conversation_state"]["current_speech_act"] == "NEW_REQUIREMENT"

    def test_decision_audit_has_before_and_after(self):
        result = self._run(
            "wire-4",
            _StubSales(
                pending_question="Will the screen be installed indoors or outdoors?",
                pending_slot="environment",
            ),
        )
        audit = result["decision_audit"]
        assert audit["selected_action"] == "ask_only"
        assert audit["question_slot"] == "environment"
        assert "conversation_state_before" in audit
        assert "conversation_state_after" in audit
        # 决定之前，这一轮的问句还没写进状态
        assert audit["conversation_state_before"]["last_question_slot"] in ("", "environment")
        assert audit["conversation_state_after"]["last_question_slot"] == "environment"

    def test_live_llm_calls_helper_exists(self):
        """§27：PERF 那行在 end_turn 之前打，不能永远显示 llm_calls=0。"""
        from src.observability.perf import PerfTracker

        perf = PerfTracker("wire-5", "hi")
        assert perf.live_llm_calls() == 0
        assert hasattr(perf, "live_llm_calls")
