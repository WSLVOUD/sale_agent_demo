"""线上日志复现（2026-09-23 客户实测）。

客户: i need a display
AI  : A display is a great place to start, we can narrow things down quickly from
      there. The setup matters a lot, since indoor and outdoor screens differ in
      cabinet build and brightness.

客户要的是"先分 LED / LCD"，拿到的却是一段 LED 需求（室内外箱体/亮度）的话术，
连问题都被吞了。日志（session_1790158720935_mlgwhtsg6）里三条线索互相叠加：

  1. ``Turn understanding: ... domain=UNKNOWN ... entry=PRODUCT_SELECTION``
     —— 第一层产品判断**做对了**；
  2. 收口层**没有**出现 ``[ProductType] 收口层拦截`` —— 类型闸门没生效；
  3. 出现 ``[Continuation] ... 本轮先承接、不问问题`` —— 承接额度又把问题压掉。

根因（三条，本文件逐条复现）：

  A. ``SalesState``（LangGraph 的 state schema）没有声明
     ``display_type_decision`` / ``product_entry`` / ``product_domain`` /
     ``turn_kind`` / ``understanding`` 等键。**LangGraph 只把声明过的键带出图**，
     于是 orchestrator 收到的 ``display_type_decision`` 恒为 ``{}`` ——
     v2.9.3 的收口层类型闸门在真实链路上**从来没生效过**
     （只在手工构造 result 的单测里"通过"）。
  B. ``script_generator`` 的"闲聊 / 无关话"分支排在类型闸门**前面**：
     这一轮被判成 ``offtopic_turn`` 后直接拿 LED 的需求问题（indoors/outdoors）
     去问客户，根本没看 ``display_type_decision``。
  C. 收口层的"承接额度"（客户口径：不要连续提问）会把这一轮的问题压掉 ——
     但不该压掉"LED 还是 LCD"这个问题本身：类型没定之前它就是唯一该问的问题。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestSalesStateCarriesTheProductTypeState:
    """A：LangGraph 的 state schema 必须声明这些键，否则它们出不了图。"""

    def test_every_key_the_type_gate_reads_survives_the_graph(self):
        from langgraph.graph import END, StateGraph

        from src.agents.sales.state import SalesState

        def _passthrough(state):
            return state

        graph = StateGraph(SalesState)
        graph.add_node("passthrough", _passthrough)
        graph.set_entry_point("passthrough")
        graph.add_edge("passthrough", END)
        app = graph.compile()

        out = app.invoke(
            {
                "session_id": "state-schema-probe",
                "current_message": "i need a display",
                "display_type_decision": {
                    "display_type": "UNKNOWN",
                    "status": "UNKNOWN",
                    "ask_customer": True,
                },
                "product_entry": "PRODUCT_SELECTION",
                "product_domain": "UNKNOWN",
                "product_subtype": "",
                "turn_kind": "PURE_CONVERSATION",
                "understanding": {"business_signal": False},
            }
        )

        assert out.get("display_type_decision", {}).get("display_type") == "UNKNOWN", (
            "display_type_decision 出不了图 → orchestrator 的类型闸门拿到的是空字典"
        )
        assert out.get("product_entry") == "PRODUCT_SELECTION"
        assert out.get("product_domain") == "UNKNOWN"
        assert out.get("turn_kind") == "PURE_CONVERSATION"
        assert out.get("understanding") == {"business_signal": False}


class TestOffTopicBranchRespectsTheTypeGate:
    """B：闲聊/无关话分支排在类型闸门前面 —— 它必须先解决 LED 还是 LCD。"""

    def _state(self):
        return {
            "intent": "need_query",
            "current_message": "i need a display",
            "offtopic_turn": True,
            "acknowledgement": "A display is a great place to start.",
            "pending_question": "Will the screen be installed indoors or outdoors?",
            "pending_slot": "environment",
            "requirements": {},
            "display_type_decision": {
                "display_type": "UNKNOWN",
                "status": "UNKNOWN",
                "source": "INFERENCE",
                "ask_customer": True,
            },
            "product_entry": "PRODUCT_SELECTION",
        }

    def test_offtopic_turn_asks_led_or_lcd_not_led_requirements(self):
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state())
        reply = str(out.get("response") or "")
        lowered = reply.lower()

        assert "led" in lowered and "lcd" in lowered, reply
        assert "indoors or outdoors" not in lowered, reply
        # 槽位由 Question Planner / Dialogue Policy 定，script_generator 不写 pending_slot
        # （架构收敛测试把这一点钉死）；这里只要求它"问的内容"是类型问题。

    def test_offtopic_turn_still_says_something_to_the_customer(self):
        """先接住客户的话，再问类型 —— 不是把整段回复丢掉。"""
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state())
        assert len(str(out.get("response") or "").strip()) > 40, out.get("response")


class TestContinuationKeepsTheTypeQuestion:
    """C：承接额度不能压掉"LED 还是 LCD"这个问题。"""

    def test_type_gate_question_is_exempt_from_the_ack_budget(self):
        from src.dialogue.continuation_budget import decide_continuation

        decision = decide_continuation(
            has_question=True,
            off_topic=True,
            ack_streak=0,
            max_ack_streak=1,
            type_gate_question=True,
        )
        assert decision.suppress_question is False, "类型问题必须问出去（计划 §五）"

    def test_plain_requirement_question_is_still_suppressed(self):
        """对照组：普通需求问题照旧受承接额度约束（客户口径不变）。"""
        from src.dialogue.continuation_budget import decide_continuation

        decision = decide_continuation(
            has_question=True, off_topic=True, ack_streak=0, max_ack_streak=1
        )
        assert decision.suppress_question is True


class TestLiveTurnEndToEnd:
    """整轮复现：``i need a display`` 的最终回复必须是 LED / LCD 的问题。"""

    def test_orchestrator_turn_ends_with_the_product_type_question(self, monkeypatch):
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        class _Sales:
            def run(self, session_id, message, has_vision=False, **_kwargs):
                return {
                    "intent": "need_query",
                    "next_action": "ask",
                    "response": (
                        "A display is a great place to start, we can narrow things down "
                        "quickly from there. The setup matters a lot, since indoor and "
                        "outdoor screens differ in cabinet build and brightness. "
                        "Is this screen going to be installed inside a building or outside?"
                    ),
                    "pending_question": "Will the screen be installed indoors or outdoors?",
                    "pending_slot": "environment",
                    "requirements": {},
                    "products": [],
                    "offtopic_turn": True,
                    "speech_act": {"act": "CASUAL", "questions": []},
                    "display_type_decision": {
                        "display_type": "UNKNOWN",
                        "status": "UNKNOWN",
                        "source": "INFERENCE",
                        "ask_customer": True,
                    },
                    "product_entry": "PRODUCT_SELECTION",
                    "product_domain": "UNKNOWN",
                    "turn_kind": "PURE_CONVERSATION",
                }

        class _Solution:
            def run(self, *args, **kwargs):
                return {"answer": "Sure.", "products": []}

        session_id = "v294-live-type-gate"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orchestrator = DualAgentOrchestrator(
                sales_agent=_Sales(), solution_agent=_Solution()
            )
            monkeypatch.setattr(orchestrator, "memory_store", memory)
            result = orchestrator.process_message("i need a display", session_id)
            reply = str(result.get("response") or "").lower()

            assert reply, "必须有回复"
            assert "led" in reply and "lcd" in reply, reply
            assert "indoors or outdoors" not in reply, reply
            assert result.get("question_slot") == "display_type", result.get("question_slot")
        finally:
            memory.clear(session_id)
