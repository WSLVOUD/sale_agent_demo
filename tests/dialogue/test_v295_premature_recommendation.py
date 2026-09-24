"""LED 链路实测问题（2026-09-24 客户会话）：需求没收齐就把型号报出来了。

客户实测：

    客户: i need a led display → AI 问室内外
    客户: indoor                → AI 问点间距
    客户: p4                    → AI: "P4 is a solid middle ground… For what you described,
                                        TW11-3216-P4.0 is the closest match, 4mm pixel pitch,
                                        500nit brightness, cabinet 640mm*480mm.
                                        In the meantime, do you have the screen dimensions…?"
                                   ↑ 型号 + 参数都报出来了，可尺寸/场景/安装都还没问

根因（两条）：

  1. classify 把 "p4"（在回答上一轮的点间距问题）误判成 ``product_question``
     → 整轮被送进 Solution 的 RAG 链路 → FAST 路径按参数命中型号并报出；
  2. 收口层没有"没在交付推荐，就不许出现型号"的兜底。

修复：用**对话状态**（上一轮问的是哪一项）判断"这是回答，不是提问"，
并在收口层加一道型号闸门（复用 ``rag.model_guard``）。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _classify_with_previous_question(monkeypatch, message, slot, llm_intent="product_question"):
    import importlib

    classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
    understanding_mod = importlib.import_module("src.dialogue.product_type_understanding")
    from src.dialogue import get_conversation_state
    from src.memory.store import memory

    session_id = f"premature-{abs(hash((message, slot))) % 9999}"
    memory.clear(session_id)
    memory.mark_first_contact_done(session_id)
    get_conversation_state(session_id).note_ai_turn(
        action="ask_only", question=f"({slot} question)", slot=slot
    )

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content=llm_intent)

    monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeLLM)
    monkeypatch.setattr(
        understanding_mod, "understand_product_type_reply", lambda *a, **k: {}
    )

    state = {
        "current_message": message,
        "messages": [],
        "session_id": session_id,
        "requirements": {},
    }
    out = classify_mod.classify(state)
    memory.clear(session_id)
    return out


class TestAnsweringOurQuestionIsNotAProductQuestion:

    def test_pitch_answer_is_not_a_product_question(self, monkeypatch):
        """客户回 "p4" 是在回答我们问的点间距 → 走需求采集，不走自由问答。"""
        out = _classify_with_previous_question(monkeypatch, "p4", "pixel_pitch")

        assert out["intent"] == "need_query", out["intent"]

    def test_size_answer_is_not_a_product_question(self, monkeypatch):
        out = _classify_with_previous_question(monkeypatch, "3*5", "size")

        assert out["intent"] == "need_query", out["intent"]

    def test_real_question_is_still_a_product_question(self, monkeypatch):
        """带问号的仍然是提问（要正常回答，不能被吞进采集）。"""
        out = _classify_with_previous_question(
            monkeypatch, "is a P4 screen any good?", "pixel_pitch"
        )

        assert out["intent"] == "product_question", out["intent"]


class TestNoModelBeforeTheGateIsReady:
    """收口层闸门：没在交付推荐 → 客户可见文本里不许出现具体型号。"""

    _REPLY = (
        "P4 is a solid middle ground for indoor work, so that gives us a good starting "
        "point. For what you described, TW11-3216-P4.0 is the closest match, 4mm pixel "
        "pitch, 500nit brightness, cabinet 640mm*480mm. In the meantime, do you have the "
        "screen dimensions, in width x height?"
    )

    def _finalize(self, monkeypatch, *, products=(), gate_ready=False):
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        class _Sales:
            def run(self, *a, **k):
                return {}

        class _Solution:
            def run(self, *a, **k):
                return {}

        session_id = f"model-gate-{gate_ready}-{len(products)}"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        orchestrator = DualAgentOrchestrator(
            sales_agent=_Sales(), solution_agent=_Solution()
        )
        monkeypatch.setattr(orchestrator, "memory_store", memory)
        try:
            result = {
                "intent": "need_query",
                "next_action": "ask",
                "response": self._REPLY,
                "pending_question": "Do you have the screen dimensions, in width x height?",
                "pending_slot": "size",
                "requirements": {"display_type": "LED"},
                "products": list(products),
                "recommendation_gate": {"ready": gate_ready},
                "display_type_decision": {
                    "display_type": "LED",
                    "status": "CONFIRMED",
                    "locked": True,
                },
                "product_entry": "LED_ENTRY",
            }
            return orchestrator._finalize_turn_response(result, session_id, "p4")
        finally:
            memory.clear(session_id)

    def test_model_is_stripped_while_collecting(self, monkeypatch):
        out = self._finalize(monkeypatch)
        reply = str(out.get("response") or "")

        assert "TW11-3216-P4.0" not in reply, reply
        assert "no visible" not in reply
        # 该问的尺寸问题还在
        assert "width x height" in reply, reply
        assert out.get("model_mentions_stripped"), out.get("model_mentions_stripped")

    def test_model_is_kept_when_we_actually_deliver_a_recommendation(self, monkeypatch):
        out = self._finalize(
            monkeypatch, products=[{"model": "TW11-3216-P4.0"}], gate_ready=True
        )
        reply = str(out.get("response") or "")

        assert "TW11-3216-P4.0" in reply, reply
