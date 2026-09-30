"""实测修复（2026-09-28）：客户说 "i need a lcd display" 时完全没有按 LCD 链路走。

客户可见原文：

    👤 i need a lcd display
    🤖 An LCD display, noted — we actually work mainly with LED screens, so tell me a bit
       more about where it's going and what you need it to do, and I'll point you in the
       right direction. [产品资料 1]
       H4930LN-B | B Series (LCD Video Wall) | LCD | indoor | size=49" | resolution=1920x1080 | …
       [产品资料 2] …

三个问题：

  1. **路由**：报产品类型（"i need a lcd display"）被 classify 判成 product_question
     → 整轮送 Solution 自由问答，LCD 需求链（`ask_purpose`）虽然算出来了却没被采用；
  2. **泄漏**：自由问答的兜底把内部检索块 `[产品资料 N] …` 原样拼进客户回复；
  3. **口径**：自由问答里自我贬低成 "we actually work mainly with LED screens"（LCD 客户）。
"""
import os
import sys
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestTypeStatementGoesToTheRequirementChain:
    """问题 1：报产品类型 → 需求采集（进对应需求链），不是自由问答。"""

    def _classify(self, monkeypatch, message: str, llm_intent: str = "product_question"):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        understanding_mod = importlib.import_module("src.dialogue.product_type_understanding")
        from src.memory.store import memory

        session_id = f"lcd-route-{abs(hash(message)) % 9999}"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)

        class _FakeLLM:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return SimpleNamespace(content=llm_intent)

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeLLM)
        monkeypatch.setattr(
            understanding_mod,
            "understand_product_type_reply",
            lambda *a, **k: {"reply": "chose", "display_type": "LCD"},
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

    def test_lcd_type_statement_is_requirement_collection(self, monkeypatch):
        out = self._classify(monkeypatch, "i need a lcd display")

        assert out["intent"] == "need_query", out.get("intent")
        assert out.get("display_type_decision", {}).get("display_type") == "LCD"

    def test_type_statement_with_a_real_question_still_answers(self, monkeypatch):
        out = self._classify(monkeypatch, "we need LCD screens, how much are they?")

        assert out["intent"] == "product_question", out.get("intent")

    def test_plain_greeting_is_untouched(self, monkeypatch):
        out = self._classify(monkeypatch, "hi there", llm_intent="greeting")

        assert out["intent"] == "greeting", out.get("intent")


class TestInternalChunksNeverReachTheCustomer:
    """问题 2：`[产品资料 N] …` 是内部块，必须在客户可见文本里被清掉。"""

    def test_strip_internal_blocks_removes_the_leaked_dump(self):
        from src.utils.text import strip_internal_blocks

        leaked = (
            "An LCD display, noted. [产品资料 1]\n"
            'H4930LN-B | B Series (LCD Video Wall) | LCD | indoor | size=49" | power=200W\n'
            "[产品资料 2]\nH4630LN-B | B Series (LCD Video Wall) | LCD | indoor\n"
        )
        cleaned = strip_internal_blocks(leaked)

        assert "产品资料" not in cleaned, cleaned
        assert "H4930LN-B" not in cleaned, cleaned
        assert cleaned.startswith("An LCD display, noted."), cleaned

    def test_customer_response_sanitizer_applies_the_cleanup(self):
        from src.rag.rerank import sanitize_customer_response

        text = "An LCD display, noted. [产品资料 1]\nH4930LN-B | LCD | size=49\""
        cleaned = sanitize_customer_response(text)

        assert "产品资料" not in cleaned, cleaned
        assert "LCD" in cleaned

    def test_others_node_never_falls_back_to_the_raw_chunks(self):
        import importlib

        others = importlib.import_module("src.agents.solution.nodes.others")
        source = open(others.__file__, encoding="utf-8").read()

        assert "answer = (\n                products_text" not in source, (
            "自由问答的兜底不允许把内部检索片段直接发给客户"
        )


class TestLcdFreeQuestionDoesNotUndermineLcd:
    """问题 3：LCD 客户面前不许自我贬低成"我们主要做 LED"。"""

    def test_business_goal_mentions_lcd_when_the_customer_is_on_lcd(self):
        import importlib

        others = importlib.import_module("src.agents.solution.nodes.others")
        source = open(others.__file__, encoding="utf-8").read()

        assert "never say we mainly" in source, (
            "自由问答要明确禁止『我们主要做 LED』这类贬低 LCD 的说法"
        )

