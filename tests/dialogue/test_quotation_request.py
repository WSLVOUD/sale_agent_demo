"""客户口径（2026-09-21）：推荐之后客户要报价 → 只回"马上发给你"。

实测：客户已经拿到推荐（"… TW11-3216-P3.0 … Shall I prepare the quotation?"），
回了一句 "yes"，系统却又讲了一遍点间距、还端出另一批 COB 型号 ——
客户要的是**报价表**，不是重新被推荐一遍。
"""
import importlib
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

RECOMMENDATION_MESSAGE = (
    "Hi Jack, for your 5m x 3m wall the TW11-3216-P3.0 fits well, with 3.076mm pitch "
    "suited to your viewing distance and a 640x480mm cabinet holding 6 modules. "
    "Shall I prepare the quotation?"
)


def _module():
    return importlib.import_module("src.agents.sales.nodes.script_generator")


class TestQuotationIntentDetection:

    def test_bare_yes_after_a_quotation_question(self):
        module = _module()
        state = {"messages": [{"role": "assistant", "content": RECOMMENDATION_MESSAGE}]}
        for message in ("yes", "好的", "可以", "sure", "please do"):
            assert module._wants_quotation(state, message) is True, message

    def test_direct_quotation_request(self):
        module = _module()
        state = {"messages": []}
        for message in ("报价", "给我报价单", "send me the price list", "quotation please"):
            assert module._wants_quotation(state, message) is True, message

    def test_yes_without_quotation_context_is_not_a_quotation_request(self):
        module = _module()
        state = {"messages": [{"role": "assistant", "content": "What screen size do you need?"}]}
        assert module._wants_quotation(state, "yes") is False

    def test_confirmation_with_more_questions_is_not_treated_as_quotation(self):
        module = _module()
        state = {"messages": [{"role": "assistant", "content": RECOMMENDATION_MESSAGE}]}
        assert module._wants_quotation(state, "yes, but I have a question about brightness") is False
        assert module._wants_quotation(state, "no thanks") is False


class TestQuotationReply:

    def _state(self, *, already_recommended=True, message="yes"):
        return {
            "current_message": message,
            "messages": [
                {"role": "assistant", "content": RECOMMENDATION_MESSAGE},
                {"role": "user", "content": message},
            ],
            "session_id": "quotation-test",
            "requirements": {},
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "already_recommended": already_recommended,
            "requirement_profile": None,
        }

    def test_after_recommendation_only_the_quotation_ack_is_sent(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module, "_dialogue_llm", lambda: None)  # 走结构化兜底
        result = module.script_generator(self._state())
        text = str(result.get("response") or "")
        lowered = text.lower()
        # ① 明确说明这就把报价发过去
        assert "quotation" in lowered
        assert any(word in lowered for word in ("right away", "moment", "shortly", "straight over"))
        # ② 不再换型号 / 不再讲课 / 不再追加问题
        assert "TW" not in text
        assert "mm" not in text
        assert "?" not in text
        assert result.get("next_action") == "ask"

    def test_before_recommendation_the_old_price_policy_still_applies(self, monkeypatch):
        """还没推荐过就问到报价 → 仍然是"先确认型号再报价"的口径。"""
        module = _module()
        monkeypatch.setattr(module, "_dialogue_llm", lambda: None)
        state = self._state(already_recommended=False, message="给我报价单")
        state["intent"] = "objection"
        result = module.script_generator(state)
        text = str(result.get("response") or "").lower()
        assert "quotation" in text or "quote" in text
        # 旧的报价口径会说"要看型号/配置"，不是"马上发给你"
        assert not any(word in text for word in ("right away", "straight over"))

    def test_direct_quotation_request_after_recommendation(self, monkeypatch):
        module = _module()
        monkeypatch.setattr(module, "_dialogue_llm", lambda: None)
        result = module.script_generator(
            self._state(message="please send me the quotation")
        )
        assert "quotation" in str(result.get("response") or "").lower()
        assert "?" not in str(result.get("response") or "")
