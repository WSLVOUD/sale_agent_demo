"""客户口径（2026-09-24）：产品类型的判断要**读语境**，不许靠关键字触发。

客户实测：

    客户: i need a display
    AI  : Are you looking for an LED display, or an LCD…?
    客户: i dont know
    AI  : That's completely fine… On our side, displays generally fall into two
          families: LED screens, or LCD options… Are you looking for an LED display,
          or an LCD…?                          ← ❌ 又把同一句问了一遍，没解释区别

客户要求：客户不知道选哪个时，**先把 LED / LCD 的区别解释一遍，再让他选**；
而且**不许用关键字触发**某个动作 —— 要结合上下文语境判断客户这句话在干什么。

所以这一版把"客户怎么回应产品类型问题"做成**语境判断**（模型看对话历史 +
我们刚问的那句），规则只做降级兜底：

    chose          客户说了要哪种                 → 确认 + 锁定
    rejects        否认我们的建议，但没说要哪种    → 问另一种
    does_not_know  不知道 / 选不出来              → 先解释区别，再让他选
    asks_meaning   问 LED / LCD 是什么、什么区别   → 先解释区别，再让他选
    delegates      让 AI 直接决定                 → 有建议就采用，没有就给建议
    unrelated      这句话跟"选哪种屏"无关          → 沿用我们上一轮的建议
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _route(message: str, **kwargs):
    from src.dialogue.product_type_router import route_display_type

    return route_display_type(message, **kwargs)


def _proposal(type_value: str = "LED"):
    return _route("here is my screen", vision_display_type=type_value, vision_reason="photo")


class TestRouterConsumesTheContextSignal:
    """路由消费语境信号（信号由模型在上下文里给出，路由不再自己猜关键词）。"""

    def test_cannot_choose_gets_an_explanation(self):
        decision = _route("i dont know", reply_signal={"reply": "does_not_know"})

        assert decision.needs_explanation is True, decision.to_dict()
        assert decision.ask_customer is True

    def test_question_about_the_difference_gets_an_explanation(self):
        decision = _route("what is the difference?", reply_signal={"reply": "asks_meaning"})

        assert decision.needs_explanation is True
        assert decision.ask_customer is True

    def test_explanation_carries_the_actual_difference(self):
        from src.dialogue.product_type_router import product_type_question

        decision = _route("i dont know", reply_signal={"reply": "does_not_know"})
        question = product_type_question(decision)

        assert "LED" in question and "LCD" in question, question
        assert "seams" in question, "要讲清区别：LED 基本无拼缝 / LCD 有拼缝"
        assert "indoors" in question or "indoor" in question, question

    def test_signal_chose_lcd_confirms_it(self):
        decision = _route(
            "go with that one", reply_signal={"reply": "chose", "display_type": "LCD"}
        )

        assert decision.display_type == "LCD"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True

    def test_signal_rejects_without_a_type_asks_the_other(self):
        decision = _route(
            "nope", current=_proposal("LED"), reply_signal={"reply": "rejects"}
        )

        assert decision.display_type == "LCD"
        assert decision.alternative is True
        assert decision.locked is False

    def test_signal_rejects_with_a_type_switches(self):
        decision = _route(
            "no",
            current=_proposal("LED"),
            reply_signal={"reply": "rejects", "display_type": "LCD"},
        )

        assert decision.display_type == "LCD"
        assert decision.status == "CONFIRMED"

    def test_unrelated_signal_adopts_our_proposal(self):
        decision = _route(
            "it is for our showroom",
            current=_proposal("LED"),
            reply_signal={"reply": "unrelated"},
        )

        assert decision.display_type == "LED"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True

    def test_dont_know_still_adopts_when_we_already_suggested(self):
        """计划 §六：我们**已经建议过** LED，客户说不知道 → 采用我们的建议（不再解释一遍）。"""
        decision = _route(
            "i dont know", current=_proposal("LED"), reply_signal={"reply": "does_not_know"}
        )

        assert decision.display_type == "LED"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True

    def test_delegate_without_a_proposal_gets_an_explanation(self):
        decision = _route("you choose for me", reply_signal={"reply": "delegates"})

        assert decision.needs_explanation is True
        assert decision.ask_customer is True


class TestContextJudge:
    """语境判断：模型看到"最近的对话 + 我们刚问的那句"，不是只看这一句话。"""

    @pytest.fixture
    def judge(self, monkeypatch):
        import importlib

        module = importlib.import_module("src.dialogue.product_type_understanding")
        captured = {}

        class _Resp:
            def __init__(self, content):
                self.content = content

        class _Fake:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, messages):
                captured["prompt"] = "\n".join(
                    str(getattr(item, "content", item)) for item in messages
                )
                return _Resp('{"reply": "does_not_know", "display_type": "", "reason": "客户说不确定"}')

        monkeypatch.setattr(module, "ChatOpenAI", _Fake)
        # 缓存按 session 走，测试里换 session_id 避免互相污染
        module.clear_cache()
        return module, captured

    def test_judge_sees_the_dialogue_and_the_question_we_asked(self, judge):
        module, captured = judge

        signal = module.understand_product_type_reply(
            "i dont know",
            session_id="judge-context",
            conversation="AI: Are you looking for an LED display, or an LCD?",
            asked_question="Are you looking for an LED display, or an LCD?",
        )

        assert signal["reply"] == "does_not_know"
        prompt = captured["prompt"]
        assert "Are you looking for an LED display" in prompt, "必须带上我们刚问的那句"
        assert "i dont know" in prompt

    def test_judge_failure_is_safe(self, monkeypatch):
        import importlib

        module = importlib.import_module("src.dialogue.product_type_understanding")

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("llm down")

        monkeypatch.setattr(module, "ChatOpenAI", _Boom)
        module.clear_cache()

        assert module.understand_product_type_reply("i dont know", session_id="judge-boom") == {}

    def test_judge_result_is_cached_per_turn(self, judge):
        module, captured = judge

        module.understand_product_type_reply("i dont know", session_id="judge-cache")
        module.understand_product_type_reply("i dont know", session_id="judge-cache")

        assert captured.get("prompt"), "第一次调用应该有 prompt"


class TestExplanationHasNoFillerOpening:
    """客户口径（2026-09-24）：解释直接说，"It's a lot to take in at once /
    The quick version:" 这类开场白不要。"""

    def _state(self):
        return {
            "intent": "need_query",
            "current_message": "i dont konw",
            "offtopic_turn": True,
            "acknowledgement": "It's a lot to take in at once.",
            "pending_question": "Will the screen be installed indoors or outdoors?",
            "pending_slot": "environment",
            "requirements": {},
            "display_type_decision": {
                "display_type": "UNKNOWN",
                "status": "UNKNOWN",
                "source": "INFERENCE",
                "needs_explanation": True,
                "ask_customer": True,
            },
            "product_entry": "PRODUCT_SELECTION",
        }

    def test_offtopic_turn_with_explanation_says_it_directly(self):
        from src.agents.sales.nodes.script_generator import script_generator

        out = script_generator(self._state())
        reply = str(out.get("response") or "")
        lowered = reply.lower()

        assert "no visible seams" in lowered, reply
        assert "take in at once" not in lowered, f"不要开场白：{reply}"
        assert "quick version" not in lowered, f"不要过渡话术：{reply}"
        assert reply.rstrip().endswith("?"), f"结尾要落在「请你选」的问题上：{reply}"

    def test_non_offtopic_turn_with_explanation_also_has_no_filler(self):
        from src.agents.sales.nodes.script_generator import script_generator

        state = self._state()
        state["offtopic_turn"] = False
        out = script_generator(state)
        reply = str(out.get("response") or "")

        assert "no visible seams" in reply.lower(), reply
        assert "take in at once" not in reply.lower(), reply


class TestLiveDontKnowTurn:
    """整轮复现：客户说不知道选哪个 → 回复必须解释区别，而不是把同一句再问一遍。"""

    @pytest.fixture
    def canned(self, monkeypatch):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")
        understanding_mod = importlib.import_module("src.dialogue.product_type_understanding")

        class _Resp:
            def __init__(self, content):
                self.content = content

        def _fake_chat(content):
            class _F:
                def __init__(self, *args, **kwargs):
                    pass

                def invoke(self, *args, **kwargs):
                    return _Resp(content)

            return _F

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _fake_chat("need_query"))
        monkeypatch.setattr(
            sales_req,
            "ChatOpenAI",
            _fake_chat('{"usage": null, "additional_requirements": []}'),
        )
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()
        # 语境判断：这里用桩把"模型读语境后的结论"固定住（真实链路由模型判）
        monkeypatch.setattr(
            understanding_mod,
            "understand_product_type_reply",
            lambda *args, **kwargs: {"reply": "does_not_know", "display_type": ""},
        )
        return None

    def test_i_dont_know_explains_the_difference_instead_of_re_asking(
        self, monkeypatch, canned
    ):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        class _Solution:
            def run(self, *args, **kwargs):
                return {"answer": "", "products": []}

        session_id = "v294-dont-know"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()
            orchestrator = DualAgentOrchestrator(
                sales_agent=runner, solution_agent=_Solution()
            )
            monkeypatch.setattr(orchestrator, "memory_store", memory)

            orchestrator.process_message("i need a display", session_id)
            second = orchestrator.process_message("i dont know", session_id)
            reply = str(second.get("response") or "")
            lowered = reply.lower()

            assert "led" in lowered and "lcd" in lowered, reply
            assert "seam" in lowered, f"要解释区别（拼缝），实际：{reply}"
            # 不能只是把菜单重复一遍
            assert lowered.count("are you looking for") <= 1, reply
            # 解释只能出现一次（销售层和收口层都别各说一遍）
            assert lowered.count("no visible seams") == 1, reply
            assert second.get("response") != "", reply
        finally:
            memory.clear(session_id)
