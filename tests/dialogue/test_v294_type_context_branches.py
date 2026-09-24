"""客户口径（2026-09-23）：产品类型判断要**联合上下文**，并把四类回应补齐。

真实会话（客户实测）：

    客户: i need a display
    AI  : …Are you looking for an LED display, or an LCD…?
    客户: （图片）like this
    AI  : It looks like an LED screen… LED looks like the better fit here,
          shall I continue in that direction?
    客户: ok
    AI  : Are you looking for an LED display, or an LCD…?      ← ❌ 又问了一遍

客户口径（四条）：

    ① 客户正面确认（ok / yes / 对 / 可以）      → 按 AI 判断走，锁定，**不再问一遍**
    ② 客户没有正面回答（去聊别的 / 只答别的项） → **默认采用 AI 识别出来的信息**，不追问
    ③ 客户说不对，并且说了是什么（"不是，要 LCD"）→ 按客户说的走
    ④ 客户只说不对、没说是什么                  → 问"那是不是要另一种"

根因（上一轮已定位）：classify 在轮内写进 memory 的 display_type_decision，
被 runner 收尾的 ``memory.clear()`` 清掉了 —— 于是下一轮读不到"我们上一轮建议过 LED"，
既无法确认、也无法采用，只能重新问一遍。
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


def _vision_proposal(type_value: str = "LED", reason: str = "large LED video wall"):
    """AI 从图片里判断出类型后，向客户确认时的那一版决策。"""
    return _route("here is my screen", vision_display_type=type_value, vision_reason=reason)


class TestDecisionSurvivesTheTurn:
    """重要信息单独保留：类型判断（含锁定）必须跨轮活下来。"""

    def test_runner_keeps_the_product_type_decision(self):
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory

        session_id = "v294-type-persist"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            class _Graph:
                def invoke(self, state):
                    # 模拟 classify 节点：轮内把第一层判断写进 memory
                    memory.set_display_type_decision(
                        session_id,
                        {"display_type": "LED", "status": "INFERRED", "source": "VISION"},
                    )
                    return dict(state, response="Understood.")

            runner = SalesAgentRunner(memory_store=memory)
            runner.graph = _Graph()
            runner.run(session_id, "like this")

            kept = memory.get_display_type_decision(session_id)
            assert kept.get("display_type") == "LED", (
                "runner 收尾的 clear() 不能把本轮的类型判断一起清掉"
            )
            assert kept.get("status") == "INFERRED"
        finally:
            memory.clear(session_id)


class TestConfirmAdoptAndDeny:
    """四类回应（客户口径 ①~④）在路由器上各自的行为。"""

    def test_ok_locks_the_type_we_proposed(self):
        proposal = _vision_proposal("LED")
        decision = _route("ok", current=proposal)

        assert decision.display_type == "LED"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True
        assert decision.ask_customer is False

    def test_ignoring_the_question_adopts_the_ai_reading(self):
        """② 客户没正面回答（去聊别的）→ 直接采用 AI 判断，不再追问。"""
        proposal = _vision_proposal("LED")
        decision = _route("it is for our conference room", current=proposal)

        assert decision.display_type == "LED"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True
        assert decision.ask_customer is False
        assert decision.subtype == ""

    def test_adoption_also_works_from_the_profile_context(self):
        """联合上下文：即使上一轮决策丢了，档案里"图片识别出的类型"也能兜住。"""
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile()
        profile.display_type = "LCD"
        profile.sources = {"display_type": "vision_explicit"}
        profile.vision_confirmation_pending = ["display_type"]

        decision = _route("ok, that works", profile=profile)

        assert decision.display_type == "LCD"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True

    def test_denial_without_a_type_asks_for_the_other_one(self):
        """④ 客户只说"不对" → 问"那是不是要另一种"，不是重新抛菜单。"""
        proposal = _vision_proposal("LED")
        decision = _route("no", current=proposal)

        assert decision.display_type == "LCD", decision.to_dict()
        assert decision.locked is False, "客户还没确认，不能锁定"
        assert decision.ask_customer is True
        assert decision.alternative is True, "要标记这是「另一种」的问法"

    def test_denial_then_yes_locks_the_other_type(self):
        proposal = _vision_proposal("LED")
        asked = _route("no", current=proposal)
        confirmed = _route("yes", current=asked)

        assert confirmed.display_type == "LCD"
        assert confirmed.status == "CONFIRMED"
        assert confirmed.locked is True

    def test_denial_that_names_a_type_switches_directly(self):
        """③ 客户说不对 + 说清是什么 → 直接按客户说的走。"""
        proposal = _vision_proposal("LED")
        decision = _route("no, we need an LCD video wall", current=proposal)

        assert decision.display_type == "LCD"
        assert decision.status == "CONFIRMED"
        assert decision.locked is True

    def test_denial_of_another_field_is_not_a_type_denial(self):
        """客户纠正的是**别的字段**（"不对，是室外的"）→ 不能当成否认产品类型。"""
        proposal = _vision_proposal("LED")
        decision = _route("no, it is outdoor", current=proposal)

        assert decision.display_type == "LED", decision.to_dict()
        assert decision.alternative is False, "这是纠正环境，不是否认 LED"

    def test_denial_plus_ifp_features_goes_to_ifp(self):
        """客户说"不对，而且我们要在上面写字"→ IFP（LCD 子类型），不是普通 LCD。"""
        proposal = _vision_proposal("LED")
        decision = _route("no, we also need to write on it", current=proposal)

        assert decision.display_type == "LCD"
        assert decision.subtype == "IFP"

    def test_confirmed_type_is_never_asked_again(self):
        """确认过的类型不再重新判断、也不再问一遍（防重复提问）。"""
        confirmed = _route("I want LED.")
        later = _route("it is for a meeting room, indoor", current=confirmed)

        assert later.display_type == "LED"
        assert later.status == "CONFIRMED"
        assert later.ask_customer is False

    def test_customer_can_change_a_confirmed_type_without_magic_words(self):
        """③ 客户明确改口（"不对，我们要 LCD"）→ 按客户说的切，不该要求他说 actually。"""
        confirmed = _route("I want LED.")
        switched = _route("no, we need an LCD video wall", current=confirmed)

        assert switched.display_type == "LCD", switched.to_dict()
        assert switched.status == "CONFIRMED"
        assert switched.locked is True

    def test_merely_mentioning_the_other_type_does_not_switch(self):
        """只是**提到**另一种类型（问句）→ 不能自己改类型（锁定仍然有效）。"""
        confirmed = _route("I want LED.")
        kept = _route("does it also come as an LCD?", current=confirmed)

        assert kept.display_type == "LED", kept.to_dict()


class TestAlternativeQuestionPhrasing:

    def test_alternative_question_names_the_other_type(self):
        from src.dialogue.product_type_router import product_type_question

        proposal = _vision_proposal("LED")
        decision = _route("no", current=proposal)
        question = product_type_question(decision).lower()

        assert "lcd" in question, question
        assert question.count("led") <= 1, "只问另一种，不再抛 LED/LCD 菜单"


class TestConfirmationKeepsTheFlowMoving:
    """客户确认后要"按这个方向继续推进"，不能被承接额度压成一句寒暄。"""

    def _turn(self, monkeypatch, speech_act, message):
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        class _Sales:
            def run(self, session_id, message, has_vision=False, **_kwargs):
                return {
                    "intent": "need_query",
                    "next_action": "ask",
                    "response": "Understood.",
                    "pending_question": "What width and height should the screen be?",
                    "pending_slot": "size",
                    "requirements": {},
                    "products": [],
                    "offtopic_turn": True,
                    "speech_act": {"speech_act": speech_act, "questions": []},
                    "display_type_decision": {
                        "display_type": "LED",
                        "status": "CONFIRMED",
                        "source": "CUSTOMER",
                        "locked": True,
                    },
                    "product_entry": "LED_ENTRY",
                    "product_domain": "LED",
                }

        class _Solution:
            def run(self, *args, **kwargs):
                return {"answer": "Sure.", "products": []}

        session_id = f"v294-speech-{speech_act.lower()}"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        orchestrator = DualAgentOrchestrator(
            sales_agent=_Sales(), solution_agent=_Solution()
        )
        monkeypatch.setattr(orchestrator, "memory_store", memory)
        try:
            return orchestrator._finalize_turn_response(
                {
                    "intent": "need_query",
                    "next_action": "ask",
                    "response": "Understood.",
                    "pending_question": "What width and height should the screen be?",
                    "pending_slot": "size",
                    "requirements": {},
                    "products": [],
                    "offtopic_turn": True,
                    "speech_act": {"speech_act": speech_act, "questions": []},
                    "display_type_decision": {
                        "display_type": "LED",
                        "status": "CONFIRMED",
                        "source": "CUSTOMER",
                        "locked": True,
                    },
                    "product_entry": "LED_ENTRY",
                    "product_domain": "LED",
                },
                session_id,
                message,
            )
        finally:
            memory.clear(session_id)

    def test_confirmation_keeps_collecting_in_the_same_turn(self, monkeypatch):
        out = self._turn(monkeypatch, "CONFIRMATION", "ok")

        assert not out.get("suppressed_question"), "确认不是闲聊，不该被承接额度压掉"
        assert out.get("pending_slot") == "size", out.get("pending_slot")

    def test_pure_chit_chat_still_gets_a_single_acknowledgement(self, monkeypatch):
        """对照组：真正的闲聊仍然只承接一条（客户口径不变）。"""
        out = self._turn(monkeypatch, "CASUAL", "do you like movies?")

        assert out.get("suppressed_question"), out.get("continuation")


class TestLiveImageConversation:
    """整轮复现客户那段会话（真图 + 真 runner）。"""

    @pytest.fixture
    def canned_llm(self, monkeypatch):
        import importlib

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
        extractor_mod = importlib.import_module("src.core.requirement_extractor")

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

    def test_ok_after_the_image_confirmation_does_not_ask_the_type_again(
        self, monkeypatch, canned_llm
    ):
        import src.vision.integration as integration
        from src.agents.sales.runner import SalesAgentRunner
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator
        from src.vision.extractor import VisionExtractor

        payload = {
            "display_type": {"value": "LED", "source": "vision_explicit", "confidence": 0.95},
            "environment": {"value": "indoor", "source": "vision_explicit", "confidence": 0.9},
            "purpose": {"value": "conference", "source": "vision_explicit", "confidence": 0.8},
        }

        class _Vision:
            def extract_many(self, images, session_id="", customer_text="", **_kwargs):
                return [VisionExtractor.from_payload(payload)]

        monkeypatch.setattr(integration, "get_vision_extractor", lambda: _Vision())

        class _Solution:
            def run(self, *args, **kwargs):
                return {"answer": "", "products": []}

        session_id = "v294-image-confirm"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            runner = SalesAgentRunner(memory_store=memory)
            runner.setup()
            orchestrator = DualAgentOrchestrator(
                sales_agent=runner, solution_agent=_Solution()
            )
            monkeypatch.setattr(orchestrator, "memory_store", memory)

            orchestrator.process_message("like this", session_id, images=[b"img"])
            proposed = memory.get_display_type_decision(session_id)
            assert proposed.get("display_type") == "LED", proposed
            assert proposed.get("status") == "INFERRED", proposed

            second = orchestrator.process_message("ok", session_id)
            reply = str(second.get("response") or "").lower()
            settled = memory.get_display_type_decision(session_id)

            assert settled.get("status") == "CONFIRMED", settled
            assert settled.get("locked") is True, settled
            assert "are you looking for an led display" not in reply, reply
            assert second.get("question_slot") != "display_type", second.get("question_slot")
        finally:
            memory.clear(session_id)
