"""计划 v2.9.3（首层产品判断 LED / LCD + IFP 归 LCD 子类型）的回归测试。

覆盖计划 §二十 的 17 个场景，以及 §十二/§十三 的状态机与锁定规则。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _route(message: str, **kwargs):
    from src.dialogue.product_type_router import route_display_type

    return route_display_type(message, **kwargs)


class TestFirstLayerRouting:

    def test_1_explicit_led(self):
        decision = _route("I need an LED display.")
        assert decision.display_type == "LED"
        assert decision.status == "CONFIRMED"
        assert decision.source == "CUSTOMER"
        assert decision.locked is True

    def test_2_explicit_lcd(self):
        decision = _route("I need an LCD display.")
        assert decision.display_type == "LCD"
        assert decision.locked is True

    def test_3_and_4_ifp_features_belong_to_lcd(self):
        for message in (
            "I need an interactive display.",
            "I need a touch screen for meetings.",
        ):
            decision = _route(message)
            assert decision.display_type == "LCD", message
            assert decision.subtype == "IFP", message
            assert decision.ask_customer is True, "推断出的类型要向客户确认"

    def test_5_plain_display_is_unknown_and_asks(self):
        decision = _route("I need a display.")
        assert decision.display_type == "UNKNOWN"
        assert decision.ask_customer is True
        assert decision.confidence == 0.0, "信息不足时不硬猜"

    def test_6_outdoor_advertising_is_led(self):
        decision = _route("I need a display for outdoor advertising.")
        assert decision.display_type == "LED"
        assert decision.source == "INFERENCE"
        assert decision.ask_customer is True

    def test_7_meeting_room_gets_a_suggestion_to_confirm(self):
        decision = _route("I need a screen for a meeting room.")
        assert decision.display_type in ("LED", "LCD")
        assert decision.status == "INFERRED"
        assert decision.ask_customer is True, "AI 建议必须让客户确认"

    def test_8_meeting_room_plus_writing_is_lcd_ifp(self):
        decision = _route("I need a screen for a meeting room and we need to write on it.")
        assert decision.display_type == "LCD"
        assert decision.subtype == "IFP"

    def test_9_huge_screen_with_far_viewing_prefers_led(self):
        decision = _route("I need a huge display for a meeting room, viewing distance is 10m.")
        assert decision.display_type == "LED"
        assert decision.ask_customer is True

    def test_14_customer_asking_what_led_is_gets_an_explanation(self):
        for message in ("What is LED?", "What is the difference between LED and LCD?"):
            decision = _route(message)
            assert decision.needs_explanation is True, message
            assert decision.display_type == "UNKNOWN", message
            assert decision.ask_customer is True


class TestVisionAndConfirmation:

    def test_10_11_vision_result_is_inferred_and_confirmed(self):
        for vision_type in ("LED", "LCD"):
            decision = _route("is this the right kind of screen?", vision_display_type=vision_type)
            assert decision.display_type == vision_type
            assert decision.source == "VISION"
            assert decision.status == "INFERRED"
            assert decision.ask_customer is True

    def test_12_vision_unknown_keeps_asking(self):
        decision = _route("is this the right kind of screen?", vision_display_type="")
        assert decision.display_type == "UNKNOWN"
        assert decision.ask_customer is True

    def test_13_customer_accepts_ai_judgement(self):
        inferred = _route("I need a screen for outdoor advertising.")  # LED / INFERRED
        adopted = _route("I don't know", current=inferred)
        assert adopted.display_type == "LED"
        assert adopted.status == "CONFIRMED"
        assert adopted.locked is True

    def test_15_customer_lets_us_choose(self):
        decision = _route("you decide for me, we need something for a meeting room")
        assert decision.display_type in ("LED", "LCD")
        assert decision.ask_customer in (True, False)  # 建议 + 确认，或直接采用


class TestLockingAndSwitching:

    def test_16_locked_led_is_not_changed_by_scene(self):
        led = _route("I want LED.")
        assert led.locked is True
        later = _route("it is for a meeting room, indoor", current=led)
        assert later.display_type == "LED", "已确认类型不能被场景改掉"
        assert later.locked is True

    def test_17_explicit_change_is_allowed(self):
        led = _route("I want LED.")
        switched = _route("Actually, I want LCD.", current=led)
        assert switched.display_type == "LCD"
        assert switched.source == "CUSTOMER"
        assert switched.locked is True

    def test_state_machine_fields_are_present(self):
        decision = _route("I need an LCD video wall.").to_dict()
        for key in (
            "display_type",
            "status",
            "source",
            "confidence",
            "reason",
            "locked",
        ):
            assert key in decision, key


class TestLegacyIsolation:

    def test_orchestrator_blocks_requirement_questions_before_type_confirmation(self):
        """收口层强约束（计划 v2.9.3 §五/§六）：类型没确认前，任何路径都不许问需求细节。

        实测：客户只发 "i need a display"，AI 却回了 LED 的需求问题
        （"…cabinet design and brightness… where it's going to live"）。
        现在收口层会把这样的需求问题换成"LED 还是 LCD"的确认问题。
        """
        from src.memory.store import memory
        from src.orchestrator import DualAgentOrchestrator

        class _S:
            def run(self, session_id, message, has_vision=False, **_kwargs):
                return {
                    "intent": "need_query",
                    "next_action": "ask",
                    "response": "A display is a great place to start.",
                    "requirements": {},
                    "products": [],
                    "pending_question": "Is this going to be installed indoors or outdoors?",
                    "pending_slot": "environment",
                }

        class _Sol:
            def run(self, *args, **kwargs):
                return {"answer": "Sure.", "products": []}

        session_id = "v293-gate-1"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        try:
            orch = DualAgentOrchestrator(sales_agent=_S(), solution_agent=_Sol())
            result = {
                "response": "A display is a great place to start.",
                "pending_question": "Is this going to be installed indoors or outdoors?",
                "pending_slot": "environment",
                "next_action": "ask",
                # 类型未确认（UNKNOWN / ask_customer）
                "display_type_decision": {
                    "display_type": "UNKNOWN",
                    "status": "UNKNOWN",
                    "ask_customer": True,
                },
            }
            out = orch._finalize_turn_response(result, session_id, "i need a display")
            assert out["question_slot"] == "display_type", out.get("question_slot")
            reply = str(out.get("response") or "").lower()
            assert "led" in reply and "lcd" in reply, reply
            assert "indoors or outdoors" not in reply, "类型没定前不能问需求细节"
        finally:
            memory.clear(session_id)

    def test_first_layer_never_returns_ifp_as_a_type(self):
        """计划 §四/§十四：第一层只有 LED / LCD / UNKNOWN，IFP 是 LCD 的子类型。"""
        from src.dialogue.product_type_router import (
            LCD,
            LED,
            SUBTYPE_IFP,
            UNKNOWN,
        )

        for message in (
            "I need an interactive display",
            "I need a touch screen",
            "I need an LCD video wall",
            "I need an LED display",
            "I need a display",
        ):
            decision = _route(message)
            assert decision.display_type in (LED, LCD, UNKNOWN), message
            if decision.subtype == SUBTYPE_IFP:
                assert decision.display_type == LCD, message

    def test_final_fallback_is_led_only_as_last_resort(self):
        from src.dialogue.product_type_router import final_fallback

        fallback = final_fallback()
        assert fallback.display_type == "LED"
        assert fallback.source == "DEFAULT"
        # 只有在"所有办法都拿不到有效信息"时才用它；默认判断不返回 DEFAULT
        assert _route("I need a display.").source != "DEFAULT"
