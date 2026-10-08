"""链路改口回归：LED ↔ LCD 双向都必须能切过去。

实测 bug（客户 2026-10）：

    客户: A 3m by 5m LED screen is a good starting point ... (LED 链路已锁定)
    AI  : Is there a particular pixel pitch you need, or should I work it out
          from the viewing distance?
    客户: no i not need a led ,i wanna a lcd
    AI  : Do you have a pixel pitch in mind, for example P2.5, P3 or P5?    ← ❌

根因（都是同一个毛病："取先出现的产品类型"，而不是"客户要哪种"）：

  1. ``product_type_router._explicit_type`` 返回**先出现**的类型。改口句里客户
     往往**先否定当前类型**再给出想要的类型：
         "no i not need a led ,i wanna a lcd"
                         ↑ led 先出现（被否定）→ 判定成"还是 LED" → 不切链路
     这个错误取值出现在两处分支：
       · "类型已锁定"分支 → 判定"客户没改口" → 永远卡在 LED；
       · "类型还没锁定"分支（全新会话 / AI 只是建议过 LED）→ 反而**确认成 LED**，
         与客户原意正好相反。
     修法：``_wanted_type`` —— 否定词只作用于它后面最近的那个类型提及，
     被否定的提及不算"客户想要的"；两处分支都用它，拿不到信号才退回原口径。

  2. ``sales.runner._extract_display_type_from_message`` 同样按
     ("IFP", "LED", "LCD") 顺序返回第一个命中的字面类型 → 同样的句子取到 LED →
     ``display_type_change`` 检测不到 → 旧链路的需求档案不会被清空。
     修法：两个类型都出现时，统一走 ``_wanted_type`` 的"客户想要哪种"口径。

  3. 改口的"触发条件"只认英文改口词（actually / instead / 换成），中文的
     "不要LED，我要LCD" 根本进不了判断。修法：把否定词也纳入触发条件 ——
     真正决定切不切的是 ``_wanted_type``，它说不出客户要什么就不切，所以放宽
     门槛是安全的。

本文件只锁**行为**（双向可切 + 中文可切 + 不该切时不许误切），不锁实现细节。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _locked(kind: str):
    """已由客户确认并锁定的产品类型判断。"""
    from src.dialogue.product_type_router import (
        SOURCE_CUSTOMER,
        STATUS_CONFIRMED,
        DisplayTypeDecision,
    )

    return DisplayTypeDecision(
        display_type=kind,
        status=STATUS_CONFIRMED,
        source=SOURCE_CUSTOMER,
        confidence=1.0,
        reason="customer explicitly named the product type",
        locked=True,
    )


# 客户改口的说法：应该切到括号里的类型
LED_TO_LCD = (
    "no i not need a led ,i wanna a lcd",
    "i don't need LED, i need LCD",
    "i do not need a led, i want a lcd",
    "no, i need an lcd instead",
    "actually i want an lcd",
    "we don't want a led display, we want an lcd video wall",
)

LCD_TO_LED = (
    "no i not need a lcd ,i wanna a led",
    "i don't need LCD, i need LED",
    "we don't want LCD, we need LED",
    "no, we need led instead",
    "actually we want an led display",
)

# 同一件事的中文说法（系统默认英文回复，但客户可以用中文说）
LED_TO_LCD_ZH = (
    "不要LED，我要LCD",
    "不需要 led，换成 lcd",
    "其实我想要lcd",
)

LCD_TO_LED_ZH = (
    "我不要lcd，我要led",
    "不需要LCD，换成LED",
)


class TestWantedTypeIgnoresNegatedMention:

    def test_negated_type_is_not_the_type_the_customer_wants(self):
        """被否定的那一次提及，不能当成客户想要的类型。"""
        from src.dialogue.product_type_router import _wanted_type

        assert _wanted_type("no i not need a led ,i wanna a lcd") == "LCD"
        assert _wanted_type("i don't need LED, i need LCD") == "LCD"
        assert _wanted_type("we don't want LCD, we need LED") == "LED"
        # 句首的 "no" 是"否认上一轮建议"，不否定后面那个名词
        assert _wanted_type("no, i need an lcd instead") == "LCD"

    def test_both_negated_or_no_mention_returns_nothing(self):
        """两边都被否定 / 只说否认不说要什么 → 不给出"想要的类型"（不切链路）。"""
        from src.dialogue.product_type_router import _wanted_type

        assert _wanted_type("no") == ""
        assert _wanted_type("not led and not lcd") == ""
        assert _wanted_type("") == ""

    def test_single_mention_is_returned_unchanged(self):
        """只提一种时行为不变（不与既有口径冲突）。"""
        from src.dialogue.product_type_router import _wanted_type

        assert _wanted_type("i need a led display") == "LED"
        assert _wanted_type("we need an lcd video wall") == "LCD"
        assert _wanted_type("actually i want lcd") == "LCD"


class TestRouterSwitchesBothDirections:

    def test_led_session_switches_to_lcd(self):
        """LED 链路锁定后客户改口要 LCD → 必须切到 LCD。"""
        from src.dialogue.product_type_router import route_display_type

        for message in LED_TO_LCD:
            decision = route_display_type(message, current=_locked("LED"))
            assert decision.display_type == "LCD", (message, decision.display_type)
            assert decision.locked is True, message
            assert decision.status == "CONFIRMED", message

    def test_lcd_session_switches_to_led(self):
        """LCD 链路锁定后客户改口要 LED → 必须切到 LED。"""
        from src.dialogue.product_type_router import route_display_type

        for message in LCD_TO_LED:
            decision = route_display_type(message, current=_locked("LCD"))
            assert decision.display_type == "LED", (message, decision.display_type)
            assert decision.locked is True, message
            assert decision.status == "CONFIRMED", message

    def test_chinese_switch_phrasing_also_switches(self):
        """中文说法（"不要LED，我要LCD"）同样必须能切 —— 不能只有英文能切。"""
        from src.dialogue.product_type_router import route_display_type

        for message in LED_TO_LCD_ZH:
            decision = route_display_type(message, current=_locked("LED"))
            assert decision.display_type == "LCD", (message, decision.display_type)
        for message in LCD_TO_LED_ZH:
            decision = route_display_type(message, current=_locked("LCD"))
            assert decision.display_type == "LED", (message, decision.display_type)

    def test_questions_and_plain_requirement_talk_never_switch(self):
        """只是提问 / 说需求 / 说场景 → 保持已确认的类型，不许被误切。"""
        from src.dialogue.product_type_router import route_display_type

        for message in (
            "can an led display do 4k?",
            "what is the brightness?",
            "we are indoor, 5m viewing distance",
            "the size is 3m by 5m",
            "how much is it?",
        ):
            decision = route_display_type(message, current=_locked("LED"))
            assert decision.display_type == "LED", (message, decision.display_type)

        for message in (
            "what is the bezel size?",
            "does it support 4k?",
            "we need it for a meeting room",
        ):
            decision = route_display_type(message, current=_locked("LCD"))
            assert decision.display_type == "LCD", (message, decision.display_type)


class TestSwitchAlsoWorksBeforeTheTypeIsLocked:

    def test_correction_on_a_fresh_session_picks_the_wanted_type(self):
        """全新会话里客户直接改口：不能把"被否定的 LED"当成他的选择。"""
        from src.dialogue.product_type_router import route_display_type

        decision = route_display_type("no i not need a led ,i wanna a lcd")
        assert decision.display_type == "LCD", decision.display_type
        assert decision.status == "CONFIRMED"

        decision = route_display_type("no i not need a lcd ,i wanna a led")
        assert decision.display_type == "LED", decision.display_type
        assert decision.status == "CONFIRMED"

    def test_correction_after_an_inferred_led_suggestion(self):
        """AI 只是**建议**过 LED（inferred，客户还没确认）→ 客户纠正要 LCD，必须切。"""
        from src.dialogue.product_type_router import (
            SOURCE_INFERENCE,
            STATUS_INFERRED,
            DisplayTypeDecision,
            route_display_type,
        )

        suggested_led = DisplayTypeDecision(
            display_type="LED",
            status=STATUS_INFERRED,
            source=SOURCE_INFERENCE,
            confidence=0.65,
            reason="very large screen or far viewing distance -> LED is usually the better fit",
            ask_customer=True,
        )
        decision = route_display_type(
            "no i not need a led ,i wanna a lcd", current=suggested_led
        )
        assert decision.display_type == "LCD", decision.display_type
        assert decision.status == "CONFIRMED"


class TestRunnerSeesTheSwitchSoTheChainResets:

    def test_display_type_extraction_follows_the_wanted_type(self):
        """runner 的提取器同样要取"客户想要的"，否则重置检测不到。"""
        from src.agents.sales.runner import _extract_display_type_from_message

        for message in LED_TO_LCD:
            assert _extract_display_type_from_message(message) == "LCD", message
        for message in LCD_TO_LED:
            assert _extract_display_type_from_message(message) == "LED", message

        # 单一类型 / IFP / 没有类型的老口径不变
        assert _extract_display_type_from_message("i need a led display") == "LED"
        assert _extract_display_type_from_message("we need 交互平板") == "IFP"
        assert _extract_display_type_from_message("do you have stock?") is None

    def test_switch_triggers_a_full_requirement_reset(self):
        """改口 → display_type_change → 旧链路需求全量清空（不能把 LED 事实带进 LCD）。"""
        from src.agents.sales.runner import _extract_display_type_from_message
        from src.rag.session_switch import detect_requirement_reset

        previous = "LED"
        message = "no i not need a led ,i wanna a lcd"
        current = _extract_display_type_from_message(message)
        assert current == "LCD"

        decision = detect_requirement_reset(
            message,
            requirements={"environment": "indoor", "pixel_pitch_mm": 3.0},
            display_type_change=(previous, current),
            use_llm=False,
        )
        assert decision.should_reset is True
        assert decision.reason == "display_type_switch"
