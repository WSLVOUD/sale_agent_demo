"""计划 v2.9.2（Greeting + 产品类型入口）的回归测试。

覆盖计划 §二十五 的 10 条：Greeting / 聚合后一次理解 / UNKNOWN 选择产品 /
LED·LCD·IFP 三个入口 / 已进 LED 链后再 Hi 不重置 / LCD→LED 切换 /
LED+LCD 两条需求 / LED 推荐链路回归。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _understand(message: str, book=None, intent: str = ""):
    from src.dialogue.turn_understanding import understand_turn

    return understand_turn(message, intent=intent, book=book)


class TestGreetingEntry:

    def test_greeting_flag_is_carried_in_the_understanding(self):
        from src.dialogue.turn_understanding import understand_turn

        # 客户把招呼和需求放在一起（前端会聚合成一个回合）
        aggregated = understand_turn("hi\ni need a display")
        assert aggregated.greeting_present is True
        # 计划 §十六：只说 "display"、没点名品类 → UNKNOWN（入口应反问品类）
        assert aggregated.product_domain == "UNKNOWN"
        # 点名 LED 才算 LED
        assert understand_turn("hi\ni need an LED display").product_domain == "LED"
        # 只有需求、没有招呼的句子不该被当成招呼
        assert understand_turn("i need a display").greeting_present is False

    def test_plain_greeting_asks_which_product(self):
        from src.dialogue.product_router import PRODUCT_SELECTION, route_product_domain
        from src.dialogue.turn_kind import PURE_CONVERSATION

        understanding = _understand("Hi")
        assert understanding.conversation_type == PURE_CONVERSATION
        assert understanding.product_domain == "UNKNOWN"
        assert route_product_domain(understanding.product_domain) == PRODUCT_SELECTION
        assert understanding.has_business_signal is False

    def test_greeting_with_business_signal_goes_to_led_entry(self):
        from src.dialogue.product_router import LED_ENTRY, route_product_domain
        from src.dialogue.turn_kind import CONVERSATION_WITH_BUSINESS_SIGNAL

        understanding = _understand("Hi, I need an outdoor LED screen.")
        assert understanding.conversation_type == CONVERSATION_WITH_BUSINESS_SIGNAL
        assert understanding.has_business_signal is True
        assert route_product_domain(understanding.product_domain) == LED_ENTRY

    def test_aggregated_messages_are_understood_once(self):
        """聚合后的一个回合（Hi + How are you?）只产生一次理解结果。"""
        from src.dialogue.turn_kind import PURE_CONVERSATION

        aggregated = "Hi, how are you?"
        understanding = _understand(aggregated)
        assert understanding.conversation_type == PURE_CONVERSATION
        assert understanding.product_domain == "UNKNOWN"


class TestProductEntries:

    def test_lcd_and_ifp_entries(self):
        from src.dialogue.product_router import (
            IFP_ENTRY,
            LCD_ENTRY,
            route_product_domain,
        )

        lcd = _understand("Hi, I need an LCD video wall.")
        ifp = _understand("Hi, I need an interactive display for a classroom.")
        assert route_product_domain(lcd.product_domain) == LCD_ENTRY
        assert route_product_domain(ifp.product_domain) == IFP_ENTRY

    def test_unknown_screen_asks_for_the_type(self):
        from src.dialogue.product_router import PRODUCT_SELECTION, route_product_domain

        understanding = _understand("Hi, I need a screen.")
        assert understanding.product_domain == "UNKNOWN"
        assert route_product_domain(understanding.product_domain) == PRODUCT_SELECTION

    def test_comparison_question_is_not_a_requirement_chain(self):
        from src.dialogue.product_router import (
            PRODUCT_COMPARISON_ENTRY,
            route_product_domain,
        )

        understanding = _understand("Which is better, LED or LCD?")
        assert understanding.comparison is True
        assert (
            route_product_domain(understanding.product_domain, comparison=True)
            == PRODUCT_COMPARISON_ENTRY
        )


class TestRequirementBook:

    def test_greeting_after_led_does_not_reset_the_led_chain(self):
        from src.dialogue.turn_understanding import RequirementBook

        book = RequirementBook()
        first = _understand("Hi, I need an outdoor LED screen for a stadium.", book=book)
        book.apply(first)
        assert book.active.product_domain == "LED"
        assert book.active.slots

        # 客户后来只说了句 Hi → 不能重置 / 清空 LED 需求
        greetings = _understand("Hi", book=book, intent="greeting")
        book.apply(greetings)
        assert book.active.product_domain == "LED"
        assert book.get("LED").slots.get("environment") == "outdoor"

    def test_switching_from_lcd_to_led_keeps_both(self):
        from src.dialogue.turn_understanding import RequirementBook

        book = RequirementBook()
        book.apply(_understand("I need an LCD video wall for the lobby.", book=book))
        assert book.active.product_domain == "LCD"
        book.apply(_understand("Actually, I want LED instead.", book=book))
        assert book.active.product_domain == "LED"
        # LCD 的信息不能被删掉（计划 §十四）
        assert book.get("LCD") is not None

    def test_two_products_create_two_requirements(self):
        from src.dialogue.turn_understanding import RequirementBook

        book = RequirementBook()
        understanding = _understand(
            "I need LED for the stadium and LCD for the meeting room.", book=book
        )
        assert understanding.multi_product is True
        book.apply(understanding)
        domains = {item.product_domain for item in book.requirements}
        assert {"LED", "LCD"} <= domains


class TestProductPolicies:

    def test_customer_greeting_always_gets_a_greeting_back(self):
        """实测反馈（2026-09-23）：客户 "hello，i need a led diplay" → 回复里没有招呼。

        两个原因：① `suppress_greeting`（首次接待刚结束别重复问候）把它一起挡掉了；
        ② 招呼只作为"可选 opening 提示"交给 LLM，模型经常忽略。
        现在 `greeting_required` 是显式要求，且不受 suppress_greeting 影响。
        """
        from src.agents.sales.nodes.script_generator import _natural_reply
        from src.dialogue.turn_kind import has_greeting_opener

        message = "hello，i need a led diplay"
        assert has_greeting_opener(message) is True

        state = {
            "session_id": "greet-always-1",
            "current_message": message,
            "requirements": {"display_type": "LED"},
            "pending_slot": "environment",
            # 关键：首次接待刚结束（以前会把招呼一起吞掉）
            "suppress_greeting": True,
            "turn_understanding": {"greeting_present": True},
            "acknowledgement": "",
            "requirement_profile": None,
            "messages": [],
        }
        text = _natural_reply(
            state,
            question="Is this going to be installed indoors or outdoors?",
            slot="environment",
            business_goal="collect the missing hard requirement",
        )
        assert text.lower().startswith(("hi", "hello")), text
        assert text.count("?") == 1

    def test_context_marks_the_greeting_as_required(self):
        from src.dialogue import build_context

        context = build_context(
            action="ASK",
            customer_message="hello, i need an led display",
            question="Is this going to be installed indoors or outdoors?",
            question_slot="environment",
            greeting_required=True,
        )
        assert context.greeting_required is True
        block = context.prompt_block()
        assert "You MUST open your reply by greeting them back" in block

    def test_no_em_dash_in_customer_text(self):
        """客户口径：破折号一律换逗号（型号里的连字符不受影响）。"""
        from src.dialogue.response_generator import normalize_customer_punctuation

        normalized = normalize_customer_punctuation(
            "One thing \u2014 the cabinet \u2014 drives brightness. TW11-3216-P3.0 fits."
        )
        assert "\u2014" not in normalized
        assert "," in normalized
        assert "TW11-3216-P3.0" in normalized

    def test_greeting_is_merged_into_the_requirement_reply(self):
        """实测 bug：客户发 "hi" + "i need a display" → 回复里招呼被丢掉。

        打招呼必须并进最终话术（同一个回合、同一条回复），而不是被 intent=need_query
        的分支吃掉；首次接待刚结束（suppress_greeting）时不重复打招呼。
        """
        from src.agents.sales.nodes.script_generator import _natural_reply

        base = {
            "session_id": "greet-merge-1",
            "current_message": "hi\ni need a display",
            "requirements": {"display_type": "LED"},
            "pending_slot": "environment",
            "suppress_greeting": False,
            "turn_understanding": {"greeting_present": True},
            "acknowledgement": "",
            "requirement_profile": None,
            "messages": [],
        }
        text = _natural_reply(
            base,
            question="Is this going to be installed indoors or outdoors?",
            slot="environment",
            business_goal="collect the missing hard requirement",
        )
        assert text.lower().startswith(("hi", "hello")), text
        assert "indoors or outdoors" in text.lower()
        assert text.count("?") == 1

        suppressed = dict(base, suppress_greeting=True, session_id="greet-merge-2")
        text2 = _natural_reply(
            suppressed,
            question="Is this going to be installed indoors or outdoors?",
            slot="environment",
            business_goal="collect the missing hard requirement",
        )
        # 客户自己打了招呼 → 即使"首次接待刚结束"，也要回应招呼（2026-09-23 实测口径）
        assert text2.lower().startswith(("hi", "hello")), text2

    def test_greeting_node_asks_for_the_product_type(self):
        """Greeting 节点：纯招呼 → 欢迎 + 问客户要找哪种显示产品（计划 §五）。"""
        from src.agents.sales.nodes.script_generator import script_generator

        state = {
            "session_id": "greet-1",
            "current_message": "Hi",
            "intent": "greeting",
            "messages": [{"role": "user", "content": "Hi"}],
            "requirements": {},
            "suppress_greeting": True,
            "product_entry": "PRODUCT_SELECTION",
            "turn_understanding": {"business_signal": {}},
        }
        out = script_generator(state)
        assert "LED display" in out.get("response", "")
        assert "LCD" in out.get("response", "")
        assert "interactive flat panel" in out.get("response", "").lower()
        assert out.get("greeting_entry") == "product_selection"

    def test_led_policy_uses_the_existing_chain(self):
        from src.dialogue.product_router import LEDPolicy, policy_for
        from src.models.requirement import RequirementProfile

        policy = policy_for("LED")
        assert isinstance(policy, LEDPolicy) and policy.implemented is True
        slots = {"display_type": "LED", "target_width_mm": 3000, "target_height_mm": 5000}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        assert policy.get_missing_requirements(profile)
        assert policy.get_next_question(profile)

    def test_lcd_and_ifp_are_placeholders_with_the_same_interface(self):
        from src.dialogue.product_router import policy_for

        for domain in ("LCD", "IFP"):
            policy = policy_for(domain)
            assert policy.implemented is False
            # 接口齐备，但不实现业务逻辑（计划 §十二）
            assert policy.get_requirement_profile() == {}
            assert policy.get_missing_requirements() == []
            assert policy.get_next_question() is None
            assert policy.can_recommend() is False
            assert policy.recommend() == []
            assert policy.validate("anything") == []
