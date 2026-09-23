"""计划 v2.9.1（Others 节点清理与重构）的回归测试。

覆盖计划 §十六 的 A（Others）、C（闲聊转需求）、D（product_domain 接口）三组，
以及 §五/§十 的轮次类型边界。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestOthersBoundaries:

    def test_pure_chat_never_goes_to_free_question(self):
        from src.dialogue.turn_kind import PURE_CONVERSATION, classify_turn_kind
        from src.agents.solution.nodes.intent import route_by_turn_kind

        for message in ("Hi", "How are you?", "Thanks!", "Have a nice day.", "你好"):
            assert classify_turn_kind(message) == PURE_CONVERSATION, message
            assert route_by_turn_kind({"turn_kind": PURE_CONVERSATION}) == "conversation"

    def test_free_question_enters_others(self):
        from src.dialogue.turn_kind import FREE_QUESTION, classify_turn_kind
        from src.agents.solution.nodes.intent import route_by_turn_kind

        for message in (
            "Can you deliver faster?",
            "Do you provide installation?",
            "Can I visit your factory?",
            "What payment methods do you accept?",
        ):
            assert classify_turn_kind(message) == FREE_QUESTION, message
        assert route_by_turn_kind({"turn_kind": FREE_QUESTION, "intent": "others"}) == "others"

    def test_confirmed_requirements_are_never_contradicted(self):
        from src.rag.model_guard import strip_environment_contradictions, requirement_summary

        source = {"indoor": True, "is_rental": False}
        assert requirement_summary(source)["environment"] == "indoor"
        text, removed = strip_environment_contradictions(
            "For an outdoor rental panel we would use a waterproof cabinet.", source
        )
        assert removed
        assert "outdoor" not in text.lower()

    def test_models_are_not_named_unless_the_customer_did(self):
        from src.rag.model_guard import strip_model_mentions

        text, removed = strip_model_mentions("The TW11-3216-P3.0 would fit well.")
        assert removed
        assert "TW11-3216-P3.0" not in text
        kept, removed_kept = strip_model_mentions(
            "The TW11-3216-P3.0 would fit well.", allow=["TW11-3216-P3.0"]
        )
        assert not removed_kept and "TW11-3216-P3.0" in kept


class TestConversationWithBusinessSignal:

    def test_chat_with_business_information_keeps_the_signals(self):
        from src.dialogue.turn_kind import (
            CONVERSATION_WITH_BUSINESS_SIGNAL,
            business_signals,
            classify_turn_kind,
        )

        message = (
            "How are you? By the way, we are actually planning an outdoor screen for a stadium."
        )
        assert classify_turn_kind(message) == CONVERSATION_WITH_BUSINESS_SIGNAL
        signals = business_signals(message)
        assert signals.get("environment") == "outdoor" or signals.get("outdoor") is True

    def test_signals_land_in_the_requirement_profile(self):
        from src.models.requirement import RequirementProfile
        from src.dialogue.turn_kind import business_signals

        message = "Hi, we are actually planning an outdoor stadium screen"
        slots = business_signals(message)
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        assert profile.environment == "outdoor"


class TestProductDomainInterface:

    def test_domains_are_detected(self):
        from src.dialogue.turn_kind import (
            IFP,
            LCD,
            LED,
            MULTI,
            UNKNOWN,
            detect_product_domain,
        )

        # 计划 v2.9.3 §五：只说 "screen / display" **不再默认 LED** —— 这是"未点名品类"；
        # LED 的倾向由 Product Type Router 从用途推断出来（要客户确认），而不是词表默认。
        from src.dialogue.product_type_router import route_display_type

        assert detect_product_domain("I need an outdoor screen for a stadium") == UNKNOWN
        inferred = route_display_type("I need an outdoor screen for a stadium")
        assert inferred.display_type == LED and inferred.status == "INFERRED"
        assert detect_product_domain("I need an LCD video wall for a lobby") == LCD
        assert detect_product_domain("interactive flat panel for a meeting room") == IFP
        assert detect_product_domain("What is the difference between LED and LCD?") == MULTI
        assert detect_product_domain("How are you?") == UNKNOWN

    def test_every_domain_passes_the_router(self):
        from src.agents.solution.nodes.intent import route_by_intent
        from src.dialogue.turn_kind import ALL_PRODUCT_DOMAINS

        allowed = {"recommendation", "product_question", "conversation", "others"}
        for domain in ALL_PRODUCT_DOMAINS:
            for intent in ("recommendation", "product_question", "conversation", "others"):
                branch = route_by_intent({"intent": intent, "product_domain": domain})
                assert branch in allowed, (domain, intent, branch)
            # 未知意图默认落到 others（Free Question），不是 conversation
            assert route_by_intent({"product_domain": domain}) == "others"

    def test_context_carries_the_domain_into_the_prompt(self):
        from src.dialogue import ResponseContext

        context = ResponseContext(action="FREE_QUESTION", product_domain="LCD")
        assert "Product domain: LCD" in context.prompt_block()
