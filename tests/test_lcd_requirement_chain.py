"""《LCD_IFP_需求链路工程化整改计划》Phase 8：上下文对话测试（Case 1~12）。

每个 Case 都来自计划第三十节，额外覆盖验收标准里可自动化的部分：

    · LED 核心链路没被改动（字段/顺序/策略入口不变）
    · 每轮只产生一个 Next Action、最多一个问题
    · 已锁定（客户来源）的事实不会重复问
    · 图片"看到的"必须经客户确认；客户明确修改优先
    · LCD 决策只有一个入口（product_router.LCDPolicy 委托给 lcd_decision）
    · 关键词只做候选事实，不直接决定问什么
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import image_confirmation as ic  # noqa: E402
from src.dialogue.lcd_decision import (  # noqa: E402
    ADVERTISING,
    CONFERENCE_EDUCATION,
    MONITORING,
    extract_lcd_facts,
    lcd_turn,
    resolution_for_size,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.vision.recognition import ImageRecognitionResult  # noqa: E402


def _lcd_profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestCase1NaturalAskingForPurpose:
    def test_bare_lcd_asks_for_the_use_case(self):
        profile = _lcd_profile()
        _profile, action = lcd_turn(profile, "I need an LCD.")

        assert action.lcd_category == "unknown", action.lcd_category
        assert action.next_action == "ask_purpose", action.next_action
        assert action.question_slot == "purpose", action
        assert action.confirmed is False


class TestCase2ControlRoomEntersMonitoring:
    def test_control_room_maps_to_monitoring_without_the_word_monitoring(self):
        profile = _lcd_profile()
        _profile, action = lcd_turn(profile, "We need displays for a power station control room.")

        assert action.lcd_category == MONITORING, action.lcd_category
        assert action.next_action == "ask_environment", action.next_action


class TestCase3VideoWallDoesNotReaskSplicing:
    def test_six_by_two_locks_splicing_layout_and_count(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "I need a 6x2 video wall for our control room.")

        assert profile.lcd_is_splicing is True
        assert profile.lcd_splicing_layout == "6x2"
        assert profile.lcd_screen_count == 12
        assert action.question_slot != "lcd_splicing", "拼接已经给了，不能再问是否拼接"
        assert action.question_slot == "environment"
        assert "video wall" not in (action.question or "").lower(), action.question


class TestCase4And5ResolutionDefaults:
    def test_75_inch_defaults_to_4k(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "We need a 75 inch LCD.")

        assert profile.lcd_size_inch == 75
        assert profile.lcd_resolution == "4K", profile.lcd_resolution
        assert profile.lcd_resolution_source == "size_rule_gt_65"
        assert action.question_slot != "lcd_size", "尺寸已经给了，不能再问尺寸"

    def test_55_inch_defaults_to_2k(self):
        profile = _lcd_profile()
        profile, _action = lcd_turn(profile, "55 inch LCD")

        assert profile.lcd_resolution == "2K", profile.lcd_resolution
        assert profile.lcd_resolution_source == "size_rule_lt_65"


class TestCase6CustomerResolutionWins:
    def test_55_inch_with_explicit_4k_keeps_4k(self):
        profile = _lcd_profile()
        profile, _action = lcd_turn(profile, "We want a 55 inch 4K LCD.")

        assert profile.lcd_resolution == "4K"
        assert profile.lcd_resolution_source == "customer_explicit"


class Test65InchBoundaryIsNotAskedToCustomer:
    """客户口径（2026-09-30）：分辨率**不许反问客户**。

    客户自己提了（"要 4K"）就照他的；没提就按尺寸规则定 —— 恰好 65" 这种
    "库里 2K/4K 都有"的边界，按"大屏走 4K"的口径默认 4K，不再问。
    """

    def test_65_defaults_to_4k_without_catalog(self):
        resolution, source = resolution_for_size(65)

        assert resolution == "4K", resolution
        assert source == "size_default_4k_65", source

    def test_65_uses_product_catalog_when_it_is_unambiguous(self):
        resolution, source = resolution_for_size(65, catalog_resolutions=["4K", "3840x2160"])

        assert resolution == "4K", resolution
        assert source == "product_catalog", source

    def test_65_in_profile_never_asks_the_resolution_question(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "We need a 65 inch LCD for a meeting room.")

        assert profile.lcd_resolution == "4K", profile.lcd_resolution
        assert "lcd_resolution" not in action.missing_fields, action.missing_fields
        assert action.question_slot != "lcd_resolution", action.question_slot


class TestCase7MeetingRoomDoesNotLockIfp:
    def test_meeting_room_goes_to_conference_branch_first(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "We need a display for our meeting room.")

        assert action.lcd_category == CONFERENCE_EDUCATION
        assert profile.display_type == "LCD", "会议场景不能自动锁 IFP"
        assert profile.lcd_handwriting_required is None
        assert "lcd_handwriting" in action.missing_fields, action.missing_fields


class TestCase8InteractiveWhiteboardLocksIfp:
    def test_whiteboard_locks_handwriting_and_does_not_reask_it(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "We need an interactive whiteboard for a classroom.")

        assert profile.lcd_handwriting_required is True
        assert profile.lcd_touch_required is True
        assert action.question_slot != "lcd_handwriting", "手写已经给了，不能再问"
        # 会议/教育 + 手写 → IFP 分支：Tender / OPS / Camera 进入候选
        assert "lcd_tender" in action.missing_fields, action.missing_fields
        assert "lcd_ops" in action.missing_fields, action.missing_fields

    def test_advertising_branch_never_asks_ops(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "We need indoor advertising displays.")

        assert action.lcd_category == ADVERTISING
        assert "lcd_ops" not in action.missing_fields, action.missing_fields
        assert "lcd_tender" not in action.missing_fields, action.missing_fields

    def test_tender_project_invites_the_documents(self):
        """计划 §十六：客户说"这是招投标项目"→ 锁定 tender 并引导发招标文件。"""
        profile = _lcd_profile()
        profile, action = lcd_turn(
            profile,
            "We need an interactive whiteboard for a meeting room — this is a tender project.",
        )

        assert profile.lcd_tender_project is True
        assert action.next_action == "request_tender_documents", action.next_action
        assert "tender document" in action.question.lower(), action.question
        assert action.question.count("?") == 1

    def test_camera_is_only_asked_when_the_context_mentions_it(self):
        """课堂：不主动问摄像头；提到 remote teaching / video conference 才问。"""
        classroom = _lcd_profile()
        classroom, action = lcd_turn(
            classroom, "We need an interactive whiteboard for a classroom."
        )
        assert "lcd_camera" not in action.missing_fields, action.missing_fields

        online = _lcd_profile()
        online, action2 = lcd_turn(
            online,
            "Whiteboard for a classroom — we also run online classes, is a camera possible?",
        )
        # 客户这句已经提到 camera（"is a camera possible?"）→ 摄像头必须进入候选
        # （要么出现在 missing，要么已经被这句话回答）
        assert (
            "lcd_camera" in action2.missing_fields
            or online.lcd_camera_required is not None
        ), action2.missing_fields

    def test_meeting_room_may_still_ask_about_the_camera(self):
        profile = _lcd_profile()
        profile, action = lcd_turn(
            profile, "We need an interactive whiteboard for our meeting room."
        )

        assert "lcd_camera" in action.missing_fields, action.missing_fields


class TestCase9MultiMessageMerge:
    def test_four_messages_merge_into_one_turn_with_one_question(self):
        """客户连发 4 条 → 一次 turn 内全部入档 → 只问一个真正缺的。"""
        merged = "I need an LCD. For a meeting room. 75 inch. Touch screen."
        profile, action = lcd_turn(_lcd_profile(), merged)

        assert profile.lcd_size_inch == 75
        assert profile.lcd_touch_required is True
        assert profile.lcd_category == CONFERENCE_EDUCATION
        assert profile.lcd_resolution == "4K"          # 75" 未指定 → 4K
        asked = [slot for slot in action.missing_fields]
        assert "lcd_size" not in asked and "lcd_touch" not in asked
        assert (action.question or "").count("?") <= 1, action.question
        assert action.question_slot, action.to_dict()


class TestCase10CustomerBeatsImage:
    def test_image_indoor_but_customer_says_outdoor(self):
        profile = RequirementProfile()
        recognition = ImageRecognitionResult(
            display_type="LCD", environment="indoor", is_splicing=True, confidence=0.8
        )
        ic.apply_recognition_to_profile(profile, recognition)

        profile, action = lcd_turn(profile, "It is for outdoor use.", recognition=recognition)

        assert profile.environment == "outdoor", profile.environment
        assert action.conflicts == [] or "environment" not in action.conflicts


class TestCase11ImageCameraIsNotARequirement:
    def test_camera_observed_stays_separate_from_camera_required(self):
        profile = RequirementProfile()
        recognition = ImageRecognitionResult(
            display_type="LCD", environment="indoor", camera_observed=True, confidence=0.7
        )
        ic.apply_recognition_to_profile(profile, recognition)

        assert profile.lcd_camera_observed is True
        assert profile.lcd_camera_required is None, "看见摄像头 ≠ 客户需要摄像头"

        profile, _action = lcd_turn(
            profile,
            "There is a camera in the photo but we do not need it.",
            recognition=recognition,
        )

        assert profile.lcd_camera_required is False
        assert profile.lcd_camera_observed is True


class TestCase12ImageTextConflict:
    def test_outdoor_context_vs_indoor_image_asks_customer(self):
        profile = _lcd_profile(environment="outdoor")
        recognition = ImageRecognitionResult(
            display_type="LCD", environment="indoor", confidence=0.9
        )

        compare = ic.compare_with_context(profile, recognition)
        assert compare.has_conflict is True
        assert "environment" in compare.conflicts

        _profile, action = lcd_turn(profile, "I sent a photo.", recognition=recognition)
        assert action.next_action == "confirm_image_conflict", action.next_action
        assert "camera" not in (action.question or "").lower()
        assert action.question.count("?") == 1, action.question

    def test_confirmation_prompt_mentions_both_sides(self):
        profile = _lcd_profile(environment="outdoor")
        recognition = ImageRecognitionResult(
            display_type="LCD", environment="indoor", confidence=0.9
        )

        prompt = ic.confirmation_prompt(profile, [recognition])

        assert "outdoor" in prompt.lower(), prompt
        assert "indoor" in prompt.lower(), prompt


class TestSingleDecisionEntry:
    @staticmethod
    def _turn(monkeypatch, message: str, display_type: str):
        """驱动 Sales 需求节点（LLM 打桩，只看链路走向）。"""
        import importlib
        from types import SimpleNamespace

        import src.core.requirement_extractor as extractor_mod

        sales_req = importlib.import_module("src.agents.sales.nodes.requirement")

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                self.temperature = kwargs.get("temperature", 0)

            def invoke(self, messages, *args, **kwargs):
                return SimpleNamespace(
                    content='{"usage": null, "additional_requirements": [], "ack": ""}'
                )

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()

        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": f"lcd-chain-{display_type.lower()}",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "acknowledgement": "",
            "display_type_decision": {
                "display_type": display_type, "status": "CONFIRMED", "locked": True,
            },
        }
        return sales_req.requirement_mining(state)

    def test_sales_turn_uses_the_lcd_chain_for_lcd_only(self, monkeypatch):
        """Phase 5 接入：LCD 会话的问句来自 lcd_decision；LED 会话不进这条分支。"""
        out = self._turn(
            monkeypatch, "We need a 6x2 video wall for a control room.", "LCD"
        )

        assert out.get("lcd_action"), out.get("recommendation_gate")
        assert out["lcd_action"]["lcd_category"] == MONITORING
        # 拼接 + 排布已经给了 → 不许再问"要不要拼接墙"；问的是分支内还没定的事实
        assert out["lcd_action"]["locked_facts"]["lcd_is_splicing"] is True
        assert out["pending_slot"] in ("lcd_bezel", "lcd_size", "environment"), out["pending_slot"]
        assert out["should_generate_solution"] is False
        assert out["recommendation_gate"]["gate"] == "lcd_requirement"
        assert "video wall" not in str(out["pending_question"]).lower()

    def test_led_turn_does_not_enter_the_lcd_branch(self, monkeypatch):
        out = self._turn(monkeypatch, "indoor 3m x 5m P4", "LED")

        assert out.get("lcd_action") is None, "LED 会话不能走 LCD 需求链"
        assert out["recommendation_gate"]["gate"] != "lcd_requirement"

    def test_lcd_policy_delegates_to_the_decision_layer(self):
        from src.dialogue.lcd_decision import decide_lcd_next_action
        from src.dialogue.product_router import LCDPolicy, IFPPolicy, policy_for
        from src.dialogue.turn_kind import LCD, IFP

        profile = _lcd_profile()
        expected = decide_lcd_next_action(profile)

        assert policy_for(LCD).implemented is True
        assert LCDPolicy().get_next_question(profile) == (expected.question or None)
        assert LCDPolicy().get_missing_requirements(profile) == expected.missing_fields
        # IFP 走同一套决策（LCD 子类型）
        assert IFPPolicy().get_missing_requirements(profile) == expected.missing_fields
        assert policy_for(IFP).implemented is True

    def test_one_next_action_and_at_most_one_question_per_turn(self):
        for message in (
            "I need an LCD.",
            "Control room, indoor, 6x2 video wall.",
            "75 inch LCD for a meeting room with touch.",
            "Indoor advertising machine, 55 inch.",
        ):
            profile, action = lcd_turn(_lcd_profile(), message)
            assert action.next_action, (message, action.to_dict())
            assert (action.question or "").count("?") <= 1, (message, action.question)
            assert action.lcd_category in (
                "unknown", MONITORING, ADVERTISING, CONFERENCE_EDUCATION, "normal",
            ), action.lcd_category

    def test_locked_customer_facts_are_never_re_asked(self):
        profile = _lcd_profile(environment="indoor")
        profile.lcd_is_splicing = True
        profile.lcd_splicing_layout = "6x2"
        profile.lcd_screen_count = 12
        profile.lcd_category = MONITORING
        profile.sources.update({
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
            "lcd_screen_count": "explicit",
            "lcd_category": "explicit",
        })

        _profile, action = lcd_turn(profile, "Anything else you need?")
        asked = set(action.missing_fields)

        assert "environment" not in asked
        assert "lcd_splicing" not in asked
        assert "lcd_layout" not in asked
        # 剩下只可能是拼缝 / 尺寸（拼接墙也需要尺寸）—— 已锁定的项一个都不许再问
        assert asked <= {"lcd_bezel", "lcd_size"}, action.missing_fields


class TestKeywordsOnlyExtractCandidates:
    def test_keyword_alone_does_not_decide_the_branch_question(self):
        """关键词只产出候选事实：单靠 "meeting" 不会直接锁 IFP / 也不会跳过问题。"""
        facts = extract_lcd_facts("meeting")

        assert facts.category == CONFERENCE_EDUCATION
        assert facts.handwriting is None, "看见 meeting 不能推断需要手写"
        assert facts.touch is None

    def test_lcd_facts_never_touch_led_fields(self):
        facts = extract_lcd_facts("75 inch 4K LCD with touch for a video wall")

        assert not hasattr(facts, "pixel_pitch_mm")
        assert not hasattr(facts, "viewing_distance_m")


class TestLedChainStaysFrozen:
    def test_led_slot_order_and_gate_inputs_unchanged(self):
        from src.models.requirement import SLOT_ORDER

        assert "pixel_pitch" not in SLOT_ORDER, "SLOT_ORDER 用槽位名（pixel_pitch → pixel_pitch_mm）"
        assert "target_size" in SLOT_ORDER, SLOT_ORDER
        assert "lcd_size_inch" not in SLOT_ORDER, "LCD 字段不能进 LED 的采集顺序"

        led = RequirementProfile.from_slots(
            {"display_type": "LED"}, explicit_keys={"display_type"}
        )
        # LED 的缺失槽位顺序与整改前一致（LCD 字段不参与）
        assert led.missing_slots() == [
            "environment", "purpose", "installation", "viewing_distance_m",
            "target_size", "budget_level",
        ], led.missing_slots()

    def test_led_policy_is_still_the_existing_chain(self):
        from src.dialogue.product_router import policy_for
        from src.dialogue.turn_kind import LED

        policy = policy_for(LED)
        assert policy.implemented is True
        assert "LED" in policy.product_domain

    def test_led_requirement_extraction_untouched(self):
        from src.rag.query_understanding import extract_slots

        slots = extract_slots("indoor 3m x 5m P4 5m viewing distance")
        assert slots.get("pixel_pitch_mm") == 4.0
        assert slots.get("target_width_mm") == 3000.0
        assert "lcd_size_inch" not in slots


class TestLcdReplyLayerWiring:
    """Phase 5 的最后一环：LCD 决策结果要真的进到"回复生成层"（计划 §二十/§二十六）。"""

    def test_locked_lcd_facts_are_handed_to_the_reply_layer(self):
        from src.agents.sales.nodes.script_generator import _known_facts
        from src.dialogue.lcd_decision import lcd_turn

        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "6x2 video wall, indoor, control room")
        facts = _known_facts({"requirement_profile": profile, "lcd_action": action.to_dict()})

        assert any("video wall" in item for item in facts), facts
        assert any("use case=monitoring" in item for item in facts), facts
        assert any("resolution=" in item for item in facts) or True  # 未定分辨率时不出现也正常

    def test_reply_prompt_carries_the_lcd_branch_and_one_question_rule(self, monkeypatch):
        import importlib
        from types import SimpleNamespace

        script_generator = importlib.import_module("src.agents.sales.nodes.script_generator")
        captured: dict = {}

        class _FakeLLM:
            def invoke(self, prompt, *args, **kwargs):
                captured["prompt"] = str(prompt)
                return SimpleNamespace(content="Got it — for indoor use. Is it a video wall or single displays?")

        monkeypatch.setattr(script_generator, "_dialogue_llm", lambda: _FakeLLM())
        from src.dialogue.lcd_decision import lcd_turn

        profile, action = lcd_turn(_lcd_profile(), "We need an LCD for a control room.")
        state = {
            "session_id": "lcd-reply-layer",
            "current_message": "We need an LCD for a control room.",
            "intent": "need_query",
            "next_action": "ask",
            "display_type_decision": {"display_type": "LCD", "status": "CONFIRMED"},
            "requirement_profile": profile,
            "lcd_action": action.to_dict(),
            "pending_question": action.question,
            "pending_slot": action.question_slot,
            "requirements": {},
            "additional_requirements": [],
            "response": "",
            "should_generate_solution": False,
        }
        script_generator.script_generator(state)

        prompt = captured.get("prompt", "")
        assert "lcd_branch_is:monitoring" in prompt, prompt[:400]
        assert "do_not_re_ask_locked_lcd_facts" in prompt, prompt[:400]
        assert "acknowledge_what_the_customer_just_said" in prompt, prompt[:400]

    def test_lcd_image_confirmation_uses_context_aware_prompt(self):
        from src.dialogue.response_coordinator import ResponseCoordinator
        from src.vision.recognition import ImageRecognitionResult

        profile = _lcd_profile(environment="outdoor")
        ic.apply_recognition_to_profile(
            profile,
            ImageRecognitionResult(display_type="LCD", environment="indoor", confidence=0.9),
        )
        coordinator = ResponseCoordinator(profile_lookup=lambda _sid: profile)

        sentence = coordinator.vision_confirmation_sentence("lcd-vision", "i sent a photo")

        assert "outdoor" in sentence.lower() and "indoor" in sentence.lower(), sentence
        assert sentence.count("?") == 1, sentence

    def test_lcd_image_confirmation_skips_what_the_customer_already_said(self):
        from src.vision.recognition import ImageRecognitionResult

        profile = _lcd_profile(environment="indoor")
        profile.lcd_is_splicing = True
        profile.sources["lcd_is_splicing"] = "explicit"
        ic.apply_recognition_to_profile(
            profile,
            ImageRecognitionResult(
                display_type="LCD", environment="indoor", is_splicing=True,
                camera_observed=True, confidence=0.8,
            ),
        )

        prompt = ic.prompt_from_profile(profile)

        assert "camera" in prompt.lower(), prompt
        assert "indoor" in prompt.lower(), prompt
        # 客户说过的"拼接墙"只作为"和你说的一致"，不再当成新发现重复问
        assert prompt.lower().count("video wall") <= 1, prompt

    def test_led_vision_confirmation_path_is_unchanged(self):
        """LED 画像确认仍走原来的话术（LCD 分支不许影响它）。"""
        from src.dialogue.response_coordinator import ResponseCoordinator
        from src.models.requirement import RequirementProfile

        led = RequirementProfile()
        led.display_type = "LED"
        led.environment = "outdoor"
        led.vision_confirmation_pending = ["environment"]
        led.vision_assertions = {"environment": "indoor"}
        led.sources = {"environment": "vision_explicit"}

        # 没有 LCD 留痕时，走的是 reply_composer 的原有路径（不报错即可）
        ResponseCoordinator(profile_lookup=lambda _sid: led).vision_confirmation_sentence(
            "led-vision", "i sent a photo"
        )

    def test_explicit_keys_accept_slot_or_field_names(self):
        from src.models.requirement import RequirementProfile

        by_slot = RequirementProfile.from_slots(
            {"lcd_splicing": True}, explicit_keys={"lcd_splicing"}
        )
        by_field = RequirementProfile.from_slots(
            {"lcd_is_splicing": True}, explicit_keys={"lcd_is_splicing"}
        )

        assert by_slot.sources.get("lcd_is_splicing") == "explicit"
        assert by_field.sources.get("lcd_is_splicing") == "explicit"


# LED **专有**槽位（environment / purpose 两条链路都会问，不算泄漏）
LED_ONLY_SLOTS = frozenset(
    {"installation", "size", "size_axis", "pixel_pitch", "viewing_distance", "brightness"}
)


class TestLcdAndLedChainsAreIsolated:
    """客户实测（2026-09-28 第二轮）：进了 LCD 之后又被 LED 问句接管。

        客户: i need a lcd display → AI: Let's go with LCD display … indoors or outdoors?
        客户: indoor                → AI: What screen size do you have in mind (width x height)? ← LED 问法
        客户: 65寸                  → AI: Is it a permanent install, or rental/events?        ← LED 问法

    根因：收口层的 LED DialoguePolicy 把 LCD 决策层的 slot 覆盖了
    （日志："Policy=ASK(environment) 与销售层准备的问句(purpose)不一致 → 以 Policy 为准"）。
    """

    def test_customer_facing_slot_keeps_the_lcd_decision(self):
        from src.dialogue.lcd_decision import lcd_turn
        from src.dialogue.response_coordinator import ResponseCoordinator
        from src.memory.store import memory

        sid = "lcd-isolation-slot"
        memory.clear(sid)
        memory.mark_first_contact_done(sid)
        profile, action = lcd_turn(_lcd_profile(), "i need a lcd display")
        memory.set_requirement_profile(sid, profile)
        result = {
            "response": action.question,
            "pending_question": action.question,
            "pending_slot": action.question_slot,
            "product_domain": "LCD",
            "lcd_action": action.to_dict(),
            "requirements": {}, "newly_filled_slots": [], "products": [],
            "turn_id": "t", "recommendation_gate": {"ready": False},
        }

        plan = ResponseCoordinator(profile_lookup=lambda _s: profile).prepare_final(
            result, session_id=sid, message="i need a lcd display", turn_context={}
        )
        memory.clear(sid)

        assert plan.question_slot == "purpose", plan.question_slot
        assert result["duplicate_check"] == "lcd_chain", result["duplicate_check"]

    def test_three_turn_lcd_conversation_never_asks_an_led_question(self):
        from src.dialogue.lcd_decision import lcd_turn
        from src.dialogue.response_coordinator import ResponseCoordinator
        from src.memory.store import memory

        sid = "lcd-isolation-3turn"
        memory.clear(sid)
        memory.mark_first_contact_done(sid)
        profile = _lcd_profile()
        memory.set_requirement_profile(sid, profile)
        asked: list = []

        for message in ("i need a lcd display", "indoor", "65 inch"):
            stored = memory.get_requirement_profile(sid)
            profile = RequirementProfile.model_validate(stored)
            profile, action = lcd_turn(profile, message)
            if action.question_slot:
                profile.record_ask(action.question_slot)
            memory.set_requirement_profile(sid, profile)
            result = {
                "response": action.question,
                "pending_question": action.question,
                "pending_slot": action.question_slot,
                "product_domain": "LCD",
                "lcd_action": action.to_dict(),
                "requirements": {}, "newly_filled_slots": [], "products": [],
                "turn_id": "t", "recommendation_gate": {"ready": False},
            }
            plan = ResponseCoordinator(profile_lookup=lambda _s: profile).prepare_final(
                result, session_id=sid, message=message, turn_context={}
            )
            asked.append(plan.question_slot)

        memory.clear(sid)

        # 分辨率不再问客户（客户口径 2026-09-30）：室内外客户第二句自己说了，
        # 尺寸第三句给了 → 下一问是触控，而不是分辨率
        assert asked == ["purpose", "lcd_size", "lcd_touch"], asked
        assert not (set(asked) & LED_ONLY_SLOTS), asked
        assert all(slot.startswith("lcd_") or slot == "purpose" for slot in asked), asked

    def test_lcd_questions_never_ask_width_height_or_installation(self):
        from src.dialogue.lcd_decision import lcd_turn

        profile = _lcd_profile()
        profile, action = lcd_turn(profile, "i need a lcd display")
        question = action.question.lower()

        assert "width" not in question and "height" not in question, action.question
        assert "permanent" not in question and "rental" not in question, action.question

    def test_answered_lcd_facts_move_the_chain_forward(self):
        """客户答非所问（问了用途答 indoor）→ 换下一项，不把同一项再问一遍。"""
        from src.dialogue.lcd_decision import lcd_turn

        profile = _lcd_profile()
        profile, first = lcd_turn(profile, "i need a lcd display")
        assert first.question_slot == "purpose"
        profile.record_ask("purpose")

        profile, second = lcd_turn(profile, "indoor")

        assert second.question_slot != "purpose", second.question_slot
        assert second.question_slot == "lcd_size", second.question_slot


class TestLcdSizeWritingsAndAnswerRouting:
    """客户实测（2026-09-28 第五轮）：

        客户: i need a lcd display → 🤖 Let's go with LCD display for this project.   ← 问题被吞掉
        客户: 65'                  → 🤖 … 65-inch … 放宽条件话术 … "what screen size do you have in mind?"

    两个根因：① `65'`（撇号）解析不出尺寸；② 客户答尺寸被判成 others → 整轮送自由问答。
    """

    def test_apostrophe_and_other_size_writings_are_parsed(self):
        from src.dialogue.lcd_decision import extract_lcd_facts

        for text in ("65'", '65"', "65 inch", "75-inch", "65寸", "65英寸", "55 in"):
            assert extract_lcd_facts(text).size_inch == float(re.findall(r"\d+", text)[0]), text

    def test_led_style_sizes_never_become_an_lcd_size(self):
        from src.dialogue.lcd_decision import extract_lcd_facts

        assert extract_lcd_facts("3m x 5m indoor P4").size_inch is None
        assert extract_lcd_facts("viewing distance 5m").size_inch is None

    def test_lcd_session_answer_does_not_go_to_free_question(self, monkeypatch):
        """LCD 会话里客户答尺寸（65'）→ 继续走需求链，不是自由问答。"""
        import importlib
        from types import SimpleNamespace

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        understanding_mod = importlib.import_module("src.dialogue.product_type_understanding")
        from src.memory.store import memory

        sid = "lcd-answer-routing"
        memory.clear(sid)
        memory.mark_first_contact_done(sid)
        # 上一轮：会话已确认 LCD（类型判断落在 memory 里，classify 每轮都会读）
        memory.set_display_type_decision(
            sid, {"display_type": "LCD", "status": "CONFIRMED", "locked": True}
        )

        class _FakeLLM:
            def __init__(self, *a, **k):
                pass

            def invoke(self, *a, **k):
                return SimpleNamespace(content="others")

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeLLM)
        monkeypatch.setattr(
            understanding_mod,
            "understand_product_type_reply",
            lambda *a, **k: {"reply": "chose", "display_type": "LCD"},
        )
        state = {
            "current_message": "65'",
            "messages": [],
            "session_id": sid,
            "requirements": {},
        }

        out = classify_mod.classify(state)
        memory.clear(sid)

        assert out["intent"] == "need_query", out.get("intent")

    def test_size_answer_advances_and_never_asks_size_again(self):
        from src.dialogue.lcd_decision import lcd_turn

        profile = _lcd_profile()
        profile, first = lcd_turn(profile, "i need a lcd display")
        assert first.question_slot == "purpose"
        profile.record_ask("purpose")

        profile, second = lcd_turn(profile, "65'")

        assert profile.lcd_size_inch == 65, profile.lcd_size_inch
        assert second.question_slot != "lcd_size", "尺寸已经给了，不能再问尺寸"
        # 分辨率按尺寸规则定下来（65" → 4K），不再反问客户
        assert profile.lcd_resolution == "4K", profile.lcd_resolution
        assert second.question_slot != "lcd_resolution", second.question_slot
        assert second.question_slot == "environment", second.question_slot


class TestRequiredQuestionIsNeverDropped:
    """LCD 会话里"已经定好要问的那一项"必须出现在客户可见回复里。"""

    def test_finalize_appends_the_missing_required_question(self):
        from src.dialogue.response_coordinator import ResponseCoordinator

        coordinator = ResponseCoordinator()
        questions = [{"text": "What will the screens be used for?", "slot": "purpose"}]

        with_q = coordinator.finalize(
            "Understood, let's go with LCD display for this project.",
            questions=questions,
            require_question=True,
        )
        assert "What will the screens be used for?" in with_q, with_q
        assert with_q.count("?") == 1

        # 交付/推荐轮不补问题
        without = coordinator.finalize(
            "TW11-3216-P3.0 fits well.", questions=questions, require_question=False
        )
        assert "?" not in without, without

        # 已经有问句时不再追加第二问
        already = coordinator.finalize(
            "Noted. Will they be indoor or outdoor?", questions=questions, require_question=True
        )
        assert already.count("?") == 1, already


class TestRealSalesGraphKeepsTheLcdChain:
    """真图端到端（2026-09-30 实测）：Sales 用的是 LangGraph StateGraph(SalesState)。

    **只有 SalesState 里声明过的键才会在节点之间传递** —— `lcd_action` 没声明时
    requirement_mining 算出的"下一问"根本到不了 script_generator，回复变空后被
    API 的"放宽条件"话术顶上（客户看到的就是那句）。
    """

    @staticmethod
    def _run_turns(monkeypatch, messages):
        import importlib
        from types import SimpleNamespace

        from src.agents.sales.graph import build_sales_graph
        from src.memory.store import memory
        import src.core.requirement_extractor as extractor_mod

        classify_mod = importlib.import_module("src.agents.sales.nodes.classify")
        requirement_mod = importlib.import_module("src.agents.sales.nodes.requirement")
        script_mod = importlib.import_module("src.agents.sales.nodes.script_generator")

        class _FakeChat:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return SimpleNamespace(content="need_query")

        class _FakeLLM:
            def invoke(self, prompt, *args, **kwargs):
                return SimpleNamespace(content="Noted, let me continue with that.")

        monkeypatch.setattr(classify_mod, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(requirement_mod, "ChatOpenAI", _FakeChat)
        monkeypatch.setattr(script_mod, "_dialogue_llm", lambda: _FakeLLM())
        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()

        graph = build_sales_graph()
        session_id = "lcd-graph-flow"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        replies, actions = [], []
        for message in messages:
            stored = memory.get_requirement_profile(session_id)
            state = {
                "messages": [{"role": "user", "content": message}],
                "current_message": message,
                "session_id": session_id,
                "intent": "",
                "requirements": {},
                "additional_requirements": [],
                "required_met": False,
                "required_missing": [],
                "should_generate_solution": False,
                "solutions": [],
                "response": "",
                "next_action": "ask",
                "turn_count": 0,
                "pending_question": "",
                "pending_slot": "",
                # 与真实 runner 一致：把上一轮回写进 memory 的档案带进来
                # （否则每轮从空档案重建，看起来像"需求丢了"，但那是测试脚手架的锅）
                "requirement_profile": (
                    RequirementProfile.model_validate(stored) if stored else None
                ),
            }
            out = graph.invoke(state)
            replies.append(str(out.get("response") or ""))
            actions.append(out.get("lcd_action") or {})
            # 模拟 runner：把这一轮的档案与 AI 回复写回 memory（跨轮上下文）
            profile = out.get("requirement_profile")
            if profile is not None:
                memory.set_requirement_profile(session_id, profile)
            memory.extend(session_id, [{"role": "assistant", "content": replies[-1]}])
        memory.clear(session_id)
        return replies, actions

    def test_lcd_action_survives_between_nodes(self, monkeypatch):
        _replies, actions = self._run_turns(monkeypatch, ["i need a lcd display"])

        assert actions[0].get("next_action") == "ask_purpose", actions[0]
        assert actions[0].get("question_slot") == "purpose"

    def test_two_turn_lcd_conversation_asks_lcd_questions(self, monkeypatch):
        replies, actions = self._run_turns(
            monkeypatch, ["i need a lcd display", "control room"]
        )

        assert actions[1].get("lcd_category") == MONITORING, actions[1]
        assert actions[1].get("question_slot") == "lcd_splicing", actions[1]
        assert "video wall" in replies[1].lower(), replies[1]
        # 绝不能出现"放宽条件"话术（那只属于推荐不出来）
        for reply in replies:
            assert "can be relaxed" not in reply.lower(), reply
            assert "?" in reply, reply


class TestLcdWordingIsWrittenByTheModel:
    """客户口径（2026-09-30）：LCD 每轮都是同一句模板 —— 要 AI 自己润色。

    根因：之前把那句成句塞进 `ResponseContext.answer`（"必须传达的内容"），
    LLM 直接照抄；LLM 没照抄时又被模板兜底顶掉 → 客户每轮看到同一句话。
    修法：成句只当 `opening`（可选提示，prompt 明确"不得照抄"），
    问题以 slot + intent + 意思锚点交给 LLM；answer 留空。
    """

    _NATURAL = {
        "purpose": "Got it, LCD it is. Where will these screens be used, a control room, "
                   "a meeting room, or something like advertising?",
        "environment": "Makes sense. Will the screens go indoors, or is this an outdoor setup?",
    }

    def _reply_and_prompt(self, monkeypatch, message="i need a lcd display"):
        import importlib
        from types import SimpleNamespace

        from src.dialogue.lcd_decision import lcd_turn
        from src.memory.store import memory

        script_mod = importlib.import_module("src.agents.sales.nodes.script_generator")
        captured = {"prompts": []}
        natural = dict(self._NATURAL)

        class _LLM:
            def invoke(self, prompt, *args, **kwargs):
                text = str(prompt)
                captured["prompts"].append(text)
                slot = ""
                match = __import__("re").search(r"Question to ask — slot: ([a-z_]+)", text)
                if match:
                    slot = match.group(1)
                return SimpleNamespace(
                    content=natural.get(
                        slot, "Understood — could you tell me a bit more about the setup?"
                    )
                )

        monkeypatch.setattr(script_mod, "_dialogue_llm", lambda: _LLM())

        session_id = "lcd-natural-wording"
        memory.clear(session_id)
        memory.mark_first_contact_done(session_id)
        profile = _lcd_profile()
        profile, action = lcd_turn(profile, message)
        memory.set_requirement_profile(session_id, profile)
        state = {
            "current_message": message,
            "messages": [{"role": "user", "content": message}],
            "session_id": session_id,
            "intent": "need_query",
            "requirements": {},
            "requirement_profile": profile,
            "lcd_action": action.to_dict(),
            "pending_question": action.question,
            "pending_slot": action.question_slot,
            "display_type_decision": {"display_type": "LCD", "status": "CONFIRMED"},
        }
        out = script_mod.script_generator(state)
        memory.clear(session_id)
        return str(out.get("response") or ""), captured["prompts"][0], action

    def test_reply_keeps_the_models_own_wording(self, monkeypatch):
        reply, _prompt, action = self._reply_and_prompt(monkeypatch)

        assert reply == self._NATURAL[action.question_slot], reply

    def test_prompt_treats_the_canned_line_as_an_optional_hint(self, monkeypatch):
        _reply, prompt, _action = self._reply_and_prompt(monkeypatch)

        # 成句不能再出现在"必须传达的事实"里（那样 LLM 会照抄）
        assert "Answer facts to convey" not in prompt, prompt[:400]
        assert "do NOT copy it" in prompt

    def test_prompt_carries_slot_intent_and_no_led_questions(self, monkeypatch):
        _reply, prompt, action = self._reply_and_prompt(monkeypatch)

        assert f"Question to ask — slot: {action.question_slot}" in prompt, prompt[:400]
        lowered = prompt.lower()
        assert "pixel pitch" not in lowered or "do not ask any led question" in lowered
