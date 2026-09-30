"""客户答"上一轮问的那一项"时必须解析出来（客户口径 2026-09-30）。

实测视频墙对话（客户口径原文见下），需求链卡死、永远不推荐：

    我们问 "How many columns and rows… something like 6x2?" → 客户答 "3x3"
    → 旧实现只在"这一句里带 wall / spliced 字样"时才把 3x3 当排布，
      排布永远没记下来 → 需求链在"问排布"上无限循环。

    我们问 "How narrow does the bezel need to be?" → 客户答 "0.88mm"
    → 旧实现要求句子里带 "bezel" 字样，于是 0.88 被丢掉，还按业务默认写成了 3.5。

根因有两个，这里分别守住：

    A. 解析不认"上一轮问的是哪一项"（本文件 TestSlotAnswerParsing 系列）
    B. 校验层不认识 LCD 槽位，"模型问错槽位"从来没被拦住，客户答的落到了别处
       （本文件 TestLcdSlotQuestionValidation）
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.lcd_decision import lcd_turn  # noqa: E402
from src.dialogue.response_validator import validate_response  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

LAYOUT_Q = "How many columns and rows are you thinking for the arrangement, something like 6x2?"
BEZEL_Q = "How narrow does the bezel need to be (for example 3.5mm)?"
SIZE_Q = "What screen size do you have in mind (in inches)?"


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


def _answer(profile, message, *, asked_slot, last_question=""):
    """模拟客户回答上一轮的问题（真实链路里 pending_slot 是 Requirement 节点写的）。"""
    profile.last_asked_slot = asked_slot
    profile.record_ask(asked_slot)
    return lcd_turn(profile, message, last_question=last_question)


@pytest.mark.parametrize(
    "answer, expected_layout, expected_count",
    [
        ("3x3", "3x3", 9),
        ("3*3", "3x3", 9),
        ("3 x 3", "3x3", 9),
        ("6x2", "6x2", 12),
        ("3 rows and 3 columns", "3x3", 9),
        ("3row and 3columns", "3x3", 9),
        ("3 columns 4 rows", "3x4", 12),
        ("3\u884c3\u5217", "3x3", 9),
    ],
)
class TestLayoutIsParsedFromTheAnsweredQuestion:
    def test_layout_variants(self, answer, expected_layout, expected_count):
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=True)
        profile, action = _answer(profile, answer, asked_slot="lcd_layout", last_question=LAYOUT_Q)

        assert profile.lcd_splicing_layout == expected_layout, profile.lcd_splicing_layout
        assert profile.lcd_screen_count == expected_count, profile.lcd_screen_count
        assert action.question_slot != "lcd_layout", "排布已经答了，不能再问排布"


class TestBareValuesAreParsedForTheAskedSlot:
    def test_bezel_without_the_word_bezel_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_bezel_without_the_word_bezel ──
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=True)
        profile, _action = _answer(profile, "0.88mm", asked_slot="lcd_bezel", last_question=BEZEL_Q)

        assert profile.lcd_bezel_mm == pytest.approx(0.88), (
            "客户答的 0.88mm 不能被业务默认值 3.5 顶掉"
        )

        # ── test_bezel_bare_number ──
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=True)
        profile, _action = _answer(profile, "0.88", asked_slot="lcd_bezel", last_question=BEZEL_Q)

        assert profile.lcd_bezel_mm == pytest.approx(0.88)

        # ── test_size_bare_number ──
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=True)
        profile, _action = _answer(profile, "65", asked_slot="lcd_size", last_question=SIZE_Q)

        assert profile.lcd_size_inch == pytest.approx(65.0)

        # ── test_meters_are_not_mistaken_for_a_layout ──
        """拼接墙口径下客户说 "3 x 5 m"（场地尺寸）时，不能当成 3x5 排布。"""
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=True)
        profile, _action = _answer(
            profile, "3 x 5 m", asked_slot="environment", last_question="Indoor or outdoor?"
        )

        assert profile.lcd_splicing_layout is None, profile.lcd_splicing_layout


class TestChainNeverLoopsForever:
    def _video_wall_profile(self):
        return _profile(
            lcd_category="monitoring",
            lcd_is_splicing=True,
            lcd_splicing_layout="3x3",
            lcd_screen_count=9,
            lcd_bezel_mm=0.88,
            lcd_size_inch=65.0,
        )

    def test_explicit_recommendation_closes_an_unanswered_environment_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_explicit_recommendation_closes_an_unanswered_environment ──
        """室内外问了两轮客户没答 → 按 LCD 业务默认（室内）收口，直接推荐。"""
        profile = self._video_wall_profile()
        profile.last_asked_slot = "environment"
        profile.record_ask("environment")          # 第一次问
        profile.record_ask("environment")          # 第二次问

        _profile_out, action = lcd_turn(profile, "\u7ed9\u6211\u63a8\u8350\u4ea7\u54c1")

        assert profile.environment == "indoor", profile.environment
        assert (profile.sources or {}).get("environment") == "recommended"
        assert action.confirmed is True, action.missing_fields
        assert action.missing_fields == []

        # ── test_layout_is_not_asked_for_a_single_display_wall ──
        profile = _profile(lcd_category="monitoring", lcd_is_splicing=False, lcd_size_inch=65.0)
        _profile_out, action = lcd_turn(profile, "single displays indoor")

        assert action.question_slot != "lcd_layout", action.question_slot

        # ── test_layout_stops_blocking_after_the_ask_limit ──
        """排布问了两轮还是没答案 → 不阻塞推荐（也不编造值）。"""
        profile = self._video_wall_profile()
        profile.lcd_splicing_layout = None
        profile.lcd_screen_count = None
        profile.last_asked_slot = "lcd_layout"
        profile.record_ask("lcd_layout")
        profile.record_ask("lcd_layout")

        _profile_out, action = lcd_turn(profile, "I don't know yet", last_question=LAYOUT_Q)

        assert "lcd_layout" not in action.missing_fields, action.missing_fields
        assert profile.lcd_splicing_layout is None, "不编造排布，只是不再阻塞"

        # ── test_full_video_wall_conversation_reaches_the_recommendation ──
        """把客户口径里的整段对话重放一遍 → 必须走到 confirmed（触发推荐）。"""
        profile = _profile()
        script = [
            ("i need a lcd display", ""),
            ("control room", "What will the screens be used for?"),
            ("video wall", "Do you need a video wall (spliced screens) or single displays?"),
            ("3x3", LAYOUT_Q),
            ("0.88mm", BEZEL_Q),
            ("65'", SIZE_Q),
            ("\u7ed9\u6211\u63a8\u8350\u4ea7\u54c1", "Will they be used indoors or outdoors?"),
        ]
        action = None
        for message, last_question in script:
            profile, action = lcd_turn(profile, message, last_question=last_question)
            profile.last_asked_slot = action.question_slot
            if action.question_slot:
                profile.record_ask(action.question_slot)

        assert profile.lcd_splicing_layout == "3x3", profile.lcd_splicing_layout
        assert profile.lcd_screen_count == 9
        assert profile.lcd_bezel_mm == pytest.approx(0.88)
        assert profile.lcd_size_inch == pytest.approx(65.0)
        assert action.confirmed is True, action.missing_fields


class TestLcdSlotQuestionValidation:
    def test_asking_a_different_lcd_slot_is_flagged_merged(self):
        """合并自 3 条同类测试（瘦身；断言全部保留）。"""

        # ── test_asking_a_different_lcd_slot_is_flagged ──
        """Python 定的是"问拼缝"，模型却问成"尺寸" → 校验必须拦下来。"""
        result = validate_response(
            "A 65-inch panel keeps things sharp. What screen size are you thinking, "
            "in inches?",
            question_slot="lcd_bezel",
            customer_message="0.88mm",
        )
        assert "question_intent_mismatch" in result.issues, result.issues

        # ── test_asking_the_right_slot_passes ──
        result = validate_response(
            "Got it. How narrow does the bezel need to be, for example 3.5mm?",
            question_slot="lcd_bezel",
            customer_message="video wall",
        )
        assert result.issues == [], result.issues

        # ── test_wrong_lcd_slot_is_detected ──
        """问的是排布，答案里却在问拼接/单体 → 判为问错槽位。"""
        result = validate_response(
            "Do you need a video wall (spliced screens) or single displays?",
            question_slot="lcd_layout",
            customer_message="control room",
        )
        assert "wrong_question_slot" in result.issues, result.issues

    @pytest.mark.parametrize(
        "slot",
        [
            "purpose",
            "environment",
            "lcd_splicing",
            "lcd_layout",
            "lcd_bezel",
            "lcd_size",
            "lcd_resolution",
            "lcd_touch",
            "lcd_handwriting",
            "lcd_tender",
            "lcd_ops",
            "lcd_camera",
        ],
    )
    def test_every_deterministic_question_passes_its_own_validation(self, slot):
        """系统给的成句必须能通过"问的就是这一项"的校验。

        实测（2026-09-30）：lcd_layout 的成句是 "How should the video wall be
        arranged (for example 6x2)?"，关键词表里没有 arrang* → 正确的问句被自己
        的校验判成"问的不是这件事"，LCD 每一轮都退化成模板话术。
        """
        from src.dialogue.lcd_decision import _question_for

        question = _question_for(slot)
        assert question, slot
        result = validate_response(question, question_slot=slot, customer_message="ok")

        assert "question_intent_mismatch" not in result.issues, (slot, question, result.issues)
        assert "wrong_question_slot" not in result.issues, (slot, question, result.issues)
