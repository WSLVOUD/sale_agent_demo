"""整改计划 §十九：必须覆盖的 LCD / IFP 测试矩阵（30 条逐条落地）。

计划原文列了 30 条必须覆盖的场景。这个文件把它们逐条写出来，作为
"旧链删除 / lcd_decision 瘦身"之后的行为护栏 —— 只允许行为不变，
不允许任何一条回退。

约定：
    · 这里只验证 LCD / IFP 的行为；
    · LED 的行为在 test_led_* 与全量回归里验证（LED 代码本次不修改）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.lcd_decision import (  # noqa: E402
    ADVERTISING,
    CONFERENCE_EDUCATION,
    MONITORING,
    NORMAL,
    UNKNOWN,
    effective_display_type,
    is_ifp_requirement,
    lcd_turn,
    resolution_for_size,
)
from src.models.requirement import RequirementProfile  # noqa: E402

CONF = {"category": CONFERENCE_EDUCATION, "confidence": 0.9, "facts": {"environment": "indoor"}}
MON = {"category": MONITORING, "confidence": 0.9, "facts": {"environment": "indoor"}}
ADV = {"category": ADVERTISING, "confidence": 0.9, "facts": {"environment": "indoor"}}
NORM = {"category": NORMAL, "confidence": 0.9}


def _p(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


# ── 1~4 Monitoring / splicing / layout / seam ───────────────────────────────

def test_01_monitoring_with_splicing():
    profile, action = lcd_turn(_p(), "a video wall for our control room", category_signal=MON)
    assert action.lcd_category == MONITORING
    assert profile.lcd_is_splicing is True
    assert action.question_slot == "lcd_layout"


def test_02_monitoring_with_layout():
    profile, action = lcd_turn(_p(), "6x2 video wall, indoor", category_signal=MON)
    assert profile.lcd_splicing_layout == "6x2"
    assert profile.lcd_screen_count == 12
    assert action.question_slot != "lcd_layout"


def test_03_monitoring_with_seam():
    profile, action = lcd_turn(
        _p(), "3.5mm bezel", last_question="How narrow does the bezel need to be?",
        category_signal=MON,
    )
    assert profile.lcd_bezel_mm == 3.5
    assert action.question_slot != "lcd_bezel"


def test_04_monitoring_without_seam_defaults_to_3_5mm():
    profile = _p(lcd_category=MONITORING, lcd_is_splicing=True,
                 lcd_splicing_layout="3x3", lcd_screen_count=9, lcd_size_inch=65.0)
    profile.record_ask("lcd_bezel")
    profile.last_asked_slot = "lcd_bezel"
    profile, action = lcd_turn(profile, "not sure", category_signal=MON)
    assert profile.lcd_bezel_mm == 3.5
    assert action.question_slot != "lcd_bezel", "问过一次没答 → 用默认值，不再重复问"


# ── 5~7 Advertising ─────────────────────────────────────────────────────────

def test_05_advertising_indoor():
    profile, action = lcd_turn(
        _p(), "advertising screen for our store, indoor", category_signal=ADV
    )
    assert action.lcd_category == ADVERTISING
    assert profile.environment == "indoor"
    assert action.question_slot == "lcd_size"


def test_06_advertising_outdoor():
    # 语境理解会把客户说的 outdoor 放进 facts（这里显式模拟这一步）
    profile, action = lcd_turn(
        _p(),
        "advertising screen, outdoor",
        category_signal={"category": ADVERTISING, "confidence": 0.9,
                         "facts": {"environment": "outdoor"}},
    )
    assert profile.environment == "outdoor"


def test_07_advertising_touch():
    profile, action = lcd_turn(
        _p(environment="indoor", lcd_size_inch=55.0), "yes it needs touch",
        last_question="Do you need touch functionality?", category_signal=ADV,
    )
    assert profile.lcd_touch_required is True
    assert action.question_slot != "lcd_touch"


def test_07b_advertising_must_ask_indoor_or_outdoor():
    """客户口径 2026-09-30：广告**必须问室内还是室外**，不许拿模型猜的顶替。

    实测：客户只说 "advertise"，语境理解顺手补了 environment=outdoor（客户没说），
    系统于是跳过室内外，最后推了一台户外广告机（DS-O-75）。
    """
    signal = {
        "category": ADVERTISING,
        "confidence": 0.9,
        "facts": {"environment": "outdoor"},
        "evidence": {"environment": "advertise"},  # 客户原话里只有这个词，推不出户外
    }
    profile, action = lcd_turn(_p(), "advertise", category_signal=signal)

    assert action.question_slot == "environment", action.question_slot
    assert (profile.sources or {}).get("environment") == "understanding", (
        "模型推的环境只能算软事实"
    )


def test_07c_advertising_honours_the_customer_stated_environment():
    """客户自己说了（inside/outside）→ 不再重复问，直接按它继续。"""
    signal = {
        "category": ADVERTISING,
        "confidence": 0.9,
        "facts": {"environment": "indoor"},
        "evidence": {"environment": "for inside use"},
    }
    profile, action = lcd_turn(
        _p(), "advertising screen for inside use", category_signal=signal
    )

    assert profile.environment == "indoor"
    assert (profile.sources or {}).get("environment") == "explicit"
    assert action.question_slot == "lcd_size", action.question_slot


def test_07d_answered_environment_is_never_downgraded_and_never_re_asked():
    """客户答过 indoor/outdoor 之后，模型再从上下文里"顺手给一次"不能把它洗成软事实。

    实测（2026-09-30，客户原文）：
        AI: 室内还是室外？→ 客户: outdoor → AI: 尺寸？→ 客户: 75'' →
        AI: **又问了一遍 "Will they be used indoors or outdoors?"**
    根因：下一轮语境理解又返回了一次 environment=outdoor（来源 understanding），
    把客户侧来源 explicit 覆盖掉了 → 广告分支的硬闸门以为"客户没答过"，于是又问。
    """
    def _sig(facts, evidence):
        return {"category": ADVERTISING, "confidence": 0.9,
                "facts": facts, "evidence": evidence}

    profile = _p()
    # 1) 客户说 advertising：模型顺手猜了 outdoor → 软事实，照问
    profile, action = lcd_turn(
        profile, "i need a lcd display for advertising",
        category_signal=_sig({"environment": "outdoor"}, {"environment": "advertising"}),
    )
    assert action.question_slot == "environment"
    assert (profile.sources or {}).get("environment") == "understanding"

    # 2) 客户回答 outdoor → 客户侧来源，下一问是尺寸
    profile, action = lcd_turn(
        profile, "outdoor",
        category_signal=_sig({"environment": "outdoor"}, {"environment": "outdoor"}),
    )
    assert (profile.sources or {}).get("environment") == "explicit"
    assert action.question_slot == "lcd_size"

    # 3) 客户答尺寸：模型又把 environment=outdoor 带回来（证据来自上一句）
    profile, action = lcd_turn(
        profile, "75''",
        category_signal=_sig(
            {"screen_size_inch": 75.0, "environment": "outdoor"},
            {"screen_size_inch": "75", "environment": "outdoor use"},
        ),
    )
    assert (profile.sources or {}).get("environment") == "explicit", "客户侧来源不许被降级"
    assert action.question_slot == "lcd_touch", action.question_slot

    # 4) 客户答触控 → 需求齐全
    profile, action = lcd_turn(
        profile, "yes i need touch",
        category_signal=_sig({"touch": True, "environment": "outdoor"},
                             {"touch": "touch", "environment": "outdoor use"}),
    )
    assert profile.lcd_touch_required is True
    assert action.confirmed is True, action.missing_fields


# ── 8~9 Normal LCD ──────────────────────────────────────────────────────────

def test_08_normal_lcd_with_size():
    profile, action = lcd_turn(_p(), "65 inch single display for our lobby",
                               category_signal=NORM)
    assert action.lcd_category == NORMAL
    assert profile.lcd_size_inch == 65.0


def test_09_normal_lcd_without_size():
    _profile, action = lcd_turn(_p(), "a single display for our lobby", category_signal=NORM)
    assert action.question_slot in ("environment", "lcd_size"), action.question_slot
    assert action.confirmed is False


# ── 10~13 Conference / Education → handwriting gate ─────────────────────────

def test_10_conference_without_handwriting_is_not_ifp():
    profile = _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=False)
    assert is_ifp_requirement(profile) is False
    assert effective_display_type(profile) == "LCD"


def test_11_conference_with_handwriting_is_ifp():
    profile = _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True)
    assert is_ifp_requirement(profile) is True
    assert effective_display_type(profile) == "IFP"


def test_12_education_with_handwriting_is_ifp():
    profile = _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True,
                 lcd_size_inch=86.0)
    assert effective_display_type(profile) == "IFP"


def test_13_education_without_handwriting_is_not_ifp():
    profile = _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=False,
                 lcd_touch_required=False)
    assert effective_display_type(profile) == "LCD"


# ── 14~16 IFP + tender / OPS / camera ──────────────────────────────────────

def test_14_ifp_tender_project():
    profile, action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True,
           lcd_size_inch=65.0),
        "yes, it is a tender project",
        last_question="Is this a tender project?",
        category_signal=CONF,
    )
    assert profile.lcd_tender_project is True
    assert action.next_action == "request_tender_documents"
    # 会议 + 手写 → IFP 分支（Tender / OPS / Camera 只有 IFP 分支才会问）
    assert is_ifp_requirement(profile) is True


def test_15_ifp_ops():
    profile, action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True,
           lcd_size_inch=65.0),
        "yes, include an OPS module",
        last_question="Do you need an OPS slot?",
        category_signal=CONF,
    )
    assert profile.lcd_ops_required is True


def test_16_ifp_camera():
    profile, _action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True,
           lcd_size_inch=65.0),
        "yes we need a camera for video calls",
        last_question="Do you need a camera?",
        category_signal=CONF,
    )
    assert profile.lcd_camera_required is True


# ── 17 camera observed ≠ camera required ───────────────────────────────────

def test_17_camera_observed_is_not_camera_required():
    from src.dialogue import image_confirmation as ic
    from src.vision.recognition import ImageRecognitionResult

    profile = _p()
    ic.apply_recognition_to_profile(
        profile,
        ImageRecognitionResult(display_type="LCD", camera_observed=True, confidence=0.9),
    )

    assert profile.lcd_camera_observed is True, "图片看到了 → observed"
    assert not profile.lcd_camera_required, "看见 ≠ 客户需要"


# ── 18~21 Resolution ───────────────────────────────────────────────────────

def test_18_explicit_resolution():
    profile, _action = lcd_turn(_p(), "we need a 55 inch 4K LCD")
    assert profile.lcd_resolution == "4K"
    assert profile.lcd_resolution_source == "customer_explicit"


def test_19_below_65_defaults_to_2k():
    assert resolution_for_size(55) == ("2K", "size_rule_lt_65")


def test_20_above_65_defaults_to_4k():
    assert resolution_for_size(86) == ("4K", "size_rule_gt_65")


def test_21_65_boundary_is_decided_deterministically():
    resolution, source = resolution_for_size(65)
    assert resolution == "4K"
    assert source in ("size_default_4k_65", "product_catalog")


# ── 22~24 多消息 / 答非当前问题 / 已锁字段 ─────────────────────────────────

def test_22_multiple_facts_in_one_message():
    profile, _action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION),
        "also we need 4K and 86 inch",
        last_question="Do you need handwriting?",
        category_signal=CONF,
    )
    assert profile.lcd_size_inch == 86.0
    assert profile.lcd_resolution == "4K"


def test_23_answering_a_different_question_is_still_recorded():
    profile, _action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION),
        "86 inch, 4K",
        last_question="Do you need handwriting?",
        category_signal=CONF,
    )
    assert profile.lcd_size_inch == 86.0
    assert profile.lcd_handwriting_required is None, "没答手写就不要替他填"


def test_24_locked_fields_are_not_asked_again():
    profile = _p(lcd_category=MONITORING, lcd_is_splicing=True,
                 lcd_splicing_layout="3x3", lcd_screen_count=9,
                 lcd_bezel_mm=0.88, lcd_size_inch=65.0, environment="indoor")
    profile.sources = {**profile.sources, "lcd_size_inch": "explicit",
                       "lcd_bezel_mm": "explicit", "environment": "explicit"}
    _profile, action = lcd_turn(profile, "go ahead", category_signal=MON)
    for slot in ("lcd_splicing", "lcd_layout", "lcd_bezel", "lcd_size", "environment"):
        assert slot not in (action.missing_fields or []), (slot, action.missing_fields)


# ── 25~28 图片链路 ─────────────────────────────────────────────────────────

def test_25_image_recognition_then_customer_confirmation():
    from src.dialogue import image_confirmation as ic
    from src.vision.recognition import ImageRecognitionResult

    profile = _p()
    ic.apply_recognition_to_profile(
        profile,
        ImageRecognitionResult(display_type="LCD", is_splicing=True, confidence=0.9),
    )
    pending = list(profile.vision_confirmation_pending or [])
    assert pending, "图片识别出的内容要请客户确认"

    prompt = ic.prompt_from_profile(profile, language="en")
    assert prompt and "LCD" in prompt, prompt


def test_26_image_recognition_then_customer_correction():
    from src.dialogue import image_confirmation as ic
    from src.vision.recognition import ImageRecognitionResult

    profile = _p(environment="outdoor")
    ic.apply_recognition_to_profile(
        profile,
        ImageRecognitionResult(display_type="LCD", environment="indoor", confidence=0.9),
    )
    compare = ic.compare_with_context(
        profile, ImageRecognitionResult(display_type="LCD", environment="indoor")
    )
    assert compare.conflicts, "图片说 indoor、客户说 outdoor → 必须记冲突"
    assert profile.environment == "outdoor", "客户的话优先"


def test_27_image_conflicts_with_history_are_kept_for_the_customer():
    from src.dialogue import image_confirmation as ic
    from src.vision.recognition import ImageRecognitionResult

    profile = _p(environment="indoor")
    profile.sources = {**profile.sources, "environment": "explicit"}
    ic.apply_recognition_to_profile(
        profile, ImageRecognitionResult(display_type="LCD", environment="outdoor")
    )
    assert profile.environment == "indoor", "客户明说的环境不被图片覆盖"


def test_28_lcd_image_environment_recognition():
    from src.vision.integration import apply_vision_to_profile
    from src.vision.schema import VISION_EXPLICIT, VisionField, VisionRequirement

    vision = VisionRequirement(
        display_type=VisionField(value="LCD", confidence=0.9, source=VISION_EXPLICIT),
        environment=VisionField(value="indoor", confidence=0.9, source=VISION_EXPLICIT),
    )
    profile, stats = apply_vision_to_profile(RequirementProfile(), vision)
    assert profile.display_type == "LCD"
    assert profile.environment == "indoor"
    assert "environment" in profile.vision_confirmation_pending


# ── 29~30 LCD → IFP / LED 图片仍走 LED ─────────────────────────────────────

def test_29_lcd_to_ifp_switch():
    profile, action = lcd_turn(
        _p(lcd_category=CONFERENCE_EDUCATION, lcd_size_inch=65.0),
        "yes, we need handwriting",
        last_question="Do you need handwriting or whiteboard capability?",
        category_signal=CONF,
    )
    assert profile.lcd_handwriting_required is True
    assert effective_display_type(profile) == "IFP"


def test_30_led_image_still_enters_the_led_chain():
    from src.dialogue.product_type_router import route_display_type
    from src.vision.pipeline import vision_display_type_payload
    from src.vision.schema import VISION_EXPLICIT, VisionField, VisionRequirement

    vision = VisionRequirement(
        display_type=VisionField(value="LED", confidence=0.95, source=VISION_EXPLICIT)
    )
    payload = vision_display_type_payload([vision])
    assert payload["display_type"] == "LED"

    decision = route_display_type("", vision_display_type=payload["display_type"])
    assert decision.display_type == "LED", decision.display_type
