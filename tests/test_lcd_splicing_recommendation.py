"""客户实测（2026-09-30）：拼接屏链路信息拿全了却不推荐，还在来回问。

客户可见原文（对话）：

    🤖 A 3x3 grid gives you a nice balanced wall… How tight does the bezel need to be,
       would something around 3.5mm work for you?
    👤 yes
    🤖 … The last piece I need is the panel size… What screen size are you thinking, in inches?
    👤 65'
    🤖 … I just need to confirm how you want them arranged, is it 3 columns by 3 rows?
    👤 yes，给我推荐
    🤖 … how narrow do you want the seam between panels to be, would something in the 3.5mm range suit…?

三个根因：
  1. 客户答 "yes" 时**没有把上一轮提出的值记下来**（3.5mm 拼缝）→ 死循环；
  2. 客户明确"给我推荐"时，可选字段（拼缝等）**没有按业务默认收口** → 继续追问；
  3. Solution 侧只有 LED 的选型路径：LCD 需求齐全时仍然被 LED 的 Gate 反问
     "Is it a permanent install, or is it for rental/events?"，永远不出型号。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.lcd_decision import (  # noqa: E402
    apply_answer_to_previous,
    decide_lcd_next_action,
    lcd_turn,
)
from src.models.requirement import RequirementProfile  # noqa: E402


def _lcd(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestYesAnswersThePreviousLcdQuestion:
    def test_yes_takes_the_value_proposed_in_the_last_question_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_yes_takes_the_value_proposed_in_the_last_question ──
        profile = _lcd(
            lcd_category="monitoring", lcd_is_splicing=True, lcd_splicing_layout="3x3",
            lcd_screen_count=9, lcd_size_inch=65.0, environment="indoor",
        )
        action = decide_lcd_next_action(profile)
        assert action.question_slot == "lcd_bezel", action.question_slot
        profile.record_ask("lcd_bezel")
        profile.last_asked_slot = "lcd_bezel"

        profile, action = lcd_turn(
            profile,
            "yes",
            last_question="How tight does the bezel need to be, would something around 3.5mm work?",
        )

        assert profile.lcd_bezel_mm == 3.5, profile.lcd_bezel_mm
        assert profile.sources.get("lcd_bezel_mm") == "explicit"
        assert action.question_slot != "lcd_bezel", "拼缝已经确认，不能再问一遍"

        # ── test_yes_answers_boolean_questions ──
        profile = _lcd(environment="indoor", lcd_category="monitoring")
        profile.record_ask("lcd_splicing")
        profile.last_asked_slot = "lcd_splicing"

        profile, _action = lcd_turn(
            profile, "yes", last_question="Do you need a video wall (spliced screens) or single displays?"
        )

        assert profile.lcd_is_splicing is True

        # ── test_no_answers_boolean_questions_negatively ──
        profile = _lcd(environment="indoor", lcd_category="monitoring")
        profile.record_ask("lcd_splicing")
        profile.last_asked_slot = "lcd_splicing"

        profile, _action = lcd_turn(profile, "no", last_question="Do you need a video wall?")

        assert profile.lcd_is_splicing is False

        # ── test_resolution_is_never_asked_so_a_bare_yes_cannot_pick_it ──
        """分辨率不再反问客户（客户口径 2026-09-30）：客户只说 "yes" 不能变成"客户要 4K"。

        档案里的 4K 是**尺寸规则**推出来的，来源必须还是规则，不能升级成
        customer_explicit —— 否则选型会把它当成客户点名要的规格。
        """
        profile = _lcd(environment="indoor", lcd_size_inch=65.0)
        profile.record_ask("lcd_resolution")
        profile.last_asked_slot = "lcd_resolution"

        profile, _action = lcd_turn(
            profile, "yes", last_question="Do you need 4K, or is 2K enough?"
        )

        assert profile.lcd_resolution == "4K", profile.lcd_resolution
        assert profile.lcd_resolution_source != "customer_explicit", (
            profile.lcd_resolution_source
        )


class TestOptionalFieldsStopLooping:
    def test_bezel_is_asked_once_then_defaulted_merged(self):
        """合并自 2 条同类测试（瘦身；断言全部保留）。"""

        # ── test_bezel_is_asked_once_then_defaulted ──
        """拼缝问过一次没得到明确值 → 按计划 §十 默认 3.5mm，不再重复问。"""
        profile = _lcd(lcd_category="monitoring", lcd_is_splicing=True,
                       lcd_splicing_layout="3x3", lcd_screen_count=9,
                       lcd_size_inch=65.0, environment="indoor")
        profile.record_ask("lcd_bezel")

        profile, action = lcd_turn(profile, "hmm, not sure yet")

        assert action.question_slot != "lcd_bezel", action.question_slot
        assert profile.lcd_bezel_mm == 3.5, profile.lcd_bezel_mm
        assert profile.sources.get("lcd_bezel_mm") == "recommended", (
            "默认值不能伪装成客户要求（计划 §八）"
        )

        # ── test_explicit_recommend_request_closes_the_chain ──
        profile = _lcd(lcd_category="monitoring", lcd_is_splicing=True,
                       lcd_splicing_layout="3x3", lcd_screen_count=9,
                       lcd_size_inch=65.0, environment="indoor")

        profile, action = lcd_turn(profile, "yes, please recommend one")

        assert action.confirmed is True, action.to_dict()
        assert action.missing_fields == [], action.missing_fields
        assert action.question_slot == "", action.question_slot


class TestLcdRecommendationPath:
    """Solution 侧必须有 LCD 的选型路径（不能再拿 LED 的 Gate / 引擎去卡）。"""

    def test_solution_gate_is_lcd_aware_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_solution_gate_is_lcd_aware ──
        from src.agents.solution.nodes.requirement import recommendation_gate_node

        profile = _lcd(
            environment="indoor", lcd_category="monitoring", lcd_is_splicing=True,
            lcd_splicing_layout="3x3", lcd_screen_count=9, lcd_size_inch=65.0,
            lcd_bezel_mm=3.5,
        )
        out = recommendation_gate_node({"requirement_profile": profile, "messages": []})

        assert out["recommendation_gate"]["ready"] is True, out["recommendation_gate"]
        assert out["next_action"] == "retrieve", out.get("next_action")
        assert "permanent install" not in str(out.get("pending_question") or "").lower()

        # ── test_lcd_selection_prefers_a_video_wall_for_a_splicing_request ──
        from src.rag.lcd_recommendation import select_lcd_candidate

        profile = _lcd(
            environment="indoor", lcd_category="monitoring", lcd_is_splicing=True,
            lcd_splicing_layout="3x3", lcd_screen_count=9, lcd_size_inch=65.0,
            lcd_bezel_mm=3.5,
        )
        candidates = [
            {"metadata": {"model": "P65", "display_type": "LCD", "display_size_inch": '65"',
                          "resolution": "1920x1080", "splicing_supported": True,
                          "brightness_nit": 350}},
            {"metadata": {"model": "H6530LN-B", "display_type": "LCD", "display_size_inch": '65"',
                          "resolution": "1920x1080", "is_splicing": True, "bazel_mm": "3.5mm",
                          "brightness_nit": 500}},
        ]

        picked = select_lcd_candidate(profile, candidates)

        assert picked is not None
        assert picked[1].get("model") == "H6530LN-B", picked[1]

        # ── test_lcd_selection_respects_a_non_splicing_request ──
        from src.rag.lcd_recommendation import select_lcd_candidate

        profile = _lcd(
            environment="indoor", lcd_category="normal", lcd_is_splicing=False,
            lcd_size_inch=65.0,
        )
        candidates = [
            {"metadata": {"model": "H6530LN-B", "display_type": "LCD", "display_size_inch": '65"',
                          "is_splicing": True, "bazel_mm": "3.5mm", "brightness_nit": 500}},
            {"metadata": {"model": "P65", "display_type": "LCD", "display_size_inch": '65"',
                          "resolution": "1920x1080", "splicing_supported": True,
                          "brightness_nit": 350}},
        ]

        picked = select_lcd_candidate(profile, candidates)

        assert picked is not None
        assert picked[1].get("model") == "P65", picked[1]

        # ── test_lcd_recommendation_never_asks_led_questions ──
        from src.agents.solution.nodes.recommend import _recommend_lcd

        profile = _lcd(
            environment="indoor", lcd_category="monitoring", lcd_is_splicing=True,
            lcd_splicing_layout="3x3", lcd_screen_count=9, lcd_size_inch=65.0,
            lcd_bezel_mm=3.5,
        )
        docs = [{"metadata": {"model": "H6530LN-B", "display_type": "LCD",
                              "display_size_inch": '65"', "is_splicing": True,
                              "bazel_mm": "3.5mm", "brightness_nit": 500}}]

        out = _recommend_lcd({"messages": [], "session_id": "t"}, profile, docs)

        answer = str(out.get("recommendation") or "")
        assert "H6530LN-B" in answer, answer
        lowered = answer.lower()
        for forbidden in ("permanent install", "rental", "pixel pitch", "viewing distance"):
            assert forbidden not in lowered, (forbidden, answer)
