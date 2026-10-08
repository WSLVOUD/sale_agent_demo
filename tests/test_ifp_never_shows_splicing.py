"""回归：IFP（会议平板）**绝不允许**出现拼接排布 / 箱体数。

客户口径（2026-10）：

    "ifp是不允许出现拼接排布/箱体数的，只有客户需要拼接屏才能出现"

实测 bug：客户先说了展览会的 **3x3 视频墙**，后来又要一块**可手写的会议室屏幕**。
系统正确地把品类切到了会议平板（选了 Omni T65-K4/K4C），但旧需求的拼接事实
（``lcd_is_splicing=True`` / ``lcd_splicing_layout=3x3`` / ``lcd_screen_count=9``）
留在了档案里，于是推荐话术变成：

    "Your 3x3 video wall layout will use 9 of these panels"   ← 9 台会议平板拼 3x3

会议平板根本无法拼接，这是硬伤。本文件锁三层防护：

  1. ``layout_text``：只有"客户要拼接屏"（``lcd_is_splicing``）才产出排布文本；
  2. ``_enforce_ifp_cannot_splice``：判定为 IFP 时清掉残留的拼接事实；
  3. ``_locked_facts``：IFP 时拼接字段一律不交给表达层。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _stale_splicing_profile():
    """先做展览会 3x3 拼接墙、后来改要会议平板的档案（复刻实测）。"""
    from src.models.requirement import RequirementProfile

    profile = RequirementProfile()
    profile.display_type = "LCD"
    profile.purpose = "exhibition"
    profile.environment = "indoor"
    profile.installation = "fixed"
    profile.lcd_category = "advertising"      # 展览会那次判的品类
    profile.lcd_is_splicing = True            # 旧需求的拼接事实
    profile.lcd_splicing_layout = "3x3"
    profile.lcd_screen_count = 9
    profile.lcd_size_inch = 65.0
    profile.sources.update(
        {
            "purpose": "explicit",
            "environment": "explicit",
            "installation": "explicit",
            "lcd_category": "understanding",
            "lcd_is_splicing": "explicit",
            "lcd_splicing_layout": "explicit",
            "lcd_screen_count": "explicit",
            "lcd_size_inch": "explicit",
        }
    )
    return profile


def _make_it_an_ifp(profile):
    """客户改口：可手写的会议室用屏 → 会议平板。"""
    profile.lcd_category = "conference_education"
    profile.purpose = "conference"
    profile.lcd_handwriting_required = True
    profile.sources["lcd_category"] = "understanding"
    profile.sources["lcd_handwriting_required"] = "explicit"
    return profile


class TestLayoutOnlyForSplicingRequirements:

    def test_layout_text_requires_a_splicing_requirement(self):
        """有排布但"不是拼接需求" → 不许产出排布文本。"""
        from src.rag.lcd_recommendation import layout_text

        profile = _stale_splicing_profile()
        assert layout_text(profile) == "3x3 layout (9 panels)"   # 拼接需求：正常

        # 只把"是否拼接"抹掉（例如 IFP 路径），排布残留也不许输出
        profile.lcd_is_splicing = None
        assert layout_text(profile) == "", "不是拼接需求就不该有排布"

        profile.lcd_is_splicing = False
        assert layout_text(profile) == "", "明确不要拼接 → 更不该有排布"

    def test_ifp_profile_never_produces_layout(self):
        from src.dialogue.lcd_decision import is_ifp_requirement
        from src.rag.lcd_recommendation import layout_text

        profile = _make_it_an_ifp(_stale_splicing_profile())
        assert is_ifp_requirement(profile) is True
        # 判定为 IFP 之后（未经 lcd_turn 清理）也不许输出排布
        profile.lcd_is_splicing = False
        assert layout_text(profile) == ""


class TestIfpClearsStaleSplicingFacts:

    def test_enforce_drops_splicing_facts_for_ifp(self):
        from src.dialogue.lcd_decision import _enforce_ifp_cannot_splice

        profile = _make_it_an_ifp(_stale_splicing_profile())
        dropped = _enforce_ifp_cannot_splice(profile)

        assert "lcd_splicing_layout" in dropped, dropped
        assert "lcd_screen_count" in dropped, dropped
        assert profile.lcd_splicing_layout is None
        assert profile.lcd_screen_count is None
        assert profile.lcd_is_splicing is False
        # 与拼接无关的事实必须原样保留
        assert profile.lcd_size_inch == 65.0
        assert profile.environment == "indoor"

    def test_non_ifp_is_left_untouched(self):
        """客户真的要拼接屏 → 什么都不许清。"""
        from src.dialogue.lcd_decision import _enforce_ifp_cannot_splice

        profile = _stale_splicing_profile()          # advertising + 3x3，不是 IFP
        assert _enforce_ifp_cannot_splice(profile) == []
        assert profile.lcd_is_splicing is True
        assert profile.lcd_splicing_layout == "3x3"
        assert profile.lcd_screen_count == 9

    def test_lcd_turn_end_to_end_drops_splicing_for_ifp(self):
        """走一遍 lcd_turn：IFP 会话里拼接事实被清掉，且不再问拼接类问题。"""
        from src.dialogue.lcd_decision import is_ifp_requirement, lcd_turn
        from src.rag.lcd_recommendation import layout_text

        profile = _make_it_an_ifp(_stale_splicing_profile())
        profile, action = lcd_turn(profile, "we need handwriting in the meeting room")

        assert is_ifp_requirement(profile) is True
        assert profile.lcd_splicing_layout is None, profile.lcd_splicing_layout
        assert profile.lcd_screen_count is None
        assert layout_text(profile) == ""
        # 拼接类槽位不该出现在"还缺什么 / 下一问"里
        assert "lcd_layout" not in (action.missing_fields or [])
        assert action.question_slot not in ("lcd_layout", "lcd_bezel", "lcd_splicing")


class TestLockedFactsHideSplicingForIfp:

    def test_locked_facts_exclude_splicing_when_ifp(self):
        from src.dialogue.lcd_decision import _locked_facts

        profile = _make_it_an_ifp(_stale_splicing_profile())
        facts = _locked_facts(profile, dict(profile.sources))
        assert "lcd_is_splicing" not in facts, facts
        assert "lcd_splicing_layout" not in facts, facts
        assert "lcd_screen_count" not in facts, facts
        # 会议平板真正需要的事实照旧给到表达层
        assert facts.get("lcd_size_inch") == 65.0

    def test_locked_facts_keep_splicing_for_a_real_video_wall(self):
        from src.dialogue.lcd_decision import _locked_facts

        profile = _stale_splicing_profile()
        facts = _locked_facts(profile, dict(profile.sources))
        assert facts.get("lcd_is_splicing") is True
        assert facts.get("lcd_splicing_layout") == "3x3"
        assert facts.get("lcd_screen_count") == 9
