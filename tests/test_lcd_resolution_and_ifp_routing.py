"""分辨率不问客户 + 要手写白板就按 IFP 选型（客户口径 2026-09-30）。

实测（客户原文）：

    会议室 + 65" + "yes i need handwriting" + "i need a lcd witch cam"
    → AI 还在问 "Are you set on 4K, or would 2K be sufficient?"
    → 最后推荐了「P65 commercial LCD display」（一台普通商用显示器）

两个问题：

    1. 分辨率不该反问客户：客户自己提了就照他的，没提就按尺寸规则定
       （>65" → 4K，<65" → 2K/1080p，恰好 65" 默认 4K）。
    2. 客户要的是**可手写的会议室交互平板（IFP）**，选型必须按 IFP 去查，
       不能从 LCD 语料里挑一台普通商用显示器。
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


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestResolutionIsNeverAsked:
    """分辨率只认客户自己提的；没提就按尺寸规则定。"""

    def test_above_65_is_4k(self):
        assert resolution_for_size(75) == ("4K", "size_rule_gt_65")

    def test_below_65_is_2k(self):
        assert resolution_for_size(55) == ("2K", "size_rule_lt_65")

    def test_exactly_65_defaults_to_4k_instead_of_asking(self):
        resolution, source = resolution_for_size(65)

        assert resolution == "4K"
        assert source == "size_default_4k_65", source

    def test_no_branch_ever_asks_the_resolution(self):
        import src.dialogue.lcd_decision as lcd

        for category, order in lcd._BRANCH_ORDER.items():
            assert "lcd_resolution" not in order, (category, order)
        assert "lcd_resolution" not in lcd._UNKNOWN_ORDER

    def test_customer_stated_resolution_is_kept(self):
        profile = _profile()
        profile, _action = lcd_turn(profile, "we need a 55 inch 4K LCD")

        assert profile.lcd_resolution == "4K"
        assert profile.lcd_resolution_source == "customer_explicit"

    def test_meeting_room_65_is_never_asked_and_gets_4k(self):
        profile = _profile()
        profile, action = lcd_turn(profile, "we need a 65 inch LCD for a meeting room")

        assert profile.lcd_resolution == "4K", profile.lcd_resolution
        assert "lcd_resolution" not in action.missing_fields
        assert action.question_slot != "lcd_resolution"


class TestIfpRequirementDetection:
    def test_handwriting_means_ifp(self):
        profile = _profile(
            lcd_category=CONFERENCE_EDUCATION, lcd_handwriting_required=True
        )

        assert is_ifp_requirement(profile) is True
        assert effective_display_type(profile) == "IFP"

    def test_touch_in_a_conference_room_means_ifp(self):
        profile = _profile(
            lcd_category=CONFERENCE_EDUCATION, lcd_touch_required=True
        )

        assert effective_display_type(profile) == "IFP"

    def test_plain_lcd_stays_lcd(self):
        profile = _profile(lcd_category=NORMAL, lcd_size_inch=65.0)

        assert is_ifp_requirement(profile) is False
        assert effective_display_type(profile) == "LCD"

    def test_spliced_video_wall_stays_lcd(self):
        profile = _profile(
            lcd_category=MONITORING, lcd_is_splicing=True, lcd_size_inch=65.0
        )

        assert effective_display_type(profile) == "LCD"

    def test_led_is_never_turned_into_ifp(self):
        profile = _profile(display_type="LED")

        assert effective_display_type(profile) == "LED"


class TestRetrievalUsesIfpWhenHandwritingIsRequired:
    class _FakeHybridSearch:
        def __init__(self):
            self.filters = None

        def search(self, query, top_k=20, filters=None, **kwargs):
            self.filters = filters
            self.query = query
            self.top_k = top_k
            return []

    def _run(self, profile):
        from src.agents.solution.nodes.retrieval import retrieval_node

        search = self._FakeHybridSearch()
        state = {
            "hybrid_search": search,
            "requirement_profile": profile,
            "messages": [{"role": "user", "content": "pls recommend for me"}],
            "requirement": {},
        }
        retrieval_node(state)
        return search

    def test_hard_filter_becomes_ifp_for_a_handwriting_meeting_room(self):
        profile = _profile(
            environment="indoor",
            lcd_category=CONFERENCE_EDUCATION,
            lcd_size_inch=65.0,
            lcd_handwriting_required=True,
            lcd_camera_required=True,
        )

        search = self._run(profile)

        assert (search.filters or {}).get("display_type") == "IFP", search.filters
        assert "interactive flat panel" in str(search.query).lower(), search.query

    def test_hard_filter_stays_lcd_for_a_plain_commercial_screen(self):
        profile = _profile(environment="indoor", lcd_category=NORMAL, lcd_size_inch=65.0)

        search = self._run(profile)

        assert (search.filters or {}).get("display_type") == "LCD", search.filters
