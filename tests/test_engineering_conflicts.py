"""v2.5 Phase 6：工程冲突统一判断（尺寸 / 比例 / 分辨率 / 环境）。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import check_feasibility, detect_engineering_conflicts, parse_resolution  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestEngineeringConflicts:

    def test_screen_bigger_than_room(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=3.0,
            target_width_mm=10000, target_height_mm=5000, room_area_sqm=25,
        )
        assert detect_engineering_conflicts(profile)
        assert check_feasibility(profile).status == "CONFLICT"

    def test_indoor_with_coarse_pitch(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=10.0,
            target_width_mm=5000, target_height_mm=3000,
        )
        assert check_feasibility(profile).status == "CONFLICT"

    def test_depth_smaller_than_screen(self):
        profile = _profile(
            environment="indoor", installation="fixed",
            target_width_mm=8000, target_height_mm=5000, room_depth_m=3,
        )
        assert check_feasibility(profile).status == "CONFLICT"

    def test_consistent_profile_has_no_conflict(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=3.0,
            target_width_mm=5000, target_height_mm=3000, room_area_sqm=80,
            room_depth_m=8,
        )
        assert detect_engineering_conflicts(profile) == []
        assert check_feasibility(profile).feasible is True

    def test_resolution_shortfall_is_stated_with_standard_size(self):
        """4m × 2m + P2.5 只能到 1600x800 → 直说达不到 4K，并给出标准尺寸。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=2.5,
            target_width_mm=4000, target_height_mm=2000,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("display resolution 3840x2160")
        )
        assert result.feasible is False and result.status == "IMPOSSIBLE"
        assert result.standard_size_m == pytest.approx((9.6, 5.4))
        assert "3840x2160" in result.message
        assert "1600x800" in result.message
