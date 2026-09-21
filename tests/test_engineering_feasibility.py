"""v2.5 Phase 6 / v2.5+：Engineering Feasibility Engine（可行 / 做不到 / 需调整）。

客户口径（2026-09-21）：
  · 分辨率一律按"屏体大约能不能达到"处理 —— 不区分输入 / 屏体，不提澄清问题；
  · 达到或超过目标都算可行；
  · 连最细的点间距也达不到 → 直接给结论话术（标准尺寸 / 更细的 P 值 / 降分辨率）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import (  # noqa: E402
    check_feasibility,
    parse_resolution,
    required_pitch_mm,
    standard_size_for_resolution,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.recommendation_service import RecommendationService  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestFeasibilityStatuses:

    def test_no_resolution_requirement_is_feasible(self):
        result = check_feasibility(_profile(environment="indoor", installation="fixed"))
        assert result.feasible is True and result.status == "FEASIBLE"
        assert result.message == ""

    def test_any_wording_is_treated_as_screen_target(self):
        """不区分"4K 输入 / 4K 屏体 / 只说 4K" —— 都按屏体大约达到处理，且不问。"""
        profile = _profile(
            environment="indoor", installation="fixed",
            target_width_mm=10000, target_height_mm=5000,
        )
        for text in (
            "support 4K input",
            "we need 4k",
            "the LED itself needs to be 3840x2160",
        ):
            result = check_feasibility(
                profile, resolution=parse_resolution(text), finest_pitch_mm=1.25
            )
            assert result.feasible is True, text
            assert result.message == "", text
            assert result.resolution_result is not None, text
            assert not hasattr(result, "question")

    def test_larger_than_target_resolution_is_feasible(self):
        """屏体拼出来比目标还高（例如 10m 宽的屏）也算达到 —— 客户口径不要求一模一样。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=1.25,
            target_width_mm=10000, target_height_mm=5000,
        )
        result = check_feasibility(
            profile,
            resolution=parse_resolution("we need 4k"),
            finest_pitch_mm=1.25,
        )
        assert result.feasible is True
        assert result.resolution_result.fit_level == "MEETS_OR_EXCEEDS"

    def test_size_too_small_is_stated_plainly(self):
        """2m × 1.125m 连最细点间距也到不了 4K → 直接说清 + 给标准尺寸。"""
        profile = _profile(
            environment="indoor", installation="fixed", purpose="conference",
            target_width_mm=2000, target_height_mm=1125,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("we need 4k"), finest_pitch_mm=1.25
        )
        assert result.feasible is False and result.status == "IMPOSSIBLE"
        assert "can't reach 3840x2160" in result.message
        assert result.standard_size_m == pytest.approx((4.8, 2.7))
        assert "4.8m x 2.7m" in result.message
        assert len(result.alternatives) >= 2

    def test_customer_fixed_coarse_pitch_is_stated_plainly(self):
        """10m × 5m 客户定死 P5 → 现在只能 2000x1000，说清需要 P2.3 或更大尺寸。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=5.0,
            target_width_mm=10000, target_height_mm=5000,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("we need 4k"), finest_pitch_mm=1.25
        )
        assert result.feasible is False and result.status == "IMPOSSIBLE"
        assert result.required_pitch_mm == pytest.approx(2.315, abs=0.01)
        assert "P2.3" in result.message
        assert len(result.alternatives) >= 3

    def test_finer_pitch_available_does_not_block(self):
        """客户没定死 P 值、目录里更细的点间距能做到 → 不打断，交给引擎选型。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=5.0,
            target_width_mm=10000, target_height_mm=5000,
        )
        profile.sources["pixel_pitch_mm"] = "inferred"
        result = check_feasibility(
            profile, resolution=parse_resolution("we need 4k"), finest_pitch_mm=1.25
        )
        assert result.feasible is True
        assert result.status == "NEED_ADJUSTMENT"
        assert result.message == ""

    def test_pitch_can_reach_target_resolution(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=1.0,
            target_width_mm=4000, target_height_mm=2250,
        )
        result = check_feasibility(
            profile,
            resolution=parse_resolution("the screen itself is 3840x2160"),
            finest_pitch_mm=1.0,
        )
        assert result.resolution_result is not None
        assert result.resolution_result.acceptable is True


class TestResolutionHelpers:

    def test_required_pitch(self):
        assert required_pitch_mm((3840, 2160), 4000, 2250) == pytest.approx(1.042, abs=0.001)

    def test_standard_size(self):
        assert standard_size_for_resolution((3840, 2160), 2.5) == pytest.approx((9.6, 5.4))
        assert standard_size_for_resolution(None, 2.5) is None


class TestCoordinatorUsesFeasibility:

    def test_service_states_size_is_too_small(self):
        """尺寸 + 分辨率不够时：不推荐，直接把结论话术交给客户（不是提问）。"""
        profile = _profile(
            environment="indoor", installation="fixed", purpose="conference",
            pixel_pitch_mm=2.5, price_preference="both",
            target_width_mm=2000, target_height_mm=1125,
        )
        profile.resolution_requirement = parse_resolution("we need 4k").to_dict()
        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        assert result["coordinator_status"] == "REJECTED"
        assert "can't reach 3840x2160" in (result["next_question"] or "")
        assert result["decision_audit"].get("feasibility")

    def test_service_blocks_on_conflict(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=2.5,
            target_width_mm=10000, target_height_mm=5000, room_area_sqm=25,
        )
        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        assert result["coordinator_status"] == "CONFLICT"
        assert result["decision_audit"].get("feasibility") or result["conflicts"]
