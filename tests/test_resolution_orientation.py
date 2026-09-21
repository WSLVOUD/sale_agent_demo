"""v2.5+：尺寸不区分宽高 + 客户分辨率优先决定点间距（实测 bug 回归）。

现象：客户说"3*5"、要 4K，系统回"我们没有任何型号能达到 3840x2160"。
根因（两条）：
  1. "室内 + 5m 视距"推出的点间距区间（P3.0~P10）被当硬约束，
     把能拼到 4K 的细点间距型号（P0.7/P0.9/P1.2）整个挡在候选之外；
  2. 尺寸按"x 是宽、y 是高"单一摆法算，没有考虑换边。
客户口径：客户要的分辨率优先决定点间距；x×y 不区分哪边是宽，
哪种摆法能拼到目标分辨率就按哪种（实在做不到才告诉他）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import check_model_feasibility, parse_resolution, size_orientations  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402
from src.rag.recommendation_engine import RecommendationEngine  # noqa: E402
from src.rag.recommendation_service import RecommendationService  # noqa: E402

TARGET_4K = "we need 4k"


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


def _base(**extra) -> RequirementProfile:
    slots = {
        "environment": "indoor",
        "installation": "fixed",
        "purpose": "church",
        "viewing_distance_m": 5,
        "price_preference": "both",
    }
    slots.update(extra)
    return _profile(**slots)


class TestSizeOrientations:

    def test_two_orientations(self):
        assert size_orientations(3000, 5000) == (
            ("as_given", 3000.0, 5000.0), ("swapped", 5000.0, 3000.0)
        )

    def test_square_has_one_orientation(self):
        assert size_orientations(3000, 3000) == (("as_given", 3000.0, 3000.0),)

    def test_missing_size(self):
        assert size_orientations(0, 5000) == ()


class TestModelFeasibilityPicksOrientation:

    def test_swapped_orientation_is_used_when_it_reaches_4k(self):
        """3m × 5m：按"5m 作宽"摆，P0.78（TW31-COB-P0.7H）能到 4K。"""
        engine = RecommendationEngine()
        model = next(m for m in engine.models if m.model == "TW31-COB-P0.7H")
        profile = _base(target_width_mm=3000, target_height_mm=5000)
        profile.resolution_requirement = parse_resolution(TARGET_4K).to_dict()
        check = check_model_feasibility(profile, model)
        assert check["acceptable"] is True
        assert check["orientation"] == "swapped"
        assert check["resolution_fit"]["actual"][0] >= 3840

    def test_no_resolution_requirement_is_not_applicable(self):
        engine = RecommendationEngine()
        model = next(m for m in engine.models if m.model == "TW31-COB-P0.7H")
        check = check_model_feasibility(
            _base(target_width_mm=3000, target_height_mm=5000), model
        )
        assert check["applicable"] is False


class TestResolutionDrivesPitch:

    def test_coarse_only_band_no_longer_blocks_fine_pitches(self):
        """室内 5m 的推荐区间是 P3.0~P10；客户要 4K 时细点间距型号必须能进候选。"""
        profile = _base(target_width_mm=3000, target_height_mm=5000)
        profile.resolution_requirement = parse_resolution(TARGET_4K).to_dict()
        result = RecommendationEngine().recommend(profile=profile, top_k=3)
        pitches = [
            r["model"] for r in result.get("recommendations") or []
        ]
        assert pitches, "应该能选出型号"
        assert all("TW31-COB" in name for name in pitches), pitches

    def test_without_resolution_the_distance_band_still_wins(self):
        """客户没提分辨率 → 口径不变（还是按视距推荐 P3.0 档）。"""
        profile = _base(target_width_mm=3000, target_height_mm=5000)
        result = RecommendationEngine().recommend(profile=profile, top_k=3)
        names = [r["model"] for r in result.get("recommendations") or []]
        assert names and all("TW31-COB" not in name for name in names), names


class TestEndToEnd:

    def test_3x5_with_4k_recommends_a_model_that_reaches_it(self):
        profile = _base(target_width_mm=3000, target_height_mm=5000)
        profile.resolution_requirement = parse_resolution(TARGET_4K).to_dict()
        assert check_recommendation_ready(profile).ready is True
        result = RecommendationService().recommend(profile)
        assert result["coordinator_status"] == "RECOMMENDED"
        models = [r["model"] for r in result["recommendations"]]
        assert models, result.get("next_question")
        assert result["size_orientation"] == "swapped"
        engine = RecommendationEngine()
        top = next(m for m in engine.models if m.model == models[0])
        check = check_model_feasibility(profile, top)
        assert check["acceptable"] is True

    def test_customer_fixed_pitch_is_respected_and_stated(self):
        """客户把 P 值定死了 → 不偷偷换细 P 值，而是直说做不到 + 给调整方向。"""
        profile = _base(
            target_width_mm=3000, target_height_mm=5000, pixel_pitch_mm=3.0
        )
        profile.resolution_requirement = parse_resolution(TARGET_4K).to_dict()
        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        message = result["next_question"] or ""
        assert "3840x2160" in message
        assert "P0.78" in message or "finer" in message

    def test_impossible_size_is_stated_plainly(self):
        profile = _base(target_width_mm=2000, target_height_mm=1125)
        profile.resolution_requirement = parse_resolution(TARGET_4K).to_dict()
        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        assert "can't reach 3840x2160" in (result["next_question"] or "")
