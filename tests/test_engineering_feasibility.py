"""v2.5 Phase 6：Engineering Feasibility Engine（可行 / 冲突 / 无法实现 / 需澄清）。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import check_feasibility, parse_resolution  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.recommendation_service import RecommendationService  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestFeasibilityStatuses:

    def test_no_resolution_requirement_is_feasible(self):
        result = check_feasibility(_profile(environment="indoor", installation="fixed"))
        assert result.feasible is True and result.status == "FEASIBLE"

    def test_input_resolution_does_not_enter_stitching(self):
        profile = _profile(environment="indoor", installation="fixed")
        result = check_feasibility(profile, resolution=parse_resolution("support 4K input"))
        assert result.feasible is True
        assert any("4K 输入" in note for note in result.notes)

    def test_ambiguous_4k_needs_one_clarification(self):
        profile = _profile(environment="indoor", installation="fixed")
        result = check_feasibility(profile, resolution=parse_resolution("we need 4k"))
        assert result.status == "NEED_CLARIFICATION"
        assert "LED itself" in result.question

    def test_ratio_conflict(self):
        """4m × 2m 物理尺寸 + 16:9 的 4K 目标 → 比例冲突，先澄清。"""
        profile = _profile(
            environment="indoor", installation="fixed",
            target_width_mm=4000, target_height_mm=2000,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("the LED itself needs to be 3840x2160")
        )
        assert result.feasible is False and result.status == "CONFLICT"
        assert result.question
        assert "physical size" in result.question.lower() or "ratio" in result.question.lower()

    def test_pitch_cannot_reach_target_resolution(self):
        """4m × 2.25m（16:9）+ P5 → 最多 800×450 像素，达不到 4K → IMPOSSIBLE + 调整方向。"""
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=5.0,
            target_width_mm=4000, target_height_mm=2250,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("the LED itself needs to be 3840x2160")
        )
        assert result.feasible is False and result.status == "IMPOSSIBLE"
        assert len(result.alternatives) >= 3
        assert result.question

    def test_pitch_can_reach_target_resolution(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=1.0,
            target_width_mm=4000, target_height_mm=2250,
        )
        result = check_feasibility(
            profile, resolution=parse_resolution("the screen itself is 3840x2160")
        )
        assert result.resolution_result is not None
        assert result.resolution_result.acceptable is True


class TestCoordinatorUsesFeasibility:

    def test_service_blocks_on_ratio_conflict(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=2.5,
            target_width_mm=4000, target_height_mm=2000,
        )
        profile.resolution_requirement = parse_resolution(
            "the LED itself needs to be 3840x2160"
        ).to_dict()
        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        assert result["coordinator_status"] == "CONFLICT"
        assert result["decision_audit"].get("feasibility")

    def test_service_blocks_when_impossible(self):
        profile = _profile(
            environment="indoor", installation="fixed", pixel_pitch_mm=5.0,
            target_width_mm=4000, target_height_mm=2250,
        )
        profile.resolution_requirement = parse_resolution(
            "the LED itself needs to be 3840x2160"
        ).to_dict()
        result = RecommendationService().recommend(profile)
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert result["coordinator_status"] == "REJECTED"
        assert result["reject_reasons"]
