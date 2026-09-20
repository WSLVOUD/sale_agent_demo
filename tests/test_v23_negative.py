"""v2.3 §17：Negative Tests —— 自相矛盾 / 缺信息 / 错单位 / 反复改需求。

覆盖计划列出的场景：

    室内 + P10
    25㎡ + 50㎡ 屏（屏比房间大）
    错误单位
    缺少 viewing distance
    Vision 错误参数
    客户反复修改需求
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.fact_validation import validate_incoming_facts  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)
from src.rag.recommendation_service import RecommendationService  # noqa: E402


def _profile(slots: dict) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestConflictingInputs:

    def test_indoor_with_p10_is_blocked(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 10.0,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        decision = check_recommendation_ready(profile)
        assert decision.status == "CONFLICT"
        assert decision.ready is False

        result = RecommendationService().recommend(profile)
        assert result["recommendations"] == []
        assert result["reject_reasons"] or result["conflicts"]

    def test_screen_bigger_than_room_is_blocked(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 10000, "target_height_mm": 5000,   # 50㎡
            "room_area_sqm": 25,                                   # 25㎡
        })
        decision = check_recommendation_ready(profile)
        assert decision.status == "CONFLICT"
        assert "size" in (decision.blocked_slots or [])
        # 冲突时给出的是澄清问句（英文），不是推荐
        assert decision.next_question

    def test_depth_smaller_than_screen_height_is_blocked(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 8000, "target_height_mm": 5000, "room_depth_m": 3,
        })
        assert check_recommendation_ready(profile).status == "CONFLICT"


class TestInvalidUnitsAndRanges:

    @pytest.mark.parametrize("facts,rejected_slot", [
        ({"viewing_distance_m": 50000}, "viewing_distance_m"),      # 5 万米
        ({"pixel_pitch_mm": 300}, "pixel_pitch_mm"),                # 300mm 点间距
        ({"target_width_mm": 5}, "target_width_mm"),                # 5mm 屏宽
        ({"room_area_sqm": 0}, "room_area_sqm"),
        ({"audience_count": 0}, "audience_count"),
        ({"environment": "inside"}, "environment"),                 # 取值不合法
        ({"installation": "buy"}, "installation"),
    ])
    def test_out_of_range_values_are_rejected(self, facts, rejected_slot):
        result = validate_incoming_facts(facts, source="llm")
        assert rejected_slot in result.rejected, result.to_dict()
        assert rejected_slot not in result.accepted
        assert result.ok is False

    def test_unknown_field_is_rejected(self):
        result = validate_incoming_facts({"totally_made_up": 1}, source="llm")
        assert "totally_made_up" in result.rejected

    def test_low_priority_source_cannot_override_customer(self):
        profile = _profile({"environment": "outdoor"})
        result = validate_incoming_facts(
            {"environment": "indoor"}, source="llm", profile=profile
        )
        assert "environment" in result.rejected
        assert profile.environment == "outdoor"


class TestMissingViewingDistance:

    def test_recommendation_still_possible_without_distance(self):
        """缺观看距离不是死路：用屏尺寸 / 人数推导，或按环境兜底。"""
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        result = RecommendationService().recommend(profile)
        assert result["recommendation_status"] in ("RECOMMENDED", "DEGRADED")
        assert result["recommendations"]
        assert result["provenance"]["entries"]["viewing_distance_m"]["source"] in (
            "derived", "inferred", "customer",
        )

    def test_calculation_asks_for_size_only(self):
        profile = _profile({"environment": "indoor", "installation": "fixed",
                            "pixel_pitch_mm": 3.0})
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert "width" in decision.missing or "size" in decision.missing


class TestVisionBadParams:

    def test_absurd_vision_values_are_dropped(self):
        from src.models.requirement import RequirementProfile as P
        from src.vision.extractor import VisionExtractor
        from src.vision.integration import apply_vision_to_profile

        vision = VisionExtractor.from_payload({
            "pixel_pitch_mm": {"value": 9999, "source": "vision_explicit"},
            "environment": {"value": "indoor", "source": "vision_explicit"},
        })
        merged, stats = apply_vision_to_profile(P(), vision)
        assert merged.pixel_pitch_mm is None, "荒谬的点间距不能入档"
        assert merged.environment == "indoor"
        assert stats["merged_fields"] >= 1

    def test_vision_pitch_is_inferred_not_confirmed(self):
        from src.engineering import build_provenance
        from src.models.requirement import RequirementProfile as P
        from src.vision.extractor import VisionExtractor
        from src.vision.integration import apply_vision_to_profile

        vision = VisionExtractor.from_payload({
            "pixel_pitch_mm": {"value": 2.5, "source": "vision_explicit"},
        })
        merged, _ = apply_vision_to_profile(P(), vision)
        entry = build_provenance(merged, {}).get("pixel_pitch_mm")
        assert entry is not None
        assert entry.source == "vision"
        assert entry.status in ("INFERRED", "CONFIRMED")


class TestCustomerRevisions:

    def test_latest_value_wins(self):
        from src.core.requirement_extractor import RequirementExtractor

        first = RequirementExtractor().extract("P3", use_llm=False, session_id="")
        assert first.pixel_pitch_mm == pytest.approx(3.0)

        second = RequirementExtractor().extract(
            "actually make it P2.5", previous_profile=first, use_llm=False, session_id=""
        )
        assert second.pixel_pitch_mm == pytest.approx(2.5)
        assert second.conflicts == []

    def test_size_revision_updates_both_axes(self):
        from src.core.requirement_extractor import RequirementExtractor

        first = RequirementExtractor().extract("5m x 3m", use_llm=False, session_id="")
        second = RequirementExtractor().extract(
            "make it 8m x 4m", previous_profile=first, use_llm=False, session_id=""
        )
        assert second.target_width_mm == pytest.approx(8000)
        assert second.target_height_mm == pytest.approx(4000)

    def test_environment_revision_keeps_customer_value(self):
        from src.core.requirement_extractor import RequirementExtractor

        first = RequirementExtractor().extract("indoor", use_llm=False, session_id="")
        second = RequirementExtractor().extract(
            "actually outdoor", previous_profile=first, use_llm=False, session_id=""
        )
        assert second.environment == "outdoor"
