"""
Phase 14：Recommendation Engine 测试（确定性，无需 LLM / 向量库）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.recommendation_engine import (  # noqa: E402
    PRICE_TIER_ORDER,
    RecommendationEngine,
)


@pytest.fixture(scope="module")
def engine():
    return RecommendationEngine()


class TestRecommendationEngine:

    def test_returns_model_level(self, engine):
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference", "viewing_distance_m": 5}
        ))
        top = result["recommendations"][0]
        assert top["model"].startswith("TW")
        assert "-P" in top["model"]  # Model 级，而不是 Series

    def test_no_hard_constraint_violation(self, engine):
        cases = [
            {"environment": "outdoor", "purpose": "advertising", "viewing_distance_m": 20},
            {"environment": "indoor", "installation": "rental", "pixel_pitch_mm": 3.9},
            {"environment": "indoor", "purpose": "showroom", "viewing_distance_m": 2},
            {"environment": "outdoor", "brightness_min": 8000},
        ]
        for slots in cases:
            result = engine.recommend(profile=RequirementProfile.from_slots(slots))
            assert result["violations"] == [], slots
            assert result["candidate_count"] > 0, slots

    def test_outdoor_never_returns_indoor(self, engine):
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "outdoor", "purpose": "advertising", "viewing_distance_m": 20}
        ))
        assert all(rec["outdoor"] for rec in result["recommendations"])

    def test_outdoor_rental_returns_empty(self, engine):
        """产品库没有户外租赁产品 → 必须返回空，不得用室内租赁替代"""
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "outdoor", "installation": "rental"}
        ))
        assert result["recommendations"] == []
        assert result["candidate_count"] == 0

    def test_explicit_pitch_is_respected(self, engine):
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "indoor", "installation": "rental", "pixel_pitch_mm": 1.95}
        ))
        assert result["recommendations"][0]["model"] == "TW11-IR-P1.95(GOB)"

    def test_unknown_model_falls_back_to_series(self, engine):
        """客户点名了库里不存在的型号时，退化到同系列同量级，而不是返回空"""
        result = engine.recommend("TW11-OD-P6.67 的箱体尺寸是多少")
        assert result["recommendations"], "应退化到 TW11-OD 系列"
        assert all(rec["series_id"] == "TW11-OD" for rec in result["recommendations"])

    def test_low_budget_prefers_low_tier(self, engine):
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "outdoor", "purpose": "advertising", "viewing_distance_m": 18,
             "budget_level": "low"}
        ))
        assert result["recommendations"][0]["price_tier"] == "low"

    def test_reasons_and_breakdown_present(self, engine):
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "outdoor", "purpose": "advertising", "viewing_distance_m": 20}
        ))
        top = result["recommendations"][0]
        assert top["reasons"]
        assert set(top["breakdown"]) == {"scene", "pitch", "size", "budget", "quality"}
        assert 0 <= top["score"] <= 100

    def test_no_budget_prefers_cheapest(self, engine):
        """客户没提预算时，默认推荐最便宜的档位（业务规则）"""
        conference = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference", "viewing_distance_m": 5}
        ))
        assert conference["recommendations"][0]["price_tier"] == "low", \
            "未指定预算时应首选最便宜的 low 档"

    def test_explicit_low_budget_prefers_low_tier(self, engine):
        """客户明确说预算低 → 低档优先"""
        result = engine.recommend(profile=RequirementProfile.from_slots(
            {"environment": "outdoor", "purpose": "advertising",
             "viewing_distance_m": 18, "budget_level": "low"}
        ))
        assert result["recommendations"][0]["price_tier"] == "low"


class TestOutdoorPitchBoundary:
    """室外屏点间距边界：P6 及以上（细间距 P2.5~P5 只用于室内/近距离）。"""

    def test_inference_floor_for_outdoor(self):
        from src.rag.parameter_inference import (
            OUTDOOR_MIN_PITCH_MM,
            infer_technical_parameters,
        )

        outdoor = infer_technical_parameters(
            {"environment": "outdoor", "viewing_distance_m": 5}
        )
        assert outdoor["pixel_pitch_min_mm"] == OUTDOOR_MIN_PITCH_MM
        assert outdoor["source"]["pixel_pitch"] == "inferred_outdoor_min_p6"

        # 室内不受影响：5m 视距仍按距离表给细间距
        indoor = infer_technical_parameters(
            {"environment": "indoor", "viewing_distance_m": 5}
        )
        assert indoor["pixel_pitch_min_mm"] < OUTDOOR_MIN_PITCH_MM

    def test_outdoor_without_distance_still_has_floor(self):
        from src.rag.parameter_inference import (
            OUTDOOR_MIN_PITCH_MM,
            infer_technical_parameters,
        )

        technical = infer_technical_parameters({"environment": "outdoor"})
        assert technical["pixel_pitch_min_mm"] == OUTDOOR_MIN_PITCH_MM

    def test_engine_never_recommends_fine_pitch_outdoor(self, engine):
        from src.rag.parameter_inference import OUTDOOR_MIN_PITCH_MM

        for distance in (5, 20, 30):
            slots = {
                "environment": "outdoor",
                "purpose": "advertising",
                "installation": "fixed",
                "viewing_distance_m": distance,
            }
            result = engine.recommend(
                profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )
            assert result["recommendations"], distance
            for rec in result["recommendations"]:
                assert rec["pixel_pitch_mm"] >= OUTDOOR_MIN_PITCH_MM - 1e-6, (distance, rec)

    def test_explicit_customer_pitch_is_still_respected(self, engine):
        """客户自己点名了更细的点间距（半户外/近距离）→ 尊重客户，不强行抬到 P6。"""
        slots = {
            "environment": "outdoor",
            "purpose": "advertising",
            "installation": "fixed",
            "pixel_pitch_mm": 3.076,
        }
        result = engine.recommend(
            profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        assert result["recommendations"]
        assert all(rec["pixel_pitch_mm"] < 6.0 for rec in result["recommendations"])

    def test_validation_flags_outdoor_fine_pitch(self):
        from src.rag.recommendation_engine import RecommendationEngine
        from src.rag.validation import validate_recommendation

        # 直接构造"室外 + 细间距"的选型结果（模拟异常路径）
        profile = RequirementProfile.from_slots(
            {
                "environment": "outdoor",
                "purpose": "advertising",
                "installation": "fixed",
                "viewing_distance_m": 20,
            },
            explicit_keys={"environment", "purpose", "installation", "viewing_distance_m"},
        )
        selection = {
            "recommendations": [{"model": "TW11-OD-P2.5"}],
            "technical_parameters": {"pixel_pitch_min_mm": 6.0, "pixel_pitch_max_mm": 8.0},
        }
        report = validate_recommendation("Recommended.", [], profile, selection, None)
        assert report["checks"]["outdoor_pitch_boundary"] is False
        assert any("室外点间距" in err for err in report["errors"])

    def test_outdoor_defaults_to_p6(self, engine):
        """业务规则：室外默认首选 P6（合格档里最细的），不管视距多远。"""
        for distance in (5, 10, 20, 30, 50):
            slots = {
                "environment": "outdoor",
                "purpose": "advertising",
                "installation": "fixed",
                "viewing_distance_m": distance,
            }
            result = engine.recommend(
                profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )
            top = result["recommendations"][0]
            assert top["model"].endswith("-P6"), (distance, top["model"])
            # 首选就是所有候选里最细的合格档
            assert top["pixel_pitch_mm"] == min(
                rec["pixel_pitch_mm"] for rec in result["recommendations"]
            ), distance

    def test_customer_pitch_overrides_outdoor_default(self, engine):
        """客户点名了点间距 → 按客户的来（P8 给 P8，P2.5 也给 P2.5）。"""
        for pitch, suffix in ((8.0, "P8"), (2.5, "P2.5")):
            slots = {
                "environment": "outdoor",
                "purpose": "advertising",
                "installation": "fixed",
                "viewing_distance_m": 20,
                "pixel_pitch_mm": pitch,
            }
            result = engine.recommend(
                profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
            )
            assert result["recommendations"], pitch
            assert result["recommendations"][0]["model"].endswith(suffix), (pitch, result["recommendations"])

    def test_indoor_pitch_preference_unchanged(self, engine):
        """室内仍然按"区间 75% 位置"的口径选，不受室外规则影响。"""
        slots = {
            "environment": "indoor",
            "purpose": "conference",
            "installation": "fixed",
            "viewing_distance_m": 5,
        }
        result = engine.recommend(
            profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        top = result["recommendations"][0]
        assert top["pixel_pitch_mm"] > 1.5, top
