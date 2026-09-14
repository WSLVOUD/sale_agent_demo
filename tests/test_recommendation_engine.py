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
