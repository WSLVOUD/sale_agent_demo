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


class TestEnvironmentPitchRules:
    """客户口径的点间距规则（2026-09-15）：

    室外：4m→P4 / 5m→P4 或 P5 / 6~20m→P5 最合适 / >30m 一律 P10
    室内：≤3m→P2.5 及以下（越近越细）/ >3m→P3 及以上
    客户明确点名点间距 → 一律以客户为准。
    """

    def _top(self, engine, environment, distance, purpose=None, pitch=None):
        purpose = purpose or ("advertising" if environment == "outdoor" else "conference")
        slots = {
            "environment": environment,
            "purpose": purpose,
            "installation": "fixed",
            "viewing_distance_m": distance,
        }
        if pitch is not None:
            slots["pixel_pitch_mm"] = pitch
        result = engine.recommend(
            profile=RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        )
        return result["recommendations"]

    # ── 室外 ────────────────────────────────────────────────────────────
    def test_outdoor_4m_recommends_p4(self, engine):
        top = self._top(engine, "outdoor", 4)[0]
        assert top["pixel_pitch_mm"] == pytest.approx(4.0, abs=0.2), top

    def test_outdoor_5m_recommends_p4_or_p5(self, engine):
        top = self._top(engine, "outdoor", 5)[0]
        pitch = top["pixel_pitch_mm"]
        assert min(abs(pitch - 4.0), abs(pitch - 5.0)) <= 0.2, top

    def test_outdoor_6_to_20m_prefers_p5(self, engine):
        for distance in (6, 8, 12, 15, 20):
            top = self._top(engine, "outdoor", distance)[0]
            assert top["pixel_pitch_mm"] == pytest.approx(5.0, abs=0.2), (distance, top)

    def test_outdoor_beyond_30m_is_p10(self, engine):
        # "超过 30m 一律 P10"（30m 本身归上一档 P6.67）
        for distance in (35, 40, 60):
            top = self._top(engine, "outdoor", distance)[0]
            assert top["pixel_pitch_mm"] == pytest.approx(10.0, abs=0.2), (distance, top)

    def test_outdoor_20_to_25m_prefers_p6(self, engine):
        for distance in (21, 25):
            top = self._top(engine, "outdoor", distance)[0]
            assert top["pixel_pitch_mm"] == pytest.approx(6.67, abs=0.2), (distance, top)

    def test_outdoor_25_to_30m_is_p8(self, engine):
        for distance in (26, 28, 30):
            top = self._top(engine, "outdoor", distance)[0]
            assert top["pixel_pitch_mm"] == pytest.approx(8.0, abs=0.2), (distance, top)

    def test_outdoor_never_recommends_finer_than_the_band(self, engine):
        """4m 场景不再推荐 P2.5/P3（细间距留给室内近距离）。"""
        recs = self._top(engine, "outdoor", 4)
        assert recs
        assert all(rec["pixel_pitch_mm"] >= 3.9 - 1e-6 for rec in recs), recs

    # ── 室内 ────────────────────────────────────────────────────────────
    def test_indoor_within_3m_prefers_p25_or_finer(self, engine):
        for distance in (1.5, 2, 3):
            top = self._top(engine, "indoor", distance)[0]
            assert top["pixel_pitch_mm"] <= 2.5 + 1e-6, (distance, top)

    def test_indoor_over_3m_prefers_p3_or_coarser(self, engine):
        for distance in (4, 5, 8, 15, 25):
            top = self._top(engine, "indoor", distance)[0]
            assert top["pixel_pitch_mm"] >= 3.0 - 1e-6, (distance, top)
            assert top["model"].endswith("-P3.0"), (distance, top)

    # ── 客户点名优先 ────────────────────────────────────────────────────
    @pytest.mark.parametrize("environment,distance,pitch,suffix", [
        ("indoor", 5, 2.0, "P2.0"),
        ("indoor", 5, 1.86, "P1.8"),
        ("indoor", 5, 4.0, "P4.0"),
        ("outdoor", 8, 4.0, "P4"),
        ("outdoor", 8, 5.0, "P5"),
        ("outdoor", 8, 2.5, "P2.5"),
        ("outdoor", 8, 10.0, "P10"),
    ])
    def test_customer_pitch_always_wins(self, engine, environment, distance, pitch, suffix):
        top = self._top(engine, environment, distance, pitch=pitch)[0]
        assert top["model"].endswith(suffix), (pitch, top)

    def test_target_recorded_in_technical_parameters(self):
        from src.rag.parameter_inference import infer_technical_parameters

        indoor = infer_technical_parameters(
            {"environment": "indoor", "viewing_distance_m": 5}
        )
        assert indoor["pitch_target_mm"] == pytest.approx(3.0)
        assert indoor["pixel_pitch_min_mm"] == pytest.approx(3.0)

        outdoor = infer_technical_parameters(
            {"environment": "outdoor", "viewing_distance_m": 8}
        )
        assert outdoor["pitch_target_mm"] == pytest.approx(5.0)

        far = infer_technical_parameters(
            {"environment": "outdoor", "viewing_distance_m": 40}
        )
        assert far["pitch_target_mm"] == pytest.approx(10.0)
