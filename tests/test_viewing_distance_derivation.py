"""v2.2：观看距离的确定性推导 + 点间距物理窗口。

背景（实测 bug）：客户说"indoor permanent 10x5m wall for around 100 viewers"，
系统给出 **P1.2**。根因不是缺规则，而是缺输入 —— 没有观看距离时"点间距"这一维
整维不参与打分，同系列所有型号完全平分，排序兜底"点间距小的优先"就把最细最贵的
型号挑了出来。

现在的做法（不是再加场景规则）：

    客户给的东西（人数 / 面积 / 进深 / 屏尺寸 / 距离）
        → 确定性公式 → 观看距离区间 [最近观众, 最远观众]
        → 物理窗口 [最远÷5, 最远÷1.5]（再用最近观众收紧上限）
        → 在窗口内选型（并列时取接近窗口首选值的型号，绝不无依据取最细）

所以客户换一种说法时，需要的是"能把数字解析出来"，而不是新增一条推荐规则。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.parameter_inference import (  # noqa: E402
    estimate_viewing_distance,
    fallback_pitch_band,
    pitch_window_for_distances,
)
from src.rag.query_understanding import extract_slots  # noqa: E402


class TestSpaceFactParsing:
    """解析层：人数 / 面积 / 进深只是"观看距离"的其它来源，不是推荐规则。"""

    @pytest.mark.parametrize("message,expected", [
        ("indoor wall for around 100 viewers", 100),
        ("about 50 people will watch it", 50),
        ("a hall with 120 seats", 120),
        ("会议室能坐 80 人", 80),
        ("for 30 attendees", 30),
    ])
    def test_audience_count(self, message, expected):
        assert extract_slots(message).get("audience_count") == expected

    @pytest.mark.parametrize("message,expected", [
        ("a 25 sqm meeting room", 25),
        ("about 40 square meters", 40),
        ("60 m2 showroom", 60),
        ("120 平米的大厅", 120),
    ])
    def test_room_area(self, message, expected):
        assert extract_slots(message).get("room_area_sqm") == pytest.approx(expected)

    @pytest.mark.parametrize("message,expected_area,expected_depth", [
        ("the room is 8m x 5m", 40, 8),
        ("场地 8米x5米", 40, 8),
        ("room 8m by 5m", 40, 8),
        ("the hall is 10m wide and 6m deep", 60, 6),
        ("5米宽8米深", 40, 8),
    ])
    def test_room_dimensions_are_not_the_screen(self, message, expected_area, expected_depth):
        """场地尺寸 ≠ 屏体尺寸：不能把 8×5m 的**房间**当成 8×5m 的屏去算箱体。"""
        slots = extract_slots(message)
        assert slots.get("room_area_sqm") == pytest.approx(expected_area)
        assert slots.get("room_depth_m") == pytest.approx(expected_depth)
        assert "target_width_mm" not in slots, slots
        assert "target_height_mm" not in slots, slots

    def test_screen_dimensions_still_parse_as_screen(self):
        """说了"屏幕"就还是屏体尺寸（别把屏当成房间）。"""
        slots = extract_slots("the screen is 10m wide and 6m deep")
        assert slots.get("target_width_mm") == pytest.approx(10000)
        assert "room_depth_m" not in slots

    @pytest.mark.parametrize("message,expected", [
        ("the hall is 8 m deep", 8),
        ("room depth 12m", 12),
        ("进深 9 米", 9),
    ])
    def test_room_depth(self, message, expected):
        assert extract_slots(message).get("room_depth_m") == pytest.approx(expected)

    def test_screen_size_is_not_mistaken_for_space(self):
        slots = extract_slots("indoor permanent LED wall 10m x 5m")
        assert slots.get("target_width_mm") == pytest.approx(10000)
        assert "audience_count" not in slots
        assert "room_area_sqm" not in slots
        assert "room_depth_m" not in slots


class TestViewingDistanceEstimation:
    """推导层：公式是封闭的，加不出新规则。"""

    def test_audience_source(self):
        estimate = estimate_viewing_distance({
            "audience_count": 100, "target_width_mm": 10000, "target_height_mm": 5000,
        })
        assert estimate.source == "audience"
        assert estimate.farthest_m > estimate.nearest_m > 0

    def test_audience_source_without_screen_size(self):
        """只有人数也要能推（实测：客户只说"大约 50 个人需要看的屏幕"）。"""
        estimate = estimate_viewing_distance({"audience_count": 50})
        assert estimate.source == "audience"
        assert estimate.farthest_m == pytest.approx(7.0)      # 5 排 × 0.9m + 2.5m
        assert estimate.nearest_m is None, "屏高未知时不做「最近观众」假设"

    def test_audience_scales_with_headcount(self):
        small = estimate_viewing_distance({"audience_count": 50})
        large = estimate_viewing_distance({"audience_count": 100})
        assert large.farthest_m > small.farthest_m

    def test_area_alone_is_enough(self):
        """只知道面积也要能推（按座位区宽深比 2:1 → 进深 = √(面积÷2)）。"""
        estimate = estimate_viewing_distance({"room_area_sqm": 50})
        assert estimate.source == "room_area"
        assert estimate.farthest_m == pytest.approx(4.5, abs=0.2)   # √25 − 0.5

    def test_area_source(self):
        estimate = estimate_viewing_distance({
            "room_area_sqm": 25, "target_width_mm": 3000, "target_height_mm": 2000,
        })
        assert estimate.source == "room_area"
        assert estimate.farthest_m == pytest.approx(25 / 3 - 0.5, abs=0.1)

    def test_depth_source(self):
        estimate = estimate_viewing_distance({"room_depth_m": 8})
        assert estimate.source == "room_depth"
        assert estimate.farthest_m == pytest.approx(7.5)

    def test_screen_size_source(self):
        estimate = estimate_viewing_distance({"target_height_m": 5})
        assert estimate.source == "screen_size"
        assert estimate.nearest_m == pytest.approx(7.5)
        assert estimate.farthest_m == pytest.approx(15.0)

    def test_explicit_distance_wins(self):
        """客户已经说了观看距离 → 不认识推导值（客户给的就是客户给的）。"""
        assert estimate_viewing_distance({"viewing_distance_m": 6}) is None

    def test_nothing_derivable_returns_none(self):
        assert estimate_viewing_distance({}) is None
        assert estimate_viewing_distance({"environment": "indoor"}) is None


class TestPitchWindow:

    def test_window_is_derived_from_farthest_viewer(self):
        low, high = pitch_window_for_distances(7.5, 15.0)
        assert low == pytest.approx(3.0)      # 15 ÷ 5
        assert high == pytest.approx(7.5)     # 最近观众 7.5m ÷ 1m/mm

    def test_window_tightens_for_close_front_row(self):
        low, high = pitch_window_for_distances(2.5, 9.0)
        assert high == pytest.approx(2.5), "最近观众只有 2.5m → 不能推 P6"
        assert low <= high

    def test_empty_input(self):
        assert pitch_window_for_distances(None, None) == (None, None)

    def test_fallback_band_never_picks_the_finest(self):
        low, high, target = fallback_pitch_band("indoor")
        assert low >= 2.0
        assert low <= target <= high


class TestEngineUsesDerivedWindow:
    """端到端：客户给的每一种说法都走同一条推导链，结果必须保守合理。"""

    def _recommend(self, message, top_k=3):
        from src.core.requirement_extractor import RequirementExtractor
        from src.rag.recommendation_engine import RecommendationEngine

        profile = RequirementExtractor().extract(message, use_llm=False, session_id="")
        result = RecommendationEngine().recommend(profile=profile, top_k=top_k)
        return profile, result

    @pytest.mark.parametrize("message", [
        "indoor permanent 10x5m wall for around 100 viewers",
        "indoor permanent 10x5m wall, about 50 people",
        "indoor fixed screen for a 25 sqm meeting room, screen 3m wide",
        "indoor permanent LED wall 10m x 5m",
        "indoor church screen 10x5m, the hall is 8 m deep",
        "I need an indoor LED display with a fixed installation",
    ])
    def test_never_recommends_an_unjustified_fine_pitch(self, message):
        _profile, result = self._recommend(message)
        assert result["recommendations"], message
        picked = result["recommendations"][0]
        assert picked["pixel_pitch_mm"] >= 2.0, (message, picked)
        assert picked["model"] not in ("TW11-3216-P1.2", "TW11-3216-P1.5"), (message, picked)

    def test_hundred_viewers_gets_p3(self):
        """实测那一条：100 人的室内 10x5m 墙 → P3（P4 作省钱备选），不是 P1.2。"""
        _profile, result = self._recommend(
            "indoor permanent 10x5m wall for around 100 viewers"
        )
        picked = result["recommendations"][0]
        assert picked["pixel_pitch_mm"] == pytest.approx(3.076, abs=0.05)
        assert picked["model"].startswith("TW11-3216-P3")
        technical = result["technical_parameters"]
        assert technical["viewing_distance_estimate"]["source"] == "audience"
        assert technical["source"]["pixel_pitch"].startswith("inferred_from_")

    def test_fifty_viewers_stays_in_the_same_window(self):
        _profile, result = self._recommend(
            "indoor permanent 10x5m wall, about 50 people"
        )
        picked = result["recommendations"][0]
        assert 2.5 <= picked["pixel_pitch_mm"] <= 4.0

    def test_no_information_falls_back_to_environment_band(self):
        _profile, result = self._recommend(
            "I need an indoor LED display with a fixed installation"
        )
        technical = result["technical_parameters"]
        assert technical["source"]["pixel_pitch"] == "fallback_environment_default"
        picked = result["recommendations"][0]
        assert 2.5 <= picked["pixel_pitch_mm"] <= 4.0


class TestExplicitInputsUnchanged:
    """回归：客户明确给了点间距 / 观看距离时，行为与以前一致。"""

    def test_explicit_pitch_is_respected(self):
        from src.rag.recommendation_engine import RecommendationEngine

        slots = {
            "display_type": "LED", "environment": "indoor", "installation": "fixed",
            "purpose": "conference", "pixel_pitch_mm": 2.5,
            "target_width_mm": 5000, "target_height_mm": 3000,
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        result = RecommendationEngine().recommend(profile=profile, top_k=3)
        assert result["recommendations"][0]["pixel_pitch_mm"] == pytest.approx(2.5)

    def test_explicit_distance_uses_business_table(self):
        from src.rag.recommendation_engine import RecommendationEngine

        slots = {
            "display_type": "LED", "environment": "indoor", "installation": "fixed",
            "purpose": "church", "viewing_distance_m": 5,
            "target_width_mm": 10000, "target_height_mm": 5000,
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        result = RecommendationEngine().recommend(profile=profile, top_k=3)
        technical = result["technical_parameters"]
        assert technical["viewing_distance_source"] == "explicit"
        assert technical["viewing_distance_estimate"] is None
        assert result["recommendations"][0]["pixel_pitch_mm"] == pytest.approx(3.076, abs=0.05)


class TestTiebreakNeverPrefersTheFinest:

    def test_tiebreak_prefers_coarser_without_a_target(self):
        from src.rag.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        fine = min(engine.models, key=lambda m: m.pixel_pitch_mm)
        coarse = max(engine.models, key=lambda m: m.pixel_pitch_mm)
        assert fine.pixel_pitch_mm < coarse.pixel_pitch_mm
        assert engine._pitch_tiebreak(fine, {}) > engine._pitch_tiebreak(coarse, {})

    def test_tiebreak_prefers_the_target_when_known(self):
        from src.rag.recommendation_engine import RecommendationEngine

        engine = RecommendationEngine()
        technical = {"pitch_target_mm": 3.0}
        near = min(engine.models, key=lambda m: abs(m.pixel_pitch_mm - 3.0))
        far = min(engine.models, key=lambda m: m.pixel_pitch_mm)
        assert near.pixel_pitch_mm != far.pixel_pitch_mm
        assert engine._pitch_tiebreak(near, technical) < engine._pitch_tiebreak(far, technical)
