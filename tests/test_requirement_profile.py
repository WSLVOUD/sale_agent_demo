"""
Phase 14：Requirement Profile / Query Understanding / 硬约束测试（无需 LLM）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile, merge_profiles  # noqa: E402
from src.rag.hard_filter import build_hard_constraints  # noqa: E402
from src.rag.query_understanding import understand_query  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402
from src.rag.parameter_inference import (  # noqa: E402
    brightness_range_for_environment,
    infer_technical_parameters,
    pitch_range_for_distance,
)


class TestQueryUnderstanding:

    def test_english_query_rewrite(self):
        u = understand_query("I need a screen for a conference room around 5m viewing distance")
        assert u.slots.get("environment") == "indoor"
        assert u.slots.get("purpose") == "conference"
        assert u.slots.get("viewing_distance_m") == 5
        assert "indoor" in u.retrieval_query and "conference room" in u.retrieval_query

    def test_multilingual_distance(self):
        for text, expected in (
            ("distancia de visión 15 metros", 15),
            ("расстояние просмотра 20 метров", 20),
            ("视距4米", 4),
            ("5m viewing distance", 5),
        ):
            u = understand_query(text)
            assert u.slots.get("viewing_distance_m") == expected, text

    def test_bare_distance_answer(self):
        """客户回答裸数值时也必须被识别为观看距离（"5m" / "about 5m" / "5米"）"""
        for text in ("5m", "about 5m", "5 meters", "大约5米", "5米", "约5m"):
            u = understand_query(text)
            assert u.slots.get("viewing_distance_m") == 5.0, text

    def test_area_and_size_are_not_distance(self):
        assert understand_query("20平米").slots.get("viewing_distance_m") is None
        u = understand_query("5m x 3m")
        assert u.slots.get("viewing_distance_m") is None
        assert u.slots.get("target_width_mm") == 5000
        assert u.slots.get("target_height_mm") == 3000

    def test_conflict_prefers_first_mention(self):
        assert understand_query("我在户外用，但希望用室内的型号").slots["environment"] == "outdoor"
        assert understand_query("室内使用，但要高亮户外屏").slots["environment"] == "indoor"

    def test_ip65_is_not_pixel_pitch(self):
        u = understand_query("防水 LED 屏 IP65")
        assert u.slots.get("waterproof") is True
        assert "pixel_pitch_mm" not in u.slots


class TestRequirementProfile:

    def test_from_slots_marks_sources(self):
        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference", "viewing_distance_m": 5},
            explicit_keys={"environment", "purpose"},
        )
        assert profile.sources["environment"] == "explicit"
        assert profile.sources["viewing_distance_m"] == "inferred"
        assert profile.is_sufficient() is True

    def test_merge_explicit_wins(self):
        base = RequirementProfile.from_slots(
            {"environment": "indoor", "viewing_distance_m": 3}, explicit_keys={"environment"}
        )
        incoming = RequirementProfile.from_slots(
            {"environment": "outdoor", "viewing_distance_m": 12}, explicit_keys={"viewing_distance_m"}
        )
        merged = base.merge(incoming)
        # environment 已是 explicit，incoming 只是 inferred → 不被覆盖
        assert merged.environment == "indoor"
        assert merged.viewing_distance_m == 12

    def test_from_legacy_maps_usage_and_location(self):
        profile = RequirementProfile.from_legacy(
            {"usage": "会议室", "location_type": "室内", "viewing_distance": "4米"}
        )
        assert profile.purpose == "conference"
        assert profile.environment == "indoor"
        assert profile.viewing_distance_m == 4.0

    def test_recommendation_ready_rules(self):
        # v2.0 Recommendation Ready Gate（见 src/rag/readiness.py）
        assert RequirementProfile.from_slots({"display_type": "LED"}).is_recommendation_ready() is False
        # 客户明确给出技术规格 → 可推荐（v2.0 Case 4）
        assert RequirementProfile.from_slots(
            {"outdoor": True, "pixel_pitch": 2.5}, explicit_keys={"outdoor", "pixel_pitch"}
        ).is_recommendation_ready() is True
        # v2.0 Case 2：室内外 + 场景仍然不够，需要安装方式与观看距离
        assert RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference"},
            explicit_keys={"environment", "purpose"},
        ).is_recommendation_ready() is False
        # v2.0 Case 3：四类核心信息齐备（且均为客户明确给出）
        full = {
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 5,
        }
        assert RequirementProfile.from_slots(full, explicit_keys=set(full)).is_recommendation_ready() is True

    def test_inferred_values_never_open_the_gate(self):
        """v2.0 防呆：规则/上下文推断出来的值不能打开推荐 Gate"""
        guessed = {
            "environment": "indoor", "purpose": "conference",
            "installation": "fixed", "viewing_distance_m": 3.0,
        }
        # 全部标记为 inferred（例如"能容纳15个人"推出来的 3-5 米）
        profile = RequirementProfile.from_slots(guessed)
        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        # 现在 environment 也必须由客户明确说出，不能靠场景推断
        assert set(decision.missing) == {"environment", "installation", "viewing_distance"}

    def test_target_size_roundtrip(self):
        profile = RequirementProfile.from_slots({"target_width_mm": 5000, "target_height_mm": 3000})
        assert profile.target_width_m == 5.0
        assert profile.target_width_mm == 5000
        facts = profile.to_facts()
        assert facts["target_width_mm"] == 5000
        assert facts["target_height_mm"] == 3000


class TestTechnicalInference:

    def test_pitch_table(self):
        assert pitch_range_for_distance(1.5) == (0.6, 1.5)
        assert pitch_range_for_distance(3) == (0.9, 2.0)
        assert pitch_range_for_distance(5) == (1.5, 3.0)
        assert pitch_range_for_distance(12) == (2.5, 5.0)
        assert pitch_range_for_distance(20) == (4.0, 8.0)
        assert pitch_range_for_distance(40) == (6.0, 10.0)

    def test_brightness_table(self):
        assert brightness_range_for_environment("outdoor") == (4500, None)
        assert brightness_range_for_environment("indoor") == (400, 800)

    def test_explicit_pitch_overrides_inference(self):
        technical = infer_technical_parameters(
            {"viewing_distance_m": 10, "pixel_pitch_mm": 2.5, "environment": "indoor"}
        )
        assert technical["pixel_pitch_min_mm"] == 2.0
        assert technical["pixel_pitch_max_mm"] == 3.0
        assert technical["source"]["pixel_pitch"] == "explicit"


class TestHardFilter:

    def test_environment_and_installation_filters(self):
        c = build_hard_constraints({"outdoor": True, "is_rental": True, "display_type": "LED"})
        where = c.chroma_where()
        assert where["outdoor"] is True
        assert where["is_rental"] is True
        assert where["display_type"] == "LED"

    def test_apply_drops_violations(self):
        c = build_hard_constraints({"indoor": True})
        outdoor_item = {"metadata": {"model": "TW11-OD-P5", "indoor": False, "outdoor": True}}
        assert c.apply([outdoor_item]) == []

    def test_accepts_requirement_profile(self):
        profile = RequirementProfile.from_slots({"environment": "outdoor", "installation": "fixed"})
        c = build_hard_constraints(profile)
        assert c.environment == "outdoor"
        assert c.installation == "fixed"
