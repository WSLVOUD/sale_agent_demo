"""v2.3 §17：Invariant Tests —— 不变量必须永远成立。

    1. 没有合法来源的 Pitch 不得进入推荐（Provenance Guard）
    2. 存在冲突时不得推荐
    3. INFERRED 不得自动变成 CONFIRMED
    4. Calculation 未 Ready 不得执行完整箱体计算
    5. 产品数据只有一个来源（json_loader）
    6. C 类（语义）说法不得直接产生具体 P 值
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import (  # noqa: E402
    check_provenance,
    classify_requirement,
    detect_engineering_conflicts,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)
from src.rag.recommendation_coordinator import (  # noqa: E402
    CONFLICT,
    RECOMMENDED,
    REJECTED,
    RecommendationCoordinator,
)
from src.rag.recommendation_service import RecommendationService  # noqa: E402


def _profile(slots: dict, confirmed: bool = True) -> RequirementProfile:
    return RequirementProfile.from_slots(
        slots, explicit_keys=set(slots) if confirmed else set()
    )


class TestProvenanceInvariant:

    def test_pitch_without_source_is_rejected(self):
        """LLM 凭空给一个 P 值（没有 sources）→ 不允许推荐。"""
        profile = _profile({
            "environment": "indoor", "installation": "fixed",
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        profile.pixel_pitch_mm = 1.2          # 值有，来源没有
        profile.sources.pop("pixel_pitch_mm", None)
        profile.field_decisions.pop("pixel_pitch", None)

        report = check_provenance(profile, {"pixel_pitch_min_mm": None})
        assert report.ok is False
        assert "pixel_pitch_mm" in report.illegal + report.missing

        outcome = RecommendationCoordinator().recommend(profile)
        assert outcome.status == REJECTED
        assert outcome.recommendations == []
        assert outcome.audit.get("selected_model") == ""

    def test_derived_pitch_carries_formula_id(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed",
            "target_width_mm": 10000, "target_height_mm": 5000,
        })
        from src.rag.parameter_inference import infer_technical_parameters

        technical = infer_technical_parameters(profile.to_facts())
        report = check_provenance(profile, technical)
        assert report.ok is True
        entry = report.entries["pixel_pitch_mm"]
        assert entry.source == "derived"
        assert entry.formula_id, "推导值必须带公式编号"

    def test_customer_pitch_keeps_customer_source(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 2.5,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        report = check_provenance(profile, {})
        assert report.entries["pixel_pitch_mm"].source == "customer"
        assert report.ok is True


class TestConflictInvariant:

    def test_conflict_blocks_recommendation(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed",
            "target_width_mm": 10000, "target_height_mm": 5000,   # 50㎡ 屏
            "room_area_sqm": 25,                                   # 25㎡ 房间
        }, confirmed=False)
        conflicts = detect_engineering_conflicts(profile)
        assert conflicts, "屏比房间大必须被识别为冲突"

        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        assert decision.status == CONFLICT

        result = RecommendationService().recommend(profile)
        assert result["recommendation_status"] == "NEED_CLARIFICATION"
        assert result["coordinator_status"] == CONFLICT
        assert result["recommendations"] == []
        assert result["conflicts"]

    def test_indoor_coarse_pitch_conflict(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 10.0,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        assert detect_engineering_conflicts(profile)

    def test_no_conflict_for_consistent_profile(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 10000, "target_height_mm": 5000, "room_area_sqm": 200,
        })
        assert detect_engineering_conflicts(profile) == []


class TestInferredStaysInferred:

    def test_scenario_derived_environment_is_not_confirmed(self):
        from src.core.requirement_extractor import RequirementExtractor

        profile = RequirementExtractor().extract(
            "we need a screen for a church", use_llm=False, session_id=""
        )
        assert profile.environment == "indoor"
        assert profile.sources["environment"] == "scenario_derived"
        assert profile.slot_source("environment") not in ("explicit", "confirmed")
        assert check_provenance(profile, {}).entries["environment"].source != "customer"
        # 场景推断出来的环境不能直接当"客户确认"用：provenance 记成 derived/inferred
        assert check_provenance(profile, {}).entries["environment"].source in (
            "derived", "inferred", "vision",
        )

    def test_inferred_value_does_not_open_the_gate(self):
        profile = RequirementProfile.from_slots({"environment": "indoor"})
        assert check_recommendation_ready(profile).ready is False


class TestCalculationInvariant:

    def test_calculation_not_ready_without_size(self):
        """两条 Gate 相互独立：尺寸延后时推荐可以放行，但计算挂起。

        （客户口径：尺寸属于硬性条件，没问过时会继续问；只有客户明确给不出来
        → DEFERRED 时，才出现"推荐 READY / 计算 DEFERRED"这种组合。）
        """
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
        })
        profile.record_ask("size")
        profile.record_ask("size")
        profile.mark_decision("size", "unknown")     # 客户两次都给不出来 → DEFERRED

        assert check_recommendation_ready(profile).ready is True
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert decision.status == "DEFERRED"
        assert decision.next_question is None

    def test_delegated_size_without_distance_defers_calculation(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
        })
        profile.mark_decision("size", "delegated")
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert decision.status in ("CONTINUE_ASKING", "DEFERRED")


class TestProductDataSourceInvariant:

    def test_only_json_loader_defines_product_data_reads(self):
        """产品数据只有一个来源：src/rag/json_loader.py。"""
        offenders = []
        for dirpath, _dirs, files in os.walk(os.path.join(project_root, "src")):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                if path.endswith(os.path.join("rag", "json_loader.py")):
                    continue
                for line in open(path, encoding="utf-8"):
                    mentions_data = "led_products" in line or "_products.json" in line
                    reads_it = any(
                        call in line for call in ("open(", "json.load", "read_text")
                    )
                    if mentions_data and reads_it:
                        offenders.append(f"{path}: {line.strip()[:80]}")
        assert not offenders, f"产品数据必须在 json_loader 里加载：{offenders}"

    def test_engine_gets_models_from_the_loader(self):
        from src.config import config
        from src.rag.json_loader import load_canonical_models
        from src.rag.recommendation_engine import RecommendationEngine

        models = load_canonical_models(config.DATA_DIR)
        engine = RecommendationEngine(models=models)
        assert len(engine.models) == len(models) > 0


class TestSemanticClassInvariant:

    def test_semantic_only_statement_has_no_physical_value(self):
        result = classify_requirement("I want a better effect, whatever you think is best")
        assert result.semantic
        assert not result.physical
        assert result.only_semantic is True

    def test_physical_statement_is_classified(self):
        result = classify_requirement("about 100 people, room 25 sqm, 10x5m screen")
        assert result.physical

    def test_semantic_text_never_yields_a_concrete_pitch(self):
        """C 类说法只能走"澄清 / 保守默认 + INFERRED"，不能直接产生 P 值。"""
        from src.rag.parameter_inference import infer_technical_parameters

        technical = infer_technical_parameters({
            "environment": "indoor", "installation": "fixed",
        })
        source = technical.get("source", {}).get("pixel_pitch")
        assert technical["pixel_pitch_min_mm"] is not None
        assert source in (
            "fallback_environment_default", "inferred_from_environment_distance",
        ), source
