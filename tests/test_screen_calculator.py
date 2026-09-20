"""
Phase 14：Screen Calculator 与确定性校验测试。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.validation import validate_recommendation  # noqa: E402
from src.tools.screen_calculator import (  # noqa: E402
    calculate_screen,
    estimate_screen,
    format_screen_spec,
)


class TestScreenCalculator:

    def test_plan_document_example(self):
        """计划文档示例：TW21-3216-P2.5 + 5m x 3m"""
        calc = calculate_screen("TW21-3216-P2.5", 5000, 3000)
        assert calc["columns"] == 8
        assert calc["rows"] == 7
        assert calc["cabinet_count"] == 56
        assert calc["actual_width_mm"] == 5120
        assert calc["actual_height_mm"] == 3360
        assert calc["modules_per_cabinet"] == 6
        assert calc["total_modules"] == 336

    def test_exact_division(self):
        calc = calculate_screen("TW21-3216-P2.5", 6400, 4800)
        assert (calc["columns"], calc["rows"]) == (10, 10)
        assert calc["actual_width_mm"] == 6400

    def test_fractional_cabinet_size(self):
        calc = calculate_screen("TW31-COB-P1.2H", 4000, 2000)
        assert (calc["columns"], calc["rows"]) == (7, 6)
        assert calc["cabinet_count"] == 42
        assert calc["actual_height_mm"] == 2025.0
        assert calc["modules_per_cabinet"] == 8
        assert calc["total_modules"] == 336

    def test_outdoor_module_count(self):
        calc = calculate_screen("TW11-OD-P6", 8000, 4000)
        assert calc["cabinet_count"] == 45
        assert calc["modules_per_cabinet"] == 18
        assert calc["total_modules"] == 810

    def test_meter_helper_and_format(self):
        calc = estimate_screen("TW11-IR-P2.6", 3, 3)
        assert calc["cabinet_count"] == 36
        text = format_screen_spec(calc)
        # 客户口径：横拼/竖拼两种排布，格式化文本要标出是哪种
        assert "36 cabinets" in text and "Horizontal tiling" in text
        from src.tools.screen_calculator import calculate_screen

        portrait = format_screen_spec(
            calculate_screen("TW11-IR-P2.6", 3000, 3000, orientation="portrait")
        )
        assert "Vertical tiling" in portrait

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            calculate_screen("TW99-999-P1.0", 1000, 1000)

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            calculate_screen("TW21-3216-P2.5", 0, 1000)


class TestDeterministicValidation:

    @pytest.fixture(scope="class")
    def selection(self):
        from src.rag.recommendation_engine import RecommendationEngine

        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "purpose": "conference", "viewing_distance_m": 5,
             "target_width_mm": 5000, "target_height_mm": 3000}
        )
        return profile, RecommendationEngine().recommend(profile=profile)

    def test_clean_recommendation_scores_full(self, selection):
        profile, result = selection
        model = result["recommendations"][0]["model"]
        calc = calculate_screen(model, 5000, 3000)
        report = validate_recommendation(
            f"The {model} is a great fit for your conference room.", [], profile, result, calc
        )
        assert report["score"] == 10.0
        assert report["checks"]["model_exists"] is True
        assert report["checks"]["no_fabricated_params"] is True

    def test_fabricated_specs_are_detected(self, selection):
        profile, result = selection
        report = validate_recommendation(
            "Our TW21-3216-P2.5 delivers 12000nit with P1.0 pitch.", [], profile, result, None
        )
        assert report["checks"]["no_fabricated_params"] is False
        assert report["score"] < 10.0

    def test_wrong_cabinet_count_is_detected(self, selection):
        profile, result = selection
        model = result["recommendations"][0]["model"]
        calc = calculate_screen(model, 5000, 3000)
        calc["cabinet_count"] = 100
        calc["total_modules"] = 999
        report = validate_recommendation("Recommended configuration.", [], profile, result, calc)
        assert report["checks"]["cabinet_count_correct"] is False
        assert report["checks"]["module_count_correct"] is False
