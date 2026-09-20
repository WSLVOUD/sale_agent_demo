"""v2.5 Phase 4 / 5：实际拼接分辨率 + 接近程度（EXACT/NEAR_MATCH/…）。"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engineering import (  # noqa: E402
    actual_pixel_resolution,
    fit_resolution,
    min_achievable_deviation,
    stitching_geometry,
)


class TestStitchingGeometry:

    def test_cabinet_grid(self):
        geometry = stitching_geometry(
            target_width_mm=4000, target_height_mm=2000,
            cabinet_width_mm=640, cabinet_height_mm=480, modules_per_cabinet=6,
        )
        assert geometry["columns"] == 6
        assert geometry["rows"] == 4
        assert geometry["cabinet_count"] == 24
        assert geometry["module_count"] == 144
        assert geometry["actual_width_mm"] == pytest.approx(3840)
        assert geometry["actual_height_mm"] == pytest.approx(1920)

    def test_actual_pixels_from_pitch(self):
        actual = actual_pixel_resolution(
            actual_width_mm=3840, actual_height_mm=1920, pixel_pitch_mm=2.5
        )
        assert actual == (1536, 768)


class TestFitLevels:

    def test_exact(self):
        fit = fit_resolution((3840, 2160), (3840, 2160))
        assert fit.fit_level == "EXACT" and fit.acceptable is True

    def test_near_match(self):
        fit = fit_resolution((3840, 2160), (3780, 2160))
        assert fit.horizontal_deviation == pytest.approx(0.0156, abs=0.001)
        assert fit.fit_level == "NEAR_MATCH" and fit.acceptable is True

    def test_acceptable_and_not_acceptable(self):
        # 比例仍是 16:9、只是像素数略少 → ACCEPTABLE
        assert fit_resolution((3840, 2160), (3600, 2025)).fit_level == "ACCEPTABLE"
        assert fit_resolution((3840, 2160), (3200, 1800)).fit_level == "NOT_ACCEPTABLE"
        assert fit_resolution((3840, 2160), (1920, 1080)).fit_level == "IMPOSSIBLE"

    def test_customer_tolerance_is_respected(self):
        strict = fit_resolution((3840, 2160), (3780, 2160), tolerance=0.001)
        assert strict.fit_level == "ACCEPTABLE"

    def test_geometry_aware_minimum_deviation(self):
        """半个模组都做不到更近时，算 NEAR_MATCH（拼到最接近了），不算失败。"""
        minimum = min_achievable_deviation(
            target_width_px=3840, target_height_px=2160,
            module_width_px=320, module_height_px=160,
        )
        assert minimum and minimum > 0.02
        fit = fit_resolution((3840, 2160), (3780, 2160), min_achievable_deviation=minimum)
        assert fit.fit_level == "NEAR_MATCH"

    def test_missing_inputs_are_impossible(self):
        assert fit_resolution(None, (3840, 2160)).fit_level == "IMPOSSIBLE"
        assert fit_resolution((3840, 2160), None).fit_level == "IMPOSSIBLE"
