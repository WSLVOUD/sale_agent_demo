"""
v2.0 Phase 1 / 二十一：产品数据完整性测试。

把 `python -m src.rag.json_loader --validate` 的检查固化成 pytest，
防止后续改数据时静默破坏 Model 级数据质量。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.json_loader import (  # noqa: E402
    load_canonical_models,
    load_structured_products,
    validate_product_data,
)

DATA_DIR = os.path.join(project_root, "data")


@pytest.fixture(scope="module")
def validation():
    return validate_product_data(DATA_DIR)


@pytest.fixture(scope="module")
def models():
    return load_canonical_models(DATA_DIR)


class TestDataValidation:

    def test_validation_passes(self, validation):
        assert validation["ok"] is True, validation["errors"]

    def test_expected_scale(self, validation):
        stats = validation["stats"]
        assert stats["series"] == 9
        assert stats["models"] == 49

    def test_environment_and_installation_split(self, validation):
        stats = validation["stats"]
        assert stats["indoor_series"] == 6
        assert stats["outdoor_series"] == 3
        assert stats["rental_series"] == 3
        assert stats["fixed_series"] == 6


class TestModelLevelData:

    def test_model_ids_unique_and_prefixed(self, models):
        names = [model.model for model in models]
        assert len(names) == len(set(names)), "Model 必须唯一"
        for name in names:
            assert name.startswith("TW")
            assert "-P" in name, f"{name} 看起来不是 Model 级编号"

    def test_every_model_has_engineering_data(self, models):
        for model in models:
            assert model.pixel_pitch_mm > 0
            assert model.module_width_mm > 0 and model.module_height_mm > 0
            assert model.cabinet_width_mm > 0 and model.cabinet_height_mm > 0
            assert model.modules_per_cabinet >= 1
            assert model.installation in ("fixed", "rental")

    def test_modules_per_cabinet_matches_geometry(self, models):
        for model in models:
            per_row = round(model.cabinet_width_mm / model.module_width_mm)
            per_col = round(model.cabinet_height_mm / model.module_height_mm)
            assert per_row * per_col == model.modules_per_cabinet, model.model

    def test_brightness_within_series_range(self, models):
        for model in models:
            assert model.brightness_min_nit <= model.brightness_nit <= model.brightness_max_nit, model.model


class TestPreviouslyKnownDataIssues:
    """v2.0 Phase 1 点名的两个数据问题，防止回归"""

    def test_outdoor_module_resolution_is_not_a_size(self, models):
        """户外系列的 module_resolution 不能是模组物理尺寸（如 320*160）"""
        outdoor = [m for m in models if m.outdoor]
        assert outdoor
        for model in outdoor:
            module_res = str(model.module_resolution)
            size_text = f"{model.module_width_mm:g}*{model.module_height_mm:g}"
            assert module_res != size_text, f"{model.model} 的 module_resolution 仍是物理尺寸"

    def test_tw21_3216_p4_cabinet_resolution_filled(self, models):
        target = next(m for m in models if m.model == "TW21-3216-P4.0")
        assert target.cabinet_resolution == "160*120"

    def test_series_ids_are_stable(self, models):
        expected = {
            "TW11-3216", "TW21-3216", "TW31-COB",
            "TW11-IR", "TW21-IRHD", "TW31-IRHD",
            "TW11-OD", "TW21-OD", "TW31-HOD",
        }
        assert {m.series_id for m in models} == expected


class TestFunctionalFlags:

    def test_flexible_flag_matches_features(self, models):
        flexible = [m.model for m in models if m.flexible]
        assert flexible, "应有柔性屏型号（TW31-IRHD 系列）"
        assert all("TW31-IRHD" in name for name in flexible)
        # 特性标签里必须真的是 flexible
        for model in models:
            if "flexible" in [f.lower() for f in model.features]:
                assert model.flexible is True

    def test_gob_flag_matches_model_name(self, models):
        gob = [m.model for m in models if m.gob]
        assert gob == ["TW11-IR-P1.95(GOB)"]

    def test_series_level_products_still_load(self):
        """Series 级数据（ProductFilter / fast_path 使用）仍需可用"""
        products = load_structured_products(DATA_DIR)
        assert len(products) == 9
        assert all(p.led_data for p in products)
