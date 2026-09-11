"""
ProductFilter 结构化过滤测试（Phase 13）

验证所有硬参数过滤的语义：
- brightness_min / max
- pixel_pitch_min / max
- outdoor / indoor / semi_outdoor
- is_rental
- display_type / exclude_ifp
- waterproof / cob / hdr
- is_splicing

这些测试不需要任何 LLM / 网络，纯 Rule-based。
"""
import pytest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.json_loader import ProductFilter, load_structured_products
from src.models.product import Product


@pytest.fixture(scope="module")
def products() -> list[Product]:
    """Load structured products once for all tests."""
    return load_structured_products(os.path.join(project_root, "data"))


class TestProductFilterBasic:
    """基础过滤"""

    def test_no_filters_returns_all(self, products):
        """无过滤参数时应返回所有产品（按 score 排序）"""
        f = ProductFilter(products)
        result = f.apply(top_k=100)
        assert len(result) == len(products)

    def test_top_k_respects_limit(self, products):
        """top_k 限制结果数量"""
        f = ProductFilter(products)
        result = f.apply(top_k=3)
        assert len(result) <= 3

    def test_empty_filter_returns_top_k(self, products):
        """无任何过滤参数时，返回前 top_k 个产品"""
        f = ProductFilter(products)
        result = f.apply(top_k=5)
        assert len(result) == 5


class TestBrightnessFilter:
    """亮度过滤"""

    def test_brightness_min_filters_low_brightness(self, products):
        """brightness_min=4000 应过滤掉所有亮度 < 4000 的产品"""
        f = ProductFilter(products)
        result = f.apply(brightness_min=4000, top_k=100)
        for p in result:
            bn = p.brightness_nit or 0
            assert bn >= 4000, f"{p.product_id} has brightness {bn} but required >= 4000"

    def test_brightness_max_filters_high_brightness(self, products):
        """brightness_max=800 应过滤掉所有亮度 > 800 的产品（一般保留室内屏）"""
        f = ProductFilter(products)
        result = f.apply(brightness_max=800, top_k=100)
        for p in result:
            bn = p.brightness_nit or 0
            assert bn <= 800, f"{p.product_id} has brightness {bn} but required <= 800"

    def test_brightness_range(self, products):
        """亮度区间 800-5000"""
        f = ProductFilter(products)
        result = f.apply(brightness_min=800, brightness_max=5000, top_k=100)
        for p in result:
            bn = p.brightness_nit or 0
            assert 800 <= bn <= 5000, f"{p.product_id} brightness {bn} out of range"


class TestPixelPitchFilter:
    """点间距过滤"""

    def test_pixel_pitch_max_filters_large_pitch(self, products):
        """pixel_pitch_max=2.0 排除点间距 > 2.0 的产品"""
        f = ProductFilter(products)
        result = f.apply(pixel_pitch_max=2.0, top_k=100)
        for p in result:
            if p.led_data and p.led_data.sub_models:
                pmin = min(s.pixel_pitch_mm for s in p.led_data.sub_models)
                assert pmin <= 2.0, f"{p.product_id} min pitch {pmin} > 2.0"

    def test_pixel_pitch_min_filters_small_pitch(self, products):
        """pixel_pitch_min=5.0 排除 max_pitch < 5.0 的产品"""
        f = ProductFilter(products)
        result = f.apply(pixel_pitch_min=5.0, top_k=100)
        for p in result:
            if p.led_data and p.led_data.sub_models:
                pmax = max(s.pixel_pitch_mm for s in p.led_data.sub_models)
                assert pmax >= 5.0, f"{p.product_id} max pitch {pmax} < 5.0"


class TestEnvironmentFilter:
    """环境过滤"""

    def test_outdoor_filter(self, products):
        """outdoor=True 排除室内专用屏"""
        f = ProductFilter(products)
        result = f.apply(outdoor=True, top_k=100)
        for p in result:
            assert "outdoor" in p.environment

    def test_indoor_filter(self, products):
        """indoor=True 排除户外专用屏"""
        f = ProductFilter(products)
        result = f.apply(indoor=True, top_k=100)
        for p in result:
            assert "indoor" in p.environment

    def test_semi_outdoor_filter(self, products):
        """semi_outdoor=True 排除非半户外屏"""
        f = ProductFilter(products)
        result = f.apply(semi_outdoor=True, top_k=100)
        # 如果 no semi_outdoor products in data, this is OK
        for p in result:
            assert "semi_outdoor" in p.environment


class TestRentalFilter:
    """租赁过滤"""

    def test_is_rental_true(self, products):
        """is_rental=True 只保留租赁屏"""
        f = ProductFilter(products)
        result = f.apply(is_rental=True, top_k=100)
        for p in result:
            assert p.is_rental is True

    def test_is_rental_false(self, products):
        """is_rental=False 只保留固定安装屏"""
        f = ProductFilter(products)
        result = f.apply(is_rental=False, top_k=100)
        for p in result:
            assert p.is_rental is False


class TestDisplayTypeFilter:
    """display_type 过滤"""

    def test_led_only(self, products):
        """display_type=LED 排除 LCD 和 IFP"""
        f = ProductFilter(products)
        result = f.apply(display_type="LED", top_k=100)
        for p in result:
            assert p.display_type == "LED"

    def test_lcd_only(self, products):
        """display_type=LCD 只保留 LCD"""
        f = ProductFilter(products)
        result = f.apply(display_type="LCD", top_k=100)
        for p in result:
            assert p.display_type == "LCD"

    def test_ifp_only(self, products):
        """display_type=IFP 只保留 IFP"""
        f = ProductFilter(products)
        result = f.apply(display_type="IFP", top_k=100)
        for p in result:
            assert p.display_type == "IFP"

    def test_exclude_ifp(self, products):
        """exclude_ifp=True 排除所有 IFP"""
        f = ProductFilter(products)
        result = f.apply(exclude_ifp=True, top_k=100)
        for p in result:
            assert p.display_type != "IFP"


class TestFeatureFilters:
    """特性 flag 过滤"""

    def test_waterproof(self, products):
        """waterproof=True 只保留防水屏"""
        f = ProductFilter(products)
        result = f.apply(waterproof=True, top_k=100)
        for p in result:
            if p.led_data:
                assert p.led_data.waterproof is True

    def test_cob(self, products):
        """cob=True 只保留 COB 工艺屏"""
        f = ProductFilter(products)
        result = f.apply(cob=True, top_k=100)
        for p in result:
            if p.led_data:
                assert p.led_data.cob is True

    def test_hdr(self, products):
        """hdr=True 只保留支持 HDR 的屏"""
        f = ProductFilter(products)
        result = f.apply(hdr=True, top_k=100)
        for p in result:
            if p.led_data:
                assert p.led_data.hdr is True

    def test_is_splicing(self, products):
        """is_splicing=True 只保留 LCD 拼接屏"""
        f = ProductFilter(products)
        result = f.apply(is_splicing=True, top_k=100)
        for p in result:
            if p.lcd_data:
                assert p.lcd_data.is_splicing is True


class TestCombinedFilters:
    """组合过滤"""

    def test_outdoor_plus_brightness(self, products):
        """outdoor + brightness_min=4500 应返回户外高亮产品"""
        f = ProductFilter(products)
        result = f.apply(outdoor=True, brightness_min=4500, top_k=100)
        for p in result:
            assert "outdoor" in p.environment
            assert (p.brightness_nit or 0) >= 4500

    def test_indoor_plus_pitch(self, products):
        """indoor + pixel_pitch_max=2.0 应返回室内小间距屏"""
        f = ProductFilter(products)
        result = f.apply(indoor=True, pixel_pitch_max=2.0, top_k=100)
        for p in result:
            assert "indoor" in p.environment
            if p.led_data and p.led_data.sub_models:
                pmin = min(s.pixel_pitch_mm for s in p.led_data.sub_models)
                assert pmin <= 2.0

    def test_complex_filter(self, products):
        """复合条件：租赁 + 户外 + 点间距 4-10mm"""
        f = ProductFilter(products)
        result = f.apply(
            is_rental=True,
            outdoor=True,
            pixel_pitch_min=4.0,
            pixel_pitch_max=10.0,
            top_k=10,
        )
        for p in result:
            assert p.is_rental is True
            assert "outdoor" in p.environment
            if p.led_data and p.led_data.sub_models:
                pmax = max(s.pixel_pitch_mm for s in p.led_data.sub_models)
                pmin = min(s.pixel_pitch_mm for s in p.led_data.sub_models)
                assert pmax >= 4.0 and pmin <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
