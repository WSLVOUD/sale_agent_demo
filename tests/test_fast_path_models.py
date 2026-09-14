"""
回归：Fast Path 必须返回 **Model 级**结果，不能返回 Series。

修复前：`有没有防水的产品` → "Here are my recommendations: TW31-COB series"
（客户买的是型号，Series 只是产品族）；`P2.5的点间距是多少` 也会把系列当成推荐。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import config  # noqa: E402
from src.rag.fast_path import fast_path_handle  # noqa: E402
from src.rag.router import classify_complexity  # noqa: E402


def _fast(query: str) -> dict:
    routing = classify_complexity(query)
    return fast_path_handle(
        query=query,
        constraints=routing.inferred_constraints,
        template_type=None,
        data_dir=config.DATA_DIR,
    )


class TestFastPathReturnsModels:

    @pytest.mark.parametrize("query", [
        "有没有防水的产品",
        "有没有租赁的屏",
        "P2.5的点间距是多少",
    ])
    def test_products_are_model_level(self, query):
        result = _fast(query)
        products = result.get("products") or []
        assert products, f"{query} 应返回匹配型号"
        for item in products:
            model = item.get("model") or item.get("product_id") or ""
            assert "-P" in model, f"返回的不是 Model 级编号: {item}"
            assert not model.endswith("series")

    def test_answer_text_uses_model_names(self):
        result = _fast("有没有防水的产品")
        assert "series" not in result["answer"].lower(), result["answer"]
        assert "-P" in result["answer"]

    def test_results_span_multiple_series(self):
        """同类需求应给出跨系列的可选项，而不是同系列相邻型号"""
        result = _fast("有没有防水的产品")
        series = {item.get("series_id") for item in result.get("products") or []}
        assert len(series) >= 2, f"应跨系列给选项，实际: {series}"

    def test_feature_filters_are_respected(self):
        rental = _fast("有没有租赁的屏")
        for item in rental.get("products") or []:
            assert item.get("installation") == "rental"

        waterproof = _fast("有没有防水的产品")
        from src.rag.json_loader import canonical_model_index

        index = canonical_model_index(config.DATA_DIR)
        for item in waterproof.get("products") or []:
            assert index[item["model"]].waterproof is True
