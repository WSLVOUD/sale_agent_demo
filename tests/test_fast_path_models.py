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


class TestFastPathDoesNotBypassSceneContext:
    """回归：已采集到场景需求的会话，不能被判成 FAST。

    实测 bug：客户已经说了"教堂 + 室内 + 5m"，后面回一句
    "video mainly. we care about price"（命中 price 参数词）被判成 FAST，
    绕过"环境 + 观看距离 → 点间距"规则表，把室内 5m 推成 P0.7H，
    而且 fast path 不会追问屏体尺寸。
    """

    @staticmethod
    def _profile():
        from src.models.requirement import RequirementProfile

        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "church",
            "installation": "fixed", "viewing_distance_m": 5,
        }
        return RequirementProfile.from_slots(slots, explicit_keys=set(slots))

    def test_bare_param_query_still_fast_without_context(self):
        """没有场景上下文时，"纯参数词"仍然是 FAST（保持老行为）。"""
        assert classify_complexity("we care about price").route.value == "fast"

    def test_scene_context_never_takes_fast_path(self):
        from src.models.legacy_adapter import profile_to_legacy

        requirements = profile_to_legacy(self._profile())
        routing = classify_complexity(
            "video mainly. we care about price", existing_requirements=requirements
        )
        assert routing.route.value != "fast", routing
        assert routing.route.value == "normal", routing

    def test_solution_runner_derives_requirements_from_profile(self):
        """Orchestrator 只传 profile 时，路由也必须看到场景上下文。"""
        from src.agents.solution.runner import routing_requirements

        assert routing_requirements(None, None) is None

        derived = routing_requirements(None, self._profile())
        assert derived, "应当从 profile 派生出一份 legacy 需求视图"
        assert derived.get("usage") == "church"
        assert derived.get("location_type") == "室内"

        # 调用方已经传了 requirements → 原样使用，不被 profile 覆盖
        explicit = {"usage": "showroom"}
        assert routing_requirements(explicit, self._profile()) is explicit
