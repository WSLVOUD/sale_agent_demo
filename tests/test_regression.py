"""
端到端回归测试（Phase 13）

测试完整请求链路的关键场景，确保端到端质量不退化：
- 端到端流程（Router → Agent）
- Fast Path 端到端
- 评估数据集基线对比
- 性能回归（延迟 / LLM 调用次数）

运行：
    pytest tests/test_regression.py -v
"""
import pytest
import sys
import os
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestRouterRegression:
    """Router 边界回归测试"""

    def test_router_all_cases_defined(self):
        """Golden Dataset 中的每个 query 都有路由测试"""
        import json

        dataset_path = os.path.join(project_root, "eval", "dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        from tests.test_router import ROUTER_TEST_CASES

        dataset_ids = {item["id"] for item in dataset.get("queries", [])}
        router_ids = {case["id"] for case in ROUTER_TEST_CASES if case["id"].startswith("q")}

        missing = dataset_ids - router_ids
        assert not missing, f"Missing router tests for: {missing}"

    def test_router_no_false_fast_for_scene(self):
        """场景推荐不应该误判为 FAST"""
        from src.rag.router import classify_complexity, QueryRoute

        for q in [
            "户外广告屏",
            "会议室 LED",
            "商场大屏",
            "演唱会舞台屏",
            "租赁临时屏",
        ]:
            r = classify_complexity(q)
            assert r.route != QueryRoute.FAST, (
                f"Scene query '{q}' should not go FAST"
            )

    def test_router_fast_path_precision(self):
        """FAST 路由的精确率：所有被路由为 FAST 的 query 确实都是参数查询"""
        from src.rag.router import classify_complexity, QueryRoute

        fast_cases = [
            "你好",
            "亮度是多少",
            "P2.5",
            "IP65防水",
            "支持HDR吗",
            "质保几年",
            "型号是什么",
        ]

        for q in fast_cases:
            r = classify_complexity(q)
            assert r.route == QueryRoute.FAST, (
                f"'{q}' should be FAST but got {r.route.value}"
            )


class TestFastPathRegression:
    """Fast Path 端到端回归"""

    def test_fast_path_produces_response(self):
        """Fast Path 必须返回非空 answer"""
        from src.rag.fast_path import fast_path_handle

        test_cases = [
            {
                "query": "P2.5 亮度是多少",
                "constraints": {"pixel_pitch": 2.5},
            },
            {
                "query": "有没有防水的 LED 屏",
                "constraints": {"waterproof": True},
            },
            {
                "query": "IP65 防水等级",
                "constraints": {"waterproof": True},
            },
            {
                "query": "亮度多少",
                "constraints": {},
            },
        ]

        for case in test_cases:
            result = fast_path_handle(
                query=case["query"],
                constraints=case["constraints"],
                template_type=None,
                data_dir=os.path.join(project_root, "data"),
            )
            assert "answer" in result, f"Fast Path answer missing for: {case['query']}"
            assert len(result["answer"]) > 0, (
                f"Fast Path answer empty for: {case['query']}"
            )


class TestEvalDatasetBaseline:
    """评估数据集基线测试"""

    def test_eval_dataset_exists(self):
        """验证 Golden Dataset 存在"""
        dataset_path = os.path.join(project_root, "eval", "dataset.json")
        assert os.path.exists(dataset_path), "eval/dataset.json not found"

    def test_eval_dataset_structure(self):
        """Golden Dataset 格式验证"""
        import json

        dataset_path = os.path.join(project_root, "eval", "dataset.json")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        assert "queries" in dataset
        assert len(dataset["queries"]) > 0

        for case in dataset["queries"]:
            assert "id" in case
            assert "query" in case

    def test_eval_report_consistency(self):
        """评估报告与数据集的 query 数量一致性"""
        import json

        dataset_path = os.path.join(project_root, "eval", "dataset.json")
        report_path = os.path.join(project_root, "eval", "report.json")

        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            # 报告中的 case 数应与数据集 query 数一致
            assert report.get("total_cases") == len(dataset["queries"]), (
                f"Mismatch: report has {report.get('total_cases')} cases "
                f"but dataset has {len(dataset['queries'])} queries"
            )


class TestArchitectureIntegrity:
    """架构完整性测试"""

    def test_solution_graph_has_all_nodes(self):
        """Solution Agent graph 包含所有必要节点"""
        from src.agents.solution.graph import build_solution_graph

        graph = build_solution_graph()
        nodes = set(graph.nodes)
        required = {
            "intent_recognition",
            "understand",
            "infer_parameters",
            "retrieve",
            "recommend",
            "reflect",
            "clarify",
        }
        missing = required - nodes
        assert not missing, f"Missing nodes in Solution graph: {missing}"

    def test_sales_graph_nodes_exist(self):
        """Sales Agent graph 节点存在"""
        from src.agents.sales.graph import build_sales_graph

        graph = build_sales_graph()
        nodes = set(graph.nodes)
        assert "classify" in nodes
        assert "requirement_mining" in nodes
        assert "router" in nodes

    def test_router_enum_values(self):
        """Router QueryRoute 枚举值正确"""
        from src.rag.router import QueryRoute

        assert QueryRoute.FAST.value == "fast"
        assert QueryRoute.NORMAL.value == "normal"
        assert QueryRoute.AGENT.value == "agent"

    def test_orchestrator_class_exists(self):
        """编排器类存在"""
        from src.orchestrator import DualAgentOrchestrator

        assert hasattr(DualAgentOrchestrator, "process_message")
        assert hasattr(DualAgentOrchestrator, "__init__")


class TestPhase2Optimization:
    """Phase 2 优化验证"""

    def test_understand_node_respects_skip_flag(self):
        """understand_node 在 requirements_skip_understand=True 时不调用 LLM"""
        from src.agents.solution.nodes.requirement import understand_node
        from src.agents.solution.state import SolutionState
        import unittest.mock as mock

        state: SolutionState = {
            "messages": [],
            "requirement": {"usage": "会议室", "indoor": True},
            "requirements_skip_understand": True,
        }

        with mock.patch(
            "src.agents.solution.nodes.requirement.get_llm"
        ) as mock_llm:
            result = understand_node(state)
            mock_llm.assert_not_called()
            assert result.get("info_sufficient") is True

    def test_parameter_inference_no_llm(self):
        """ParameterInference.extract_constraints 不调用 LLM（纯规则）"""
        from src.rag.parameter_inference import ParameterInference
        import unittest.mock as mock

        pi = ParameterInference()
        with mock.patch(
            "src.rag.parameter_inference.get_llm"
        ) as mock_llm:
            result = pi.extract_constraints("户外广告屏 P3")
            mock_llm.assert_not_called()
            assert result.get("outdoor") is True
            assert result.get("pixel_pitch") == 3.0


class TestPhase11Optimization:
    """Phase 11 性能优化验证"""

    def test_embeddings_cached(self):
        """Embedding 模型只加载一次（缓存生效）"""
        from src.core.embeddings import (
            get_embeddings,
            reset_embeddings_cache,
        )
        import unittest.mock as mock

        reset_embeddings_cache()

        with mock.patch(
            "src.core.embeddings.HuggingFaceEmbeddings"
        ) as mock_cls:
            mock_cls.return_value = mock.MagicMock()

            # 第一次调用
            get_embeddings()
            assert mock_cls.call_count == 1

            # 第二次调用（应该复用缓存）
            get_embeddings()
            assert mock_cls.call_count == 1, (
                "Embedding should be cached after first load"
            )

        reset_embeddings_cache()

    def test_product_filter_no_network(self):
        """ProductFilter 不需要网络（纯内存过滤）"""
        import unittest.mock as mock

        with mock.patch("src.core.llm.get_llm") as mock_llm:
            from src.rag.json_loader import ProductFilter, load_structured_products

            products = load_structured_products(
                os.path.join(project_root, "data")
            )
            pf = ProductFilter(products)
            result = pf.apply(brightness_min=4000, top_k=5)

            mock_llm.assert_not_called()
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
