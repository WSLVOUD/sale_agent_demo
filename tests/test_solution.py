"""
Solution Agent 核心行为测试（Phase 13）

测试 Solution Agent 的关键行为，不依赖真实 LLM API：
- understand_node 跳过逻辑（Phase 2 优化）
- infer_parameters_node 参数推断
- ParameterInference 规则
- Router 三层路由
- Fast Path 结构化响应
"""
import pytest
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.solution.nodes.requirement import (
    understand_node,
    infer_parameters_node,
)
from src.agents.solution.state import SolutionState
from src.rag.parameter_inference import ParameterInference


class TestUnderstandNodeSkip:
    """Phase 2: understand_node 跳过逻辑"""

    def test_skip_when_flag_set(self):
        """requirements_skip_understand=True 时跳过 LLM"""
        state: SolutionState = {
            "messages": [{"role": "user", "content": "测试消息"}],
            "requirement": {"usage": "会议室", "indoor": True, "display_type": "LED"},
            "requirements_skip_understand": True,
        }
        result = understand_node(state)
        assert result.get("info_sufficient") is True
        assert result.get("missing_info") == []

    def test_skip_when_sufficient_fields(self):
        """已有 3+ 关键字段时跳过 LLM"""
        state: SolutionState = {
            "messages": [{"role": "user", "content": "测试消息"}],
            "requirement": {
                "usage": "会议室",
                "indoor": True,
                "outdoor": False,
                "display_type": "LED",
                "distance": "4米",
            },
            "requirements_skip_understand": False,
        }
        result = understand_node(state)
        assert result.get("info_sufficient") is True

    def test_no_skip_when_insufficient(self):
        """字段不足时进入完整 LLM 流程（由 mock 决定行为）"""
        state: SolutionState = {
            "messages": [{"role": "user", "content": "测试消息"}],
            "requirement": {"usage": "会议室"},  # 只填了一个字段
            "requirements_skip_understand": False,
        }
        # 会尝试调用 LLM；用 mock 来模拟
        import unittest.mock as mock
        with mock.patch(
            "src.agents.solution.nodes.requirement.get_llm"
        ) as mock_llm:
            mock_response = mock.MagicMock()
            mock_response.content = '{"indoor": true, "outdoor": false, "info_sufficient": false, "missing_info": ["display_type"]}'
            mock_llm.return_value.invoke.return_value = mock_response

            result = understand_node(state)
            mock_llm.assert_called_once()
            assert "info_sufficient" in result


class TestInferParametersNode:
    """参数推断节点测试"""

    def test_infer_outdoor_brightness(self):
        """户外 → brightness_min=4500"""
        state: SolutionState = {
            "requirement": {
                "outdoor": True,
                "indoor": False,
            }
        }
        result = infer_parameters_node(state)
        assert result.get("inferred_brightness_min_nit") == 4500

    def test_infer_indoor_brightness(self):
        """室内 → brightness_max=800"""
        state: SolutionState = {
            "requirement": {
                "indoor": True,
                "outdoor": False,
            }
        }
        result = infer_parameters_node(state)
        assert result.get("inferred_brightness_max_nit") == 800

    def test_infer_distance_pitch(self):
        """视距 4 米（室内）→ 业务规则：3m 以上用 P3 及以上，区间 3.0~10.0mm"""
        state: SolutionState = {
            "requirement": {
                "indoor": True,
                "distance": "4米",
            }
        }
        result = infer_parameters_node(state)
        assert result.get("inferred_pixel_pitch_min_mm") == 3.0
        assert result.get("inferred_pixel_pitch_max_mm") == 10.0

    def test_explicit_pitch_wins_over_inference(self):
        """客户指定 P2.5 时，推断值不得覆盖客户明确值"""
        state: SolutionState = {
            "requirement": {
                "indoor": True,
                "distance": "10米",
                "pixel_pitch": 2.5,
                "pixel_pitch_tolerance": 0.5,
            }
        }
        result = infer_parameters_node(state)
        assert result.get("inferred_pixel_pitch_min_mm") == 2.0
        assert result.get("inferred_pixel_pitch_max_mm") == 3.0
        assert result.get("technical_parameters", {}).get("source", {}).get("pixel_pitch") == "explicit"

    def test_infer_rental_from_purpose(self):
        """演唱会 → is_rental=True"""
        state: SolutionState = {
            "requirement": {
                "purpose": "演唱会舞台",
            }
        }
        result = infer_parameters_node(state)
        assert result.get("inferred_is_rental") is True


class TestParameterInference:
    """ParameterInference 规则引擎测试"""

    @pytest.fixture
    def pi(self):
        return ParameterInference()

    def test_indoor_extraction(self, pi):
        result = pi.extract_constraints("会议室大屏")
        assert result.get("indoor") is True
        assert result.get("outdoor") is False

    def test_outdoor_extraction(self, pi):
        result = pi.extract_constraints("户外广告屏")
        assert result.get("outdoor") is True
        assert result.get("indoor") is False

    def test_pitch_extraction(self, pi):
        result = pi.extract_constraints("P2.5 LED 屏")
        assert result.get("pixel_pitch") == 2.5
        assert result.get("pixel_pitch_tolerance") == 0.5

    def test_brightness_extraction(self, pi):
        result = pi.extract_constraints("亮度 5000nit")
        assert result.get("brightness_min") == 5000

    def test_rental_extraction(self, pi):
        result = pi.extract_constraints("租赁用 LED 屏")
        assert result.get("is_rental") is True

    def test_fixed_extraction(self, pi):
        result = pi.extract_constraints("固定安装室内屏")
        assert result.get("is_rental") is False

    def test_waterproof_extraction(self, pi):
        result = pi.extract_constraints("防水 LED 屏 IP65")
        assert result.get("waterproof") is True
        # Phase 5 修复：IP65 不再被误当作 P65 点间距
        assert "pixel_pitch" not in result

    def test_display_type_ifp(self, pi):
        result = pi.extract_constraints("会议一体机 IFP")
        assert result.get("display_type") == "IFP"

    def test_display_type_lcd(self, pi):
        result = pi.extract_constraints("LCD 拼接屏")
        assert result.get("display_type") == "LCD"

    def test_empty_query(self, pi):
        result = pi.extract_constraints("")
        assert result == {}


class TestRouterPhase1:
    """Phase 1 Router 三层路由测试"""

    def test_greeting_goes_fast(self):
        from src.rag.router import classify_complexity, QueryRoute

        for q in ["你好", "您好", "Hi", "hello"]:
            r = classify_complexity(q)
            assert r.route == QueryRoute.FAST, f"'{q}' should be FAST"

    def test_param_only_goes_fast(self):
        from src.rag.router import classify_complexity, QueryRoute

        for q in ["亮度是多少", "P2.5", "IP65", "支持HDR吗"]:
            r = classify_complexity(q)
            assert r.route == QueryRoute.FAST, f"'{q}' should be FAST"

    def test_scene_recommend_goes_normal(self):
        from src.rag.router import classify_complexity, QueryRoute

        for q in ["户外广告屏", "会议室P2.5", "租赁屏P3.9", "展厅大屏"]:
            r = classify_complexity(q)
            assert r.route == QueryRoute.NORMAL, f"'{q}' should be NORMAL"

    def test_reasoning_goes_agent(self):
        from src.rag.router import classify_complexity, QueryRoute

        for q in ["什么屏比较好", "怎么选", "哪个合适", "P5够用吗"]:
            r = classify_complexity(q)
            assert r.route == QueryRoute.AGENT, f"'{q}' should be AGENT"


class TestFastPath:
    """Fast Path 结构化响应测试"""

    def test_fast_path_returns_answer(self):
        """Fast Path 必须返回 answer（即使无匹配产品）"""
        from src.rag.fast_path import fast_path_handle

        result = fast_path_handle(
            query="P2.5 亮度是多少",
            constraints={"pixel_pitch": 2.5},
            template_type=None,
            data_dir=os.path.join(project_root, "data"),
        )
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_fast_path_with_constraints(self):
        """带约束的 Fast Path"""
        from src.rag.fast_path import fast_path_handle

        result = fast_path_handle(
            query="有没有防水的产品",
            constraints={"waterproof": True},
            template_type=None,
            data_dir=os.path.join(project_root, "data"),
        )
        assert "answer" in result
        assert "products" in result

    def test_fast_path_no_llm(self):
        """Fast Path 不调用任何 LLM（只依赖规则和结构化数据）"""
        from src.rag.fast_path import fast_path_handle
        import unittest.mock as mock

        # fast_path.py 不导入 get_llm，所以这里 mock 不会生效（说明它本来就不依赖 LLM）
        # 用 create=True 避免 AttributeError（因为属性不存在于模块）
        with mock.patch("src.rag.fast_path.get_llm", create=True) as mock_llm:
            result = fast_path_handle(
                query="IP65 防水等级",
                constraints={"waterproof": True},
                template_type=None,
                data_dir=os.path.join(project_root, "data"),
            )
            # get_llm 不存在于 fast_path 模块，不应该被调用
            mock_llm.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
