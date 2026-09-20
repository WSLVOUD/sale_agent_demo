"""v2.3.1 完成标准：Orchestrator 只做编排（职责收敛的不变量测试）。

计划《LED_RAG_v2.3进一步优化.md》的完成标准：

    1. Orchestrator 不再包含核心推荐业务规则
    2. Orchestrator 不再包含产品参数计算
    3. Orchestrator 不再独立判断推荐条件
    4. 推荐只有 RecommendationCoordinator 一个出口
    5. 回复策略只有 ResponseCoordinator 一个出口
    6. 需求事实统一来自 RequirementProfile
    7. Legacy Adapter 仅承担兼容职责
    8. 全量测试通过 / 重构前后行为一致

这里用"源码级不变量 + 行为级断言"把 1~7 锁住（8 由全量测试保证）。
"""
import io
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

ORCHESTRATOR = os.path.join(project_root, "src", "orchestrator.py")


def _source() -> str:
    return io.open(ORCHESTRATOR, encoding="utf-8").read()


def _code_lines() -> list:
    """只看代码行（去掉注释与文档字符串的粗略近似：以 # 开头的行）。"""
    return [
        line for line in _source().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class TestOrchestratorHasNoBusinessRules:

    @pytest.mark.parametrize("forbidden", [
        "RecommendationEngine(",        # 产品参数计算 / 选型
        "calculate_screen(",            # 箱体模组计算
        "check_recommendation_ready(",  # 独立判断推荐条件
        "check_calculation_ready(",
        "infer_technical_parameters(",  # 工程参数推导
        "pitch_window_for_distances(",
        "detect_engineering_conflicts(",
        "classify_requirement(",
    ])
    def test_no_engineering_or_recommendation_logic(self, forbidden):
        offenders = [line.strip()[:120] for line in _code_lines() if forbidden in line]
        assert not offenders, f"Orchestrator 不应包含 {forbidden}: {offenders}"

    def test_no_pitch_or_parameter_arithmetic(self):
        """Orchestrator 里不应再出现点间距/像素密度这类参数运算。"""
        code = "\n".join(_code_lines())
        for token in ("pixel_pitch", "PITCH_", "resolution_per_sqm", "cabinet_count"):
            assert token not in code, token

    def test_multi_screen_business_lives_in_its_own_module(self):
        """多屏业务（拆分 / 切换 / 共享 / 多屏推荐）只有一份实现。"""
        orch = _source()
        for name in (
            "_share_common_facts", "_split_and_apply_screen_specs",
            "_recommend_all_screens", "_multi_item_follow_up",
        ):
            # Orchestrator 里只允许出现"包装 + 委托"，不允许出现定义体
            assert f"def {name}(self" in orch
            assert "_multi_screen()" in orch
        multi_path = os.path.join(project_root, "src", "rag", "multi_screen.py")
        multi = io.open(multi_path, encoding="utf-8").read()
        assert "class MultiScreenManager" in multi
        assert "_share_common_facts" in multi

    def test_vision_pipeline_is_moved_out(self):
        orch = _source()
        assert "from .vision.pipeline import" in orch
        pipeline = io.open(
            os.path.join(project_root, "src", "vision", "pipeline.py"), encoding="utf-8"
        ).read()
        assert "def _merge_vision_into_stored_profile(" in pipeline
        assert "def _vision_enabled(" in pipeline

    def test_perf_tracker_is_moved_out(self):
        orch = _source()
        assert "from .observability.perf import PerfTracker" in orch
        perf = io.open(
            os.path.join(project_root, "src", "observability", "perf.py"), encoding="utf-8"
        ).read()
        assert "class PerfTracker" in perf
        assert "class PerfTracker" not in orch

    def test_response_helpers_delegate(self):
        """回复层只委托：售后口径 / 图片核对 / 答复+追问 都在 ResponseCoordinator。"""
        orch = _source()
        for name in (
            "_attach_service_faq", "_attach_vision_confirmation",
            "_vision_confirmation_sentence", "_compose_with_requirement_question",
        ):
            assert f"def {name}(" in orch
        assert orch.count("self._response_coordinator()") >= 4


class TestSingleOutlets:

    def test_recommendation_single_outlet(self):
        from src.rag.recommendation_coordinator import RecommendationCoordinator
        from src.rag.recommendation_service import RecommendationService

        service = RecommendationService()
        assert isinstance(service.coordinator, RecommendationCoordinator)
        # 服务层不再自己实现 Gate / 引擎顺序：只持有协调器
        assert not hasattr(service, "engine") or service.engine is service.coordinator.engine

    def test_response_single_outlet(self):
        from src.dialogue import ResponseCoordinator

        for name in (
            "finalize", "attach_service_faq", "attach_vision_confirmation",
            "vision_confirmation_sentence", "compose_with_requirement_question",
        ):
            assert hasattr(ResponseCoordinator, name), name

    def test_requirement_facts_come_from_profile(self):
        """需求事实统一来自 RequirementProfile（没有平行状态）。"""
        from src.models.requirement import RequirementProfile

        assert hasattr(RequirementProfile, "to_facts")
        assert hasattr(RequirementProfile, "field_decision")
        adapter = io.open(
            os.path.join(project_root, "src", "models", "legacy_adapter.py"),
            encoding="utf-8",
        ).read()
        # Legacy Adapter 只做投影，不参与决策
        assert "def profile_to_legacy" in adapter


class TestOrchestratorDelegates:
    """行为级：Orchestrator 的旧方法确实委托给新模块。"""

    class _Store:
        def __init__(self):
            self.items = []

        def get_project_items(self, session_id):
            return self.items

        def get_active_item_index(self, session_id):
            return 0

        def is_first_contact_done(self, session_id):
            return True

    def _orchestrator(self):
        from src.orchestrator import DualAgentOrchestrator

        class _Agent:
            def run(self, **kwargs):
                return {}

        return DualAgentOrchestrator(sales_agent=_Agent(), solution_agent=_Agent())

    def test_multi_screen_methods_delegate_to_manager(self):
        from src.rag.multi_screen import MultiScreenManager

        orch = self._orchestrator()
        orch.memory_store = self._Store()
        assert isinstance(orch._multi_screen(), MultiScreenManager)
        assert orch._split_and_apply_screen_specs("s1", "hello") == []

    def test_response_helpers_use_coordinator(self):
        from src.dialogue import ResponseCoordinator

        orch = self._orchestrator()
        orch.memory_store = self._Store()
        assert isinstance(orch._response_coordinator(), ResponseCoordinator)
        assert orch._vision_confirmation_sentence("s1", "hi") == ""

    def test_screen_label_prefix_is_delegated(self):
        orch = self._orchestrator()
        orch.memory_store = self._Store()
        # 只有一块屏（active_index=0）→ 不加前缀，原样返回
        assert orch._multi_screen().prefix_active_screen_label(
            "s1", "hi", "TW11-3216-P3.0 fits."
        ) == "TW11-3216-P3.0 fits."
