"""计划 2.0（Phase 12-1）：Solution 不得自己重建需求。

计划原文：

    步骤 2  RequirementProfile = 唯一业务需求来源；Solution 不允许重新建立自己的需求模型
    步骤 3  legacy requirement 改为**只读兼容**（只能读取 / 转换，不能修改、重新推导、重新抽取）
    步骤 4  关闭 history → requirement 重新抽取；没有 profile 时返回 REQUIREMENT_NOT_READY

核对到的两处"第二套需求逻辑"（都在 `solution/runner.py::_build_initial_state`）：

    · 有旧 requirements 字典、没有 profile → 按**中文关键词**重新推断室内外
      （outdoor_usages / indoor_usages 两份关键词表）
    · 两者都没有 → **逐条扫 history** 调 `_extract_requirements` 重新抽取需求

本轮把这两条路关闭：没有 profile 就只原样带上调用方给的旧字典，并标记
`REQUIREMENT_NOT_READY`，交由上层处理。
"""
import os
import re
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _read(rel_path: str) -> str:
    with open(os.path.join(project_root, rel_path), encoding="utf-8") as handle:
        return handle.read()


def _code_only(text: str) -> str:
    """去掉注释行，只留代码（注释里提到旧实现不算违规）。"""
    return "\n".join(line.split("#", 1)[0] for line in str(text or "").splitlines())


class TestSolutionDoesNotRebuildRequirements:

    def _solution_runner(self):
        from src.agents.solution.runner import SolutionAgentRunner

        # documents=None → 不加载 sparse/bm25 索引（只测 state 构造，不跑检索）
        return SolutionAgentRunner(vectorstore=None)

    def test_no_profile_means_requirement_not_ready(self):
        """没有 RequirementProfile → 明确标记未就绪，不再自己抽需求。"""
        runner = self._solution_runner()
        state = runner._build_initial_state(
            "i need an indoor led screen 3m x 5m",
            history=[
                {"role": "user", "content": "we need an outdoor church screen"},
                {"role": "assistant", "content": "sure"},
            ],
            requirements=None,
            additional_requirements=None,
            profile=None,
            session_id="sol-no-profile",
            intent="need_query",
        )

        assert state.get("requirement") == {}, state.get("requirement")
        assert state.get("requirement_not_ready") is True

    def test_legacy_dict_is_passed_through_without_re_derivation(self):
        """调用方给了旧字典、但没有 profile → 原样带过，不做关键词推断。"""
        runner = self._solution_runner()
        legacy = {"display_type": "LED", "usage": "演唱会"}  # 以前会被推成 outdoor

        state = runner._build_initial_state(
            "whatever",
            history=[],
            requirements=legacy,
            additional_requirements=None,
            profile=None,
            session_id="sol-legacy",
            intent="need_query",
        )

        assert state.get("requirement") == legacy, state.get("requirement")
        assert state.get("requirement_not_ready") is True

    def test_profile_path_is_unchanged(self):
        """正常链路：有 profile → 仍然是它的只读投影。"""
        from src.models.requirement import RequirementProfile

        runner = self._solution_runner()
        slots = {"display_type": "LED", "environment": "indoor"}
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))

        state = runner._build_initial_state(
            "whatever",
            history=[],
            requirements={"display_type": "LED"},
            additional_requirements=None,
            profile=profile,
            session_id="sol-profile",
            intent="need_query",
        )

        assert state.get("requirement"), "投影不能是空的"
        assert state.get("requirement_not_ready") in (None, False)


class TestNoHistoryExtractionLeft:

    def test_solution_runner_has_no_history_requirement_extraction(self):
        code = _code_only(_read("src/agents/solution/runner.py"))
        assert "_extract_requirements(" not in code, "Solution 不该再从 history 抽取需求"
        assert "outdoor_usages" not in code and "indoor_usages" not in code, (
            "中文关键词重新推断室内外应已移除"
        )

    def test_production_callers_always_pass_a_profile(self):
        """所有生产调用点都必须把 profile 传进来（否则会退化成 NOT_READY）。"""
        for rel in (
            "src/agents/sales/nodes/router.py",
            "src/orchestrator.py",
        ):
            text = _read(rel)
            for call in re.finditer(r"(solution_runner|solution_agent)\.run\(", text):
                chunk = text[call.start() : call.start() + 900]
                assert "profile=" in chunk, f"{rel} 的调用点没传 profile"
