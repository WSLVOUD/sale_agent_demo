"""
Phase 12/14：LLM 调用预算回归测试。

计划文档 Phase 12 声称"推荐链路的 LLM 调用从约 9~11 次降到 3~4 次"。
这里用计数假 LLM 把该结论固化成断言，避免后续改动悄悄把 LLM 又加回来。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class _CountingLLM:
    """最小可用的假 LLM：只记录调用次数。"""

    def __init__(self, content: str = "TW21-3216-P2.5 fits your conference room well."):
        self.calls = 0
        self._content = content

    def invoke(self, prompt, *args, **kwargs):
        self.calls += 1

        class _Response:
            content = self._content

        return _Response()


def _state(**overrides):
    from src.models.requirement import RequirementProfile

    slots = {
        "environment": "indoor",
        "purpose": "conference",
        "viewing_distance_m": 5,
        "installation": "fixed",
    }
    state = {
        "requirement": {"usage": "会议室"},
        "understood_slots": slots,
        # 推荐链路要求客户明确说出全部关键字段（confirmed）
        "requirement_profile": RequirementProfile.from_slots(
            slots, explicit_keys=set(slots)
        ),
        "products": [],
        "messages": [],
        "current_message": "会议室5米视距用，要LED屏",
    }
    state.update(overrides)
    # 从最终的 understood_slots 构建 confirmed profile（含尺寸覆盖）
    final_slots = dict(state.get("understood_slots") or {})
    state["requirement_profile"] = RequirementProfile.from_slots(
        final_slots, explicit_keys=set(final_slots)
    )
    return state


class TestLlmBudget:

    def test_parameter_inference_never_calls_llm(self, monkeypatch):
        """工程参数推断必须是纯 Python（LLM 不参与工程计算）"""
        import src.rag.parameter_inference as pi

        fake = _CountingLLM()
        monkeypatch.setattr(pi, "get_llm", lambda *a, **k: fake)

        result = pi.parameter_inference_node(_state())

        assert fake.calls == 0
        assert result["inferred_pixel_pitch_min_mm"] == 1.5
        assert result["inferred_pixel_pitch_max_mm"] == 3.0

    def test_reflection_never_calls_llm(self, monkeypatch):
        """Reflection 已改为确定性校验，不应再调用 LLM"""
        import src.agents.solution.nodes.reflection as reflection

        fake = _CountingLLM()
        monkeypatch.setattr(reflection, "get_llm", lambda *a, **k: fake)

        state = _state(
            recommendation="TW21-3216-P2.5 is a great fit.",
            products=[],
        )
        result = reflection.reflection_node(state)

        assert fake.calls == 0
        assert result["needs_refine"] is False
        assert "validation_report" in result

    def test_recommend_uses_exactly_one_llm_call(self, monkeypatch):
        """推荐环节：选型/计算/校验全部离线完成，只留 1 次表达调用"""
        import src.agents.solution.nodes.recommend as recommend

        fake = _CountingLLM()
        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: fake)

        result = recommend.recommend_node(_state())

        assert fake.calls == 1, "推荐链路应只有一次 LLM 表达调用"
        assert result["recommendation_result"]["recommendations"], "选型结果应来自引擎"
        assert len(result["recommendation_result"]["recommendations"]) == 3
        assert result["products"], "应带出对应的产品证据"
        assert len(result["products"]) <= 3

    def test_full_recommendation_path_budget(self, monkeypatch):
        """推荐 + 校验整条链路 ≤ 2 次 LLM 调用（实际为 1 次）"""
        import src.agents.solution.nodes.recommend as recommend
        import src.agents.solution.nodes.reflection as reflection

        fake = _CountingLLM()
        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: fake)
        monkeypatch.setattr(reflection, "get_llm", lambda *a, **k: fake)

        state = recommend.recommend_node(_state())
        state = reflection.reflection_node(state)

        assert fake.calls == 1
        assert state["needs_refine"] is False

    def test_llm_failure_degrades_to_template(self, monkeypatch):
        """LLM 不可用时仍要给出可用的销售方案（型号 + 工程配置）"""
        import src.agents.solution.nodes.recommend as recommend

        class _FailingLLM:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("network down")

        monkeypatch.setattr(recommend, "get_llm", lambda *a, **k: _FailingLLM())

        state = _state(
            understood_slots={
                "environment": "indoor",
                "purpose": "conference",
                "viewing_distance_m": 5,
                "installation": "fixed",
                "target_width_mm": 5000,
                "target_height_mm": 3000,
            }
        )
        result = recommend.recommend_node(state)

        # 未指定预算 → 默认最便宜 → TW11-3216 系列（low 档）
        assert "TW11-3216-P2.5" in result["recommendation"]
        assert result["screen_calculation"]["cabinet_count"] == 56
        assert "56" in result["recommendation"]
