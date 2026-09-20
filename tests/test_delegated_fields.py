"""v2.1 Phase 8：DELEGATED（客户授权 AI 决定）的行为（计划 4.2 / 14 / 15 / 19 节）。

客户说 "You decide" / "No range, you recommend." 之后：

    · 不再追问同一个字段（计划 Case 1）
    · 参数必须走 **Python 确定性推导**，不许 LLM 编一个当事实（计划 14 / 15）
    · 推导不出来的（例如没有观看距离 → 推不出尺寸）→ 转问"推导需要的输入"或延后
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.parameter_inference import suggest_screen_size  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)


def _base(**overrides) -> RequirementProfile:
    slots = {
        "display_type": "LED", "environment": "indoor", "purpose": "church",
        "content_type": "mixed", "installation": "fixed", "price_preference": "price",
    }
    slots.update(overrides)
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestDelegatedIsNeverAskedAgain:

    def test_delegated_size_is_not_asked(self):
        profile = _base(pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        decision = check_recommendation_ready(profile)
        assert decision.ready is True, decision.missing
        assert "size" not in (decision.missing or [])

    def test_delegated_pitch_is_not_asked(self):
        profile = _base()
        profile.mark_decision("pixel_pitch", "delegated")
        decision = check_recommendation_ready(profile)
        assert "pixel_pitch" not in (decision.missing or [])
        assert "viewing_distance" not in (decision.missing or [])

    def test_delegated_pitch_still_recommends_a_model(self):
        """点间距授权给 AI → 由环境 / 场景确定性推导，不能卡在追问上。"""
        from src.rag.recommendation_service import RecommendationService

        profile = _base(target_width_mm=5000, target_height_mm=3000)
        profile.mark_decision("pixel_pitch", "delegated")
        result = RecommendationService().recommend(profile)
        assert result["recommendations"], result.get("missing_fields")
        assert all(rec["indoor"] for rec in result["recommendations"])


class TestDelegatedSizeUsesDeterministicInference:

    def test_suggest_screen_size_is_deterministic(self):
        facts = {"viewing_distance_m": 6.0}
        first = suggest_screen_size(facts)
        second = suggest_screen_size(facts)
        assert first == second
        width, height = first
        assert height == pytest.approx(2.0)          # 观看距离 6m ÷ 3
        assert width == pytest.approx(2.0 * 16 / 9, abs=0.01)

    def test_no_distance_means_no_guess(self):
        assert suggest_screen_size({"viewing_distance_m": None}) is None
        assert suggest_screen_size({}) is None

    def test_calculation_gate_uses_derived_size(self):
        profile = _base(viewing_distance_m=6.0, pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        decision = check_calculation_ready(profile)
        assert decision.ready is True
        assert decision.derived_size_m, "必须给出确定性推导出来的参考尺寸"
        width_m, height_m = decision.derived_size_m
        assert height_m == pytest.approx(2.0)
        assert width_m == pytest.approx(3.56, abs=0.01)

    def test_derived_size_is_not_written_into_customer_facts(self):
        profile = _base(viewing_distance_m=6.0, pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        check_calculation_ready(profile)
        assert profile.target_width_mm is None
        assert profile.has_target_size is False

    def test_delegated_size_without_distance_asks_for_distance(self):
        """推导尺寸需要观看距离 → 问观看距离，而不是回头再问尺寸。"""
        profile = _base(pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        decision = check_calculation_ready(profile)
        assert decision.ready is False
        assert decision.status == "CONTINUE_ASKING"
        assert "width" not in (decision.missing or [])
        assert decision.next_question

    def test_end_to_end_calculation_from_delegated_size(self):
        from src.agents.solution.nodes.recommend import recommend_node

        profile = _base(viewing_distance_m=6.0, pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        state = {
            "requirement": {},
            "products": [],
            "messages": [{"role": "user", "content": "indoor church LED screen, you decide the size"}],
            "requirement_profile": profile,
            "understood_language": "en",
            "additional_requirements": [],
        }
        out = recommend_node(state)
        assert out["products"], "授权尺寸后必须照常推荐"
        assert out["screen_calculation"], "推导出参考尺寸后必须算箱体"
        assert out["screen_calculation"]["cabinet_count"] > 0


class TestDelegatedWidthCase:
    """计划 Case 1：客户说 "No range, you recommend." 之后不再问宽度。"""

    def test_width_delegated_skips_the_question(self):
        from src.agents.sales.question_planner import plan_next_question

        profile = _base(pixel_pitch_mm=3.0)
        profile.mark_decision("width", "delegated")
        profile.mark_decision("height", "delegated")
        plan = plan_next_question(profile) or {}
        assert plan.get("slot") != "target_size"
        assert "width" not in (check_recommendation_ready(profile).missing or [])

    def test_size_delegation_covers_width_and_height(self):
        profile = _base(pixel_pitch_mm=3.0)
        profile.mark_decision("size", "delegated")
        assert profile.is_delegated("width") is True
        assert profile.is_delegated("height") is True


@pytest.fixture
def sales_node(monkeypatch):
    """把需求挖掘链路上的 LLM 全换成假实现（只验证规则 + 状态机）。"""
    import importlib

    sales_req = importlib.import_module("src.agents.sales.nodes.requirement")
    extractor_mod = importlib.import_module("src.core.requirement_extractor")

    class _Response:
        content = '{"usage": "church", "location_type": "室内", "additional_requirements": [], "ack": ""}'

    class _FakeChat:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeChat)
    monkeypatch.setattr(
        extractor_mod.RequirementExtractor,
        "_llm_semantic_extract",
        lambda self, message, rule_slots, session_id="": {},
    )
    extractor_mod.RequirementExtractor._semantic_cache.clear()
    return sales_req


class TestDelegationThroughSalesNode:
    """计划 Case 4 走真实节点：一句话授权尺寸 + 点间距 → 直接进入推荐。"""

    def _turn(self, module, message, profile=None):
        state = {
            "messages": [{"role": "user", "content": message}],
            "current_message": message,
            "session_id": "delegated-session",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
        }
        if profile is not None:
            state["requirement_profile"] = profile
        return module.requirement_mining(state)

    def test_authorising_size_and_pitch_recommends(self, sales_node):
        turn = self._turn(
            sales_node,
            "Indoor church, fixed installation. You decide the size and pitch.",
        )
        profile = turn["requirement_profile"]
        assert profile.environment == "indoor"
        assert profile.installation == "fixed"
        assert profile.is_delegated("size") is True
        assert profile.is_delegated("pixel_pitch") is True
        assert turn["recommendation_gate"]["status"] in ("READY", "DEGRADED_READY")
        assert turn["should_generate_solution"] is True
        assert turn["pending_slot"] != "pixel_pitch"
        assert turn["pending_slot"] != "size"
