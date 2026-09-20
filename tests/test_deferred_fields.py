"""v2.1 Phase 8：DEFERRED 字段的行为（计划 4.4 / 8 / 13 / 19 / 21 节）。

DEFERRED = 客户给不出来 / 不愿再被问 → **不再追问**；
但它只阻塞"确实依赖它的 Action"，而不是阻塞整个销售流程：

    Recommendation: 照常推荐（必要时降级）
    Calculation:    只有依赖尺寸的计算才挂起

而且 DEFERRED **绝不等于 CONFIRMED**：推荐话术里不能把它当成客户说过的事实。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.sales.question_planner import plan_next_question  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)


def _defer(profile: RequirementProfile, slot: str, times: int = 2) -> RequirementProfile:
    """模拟"客户连续说不知道" → 落成 DEFERRED。"""
    for _ in range(times):
        profile.record_ask(slot)
    profile.mark_decision(slot, "unknown")
    return profile


def _hard_conditions(**overrides) -> RequirementProfile:
    slots = {
        "display_type": "LED", "environment": "indoor", "purpose": "church",
        "content_type": "mixed", "installation": "fixed", "price_preference": "price",
        "pixel_pitch_mm": 3.0,
    }
    slots.update(overrides)
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestDeferredIsNeverAskedAgain:

    def test_recommendation_gate_does_not_ask_deferred_size(self):
        profile = _defer(_hard_conditions(), "size")
        decision = check_recommendation_ready(profile)
        assert "size" not in decision.missing, decision.missing
        assert decision.next_question is None or "size" not in (decision.next_question or "")

    def test_question_planner_skips_deferred_slot(self):
        profile = _defer(_hard_conditions(), "size")
        plan = plan_next_question(profile) or {}
        assert plan.get("slot") != "target_size"

    def test_deferred_viewing_distance_is_not_asked(self):
        profile = _defer(_hard_conditions(), "viewing_distance")
        decision = check_recommendation_ready(profile)
        assert "viewing_distance" not in (decision.missing or [])


class TestDeferredOnlyBlocksDependentActions:

    def test_recommendation_ready_but_calculation_deferred(self):
        """计划 13 节：Recommendation 与 Calculation 必须各自判断。"""
        profile = _defer(_hard_conditions(), "size")
        recommendation = check_recommendation_ready(profile)
        calculation = check_calculation_ready(profile)

        assert recommendation.ready is True
        assert recommendation.status in ("READY", "DEGRADED_READY")
        assert calculation.ready is False
        assert calculation.status == "DEFERRED"
        assert calculation.next_question is None, "已经 DEFERRED 的字段不许再问"

    def test_recommendation_still_returns_products(self):
        from src.rag.recommendation_service import RecommendationService

        profile = _defer(_hard_conditions(), "size")
        result = RecommendationService().recommend(profile)
        assert result["recommendations"], "尺寸延后不该阻止选型"
        assert result["recommendation_status"] in ("RECOMMENDED", "DEGRADED")

    def test_deferred_distance_does_not_block_calculation_when_size_known(self):
        """尺寸是硬性计算输入；观看距离延后不影响箱体计算。"""
        profile = _defer(_hard_conditions(target_width_mm=5000, target_height_mm=3000),
                         "viewing_distance")
        assert check_calculation_ready(profile).ready is True


class TestDeferredIsNotConfirmed:

    def test_deferred_slot_is_not_in_confirmed_basis(self):
        profile = _defer(_hard_conditions(), "size")
        basis = profile.requirement_basis()
        assert "width" not in basis["confirmed"]
        assert "height" not in basis["confirmed"]

    def test_profile_does_not_invent_a_value(self):
        profile = _defer(_hard_conditions(), "size")
        assert profile.target_width_mm is None
        assert profile.target_height_mm is None
        assert profile.has_target_size is False


class TestDeferredStopsAskingAfterTwoTries:

    def test_unknown_twice_becomes_deferred(self):
        profile = _hard_conditions()
        profile.record_ask("installation")
        assert profile.mark_decision("installation", "unknown") == "UNKNOWN"
        profile.record_ask("installation")
        assert profile.mark_decision("installation", "unknown") == "DEFERRED"
        decision = check_recommendation_ready(profile)
        assert "installation" not in (decision.missing or [])

    def test_declined_is_also_never_asked(self):
        profile = _hard_conditions()
        profile.mark_decision("installation", "declined")
        decision = check_recommendation_ready(profile)
        assert "installation" not in (decision.missing or [])
