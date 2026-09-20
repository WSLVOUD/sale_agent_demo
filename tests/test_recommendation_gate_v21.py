"""v2.1 Phase 8：Recommendation / Calculation Gate（计划 11 / 12 / 13 / 18 / 24 节）。

直接跑计划第 18 节的 5 个典型场景，外加第 24 节的验收标准：

    READY           所有推荐必需条件满足
    DEGRADED_READY  非关键参数未知，仍可合理推荐
    CONTINUE_ASKING 有一个真正值得问、且没到上限的关键参数
    BLOCKED         无法推导 + 客户没授权 + 该 Action 必需
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_calculation_ready,
    check_recommendation_ready,
)


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


CHURCH_FIXED = {
    "display_type": "LED", "environment": "indoor", "purpose": "church",
    "content_type": "mixed", "installation": "fixed", "price_preference": "price",
}


class TestCase1CustomerDelegatesSize:
    """计划 Case 1：客户说 "No range, you recommend." → 不再问尺寸。"""

    def test_width_delegated_never_asked_again(self):
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=3.0)
        profile.mark_decision("width", "delegated")
        profile.mark_decision("height", "delegated")

        decision = check_recommendation_ready(profile)
        assert decision.ready is True
        assert "size" not in decision.missing
        assert profile.field_decision("width") == "DELEGATED"


class TestCase2UnknownTwiceThenDeferred:
    """计划 Case 2：同一字段问两次仍不知道 → DEFERRED，之后不再问。"""

    def test_unknown_then_deferred_question_flow(self):
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=3.0)

        first = check_recommendation_ready(profile, variant_seed=0)
        assert first.status == "CONTINUE_ASKING"
        assert "size" in first.missing
        # 追问记账发生在"问完之后"（节点里记录），客户第一次说"不知道" → UNKNOWN
        profile.record_ask("size")
        assert profile.mark_decision("size", "unknown") == "UNKNOWN"

        second = check_recommendation_ready(profile, variant_seed=0)
        assert second.status == "CONTINUE_ASKING"
        assert "size" in second.missing
        assert second.next_question != first.next_question, "第二次要换降门槛的问法"
        assert any(
            word in second.next_question.lower()
            for word in ("rough", "approximately", "estimate", "about")
        ), second.next_question

        # 第二次仍不知道 → DEFERRED：不再问，也不再阻塞推荐
        profile.record_ask("size")
        assert profile.mark_decision("size", "unknown") == "DEFERRED"
        third = check_recommendation_ready(profile, variant_seed=0)
        assert third.ready is True
        assert "size" not in third.missing
        assert check_calculation_ready(profile).status == "DEFERRED"


class TestCase3PitchKnownDistanceUnknown:
    """计划 Case 3：有了 P 值，观看距离延后 → 推荐照样 READY。"""

    def test_recommendation_ready_with_deferred_distance(self):
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=2.5,
                           target_width_mm=5000, target_height_mm=3000)
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_decision("viewing_distance", "unknown")

        decision = check_recommendation_ready(profile)
        assert decision.ready is True
        assert decision.status == "READY", decision.reason
        # 尺寸齐备 → 计算不受观看距离延后影响
        assert check_calculation_ready(profile).ready is True


class TestCase4DelegateMultipleFields:
    """计划 Case 4：Indoor church, fixed installation. You decide the size and pitch."""

    def test_profile_ends_up_in_expected_states(self):
        profile = RequirementProfile()
        values = {
            "display_type": "LED", "environment": "indoor", "purpose": "church",
            "installation": "fixed",
        }
        profile = profile.merge(
            RequirementProfile.from_slots(values, explicit_keys=set(values))
        )
        profile.mark_decision("size", "delegated")
        profile.mark_decision("pixel_pitch", "delegated")

        assert profile.field_decision("environment") == "CONFIRMED"
        assert profile.field_decision("purpose") == "CONFIRMED"
        assert profile.field_decision("installation") == "CONFIRMED"
        assert profile.field_decision("size") == "DELEGATED"
        assert profile.field_decision("pixel_pitch") == "DELEGATED"

        decision = check_recommendation_ready(profile)
        assert decision.ready is True
        assert "size" not in decision.missing
        assert "pixel_pitch" not in decision.missing

    def test_engine_derives_parameters_and_recommends(self):
        from src.rag.recommendation_service import RecommendationService

        profile = _profile(**CHURCH_FIXED)
        profile.mark_decision("size", "delegated")
        profile.mark_decision("pixel_pitch", "delegated")
        result = RecommendationService().recommend(profile)
        assert result["recommendations"], result.get("missing_fields")
        assert all(rec["indoor"] for rec in result["recommendations"])


class TestCase5TrulyBlocked:
    """计划 Case 5：环境无法判断 → BLOCKED，只说明这一件事。"""

    def _unresolvable_environment(self) -> RequirementProfile:
        profile = RequirementProfile()
        profile.record_ask("environment")
        profile.mark_decision("environment", "unknown")     # 第一次：还能降门槛再问
        profile.record_ask("environment")
        profile.mark_decision("environment", "unknown")     # 第二次仍不知道 → DEFERRED
        return profile

    def test_environment_is_blocked_with_clear_message(self):
        decision = check_recommendation_ready(self._unresolvable_environment())
        assert decision.ready is False
        assert decision.status == "BLOCKED"
        assert decision.blocked_slots == ["environment"]
        message = (decision.next_question or "").lower()
        assert "indoors or outdoors" in message

    def test_customer_cannot_delegate_environment(self):
        """室内外是产品硬过滤必需条件 → 客户说"你决定"也不能替他决定。"""
        profile = _profile(**CHURCH_FIXED)
        profile.field_decisions.pop("environment", None)
        profile.environment = None
        profile.sources.pop("environment", None)
        profile.mark_decision("environment", "delegated")
        decision = check_recommendation_ready(profile)
        assert decision.ready is False
        assert decision.status in ("CONTINUE_ASKING", "BLOCKED")

    def test_first_dont_know_still_asks_easier(self):
        profile = RequirementProfile()
        profile.record_ask("environment")
        profile.mark_decision("environment", "unknown")
        decision = check_recommendation_ready(profile)
        assert decision.status == "CONTINUE_ASKING"
        assert "environment" in decision.missing


class TestGateStatuses:
    """计划 12 / 24 节：Gate 只返回四种状态，且各司其职。"""

    def test_missing_hard_condition_continues_asking(self):
        decision = check_recommendation_ready(RequirementProfile())
        assert decision.status == "CONTINUE_ASKING"
        assert decision.ready is False

    def test_full_hard_conditions_ready(self):
        profile = _profile(
            **CHURCH_FIXED, pixel_pitch_mm=3.0,
            target_width_mm=5000, target_height_mm=3000,
        )
        decision = check_recommendation_ready(profile)
        assert decision.status == "READY"
        assert decision.ready is True
        assert decision.deferred_slots == []
        assert decision.blocked_slots == []

    def test_degraded_ready_reports_deferred_slots(self):
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=3.0)
        profile.record_ask("size")
        profile.record_ask("size")
        profile.mark_decision("size", "unknown")
        decision = check_recommendation_ready(profile)
        assert decision.status == "DEGRADED_READY"
        assert decision.ready is True
        assert "size" in decision.deferred_slots

    def test_gate_log_payload_shape(self):
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=3.0,
                           target_width_mm=5000, target_height_mm=3000)
        payload = check_recommendation_ready(profile).to_dict()
        for key in ("ready", "status", "missing", "deferred_slots",
                    "blocked_slots", "derived_size_m"):
            assert key in payload

    def test_calculation_gate_independent_of_recommendation(self):
        """计划 13 节：尺寸 DEFERRED 时推荐 READY、计算 DEFERRED。"""
        profile = _profile(**CHURCH_FIXED, pixel_pitch_mm=3.0)
        profile.record_ask("size")
        profile.record_ask("size")
        profile.mark_decision("size", "unknown")
        assert check_recommendation_ready(profile).ready is True
        assert check_calculation_ready(profile).status == "DEFERRED"
