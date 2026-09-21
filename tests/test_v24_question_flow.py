"""v2.4：提问顺序随机化 + "一轮走完再回头问硬性条件"。

客户口径（2026-09-20）：

  1. 提问顺序**随机**（同一会话内稳定可复现）；
  2. 客户说"不知道"或没回答 → 这个问题本轮不再重复问，换下一个；
  3. 所有问题随机问完一遍后，才回头问还缺的**硬性条件**，并说明为什么需要；
  4. 硬性条件仍然"每个字段最多两次接触"；第二次还拿不到 → DEFERRED；
  5. 只有客户**明确要推荐**时才跳过随机轮（齐了就直接推荐）。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ASK_POOL,
    HARD_SLOTS,
    hard_recap_pending,
    next_question_plan,
    pass1_complete,
    pass1_pending,
    random_order,
    shuffled_slots,
)
from src.models.requirement import RequirementProfile  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestRandomOrder:

    def test_order_is_a_permutation_of_the_pool(self):
        for session in ("s1", "s2", "abc", ""):
            order = random_order(session)
            assert sorted(order) == sorted(ASK_POOL)

    def test_order_is_stable_within_a_session(self):
        assert random_order("session-A") == random_order("session-A")

    def test_order_differs_between_sessions(self):
        orders = {tuple(random_order(f"session-{index}")) for index in range(12)}
        assert len(orders) > 1, "不同会话应该是不同的顺序"

    def test_order_is_not_the_old_fixed_priority(self):
        fixed = ["environment", "purpose", "installation", "price_preference",
                 "pixel_pitch", "viewing_distance", "size"]
        assert any(list(random_order(f"s{i}")) != fixed for i in range(12))

    def test_shuffled_slots_is_deterministic_by_seed(self):
        assert shuffled_slots(ASK_POOL, seed="x") == shuffled_slots(ASK_POOL, seed="x")
        assert shuffled_slots(ASK_POOL, seed_override=7) == shuffled_slots(
            ASK_POOL, seed_override=7
        )


class TestPass1NoRepeat:

    def test_answered_once_keeps_environment_as_the_first_question(self):
        """v2.6 §12/§13：环境没定 → 仍然是第一问（问满两次才让位）。"""
        profile = _profile(display_type="LED")
        first = next_question_plan(profile, session_id="p1")
        assert first is not None
        profile.record_ask(first.slot)          # 问过但客户没回答
        pending = pass1_pending(profile, "p1")
        assert first.slot not in pending, "问过没答的问题，本轮不再问"
        second = next_question_plan(profile, session_id="p1")
        assert second is not None
        assert second.slot == "environment", "环境没定 → 仍然是第一问（v2.6 §12）"
        # 复问统一用直问：客户反馈过 "That's okay — most installations are indoors…"
        # 这种降门槛说法紧接着再出现，观感更差。
        assert second.easier is False

    def test_environment_gate_releases_after_two_contacts(self):
        """问满两次仍拿不到 → 环境让位给其它问题（不会死循环问同一项）。"""
        profile = _profile(display_type="LED")
        first = next_question_plan(profile, session_id="p1b")
        assert first is not None
        profile.record_ask(first.slot)
        profile.record_ask(first.slot)
        third = next_question_plan(profile, session_id="p1b")
        assert third is None or third.slot != "environment"

    def test_unknown_slot_is_not_asked_again_in_pass1(self):
        profile = _profile(display_type="LED")
        first = next_question_plan(profile, session_id="p2")
        profile.record_ask(first.slot)
        profile.mark_decision(first.slot, "unknown")
        assert first.slot not in pass1_pending(profile, "p2")

    def test_derivable_slot_is_not_in_the_pool(self):
        """客户给了观看距离 → 点间距可推导，不该再问点间距。"""
        profile = _profile(display_type="LED", viewing_distance_m=8)
        assert "pixel_pitch" not in pass1_pending(profile, "p3")

    def test_cross_slot_rule_skips_distance_when_pitch_known(self):
        profile = _profile(display_type="LED", pixel_pitch_mm=3.0)
        assert "viewing_distance" not in pass1_pending(profile, "p4")


class TestPass2HardRecap:

    def test_pass1_complete_detection(self):
        profile = _profile(display_type="LED")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        assert pass1_complete(profile) is True

    def test_hard_recap_only_hard_conditions(self):
        profile = _profile(display_type="LED")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        pending = hard_recap_pending(profile, "p5")
        assert pending, "一轮走完后应该回头问硬性条件"
        assert set(pending) <= set(HARD_SLOTS)

    def test_recap_carries_the_why(self):
        profile = _profile(display_type="LED", environment="indoor")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        plan = next_question_plan(profile, session_id="p6")
        assert plan is not None
        assert plan.reason == "hard_condition_recap"
        assert plan.why, "复问硬性条件必须说明为什么需要知道"
        assert plan.slot in HARD_SLOTS

    def test_recap_uses_easier_wording(self):
        profile = _profile(display_type="LED", environment="indoor")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        plan = next_question_plan(profile, session_id="p7")
        assert plan.easier is True

    def test_soft_slots_are_not_recapped(self):
        profile = _profile(display_type="LED", environment="indoor")
        for slot in ASK_POOL:
            profile.record_ask(slot)
        plan = next_question_plan(profile, session_id="p8")
        assert plan.slot not in ("purpose", "price_preference")

    def test_recap_stops_after_two_contacts(self):
        profile = _profile(display_type="LED")
        for slot in ASK_POOL:
            profile.record_ask(slot)      # 第一轮
            profile.record_ask(slot)      # 复问
        assert hard_recap_pending(profile, "p9") == []
        assert next_question_plan(profile, session_id="p9") is None


class TestImmediateRecommendation:

    def test_hard_conditions_complete_does_not_auto_recommend(self):
        """光把硬性条件凑齐不触发推荐：还要把随机轮走完（软问题也在里面）。"""
        profile = _profile(
            display_type="LED", environment="indoor", installation="fixed",
            pixel_pitch_mm=3.0, target_width_mm=5000, target_height_mm=3000,
        )
        assert pass1_complete(profile) is False
        plan = next_question_plan(profile, session_id="r1")
        assert plan is not None
        assert plan.slot in ("purpose", "price_preference", "viewing_distance")

    def test_customer_asking_for_recommendation_skips_pass1(self):
        profile = _profile(
            display_type="LED", environment="indoor", installation="fixed",
            pixel_pitch_mm=3.0, target_width_mm=5000, target_height_mm=3000,
        )
        plan = next_question_plan(
            profile, session_id="r2", customer_wants_recommendation=True
        )
        assert plan is None, "硬性条件齐 + 客户要推荐 → 直接推荐"

    def test_customer_asking_for_recommendation_still_asks_missing_hard(self):
        profile = _profile(display_type="LED", environment="indoor")
        plan = next_question_plan(
            profile, session_id="r3", customer_wants_recommendation=True
        )
        assert plan is not None
        assert plan.slot in HARD_SLOTS
        assert plan.why, "客户要推荐但缺硬性条件 → 问它并说明原因"

    def test_exclude_skips_vision_confirmed_slots(self):
        profile = _profile(display_type="LED")
        plan = next_question_plan(
            profile, session_id="r4", exclude={"installation"}
        )
        assert plan is not None and plan.slot != "installation"
