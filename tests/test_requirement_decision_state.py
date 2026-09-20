"""v2.1 Phase 8：RequirementProfile 字段决策状态（计划 3 / 6 / 19 节）。

字段状态：MISSING / CONFIRMED / INFERRED / UNKNOWN / DELEGATED / DECLINED / DEFERRED。
核心口径：

    Missing 不等于 Blocked
    Unknown 不等于 Ask Forever
    Delegated 不等于 Guess
    Deferred 不等于 Confirmed
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import (  # noqa: E402
    MAX_ASKS_PER_SLOT,
    RequirementProfile,
    canonical_slot,
)


class TestStateBasics:

    def test_missing_by_default(self):
        profile = RequirementProfile()
        assert profile.field_decision("viewing_distance") == "MISSING"
        assert profile.field_decision("pixel_pitch") == "MISSING"

    def test_value_from_customer_is_confirmed(self):
        profile = RequirementProfile.from_slots(
            {"viewing_distance_m": 5}, explicit_keys={"viewing_distance_m"}
        )
        assert profile.field_decision("viewing_distance") == "CONFIRMED"

    def test_value_from_system_is_inferred(self):
        profile = RequirementProfile.from_slots({"viewing_distance_m": 5})
        assert profile.field_decision("viewing_distance") == "INFERRED"

    def test_delegated(self):
        profile = RequirementProfile()
        assert profile.mark_decision("size", "delegated") == "DELEGATED"
        assert profile.is_delegated("size") is True
        assert profile.is_exhausted("size") is False

    def test_declined(self):
        profile = RequirementProfile()
        assert profile.mark_decision("installation", "declined") == "DECLINED"
        assert profile.is_declined("installation") is True
        assert profile.is_exhausted("installation") is True

    def test_unknown_then_deferred_after_two_asks(self):
        profile = RequirementProfile()
        profile.record_ask("viewing_distance")
        assert profile.mark_decision("viewing_distance", "unknown") == "UNKNOWN"
        assert profile.is_unknown("viewing_distance") is False, "第一次不知道还能降门槛再问"
        profile.record_ask("viewing_distance")
        assert profile.mark_decision("viewing_distance", "unknown") == "DEFERRED"
        assert profile.is_deferred("viewing_distance") is True
        assert profile.is_exhausted("viewing_distance") is True

    def test_single_ask_slot_defers_immediately(self):
        """点间距问一次就够：客户不知道就直接转问观看距离（计划 Case 2/3）。"""
        profile = RequirementProfile()
        profile.record_ask("pixel_pitch")
        assert profile.mark_decision("pixel_pitch", "unknown") == "DEFERRED"
        assert profile.is_unknown("pixel_pitch") is True

    def test_slot_alias_writes_one_canonical_key(self):
        profile = RequirementProfile()
        profile.record_ask("viewing_distance")
        assert profile.ask_count("viewing_distance_m") == 1
        assert profile.ask_counts == {"viewing_distance_m": 1}
        assert canonical_slot("viewing_distance") == "viewing_distance_m"


class TestStateTransitions:

    def test_customer_value_cancels_previous_decision(self):
        """客户之前说"你决定尺寸"，后来又真的给了尺寸 → 决策作废、以值为准。"""
        profile = RequirementProfile()
        profile.mark_decision("size", "delegated")
        sizes = {"target_width_mm": 5000, "target_height_mm": 3000}
        merged = profile.merge(
            RequirementProfile.from_slots(sizes, explicit_keys=set(sizes))
        )
        assert merged.has_target_size
        assert merged.is_delegated("size") is False
        assert merged.is_delegated("width") is False
        assert merged.field_decision("width") == "CONFIRMED"

    def test_merge_keeps_customer_decisions(self):
        profile = RequirementProfile()
        profile.mark_decision("pixel_pitch", "delegated")
        merged = profile.merge(RequirementProfile())
        assert merged.field_decision("pixel_pitch") == "DELEGATED"

    def test_merge_drops_decision_when_value_arrives(self):
        profile = RequirementProfile()
        profile.mark_decision("viewing_distance", "deferred")
        incoming = RequirementProfile.from_slots(
            {"viewing_distance_m": 8}, explicit_keys={"viewing_distance_m"}
        )
        merged = profile.merge(incoming)
        assert merged.field_decision("viewing_distance") == "CONFIRMED"
        assert merged.is_deferred("viewing_distance") is False

    def test_delegated_size_propagates_to_width_and_height(self):
        """客户说"尺寸你决定" → 不能因为 width / height 没单独记录又回去问尺寸。"""
        profile = RequirementProfile()
        profile.mark_decision("size", "delegated")
        assert profile.field_decision("width") == "DELEGATED"
        assert profile.field_decision("height") == "DELEGATED"

    def test_max_asks_is_two(self):
        profile = RequirementProfile()
        profile.record_ask("installation")
        profile.record_ask("installation")
        assert profile.ask_count("installation") == MAX_ASKS_PER_SLOT
        assert profile.is_exhausted("installation") is True

    def test_downstream_slots_excludes_decided(self):
        profile = RequirementProfile()
        profile.mark_decision("pixel_pitch", "delegated")
        profile.mark_decision("installation", "deferred")
        allowed = profile.downstream_slots()
        assert "pixel_pitch" not in allowed
        assert "installation" not in allowed
        assert "size" in allowed


class TestBasisReflectsDecisions:

    def test_deferred_counts_as_unknown_basis(self):
        """DEFERRED 是"客户给不了"，推荐依据里必须体现（不能算确认）。"""
        profile = RequirementProfile.from_slots(
            {"environment": "indoor"}, explicit_keys={"environment"}
        )
        profile.record_ask("viewing_distance")
        profile.record_ask("viewing_distance")
        profile.mark_decision("viewing_distance", "unknown")
        basis = profile.requirement_basis()
        assert "viewing_distance" in basis["unknown"]
        assert "viewing_distance" not in basis["confirmed"]
