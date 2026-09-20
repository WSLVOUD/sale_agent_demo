"""v2.2.3：发给客户的追问必须是"人话"，而且不能问已经能推导出来的东西。

实测日志（客户"需要50人观看的屏幕"之后）：

    To recommend the right products for you, could you tell me: distance?

两个问题：
  1. 内部槽位名 `distance` 被直接抛给客户（`clarify_node` 的旧分支用 LLM 的
     missing_info 拼问句）；
  2. 客户已经给了人数，观看距离本来就能推导，仍然在追问。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import (  # noqa: E402
    check_recommendation_ready,
    human_label,
    question_for,
)


RAW_SLOT_NAMES = (
    "distance", "viewing_distance", "viewing_distance_m", "size", "target_size",
    "width", "height", "pixel_pitch", "pitch", "environment", "installation",
    "unknown_slot",
)


class TestQuestionNeverLeaksSlotNames:

    @pytest.mark.parametrize("slot", RAW_SLOT_NAMES)
    def test_question_is_human_readable(self, slot):
        question = question_for(slot, "en", 0) or ""
        assert question.endswith("?"), (slot, question)
        lowered = question.lower()
        # 不能出现 "could you tell me: distance?" 这种把字段名抛给客户的说法
        assert f": {slot}?" not in lowered, (slot, question)
        assert not lowered.rstrip("?").endswith(slot), (slot, question)
        assert f"could you tell me {slot}" not in lowered, (slot, question)

    @pytest.mark.parametrize("slot", RAW_SLOT_NAMES)
    def test_human_label_never_returns_the_raw_name(self, slot):
        label = human_label(slot)
        assert label != slot
        if label:
            assert "?" not in label

    def test_unknown_slot_falls_back_to_a_generic_sentence(self):
        question = question_for("slot_that_does_not_exist", "en", 0) or ""
        assert "slot_that_does_not_exist" not in question
        assert "?" in question

    def test_distance_slot_is_phrased_for_a_customer(self):
        question = (question_for("distance", "en", 0) or "").lower()
        assert "how far" in question or "viewing distance" in question


class TestClarifyNodeUsesTheGate:
    """`clarify_node` 必须用确定性 Gate 的问句，而不是 LLM 给的 missing_info。"""

    def _state(self, **overrides):
        state = {
            "requirement": {"purpose": "church"},
            "messages": [{"role": "user", "content": "需要50人观看的屏幕"}],
            "pending_question": "",
            "missing_info": ["distance"],
        }
        state.update(overrides)
        return state

    def test_legacy_branch_translates_the_slot_name(self):
        from src.agents.solution.nodes.requirement import clarify_node

        out = clarify_node(self._state())
        question = out.get("pending_question") or ""
        assert question, out
        assert "distance?" not in question.lower()
        assert "how far" in question.lower() or "viewing distance" in question.lower()

    def test_profile_wins_over_missing_info(self):
        from src.agents.solution.nodes.requirement import clarify_node
        from src.rag.query_understanding import extract_slots, merge_slots

        slots = {}
        for message in ("we need an indoor LED screen, permanent installation",
                        "需要50人观看的屏幕"):
            slots = merge_slots(slots, extract_slots(message))
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        decision = check_recommendation_ready(profile)

        out = clarify_node(self._state(
            requirement_profile=profile,
            recommendation_gate=decision.to_dict(),
            missing_info=["distance"],
        ))
        question = out.get("pending_question") or ""
        assert question == decision.next_question, out
        assert "how far viewers" not in question.lower(), "已经有 50 人了，不该再问观看距离"

    def test_ready_profile_asks_nothing(self):
        from src.agents.solution.nodes.requirement import clarify_node

        slots = {
            "display_type": "LED", "environment": "indoor", "purpose": "church",
            "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 10000, "target_height_mm": 5000,
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        assert check_recommendation_ready(profile).ready is True

        out = clarify_node(self._state(
            requirement_profile=profile, recommendation_gate={"ready": True},
        ))
        assert out.get("pending_question") == ""
        assert out.get("waiting_for_clarification") is False


class TestAudienceCountSuppressesTheDistanceQuestion:
    """客户给了人数 → 观看距离推导得出来 → Gate 不能再问"站多远"。"""

    def _gate(self, message: str):
        from src.rag.query_understanding import extract_slots, merge_slots

        slots = {}
        for item in ("we need an indoor LED screen, permanent installation", message):
            slots = merge_slots(slots, extract_slots(item))
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        return profile, check_recommendation_ready(profile)

    @pytest.mark.parametrize("message", [
        "需要50人观看的屏幕",
        "about 50 people",
        "50 viewers will watch it",
    ])
    def test_distance_not_asked(self, message):
        profile, decision = self._gate(message)
        assert profile.audience_count == 50, profile.to_facts()
        assert "viewing_distance" not in (decision.missing or [])
        assert "how far" not in (decision.next_question or "").lower()
