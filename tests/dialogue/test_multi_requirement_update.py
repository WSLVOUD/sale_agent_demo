"""v2.6 §11：客户一次给多个信息 → 一次更新，不连问四个问题。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    canonical_slots,
    explicit_slots,
    get_conversation_state,
    next_question_plan,
    reset_conversation_state,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402

MESSAGE = "Indoor, 3x5m, P3, fixed installation."


def _raw_slots(message: str) -> dict:
    """生产链路的做法：档案用**原始槽位**构建（宽高分开，信息不丢）。"""
    return {
        key: value
        for key, value in (extract_slots(message) or {}).items()
        if not str(key).startswith("_")
    }


class TestMultiRequirementUpdate:

    def test_all_four_fields_are_extracted_at_once(self):
        slots = canonical_slots(explicit_slots(MESSAGE))
        assert slots.get("environment") == "indoor"
        assert "size" in slots
        assert slots.get("pixel_pitch") == 3.0
        assert slots.get("installation") == "fixed"

    def test_profile_records_every_field(self):
        raw = _raw_slots(MESSAGE)
        profile = RequirementProfile.from_slots(raw, explicit_keys=set(raw))
        assert getattr(profile, "environment", "") == "indoor"
        assert getattr(profile, "pixel_pitch_mm", None) == 3.0
        assert getattr(profile, "installation", "") == "fixed"
        assert getattr(profile, "target_width_mm", None) or getattr(
            profile, "target_height_mm", None
        )

    def test_at_most_one_follow_up_question(self):
        """§26 Case 6：不要连续提四个问题。"""
        raw = _raw_slots(MESSAGE)
        covered = canonical_slots(explicit_slots(MESSAGE))
        profile = RequirementProfile.from_slots(raw, explicit_keys=set(raw))
        session_id = "multi-update-1"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        if plan is not None:
            assert plan.slot not in covered, "已经给过的字段不能再问"
