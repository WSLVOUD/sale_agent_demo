"""档案 → 槽位名映射（v2.7 §18 Answer Coverage 的输入）。

两件事必须锁住：

1. 档案里的字段**一个都不能漏**（``price_preference`` / ``content_type`` 曾经漏掉，
   导致"客户答了价格取向"被判成没回答 → 每轮压掉问题 → 对话卡死）；
2. 映射表住在对话层，Orchestrator 只委托
   （边界不变量见 ``tests/test_v231_orchestrator_boundary.py``）。
"""
import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.profile_slots import FIELD_TO_SLOT, profile_slot_map  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

CHURCH_SLOTS = {
    "display_type": "LED",
    "environment": "indoor",
    "purpose": "church",
    "installation": "fixed",
    "viewing_distance_m": 3.0,
    "target_width_mm": 3000,
    "target_height_mm": 5000,
    "price_preference": "quality",
    "content_type": "mixed",
}


class TestProfileSlotMap:

    def test_all_answer_slots_are_visible(self):
        profile = RequirementProfile.from_slots(CHURCH_SLOTS, explicit_keys=set(CHURCH_SLOTS))
        slots = profile_slot_map(profile)
        assert slots["price_preference"] == "quality"
        assert slots["content_type"] == "mixed"
        assert slots["installation"] == "fixed"
        assert slots["environment"] == "indoor"
        assert slots["viewing_distance"] == 3.0
        assert slots["size"], "尺寸（宽/高）也要映射进 scoreboard"
        assert slots["pixel_pitch"] if "pixel_pitch" in slots else True

    def test_empty_and_meta_fields_are_skipped(self):
        slots = profile_slot_map(RequirementProfile.from_slots({}, explicit_keys=set()))
        for name in ("sources", "conflicts", "conflict_slots", "ask_counts", "field_decisions"):
            assert name not in slots, name
        assert all(value not in (None, "", [], {}) for value in slots.values())

    def test_none_profile_is_safe(self):
        assert profile_slot_map(None) == {}


class TestOrchestratorOnlyDelegates:

    def test_mapping_table_lives_in_dialogue_layer(self):
        source = io.open(
            os.path.join(project_root, "src", "orchestrator.py"), encoding="utf-8"
        ).read()
        assert "from .dialogue.profile_slots import profile_slot_map" in source
        assert "FIELD_TO_SLOT" not in source

    def test_every_mapped_field_is_a_real_profile_field(self):
        profile = RequirementProfile.from_slots(CHURCH_SLOTS, explicit_keys=set(CHURCH_SLOTS))
        dumped = set(profile.model_dump().keys())
        for field in FIELD_TO_SLOT:
            assert field in dumped, f"{field} 不是 RequirementProfile 的字段"
