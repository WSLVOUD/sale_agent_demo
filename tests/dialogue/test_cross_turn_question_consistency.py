"""计划 v2.9 §二十三（Phase 15）：跨轮一致性 —— 已答过的字段不能再问。

    第一轮  Customer: 3m x 5m indoor LED   → 系统应该问 viewing_distance
    第二轮  Customer: 5 meters             → 不能再次问 viewing_distance，
                                            必须换成新的 QuestionSpec（另一个槽位）

同时验证 §二十四（Phase 16）：客户答非所问时，有效信息照样进档案。
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _profile_from(*messages: str):
    """把若干轮客户消息合并成一个 RequirementProfile（用项目真实的确定性抽取）。"""
    from src.models.requirement import RequirementProfile
    from src.rag.query_understanding import extract_slots

    slots: dict = {}
    explicit: set = set()
    for message in messages:
        for key, value in (extract_slots(message) or {}).items():
            if str(key).startswith("_"):
                continue
            slots[key] = value
            explicit.add(key)
    return RequirementProfile.from_slots(slots, explicit_keys=explicit)


def _candidate_slots(profile) -> list:
    from src.agents.sales.question_planner import candidate_questions

    # 计划 v2.9 §四：Policy / QuestionSpec 用对话层槽位名（dialogue_slot）
    return [
        str(item.get("dialogue_slot") or item.get("slot") or "")
        for item in candidate_questions(profile)
    ]


class TestCrossTurnQuestionConsistency:

    def test_answering_the_question_removes_it_from_candidates(self):
        first = _profile_from("3m x 5m indoor LED")
        first_slots = _candidate_slots(first)
        assert "viewing_distance" in first_slots, first_slots

        # 第二轮客户回答 "5 meters" → 这一项已经确认，不能再出现在候选里
        second = _profile_from("3m x 5m indoor LED", "5 meters")
        assert second.viewing_distance_m == 5.0
        second_slots = _candidate_slots(second)
        assert "viewing_distance" not in second_slots, second_slots
        # 已经确认过的 environment / size 也不能再问
        assert "environment" not in second_slots, second_slots
        assert "size" not in second_slots, second_slots

    def test_confirmed_slots_never_come_back(self):
        """三个已确认的硬条件（环境 / 尺寸 / 视距）都不能再进候选。"""
        second = _profile_from("3m x 5m indoor LED", "5 meters")
        second_slots = _candidate_slots(second)
        for answered in ("environment", "size", "viewing_distance"):
            assert answered not in second_slots, (answered, second_slots)

    def test_answer_to_another_slot_is_still_saved(self):
        """§二十四：客户答的是 mall（没回答 viewing distance）→ mall 也要存下来。"""
        profile = _profile_from("indoor 3m x 5m LED", "it will be installed in a shopping mall")
        assert profile.purpose or profile.content_type or profile.installation
        # viewing_distance 仍然缺失 → 还允许问（不是丢信息导致的重复）
        assert "viewing_distance" in _candidate_slots(profile)
