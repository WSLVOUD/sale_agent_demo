"""v2.6 §12/§13：环境仍然是第一优先级，明显场景推断保留。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    environment_needs_asking,
    get_conversation_state,
    next_question_plan,
    reset_conversation_state,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.query_understanding import extract_slots  # noqa: E402


def _profile_from_message(message: str) -> RequirementProfile:
    slots = {
        key: value
        for key, value in (extract_slots(message) or {}).items()
        if not str(key).startswith("_")
    }
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


def _next_slot(profile: RequirementProfile, session_id: str):
    reset_conversation_state(session_id)
    plan = next_question_plan(
        profile, session_id=session_id, conversation=get_conversation_state(session_id)
    )
    return getattr(plan, "slot", None)


class TestEnvironmentFirst:

    def test_case1_generic_request_asks_environment_first(self):
        """§26 Case 1：i need a display → 只问 indoor / outdoor。"""
        assert _next_slot(_profile_from_message("i need a display"), "env-case1") == "environment"

    def test_case2_size_only_still_asks_environment(self):
        """§26 Case 2：3*5 → 记下尺寸，环境仍然第一问。"""
        assert _next_slot(_profile_from_message("3*5"), "env-case2") == "environment"

    def test_pitch_alone_does_not_skip_environment(self):
        """§13：随机只发生在同等重要的候选之间，environment 未知时不许被跳过。"""
        assert _next_slot(_profile_from_message("P3"), "env-case3") == "environment"

    def test_obvious_scenario_keeps_inferring(self):
        """§12：明显场景（教堂 / 会议室 / 户外广告）继续允许推断，不再问室内外。"""
        profile = _profile_from_message("led screen for a church")
        assert environment_needs_asking(profile) is False
        assert _next_slot(profile, "env-case4") != "environment"

    def test_customer_said_indoor_never_asked_again(self):
        profile = _profile_from_message("indoor")
        assert environment_needs_asking(profile) is False
        assert _next_slot(profile, "env-case5") != "environment"


class TestEnvironmentAfterWrongAnswer:
    """真实日志复现（2026-09-21 14:50）：问过 indoor/outdoor、客户回 3×5。

    v2.7 §19.1（Phase 8 Duplicate Question Firewall）：上一轮问的就是这一项、
    客户又没给出相关新信息 → **本轮不许重复问同一项**；它留到
    "其它问题问完后的硬性条件复问"再回来。

    （说明：本轮迭代早期按 v2.6 §12 的严格读法改成过"环境每轮必问"；
      v2.7 §19.1 明确写"绝对禁止"，所以以 v2.7 为准。）
    """

    def _profile_asked_once(self, message: str = "3*5") -> RequirementProfile:
        profile = _profile_from_message(message)
        profile.last_asked_slot = "environment"
        profile.record_ask("environment")
        return profile

    def test_environment_can_move_on_when_unanswered(self):
        """客户口径（2026-09-21）：每个问题没答出来**都可以跳转**（硬性条件也一样）；
        硬性条件只在"要推荐 / 要算方案"时才必须满足。

        "不连续提问"由 continuation_budget 控制：客户没答时先承接（最多 1 条），
        第 2 条必须拉回需求问题 —— 见 tests/dialogue/test_ack_streak.py。
        """
        profile = self._profile_asked_once()
        session_id = "env-after-wrong-1"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", None) != "environment", "没答出来 → 可以跳转"
        assert plan is not None

    def test_environment_comes_back_with_the_plain_form(self):
        """其它都问不到了 → 环境作为复问回来（直问，不用降门槛说法）。"""
        slots = {
            "display_type": "LED",
            "purpose": "church",
            "installation": "fixed",
            "target_width_mm": 3000,
            "target_height_mm": 5000,
            "pixel_pitch_mm": 3.0,
            "price_preference": "both",
            "content_type": "mixed",
        }
        profile = RequirementProfile.from_slots(slots, explicit_keys=set(slots))
        profile.last_asked_slot = "environment"
        profile.record_ask("environment")
        assert environment_needs_asking(profile) is True, "环境仍然是唯一没定的硬性条件"
        session_id = "env-after-wrong-2"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", None) == "environment"
        question = (getattr(plan, "question", "") or "")
        assert "indoor" in question.lower()

    def test_two_attempts_is_the_cap(self):
        """问满两次仍拿不到 → 不再无限追问（交给 Gate 的 DEFERRED / BLOCKED）。"""
        profile = self._profile_asked_once()
        profile.record_ask("environment")
        assert environment_needs_asking(profile) is False
        session_id = "env-after-wrong-3"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", None) != "environment"
