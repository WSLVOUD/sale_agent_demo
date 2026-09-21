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

    计划 §10 / §12 / §13 / §26 Case 2 要求：环境仍然未知 → **本轮唯一的问题
    还是 indoor / outdoor**，不能被随机轮换掉（日志里当时问成了 P 值）。
    """

    def _profile_asked_once(self, message: str = "3*5") -> RequirementProfile:
        profile = _profile_from_message(message)
        profile.last_asked_slot = "environment"
        profile.record_ask("environment")
        return profile

    def test_environment_is_still_the_question(self):
        profile = self._profile_asked_once()
        session_id = "env-after-wrong-1"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        assert getattr(plan, "slot", None) == "environment"
        assert "indoor" in (getattr(plan, "question", "") or "").lower()

    def test_question_is_the_plain_form_not_the_softened_one(self):
        """客户反馈过：不要换成 "That's okay — most installations are indoors…"。"""
        profile = self._profile_asked_once()
        session_id = "env-after-wrong-2"
        reset_conversation_state(session_id)
        plan = next_question_plan(
            profile, session_id=session_id, conversation=get_conversation_state(session_id)
        )
        question = (getattr(plan, "question", "") or "")
        assert "that's okay" not in question.lower()
        assert "most installations are indoors" not in question.lower()

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
