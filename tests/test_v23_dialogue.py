"""v2.3 §19~§23：对话层（Question Planner / Response Planner / Conversation State）。

要求：一次只问一个问题、不重复、不暴露内部字段、推荐第一轮短追问再展开。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ConversationState,
    ResponseCoordinator,
    get_conversation_state,
    plan_question,
    plan_response,
    reset_conversation_state,
)
from src.dialogue.response_planner import (  # noqa: E402
    ACKNOWLEDGE,
    ASK,
    ASK_ONE_CLARIFICATION,
    ASK_ONE_QUESTION,
    CONFLICT,
    EXPLAIN_CONFLICT,
    EXPLAIN_WHY,
    KEY_REASON,
    RECOMMEND,
    RECOMMEND_ACTION,
    REASSURE,
    SHORT_SUMMARY,
    UNKNOWN,
)
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import check_recommendation_ready  # noqa: E402


def _profile(slots: dict) -> RequirementProfile:
    return RequirementProfile.from_slots(slots, explicit_keys=set(slots))


class TestResponseStrategies:

    def test_ask_strategy(self):
        profile = _profile({"environment": "indoor"})
        decision = check_recommendation_ready(profile)
        plan = plan_response(decision, profile=profile)
        assert plan.action == ASK
        assert plan.blocks == [ACKNOWLEDGE, EXPLAIN_WHY, ASK_ONE_QUESTION]
        assert plan.one_question is True
        assert "ONE question" in " ".join(plan.prompt_lines())

    def test_recommend_strategy_is_short_first(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 3.0,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        decision = check_recommendation_ready(profile)
        plan = plan_response(decision, profile=profile)
        assert plan.action == RECOMMEND_ACTION
        assert plan.blocks == [SHORT_SUMMARY, RECOMMEND, KEY_REASON]
        assert plan.expand_only_when_asked is True
        assert any("asks why" in line for line in plan.prompt_lines())

    def test_conflict_strategy(self):
        profile = _profile({
            "environment": "indoor", "installation": "fixed", "pixel_pitch_mm": 10.0,
            "target_width_mm": 5000, "target_height_mm": 3000,
        })
        decision = check_recommendation_ready(profile)
        plan = plan_response(decision, profile=profile)
        assert plan.action == CONFLICT
        assert plan.blocks == [ACKNOWLEDGE, EXPLAIN_CONFLICT, ASK_ONE_CLARIFICATION]

    def test_unknown_strategy(self):
        plan = plan_response(None, unknown=True, question_plan=type(
            "Q", (), {"slot": "viewing_distance", "question": "How far away?", "reason": ""}
        )())
        assert plan.action == UNKNOWN
        assert plan.blocks == [REASSURE, "OFFER_SIMPLE_ALTERNATIVE", ASK_ONE_QUESTION]

    def test_prompt_lines_never_expose_internal_names(self):
        profile = _profile({"environment": "indoor"})
        plan = plan_response(check_recommendation_ready(profile), profile=profile)
        text = " ".join(plan.prompt_lines())
        assert "field names" in text  # 明确要求不得暴露内部字段名


class TestQuestionPlanner:

    def test_question_is_phrased_once_and_rotated(self):
        profile = _profile({"environment": "indoor"})
        decision = check_recommendation_ready(profile, variant_seed=0)
        conversation = ConversationState(session_id="qp-1")

        first = plan_question(decision, profile, seed=0, conversation=conversation)
        assert first.slot == "purpose"
        assert first.question
        conversation.note_turn(question=first.question, slot=first.slot, action=first.action)

        second = plan_question(decision, profile, seed=0, conversation=conversation)
        assert second.question != first.question, "同一句话不能连着问两遍"
        assert second.reused is True

    def test_slot_override_wins(self):
        """调用方已经调整过槽位（例如图片刚确认过安装方式）→ 以调用方为准。"""
        profile = _profile({"environment": "indoor"})
        decision = check_recommendation_ready(profile)
        plan = plan_question(
            decision, profile, slot="pixel_pitch", question="Which pitch?", seed=0
        )
        assert plan.slot == "pixel_pitch"
        assert plan.question == "Which pitch?"

    def test_no_question_when_decision_is_empty(self):
        plan = plan_question(None)
        assert plan.question == ""


class TestConversationState:

    def test_records_turn(self):
        reset_conversation_state("cs-1")
        state = get_conversation_state("cs-1")
        state.note_turn(
            answer="about 8 meters", question="How far?", slot="viewing_distance",
            action="ASK", intent="need_query", stage="collecting",
        )
        assert state.last_question == "How far?"
        assert state.last_slot == "viewing_distance"
        assert state.last_answer == "about 8 meters"
        assert state.asked_before("How far?") is True
        assert state.question_is_repeating is False

    def test_repeat_detection(self):
        state = ConversationState(session_id="cs-2")
        state.note_turn(question="How far?", slot="viewing_distance")
        state.note_turn(question="How far?", slot="viewing_distance")
        assert state.repeated_question_count == 1
        assert state.question_is_repeating is True

    def test_state_is_per_session(self):
        reset_conversation_state("cs-a")
        reset_conversation_state("cs-b")
        get_conversation_state("cs-a").note_turn(question="Q-A", slot="size")
        assert get_conversation_state("cs-b").last_question == ""


class TestResponseCoordinator:

    class _Profile:
        vision_confirmation_pending = []
        conflicts: list = []

    def test_faq_is_attached_when_asked(self):
        coordinator = ResponseCoordinator()
        out = coordinator.finalize("TW11-3216-P3.0 looks like the best fit.", message="你们包安装吗？")
        assert "installation" in out.lower()
        assert "TW11" in out
        # 标准口径：不提供现场安装
        assert "do not provide on-site installation" in out.lower()

    def test_unrelated_turn_untouched(self):
        coordinator = ResponseCoordinator()
        assert coordinator.finalize("Plain reply.", message="教堂室内 P3") == "Plain reply."

    def test_vision_confirmation_uses_lookup(self):
        class _P:
            vision_confirmation_pending = ["environment"]
            environment = "indoor"

        coordinator = ResponseCoordinator(profile_lookup=lambda _sid: _P())
        out = coordinator.finalize("Sure.", session_id="s-1", message="i need this")
        assert out.endswith("Sure.")

    def test_orchestrator_delegates(self):
        """Orchestrator 的回复组装已经委托给 ResponseCoordinator（§13）。"""
        import io
        import re

        source = io.open("src/orchestrator.py", encoding="utf-8").read()
        assert "_response_coordinator().finalize" in source
        assert re.search(r"class\s+\w*Orchestrator", source)
