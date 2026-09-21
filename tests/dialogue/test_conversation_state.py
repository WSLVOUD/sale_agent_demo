"""v2.6 §5~§8：ConversationState（双状态架构的"当前对话在做什么"那一半）。"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import (  # noqa: E402
    ConversationState,
    get_conversation_state,
    reset_conversation_state,
)


class TestConversationStateV26:

    def test_ai_turn_records_question_slot_and_response(self):
        state = ConversationState(session_id="cs-ai")
        state.note_ai_turn(
            action="ask_only",
            question="What pixel pitch are you considering?",
            slot="pixel_pitch",
            response="What pixel pitch are you considering?",
            turn_id="turn-1",
        )
        assert state.last_ai_action == "ask_only"
        assert state.last_ai_question == "What pixel pitch are you considering?"
        assert state.last_question_slot == "pixel_pitch"
        assert state.last_slot == "pixel_pitch", "旧字段保持兼容"
        assert state.last_response
        assert state.last_turn_id == "turn-1"

    def test_customer_turn_records_answer_slot_and_turn_id(self):
        state = ConversationState(session_id="cs-customer")
        state.note_customer_turn(text="P3", answer_slot="pixel_pitch", turn_id="turn-2")
        assert state.current_answer_slot == "pixel_pitch"
        assert state.current_turn_id == "turn-2"
        assert state.pending_customer_answer == "P3"
        assert state.turn_index == 1

    def test_question_slot_is_remembered_for_next_turn(self):
        """§8：AI 问过的那一项必须留着，下一轮才知道客户在答什么。"""
        state = ConversationState(session_id="cs-memory")
        state.note_ai_turn(question="What is the viewing distance?", slot="viewing_distance")
        assert state.last_question_slot == "viewing_distance"

    def test_state_is_per_session(self):
        reset_conversation_state("cs-v26-a")
        reset_conversation_state("cs-v26-b")
        get_conversation_state("cs-v26-a").note_ai_turn(
            question="Indoor or outdoor?", slot="environment"
        )
        assert get_conversation_state("cs-v26-b").last_question_slot == ""

    def test_to_dict_exposes_v26_fields(self):
        state = ConversationState(session_id="cs-dict")
        state.note_ai_turn(action="ask_only", question="Q?", slot="size")
        payload = state.to_dict()
        for key in (
            "last_ai_action",
            "last_ai_question",
            "last_question_slot",
            "current_speech_act",
            "current_answer_slot",
            "pending_customer_question",
            "pending_customer_answer",
            "last_turn_id",
            "current_turn_id",
            "last_response",
            "registry",
        ):
            assert key in payload, key

    def test_requirement_profile_and_conversation_state_are_separate(self):
        """§7：档案记"客户有什么需求"，对话状态记"客户现在在做什么"。"""
        from src.models.requirement import RequirementProfile

        profile = RequirementProfile.from_slots(
            {"environment": "indoor", "pixel_pitch_mm": 3.0},
            explicit_keys={"environment", "pixel_pitch_mm"},
        )
        state = ConversationState(session_id="cs-separate")
        state.note_ai_turn(question="How far will people sit?", slot="viewing_distance")

        assert getattr(profile, "environment", "") == "indoor"
        assert state.last_question_slot == "viewing_distance"
        assert not hasattr(profile, "last_question_slot")
