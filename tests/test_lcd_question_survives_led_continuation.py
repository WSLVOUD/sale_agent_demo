"""LCD 的下一问不被 LED 时代的"承接额度"吃掉（客户口径 2026-09-30）。

实测（客户原文）：

    AI: What will the screens be used for (for example a control room, a meeting room…)?
    客户: meeting
    AI: A conference setup makes a lot of sense for an LCD display. Knowing the room size
        helps me point you to the right option.        ← 只接住了一句，**问题没了**
    客户: 他为什么问了一个询问房间的尺寸？

根因链：

    1. "meeting" 被判成"与需求无关的闲聊"（is_offtopic_message）→ offtopic_turn=True；
    2. LED 时代的"承接额度"规则（continuation_budget）→ 本轮只承接、**不问问题**，
       并把 pending_question / pending_slot 清空；
    3. 回复分支发现没有待问项 → 只回了一句接话。
       （那句接话由 LLM 自由生成，于是冒出"room size"这种和真正要问的"屏幕尺寸"不一致的内容。）

修法（只动 LCD 一侧）：

    · LCD 会话不进"无关闲聊"那套分支（LED 侧照旧）；
    · LCD 决策层算出的下一问不受承接额度约束（LED 侧照旧）。
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root := project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.sales.nodes import requirement as sales_req  # noqa: E402
from src.dialogue.continuation_budget import apply_continuation_budget  # noqa: E402
from src.dialogue.response_coordinator import ResponseCoordinator  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _lcd_profile() -> RequirementProfile:
    profile = RequirementProfile.from_slots(
        {
            "display_type": "LCD",
            "lcd_category": "conference_education",
            "environment": "indoor",
            "purpose": "conference",
        },
        explicit_keys={"display_type", "lcd_category", "environment", "purpose"},
    )
    profile.last_asked_slot = "purpose"
    profile.record_ask("purpose")
    return profile


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import src.core.requirement_extractor as extractor_mod
    import src.dialogue.lcd_category_understanding as lcu

    monkeypatch.setattr(
        extractor_mod.RequirementExtractor,
        "_llm_semantic_extract",
        lambda self, message, rule_slots, session_id="": {},
    )
    extractor_mod.RequirementExtractor._semantic_cache.clear()
    monkeypatch.setattr(
        lcu,
        "understand_lcd_category",
        lambda message, **kwargs: {
            "category": "conference_education",
            "confidence": 0.9,
            "facts": {"environment": "indoor"},
        },
    )

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content="")

    monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeLLM)
    monkeypatch.setattr("src.core.llm.get_llm", lambda *a, **k: _FakeLLM())
    yield


class TestScenarioAnswerIsNotOffTopicInLcd:
    def test_meeting_is_not_marked_off_topic_for_an_lcd_session(self):
        profile = _lcd_profile()
        state = {
            "messages": [{"role": "user", "content": "meeting"}],
            "current_message": "meeting",
            "session_id": "lcd-offtopic-1",
            "requirements": {},
            "additional_requirements": [],
            "intent": "need_query",
            "next_action": "ask",
            "should_generate_solution": False,
            "response": "",
            "pending_question": "",
            "pending_slot": "",
            "acknowledgement": "",
            "display_type_decision": {"display_type": "LCD", "status": "CONFIRMED", "locked": True},
            "requirement_profile": profile,
        }

        out = sales_req.requirement_mining(state)

        assert not out.get("offtopic_turn"), "LCD 会话不该走'无关闲聊'那条路"
        assert out.get("pending_slot") == "lcd_size", out.get("pending_slot")


class TestLcdQuestionIsNotSuppressed:
    def test_continuation_budget_does_not_clear_an_lcd_question(self):
        data = {
            "offtopic_turn": True,  # 就算被标成闲聊，LCD 的问句也不许被清掉
            "pending_question": "What screen size do you have in mind (in inches)?",
            "pending_slot": "lcd_size",
        }

        slot, questions = apply_continuation_budget(
            data,
            question_slot="lcd_size",
            questions=[{"text": data["pending_question"], "slot": "lcd_size"}],
            protect_question=True,
        )

        assert slot == "lcd_size"
        assert questions, "LCD 的下一问必须留着"
        assert data.get("pending_slot") == "lcd_size", "不许被清空"
        assert not data.get("suppressed_question")

    def test_led_behaviour_is_unchanged_without_the_protection_flag(self):
        """不传 protect_question 时，承接额度的老行为完全不变。"""
        data = {
            "offtopic_turn": True,
            "pending_question": "How far away will viewers be?",
            "pending_slot": "viewing_distance",
        }

        slot, questions = apply_continuation_budget(
            data,
            question_slot="viewing_distance",
            questions=[{"text": data["pending_question"], "slot": "viewing_distance"}],
        )

        assert slot == ""
        assert questions == []
        assert data.get("suppressed_question")

    def test_prepare_final_keeps_the_lcd_question(self):
        profile = _lcd_profile()
        coordinator = ResponseCoordinator(profile_lookup=lambda _s: profile)
        result = {
            "response": "A conference setup makes a lot of sense for an LCD display.",
            "pending_question": "What screen size do you have in mind (in inches)?",
            "pending_slot": "lcd_size",
            "product_domain": "LCD",
            "lcd_action": {"confirmed": False, "question_slot": "lcd_size", "missing_fields": ["lcd_size"]},
            "dialogue_action": {
                "action": "ask_only",
                "target_slot": "lcd_size",
                "question": "What screen size do you have in mind (in inches)?",
                "source": "lcd_chain",
            },
            "offtopic_turn": True,  # 最坏情况：被标成闲聊
            "requirements": {},
            "newly_filled_slots": [],
            "products": [],
            "turn_id": "t",
            "recommendation_gate": {"ready": False},
        }

        plan = coordinator.prepare_final(
            result, session_id="lcd-offtopic-2", message="meeting", turn_context={}
        )

        assert plan.question_slot == "lcd_size", plan.question_slot
        assert plan.questions, plan.questions
