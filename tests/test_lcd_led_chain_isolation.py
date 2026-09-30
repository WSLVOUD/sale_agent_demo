"""LCD / LED 两条链路彻底隔离（客户口径 2026-09-30）。

客户原话：

    「完全帮我把两个链路都隔离开，而且不要改动任何现有的 led 链路，
      单纯的就是把 lcd 里掺杂的 led 链路隔离开。」

实测病根：LCD 决策层算好了"下一问"，**动作**却由通用层（给 LED 场景调出来的
SpeechAct + DialoguePolicy）决定 —— 客户答一个 "no" 被判成 CORRECTION →
clarify_only → LCD 那一问被吞 → 回复为空 → 兜底话术顶上。

这里守住三件事：

    A. 这一轮的**动作**由 LCD 决策层自己给（ask_only / recommend_only）；
    B. 通用层把动作判成 clarify_only 也压不过 LCD（隔离，不是"凑巧一致"）；
    C. LED 侧一行不变（没有 lcd_chain 的动作，也不会被 LCD 的逻辑碰到）。
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.sales.nodes import requirement as sales_req  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _lcd_profile(**slots) -> RequirementProfile:
    base = {
        "display_type": "LCD",
        "purpose": "conference",
        "environment": "indoor",
        "installation": "fixed",
        "lcd_category": "conference_education",
        "lcd_size_inch": 65.0,
        "lcd_resolution": "4K",
        "lcd_handwriting_required": True,
        "lcd_tender_project": False,
        "lcd_camera_required": True,
    }
    base.update(slots)
    return RequirementProfile.model_validate(base)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """需求抽取 / 接话 / 语境理解的 LLM 一律打桩。"""
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
        lambda message, **kwargs: {"category": "conference_education", "confidence": 0.95},
    )

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content="")

    monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeLLM)
    monkeypatch.setattr("src.core.llm.get_llm", lambda *a, **k: _FakeLLM())
    yield


def _lcd_state(profile, message="no,personal"):
    return {
        "messages": [{"role": "user", "content": message}],
        "current_message": message,
        "session_id": "lcd-led-isolation",
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


class TestLcdOwnsTheTurnAction:
    def test_lcd_action_comes_from_the_lcd_chain(self):
        profile = _lcd_profile()
        profile.sources = {key: "explicit" for key in profile.model_dump()}
        profile.last_asked_slot = "lcd_tender"

        out = sales_req.requirement_mining(_lcd_state(profile))

        action = out.get("dialogue_action") or {}
        assert action.get("source") == "lcd_chain", action
        assert action.get("action") == "ask_only", action
        assert action.get("target_slot") == "lcd_ops", action
        assert out.get("pending_slot") == "lcd_ops"

    def test_generic_clarify_only_cannot_override_the_lcd_question(self, monkeypatch):
        """通用层（LED 口径）判成 clarify_only 也压不过 LCD 决策层。"""

        def _generic_policy(*args, **kwargs):
            return {
                "speech_act": {"speech_act": "CORRECTION", "customer_questions": []},
                "dialogue_action": {
                    "action": "clarify_only",
                    "target_slot": "",
                    "question": "",
                    "reason": "correction_or_conflict",
                    "priority": 1,
                    "priority_label": "correction_or_conflict",
                },
            }

        monkeypatch.setattr("src.dialogue.decide_speech_policy", _generic_policy)
        profile = _lcd_profile()
        profile.sources = {key: "explicit" for key in profile.model_dump()}
        profile.last_asked_slot = "lcd_tender"

        out = sales_req.requirement_mining(_lcd_state(profile))

        action = out.get("dialogue_action") or {}
        assert action.get("action") == "ask_only", action
        assert action.get("source") == "lcd_chain", action
        assert out.get("pending_slot") == "lcd_ops"

    def test_complete_lcd_requirements_report_a_recommend_action(self):
        profile = _lcd_profile(lcd_ops_required=False)
        profile.sources = {key: "explicit" for key in profile.model_dump()}

        out = sales_req.requirement_mining(_lcd_state(profile, message="\u7ed9\u6211\u63a8\u8350"))

        action = out.get("dialogue_action") or {}
        assert action.get("action") == "recommend_only", action
        assert out.get("should_generate_solution") is True


class TestLedSideIsUntouched:
    def test_led_turn_never_carries_an_lcd_action(self):
        profile = RequirementProfile.model_validate(
            {
                "display_type": "LED",
                "environment": "indoor",
                "purpose": "conference",
                "target_width_m": 4.0,
                "target_height_m": 3.0,
                "pixel_pitch_mm": 4.0,
            }
        )
        state = _lcd_state(profile, message="4m x 3m indoor")
        state["display_type_decision"] = {"display_type": "LED", "status": "CONFIRMED", "locked": True}

        out = sales_req.requirement_mining(state)

        action = out.get("dialogue_action") or {}
        assert action.get("source") != "lcd_chain", action
        assert not out.get("lcd_action"), "LED 会话不该有 LCD 决策结果"

    def test_lcd_domain_stays_empty_for_led(self):
        profile = RequirementProfile.model_validate({"display_type": "LED"})
        state = _lcd_state(profile)
        state["display_type_decision"] = {"display_type": "LED", "status": "CONFIRMED", "locked": True}

        assert sales_req._lcd_domain(state, profile) == ""


class TestSolutionSideIsolation:
    def test_lcd_profile_skips_the_structured_led_product_query(self):
        from src.agents.solution.runner import _is_lcd_like_profile

        assert _is_lcd_like_profile(RequirementProfile.model_validate({"display_type": "LCD"})) is True
        assert _is_lcd_like_profile(RequirementProfile.model_validate({"display_type": "IFP"})) is True
        assert _is_lcd_like_profile(RequirementProfile.model_validate({"display_type": "LED"})) is False
        assert _is_lcd_like_profile(None) is False
