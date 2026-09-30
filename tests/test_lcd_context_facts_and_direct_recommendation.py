"""需求齐了直接推荐 + 需求事实由语境整理、关键词只兜底（客户口径 2026-09-30）。

实测（客户原文）：

    会议室 + 65" + 摄像头 + 手写 + 无 OPS
    → 客户看到的是 "Before I finalize it, just confirm for me that the LCD
      requirement is fully covered on your end…"，**型号永远出不来**。
    → 客户又说 "给我推荐"，还是这句话。

    以及："你之前改的哪个 cam，需要摄像头，你又是用的关键词去触发 ai，
    不允许用关键词去触发，以后的所有情况都是，只允许是结合上下文语境去整理
    需要的信息，关键词只能是兜底。"

这里守住两件事：

    A. 需求满足（lcd_action.confirmed）→ 保留 router/Solution 产出的推荐，不许覆盖；
    B. 需求事实优先由**语境理解**给出（模型读对话），正则只在它没覆盖时兜底。
"""
import os
import sys
from types import SimpleNamespace

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue import lcd_category_understanding as lcu  # noqa: E402
from src.dialogue.lcd_decision import extract_lcd_facts, lcd_turn  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


class TestConfirmedLcdKeepsTheRecommendation:
    """需求齐了 → 直接把型号给客户，不许再拿"我在准备推荐"把推荐盖掉。"""

    def _state(self, **extra):
        state = {
            "session_id": "lcd-direct-reco",
            "current_message": "给我推荐",
            "intent": "need_query",
            "next_action": "trigger_solution",
            "display_type_decision": {"display_type": "LCD", "status": "CONFIRMED"},
            "requirement_profile": _profile(lcd_category="conference_education"),
            "lcd_action": {
                "confirmed": True,
                "question": "",
                "question_slot": "",
                "lcd_category": "conference_education",
                "missing_fields": [],
            },
            "pending_question": "",
            "pending_slot": "",
            "requirements": {},
            "additional_requirements": [],
            "response": (
                "Omni T65-K4/K4C is the closest fit for your requirement: 65\" panels, "
                "3840x2160 UHD@60Hz resolution. Shall I prepare the quotation?"
            ),
            "solutions": [{"metadata": {"model": "Omni T65-K4/K4C"}}],
            "should_generate_solution": True,
        }
        state.update(extra)
        return state

    def test_router_recommendation_is_not_overwritten(self, monkeypatch):
        import importlib

        sg = importlib.import_module("src.agents.sales.nodes.script_generator")

        def _boom(*_args, **_kwargs):  # pragma: no cover - 不该被调到
            raise AssertionError("需求已齐、推荐已产出 → 不该再调用表达层重写回复")

        monkeypatch.setattr(sg, "_natural_reply", _boom)

        out = sg.script_generator(self._state())

        assert out["response"].startswith("Omni T65-K4/K4C"), out["response"]
        assert "confirm" not in out["response"].lower()

    def test_router_fallback_text_is_also_kept_instead_of_a_confirm_question(self, monkeypatch):
        """router 跑了但没拿到型号（solutions 为空）→ 保留它的兜底话，不要改问需求。"""
        import importlib

        sg = importlib.import_module("src.agents.sales.nodes.script_generator")

        def _boom(*_args, **_kwargs):  # pragma: no cover - 不该被调到
            raise AssertionError("router 已经产出回复 → 不该再调用表达层")

        monkeypatch.setattr(sg, "_natural_reply", _boom)

        out = sg.script_generator(
            self._state(
                solutions=[],
                response="Based on what you described, I've shortlisted some suitable screens.",
            )
        )

        assert out["response"].startswith("Based on what you described"), out["response"]
        assert "?" not in out["response"]

    def test_without_a_recommendation_it_still_says_something_useful(self, monkeypatch):
        """router 没产出（比如 Solution 失败）时，兜底说一句"正在准备推荐"，但不能问需求。"""
        import importlib

        sg = importlib.import_module("src.agents.sales.nodes.script_generator")

        monkeypatch.setattr(
            sg,
            "_natural_reply",
            lambda *_a, **_k: "Understood, I'm putting the recommendation together now.",
        )

        out = sg.script_generator(
            self._state(next_action="ask", response="", solutions=[])
        )

        assert out["response"], "兜底也不能给空回复"
        assert "?" not in out["response"]


class TestFactsComeFromContext:
    """需求事实优先由语境理解给出；正则只是兜底。"""

    def test_parse_returns_both_category_and_facts(self):
        parsed = lcu._parse(
            '{"category": "conference_education", "confidence": 0.9,'
            ' "scene": "meeting room", "reason": "meetings",'
            ' "facts": {"handwriting": true, "camera": true, "screen_size_inch": 65,'
            ' "resolution": "4k"}, "evidence": {"camera": "i need a cam"}}'
        )

        assert parsed["category"] == "conference_education"
        assert parsed["facts"]["handwriting"] is True
        assert parsed["facts"]["camera"] is True
        assert parsed["facts"]["screen_size_inch"] == 65.0
        assert parsed["facts"]["resolution"] == "4K"
        assert parsed["evidence"]["camera"] == "i need a cam"

    def test_parse_keeps_facts_even_when_the_category_is_unclear(self):
        parsed = lcu._parse('{"category": "nonsense", "facts": {"camera": true}}')

        assert parsed["category"] == "unknown"
        assert parsed["facts"]["camera"] is True

    def test_parse_drops_bogus_fact_values(self):
        parsed = lcu._parse(
            '{"category": "normal", "facts": {"screen_size_inch": 9999,'
            ' "bezel_mm": -3, "resolution": "8K", "unknown_field": 1}}'
        )

        assert parsed["facts"] == {}, parsed["facts"]

    def test_understanding_covers_phrasings_the_regex_would_miss(self):
        """客户说"想在会上写写画画"（一个关键词都没有）→ 语境理解给出 handwriting。"""
        message = "the team would like to sketch on it during the meeting"
        signal = {"category": "conference_education", "facts": {"handwriting": True}}

        facts = extract_lcd_facts(message, fact_signal=signal)

        assert facts.handwriting is True, "正则认不出来，但语境理解认得出来"
        assert facts.touch is True, "要手写就一定带触控"

    def test_camera_can_be_understood_without_the_word_camera(self):
        message = "we also want a lens on top for video calls"
        signal = {"facts": {"camera": True}}

        facts = extract_lcd_facts(message, fact_signal=signal)

        assert facts.camera is True

    def test_a_negative_answer_is_a_valid_fact(self):
        """客户说"不用"是有效答案（false），不是"没提到"。"""
        signal = {"facts": {"ops": False, "tender": False}}

        facts = extract_lcd_facts("no thanks", fact_signal=signal)

        assert facts.ops is False
        assert facts.tender is False

    def test_regex_still_works_when_the_model_is_unavailable(self):
        """模型不可用（没有语境信号）→ 全部退回正则兜底，链路不能死。"""
        facts = extract_lcd_facts(
            "We need a 6x2 video wall, indoor, 3.5mm bezel, with a camera"
        )

        assert facts.is_splicing is True
        assert facts.layout == "6x2"
        assert facts.environment == "indoor"
        assert facts.bezel_mm == 3.5
        assert facts.camera is True

    def test_lcd_turn_applies_the_context_facts_end_to_end(self):
        profile = _profile()
        signal = {
            "category": "conference_education",
            "facts": {"handwriting": True, "screen_size_inch": 65.0, "camera": True},
        }

        profile, action = lcd_turn(
            profile,
            "the team would like to sketch on it during the meeting",
            category_signal=signal,
        )

        assert profile.lcd_handwriting_required is True
        assert profile.lcd_size_inch == 65.0
        assert profile.lcd_camera_required is True
        assert action.lcd_category == "conference_education"

    def test_context_facts_win_over_the_regex(self):
        """模型读了上下文说没有摄像头 → 压过"这一句里出现了 camera"的兜底判断。"""
        signal = {"facts": {"camera": False}}

        facts = extract_lcd_facts("do we need a camera?", fact_signal=signal)

        assert facts.camera is False


class TestUnderstandingCallIsUsedByTheSalesNode:
    def test_sales_node_applies_the_understanding_facts(self, monkeypatch):
        """销售节点必须把语境理解的 facts 交给 lcd_turn —— 不是另起一套关键词。"""
        import src.dialogue.lcd_category_understanding as understanding
        from src.agents.sales.nodes import requirement as sales_req

        captured = {}

        def _fake_understand(message, **kwargs):
            captured["message"] = message
            return {
                "category": "conference_education",
                "confidence": 0.9,
                "facts": {"camera": True},
            }

        monkeypatch.setattr(understanding, "understand_lcd_category", _fake_understand)

        profile = _profile()
        state = {
            "session_id": "ctx-facts-sales",
            "current_message": "i need a cam",
            "display_type_decision": {"display_type": "LCD", "status": "CONFIRMED"},
            "requirement_profile": profile,
        }

        action = sales_req._lcd_requirement_action(state, profile, "i need a cam")

        assert action is not None, action
        assert captured["message"] == "i need a cam"
        assert profile.lcd_camera_required is True, "语境理解的 camera=True 要落到档案"
