"""类型必须**客户确认**才进需求链 + 已核对过的图片事实不再问（客户口径 2026-09-30）。

实测 bug（session_1790746595164_oxuj22ybj）：

    客户发了一张明显是 LCD 的图（有拼缝、有边框）→ 图片判成 LED →
    `Turn understanding: domain=LED status=INFERRED locked=False entry=LED_ENTRY`
    → 整轮按 LED 口径采集需求 → 客户回 "no,this is lcd" 才纠正回来。
    纠正后又说了 LED 口径的话（"The cabinet and brightness…"），
    而且把图片已经识别出来、客户已经核对过的"室内"又问了一遍。

这里守住四件事：

    1. 类型只是"图片识别 / 场景推断"（INFERRED）时，**不进** LED / LCD 需求链；
    2. 客户确认过（或客户没反对、图片结果已被核对 accepted）才算数；
    3. 客户已核对过的图片事实（vision_accepted）不再重复问；
    4. environment 的问句意图里不许带 LED 口径的"箱体 / 亮度"。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.sales.nodes import requirement as sales_req  # noqa: E402
from src.dialogue.lcd_decision import lcd_turn  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402
from src.rag.readiness import question_intent  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


def _state(message: str, decision: dict, **extra) -> dict:
    state = {
        "messages": [{"role": "user", "content": message}],
        "current_message": message,
        "session_id": "type-gate-test",
        "requirements": {},
        "additional_requirements": [],
        "intent": "need_query",
        "next_action": "ask",
        "should_generate_solution": False,
        "response": "",
        "pending_question": "",
        "pending_slot": "",
        "acknowledgement": "",
        "display_type_decision": decision,
    }
    state.update(extra)
    return state


class TestTypeGate:
    @pytest.fixture(autouse=True)
    def _no_llm(self, monkeypatch):
        """需求抽取 / 接话里的 LLM 一律打桩（这里测的是闸门，不是抽取）。"""
        from types import SimpleNamespace

        import src.core.requirement_extractor as extractor_mod

        monkeypatch.setattr(
            extractor_mod.RequirementExtractor,
            "_llm_semantic_extract",
            lambda self, message, rule_slots, session_id="": {},
        )
        extractor_mod.RequirementExtractor._semantic_cache.clear()

        class _FakeLLM:
            def __init__(self, *args, **kwargs):
                pass

            def invoke(self, *args, **kwargs):
                return SimpleNamespace(content="")

        monkeypatch.setattr(sales_req, "ChatOpenAI", _FakeLLM)
        monkeypatch.setattr("src.core.llm.get_llm", lambda *a, **k: _FakeLLM())
        yield

    def test_image_inferred_led_does_not_enter_the_led_chain(self):
        """图片判成 LED 但客户还没确认 → 这一轮只问类型，不采集 LED 需求。"""
        state = _state(
            "i need a display,like this",
            {"display_type": "LED", "status": "INFERRED", "source": "vision",
             "ask_customer": True, "locked": False},
        )

        out = sales_req.requirement_mining(state)

        assert out["pending_slot"] == "display_type", out.get("pending_slot")
        assert out["pending_question"], "必须把类型确认问题问出去"
        assert out["should_generate_solution"] is False
        assert out["recommendation_gate"]["gate"] == "product_type"
        # 关键：没有把 LED 的硬性条件（点间距 / 视距 / 尺寸）塞进这一轮
        assert not out.get("lcd_action")
        assert "pixel_pitch" not in str(out.get("pending_question") or "")

    def test_confirmed_led_enters_the_chain(self):
        state = _state(
            "led",
            {"display_type": "LED", "status": "CONFIRMED", "locked": True},
        )

        out = sales_req.requirement_mining(state)

        assert out.get("pending_slot") != "display_type", out.get("pending_slot")
        assert out.get("recommendation_gate", {}).get("gate") != "product_type"

    def test_inferred_lcd_does_not_enter_the_lcd_chain(self):
        state = _state(
            "we need screens for a meeting room",
            {"display_type": "LCD", "status": "INFERRED", "source": "inference",
             "ask_customer": True, "locked": False},
        )

        out = sales_req.requirement_mining(state)

        assert out["pending_slot"] == "display_type"
        assert not out.get("lcd_action"), "类型没确认就不许进 LCD 需求链"

    def test_profile_customer_source_counts_as_confirmed(self):
        profile = _profile()
        profile.sources["display_type"] = "explicit"
        state = _state(
            "we need a video wall for a control room",
            {"display_type": "LCD", "status": "UNKNOWN", "locked": False},
            requirement_profile=profile,
        )

        assert sales_req._type_confirmed(state, profile) is True

    def test_vision_accepted_type_counts_as_confirmed(self):
        """客户没正面回答 → 按识别结果走，也算已定（客户口径）。"""
        profile = _profile()
        profile.sources["display_type"] = "vision_accepted"
        state = _state(
            "how much is it?",
            {"display_type": "LCD", "status": "INFERRED", "locked": False},
            requirement_profile=profile,
        )

        assert sales_req._type_confirmed(state, profile) is True

    def test_vision_explicit_type_alone_is_not_enough(self):
        """图片刚看到、还没跟客户核对（vision_explicit）→ 不算确认。"""
        profile = _profile()
        profile.sources["display_type"] = "vision_explicit"
        state = _state(
            "like this",
            {"display_type": "LCD", "status": "INFERRED", "locked": False},
            requirement_profile=profile,
        )

        assert sales_req._type_confirmed(state, profile) is False

    def test_lcd_domain_requires_confirmation(self):
        profile = _profile()
        profile.sources["display_type"] = "vision_explicit"

        inferred = _state(
            "", {"display_type": "LCD", "status": "INFERRED", "locked": False},
            requirement_profile=profile,
        )
        assert sales_req._lcd_domain(inferred, profile) == ""

        confirmed = _state(
            "", {"display_type": "LCD", "status": "CONFIRMED", "locked": True},
            requirement_profile=profile,
        )
        assert sales_req._lcd_domain(confirmed, profile) == "LCD"


class TestConfirmedImageFactsAreNotAskedAgain:
    def _monitoring_profile(self, environment_source: str) -> RequirementProfile:
        profile = _profile(
            lcd_category="monitoring",
            lcd_is_splicing=True,
            lcd_splicing_layout="3x3",
            lcd_screen_count=9,
            lcd_bezel_mm=3.5,
            lcd_size_inch=65.0,
            environment="indoor",
        )
        profile.sources["environment"] = environment_source
        return profile

    def test_vision_accepted_fact_is_not_asked_again(self):
        """图片说 indoor，客户核对过没反对 → 不能再问"室内还是室外"。"""
        profile = self._monitoring_profile("vision_accepted")

        _profile_out, action = lcd_turn(profile, "65 inch")

        assert "environment" not in action.missing_fields, action.missing_fields
        assert action.question_slot != "environment", action.question_slot

    def test_vision_explicit_fact_still_needs_the_confirmation_round(self):
        """刚识别出来、还没核对（vision_explicit）→ 仍然不算客户已经回答。"""
        profile = self._monitoring_profile("vision_explicit")

        _profile_out, action = lcd_turn(profile, "65 inch")

        assert "environment" in action.missing_fields, action.missing_fields


class TestLedWordingDoesNotLeakIntoLcd:
    def test_environment_intent_has_no_led_only_wording(self):
        """environment 的问句意图交给表达层 LLM，不能带 LED 口径的词。"""
        intent = question_intent("environment")

        assert "箱体" not in intent, intent
        assert "亮度" not in intent, intent
