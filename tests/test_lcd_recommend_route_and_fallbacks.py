"""需求齐了必须走推荐 + 兜底话术不能串 LED 口径（客户口径 2026-09-30）。

实测（客户原文）：

    会议室 / 65" / 手写 / 无招标 —— 需求都答对了
    → AI 回："Let's take a slightly different angle, if one of the requirements can be
      relaxed (for example the pixel pitch, the screen size, or the viewing distance)…"
    → 客户："需求都是正确的为什么会冒出这句话？而且这句话是 led 使用的。"

两个根因：

    A. 需求齐了以后，Solution 侧按**客户最后一句的表面意思**判意图 ——
       "no,personal" 这种光秃秃的回答被判成 others → 走自由问答 → 压根没进 LCD 选型 →
       没有型号、回复为空 → 掉进"放宽条件"兜底。
    B. 那句兜底话术是 **LED 口径**（点间距 / 观看距离），被 LCD 会话复用了。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.solution.nodes.intent import (  # noqa: E402
    _lcd_requirements_are_complete,
    intent_node,
)
from src.models.requirement import RequirementProfile  # noqa: E402


def _profile(**slots) -> RequirementProfile:
    base = {"display_type": "LCD"}
    base.update(slots)
    return RequirementProfile.from_slots(base, explicit_keys=set(base))


def _complete_ifp_profile() -> RequirementProfile:
    return _profile(
        lcd_category="conference_education",
        lcd_size_inch=65.0,
        lcd_resolution="4K",
        lcd_handwriting_required=True,
        lcd_tender_project=False,
        lcd_ops_required=False,
        lcd_camera_required=False,
        environment="indoor",
    )


class TestRecommendationRouteWhenRequirementsAreComplete:
    def test_bare_answer_after_complete_requirements_still_recommends_merged(self):
        """合并自 4 条同类测试（瘦身；断言全部保留）。"""

        # ── test_bare_answer_after_complete_requirements_still_recommends ──
        """客户最后一句只是回答我们的问题（"no,personal"）→ 这一轮必须给推荐。"""
        state = {
            "messages": [{"role": "user", "content": "no,personal"}],
            "requirement_profile": _complete_ifp_profile(),
            "intent": "",
        }

        out = intent_node(state)

        assert out["intent"] == "recommendation", out["intent"]

        # ── test_caller_product_question_is_still_respected ──
        """客户真在问问题 → 先回答问题，不要拿推荐打断。"""
        state = {
            "messages": [{"role": "user", "content": "does it support 4K input?"}],
            "requirement_profile": _complete_ifp_profile(),
            "intent": "product_question",
        }

        out = intent_node(state)

        assert out["intent"] == "product_question", out["intent"]

        # ── test_incomplete_requirements_do_not_force_a_recommendation ──
        state = {
            "messages": [{"role": "user", "content": "yes i need handwriting"}],
            "requirement_profile": _profile(
                lcd_category="conference_education", lcd_size_inch=65.0
            ),
            "intent": "",
        }

        assert _lcd_requirements_are_complete(state) is False

        # ── test_led_is_not_touched ──
        """LED 链路保持原样（客户口径：LED 那条不许动）。"""
        state = {
            "messages": [{"role": "user", "content": "no"}],
            "requirement_profile": _profile(
                display_type="LED",
                environment="indoor",
                target_width_m=4.0,
                target_height_m=3.0,
                pixel_pitch_mm=4.0,
            ),
            "intent": "",
        }

        assert _lcd_requirements_are_complete(state) is False


class TestRouterPassesTheRecommendationIntent:
    def test_router_tells_the_solution_agent_it_is_a_recommendation(self):
        from src.agents.sales.nodes.router import router

        captured = {}

        class _FakeRunner:
            def run(self, **kwargs):
                captured.update(kwargs)
                return {"answer": "Omni T65-K4/K4C ...", "products": []}

        state = {
            "should_generate_solution": True,
            "solution_runner": _FakeRunner(),
            "requirements": {},
            "additional_requirements": [],
            "current_message": "no,personal",
            "messages": [{"role": "user", "content": "no,personal"}],
            "requirement_profile": _complete_ifp_profile(),
            "already_recommended": False,
            "previous_recommended_models": [],
            "response": "",
        }

        router(state)

        assert captured.get("intent") == "recommendation", captured.get("intent")


class TestFallbacksUseTheRightProductFamily:
    def test_lcd_relaxation_has_no_led_wording_merged(self):
        """合并自 5 条同类测试（瘦身；断言全部保留）。"""

        # ── test_lcd_relaxation_has_no_led_wording ──
        from src.rag.reply_composer import relaxation_answer

        for seed in range(4):
            text = relaxation_answer(seed=seed, product_family="lcd").lower()
            assert "pixel pitch" not in text, text
            assert "viewing distance" not in text, text

        # ── test_led_relaxation_keeps_the_original_wording ──
        from src.rag.reply_composer import relaxation_answer

        assert "pitch" in relaxation_answer().lower()

        # ── test_product_fallback_no_longer_names_the_model（契约更新 2026-10）──
        # 旧契约：正文被清空时报出型号（"Based on your requirements, the closest
        # match is …"）。客户口径已改：那句话在闲聊轮反复冒出来、把真正的回答盖掉
        # （客户原话"总是遮挡了该回答的话"），客户只说了句 "yes" 也会被报一个型号
        # → 等于凭空给一个匹配结果。现在两个调用点只邀请客户补充条件，
        # 函数本身废弃、永远返回空串，不再产生任何型号文案。
        from src.rag.reply_composer import product_fallback_answer

        assert product_fallback_answer(
            [
                {
                    "metadata": {
                        "model": "Omni T65-K4/K4C",
                        "display_size_inch": '65"',
                        "resolution": "3840x2160",
                    }
                }
            ]
        ) == ""
        assert product_fallback_answer([]) == ""

        # ── test_no_product_rewrite_follows_the_family ──
        from src.rag.rerank import sanitize_customer_response

        rewritten = sanitize_customer_response(
            "Sorry, I could not find a matching product for you.",
            product_family="lcd",
        )

        assert "pixel pitch" not in rewritten.lower(), rewritten
        assert "panel size" in rewritten.lower() or "bezel" in rewritten.lower(), rewritten

        # ── test_family_detection_for_a_handwriting_meeting_room ──
        from src.utils.product_family import product_family_of

        assert product_family_of(_complete_ifp_profile()) == "ifp"
        assert product_family_of(_profile(display_type="LED")) == "led"
        assert product_family_of(
            _profile(lcd_category="monitoring", lcd_is_splicing=True)
        ) == "lcd"
