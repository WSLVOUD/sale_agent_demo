"""推荐完产品之后：客户问什么就答什么（客户口径 2026-09-30）。

实测（客户原文）：

    推荐完 DS-O-75 →
    客户: 你们包安装吗？  → 答了安装口径，**后面又跟了一整段产品介绍**
    客户: 你们的交付日期是多久 → 交期没答，**只有产品介绍**
    客户: 今天天气真不错   → **还是产品介绍**

要求：

    1. 问售后（安装）就只答售后，后面不要再出现产品话术；
    2. 交期问题要在 **LCD 链路里**有自己的回答流程（口径类似 LED，但不套 LED 的话术）；
    3. 推荐过一次之后**不要再一直推荐** —— 客户说什么就答什么，
       而且不许编造没有灌输过的内容，话术要对我方有利。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dialogue.response_coordinator import ResponseCoordinator  # noqa: E402
from src.models.requirement import RequirementProfile  # noqa: E402

PITCH = (
    "Thank you for choosing the DS-O-75 commercial LCD display for your advertising "
    "needs. Its 75-inch panel matches your required screen size, and it delivers "
    "1920x1080p resolution with an optional 3840x2160p upgrade. With 2000nit "
    "brightness, this commercial monitor ensures your advertising content stays vivid."
)


def _lcd_profile() -> RequirementProfile:
    profile = RequirementProfile.from_slots(
        {
            "display_type": "LCD",
            "environment": "outdoor",
            "lcd_category": "advertising",
            "lcd_size_inch": 75.0,
            "lcd_touch_required": True,
        },
        explicit_keys={"display_type", "environment", "lcd_category",
                       "lcd_size_inch", "lcd_touch_required"},
    )
    return profile


@pytest.fixture
def coordinator():
    profile = _lcd_profile()
    return ResponseCoordinator(profile_lookup=lambda _sid: profile)


class TestInstallationQuestion:
    def test_only_the_installation_policy_is_replied(self, coordinator):
        out = coordinator.finalize(
            PITCH, session_id="s1", message="你们包安装吗？", product_family="lcd"
        )

        assert "do not provide on-site installation" in out
        assert "installation guide" in out
        assert "DS-O-75" not in out, "答完售后不要再来一遍产品介绍"
        assert "2000nit" not in out


class TestDeliveryQuestion:
    def test_lcd_has_its_own_delivery_answer(self, coordinator):
        out = coordinator.finalize(
            PITCH, session_id="s1", message="你们的交付日期是多久", product_family="lcd"
        )

        assert "15" in out and "30" in out, out
        assert "LCD panels" in out, "LCD 交期要有自己的口径，不套 LED 的说法"
        assert "DS-O-75" not in out, "交期问题只答交期"

    def test_led_branch_is_untouched(self, coordinator):
        """LED 侧不经过这条（它自己那条链已有交期口径）→ 原样返回。"""
        out = coordinator.finalize(
            PITCH, session_id="s1", message="你们的交付日期是多久", product_family="led"
        )

        assert out == PITCH


class TestOffTopicChat:
    def test_chat_is_acknowledged_without_a_product_pitch(self, coordinator):
        out = coordinator.finalize(
            PITCH,
            session_id="s1",
            message="今天天气真不错",
            product_family="lcd",
            offtopic_turn=True,
        )

        assert "DS-O-75" not in out, "闲聊不要拿产品介绍顶上"
        assert "2000nit" not in out
        assert out.strip(), "也不能给空回复"


class TestNoRepeatedRecommendation:
    def test_completed_requirements_do_not_force_a_recommendation_after_delivery(self):
        from src.agents.solution.nodes.intent import intent_node

        state = {
            "messages": [{"role": "user", "content": "你们包安装吗？"}],
            "requirement_profile": _lcd_profile(),
            "already_recommended": True,
            "intent": "",
        }

        assert intent_node(dict(state))["intent"] != "recommendation"

    def test_first_time_completion_still_recommends(self):
        from src.agents.solution.nodes.intent import intent_node

        state = {
            "messages": [{"role": "user", "content": "yes i need touch"}],
            "requirement_profile": _lcd_profile(),
            "already_recommended": False,
            "intent": "",
        }

        assert intent_node(dict(state))["intent"] == "recommendation"
