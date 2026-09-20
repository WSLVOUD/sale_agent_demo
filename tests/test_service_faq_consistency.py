"""v2.2.5：服务口径不能自相矛盾（实测日志）。

客户日志里出现过这样一段回复：

    We usually don't provide on-site installation ... but every order includes an
    installation guide that ships with your goods. **Yes, installation is included.**

前一句是公司口径（不提供现场安装，只随货给安装指导说明书），后一句是模型把
"随货说明书"说成了"包安装"。这里锁死：任何来源的这类说法都要被清掉。
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.service_faq import (  # noqa: E402
    FACTS,
    FAQ_INSTALLATION,
    FAQ_WARRANTY,
    detect_service_faq,
    sanitize_service_reply,
    strip_contradictory_installation_claims,
)


CONTRADICTING_REPLIES = (
    "Yes, installation is included.",
    "Installation is included with every order.",
    "The installation will be provided by our team.",
    "We can provide on-site installation for you.",
    "We'll install the screen for you.",
    "价格里含安装，包安装。",
)


class TestInstallationClaimsAreStripped:

    @pytest.mark.parametrize("reply", CONTRADICTING_REPLIES)
    def test_stripped(self, reply):
        cleaned = strip_contradictory_installation_claims(reply)
        lowered = cleaned.lower()
        assert "included" not in lowered or "installation guide" in lowered
        assert "包安装" not in cleaned
        assert "on-site installation for you" not in lowered

    def test_keeps_the_true_statement(self):
        reply = (
            "In general we do not provide on-site installation, it is usually "
            "cheaper to use a local installer. Yes, installation is included."
        )
        cleaned = strip_contradictory_installation_claims(reply)
        assert "do not provide on-site installation" in cleaned
        assert "installation is included" not in cleaned.lower()

    def test_keeps_installation_guide_mention(self):
        reply = "Every order includes an installation guide that ships with your goods."
        assert strip_contradictory_installation_claims(reply) == reply

    def test_sanitize_only_touches_installation_kind(self):
        reply = "Yes, installation is included."
        assert sanitize_service_reply(reply, FAQ_INSTALLATION) != reply
        assert sanitize_service_reply(reply, FAQ_WARRANTY) == reply


class TestWholeReplySanitizer:
    """编排层：客户问安装 → 回复里矛盾的说法必须先被清掉，再接标准回答。"""

    def test_faq_kind_detected(self):
        assert detect_service_faq("Do you also install it?") == FAQ_INSTALLATION
        assert detect_service_faq("你们包安装吗") == FAQ_INSTALLATION

    def test_contradiction_removed_before_attaching(self):
        response = "Yes, installation is included."
        cleaned = sanitize_service_reply(response, detect_service_faq("do you install it?"))
        assert cleaned == ""
        # 标准回答本身必须说清"不提供现场安装"
        fact = FACTS[FAQ_INSTALLATION].lower()
        assert "do not provide on-site installation" in fact
        assert "installation guide" in fact
