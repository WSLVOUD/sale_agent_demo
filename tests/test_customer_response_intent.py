"""v2.1 Phase 8：Customer Response Intent（计划 5 / 19 节）。

覆盖《LED_RAG_v2.1_客户决策状态与灵活追问优化实施计划》第 19 节要求的意图识别：
    I don't know / Not sure            → UNKNOWN
    You decide / You recommend         → DELEGATED
    I don't care / doesn't matter      → DECLINED
    客户直接给参数                      → 由抽取链记录（这里验证"不误判成决策"）
    Actually …                          → CORRECTION
    一句话多个决策                      → 逐槽位返回（计划 7.1 / 7.2）
"""
import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.customer_response import (  # noqa: E402
    DECLINED,
    DELEGATED,
    UNKNOWN,
    detect_response_intents,
    is_customer_correction,
    slot_decisions,
)


def _intent(message: str, last_asked: str = "viewing_distance") -> tuple[str, str]:
    """取"这句话对上一轮那个槽位"的意图（槽位, 意图）。"""
    items = detect_response_intents(message, last_asked_slot=last_asked)
    assert items, f"未识别出任何意图：{message!r}"
    return items[0].slot, items[0].intent


class TestUnknownIntent:

    @pytest.mark.parametrize("message", [
        "I don't know",
        "I dont know yet",
        "I have no idea",
        "Not sure",
        "I'm not sure",
        "I don't have the measurements",
        "hard to say",
        "不知道",
        "不太清楚",
    ])
    def test_unknown(self, message):
        assert _intent(message)[1] == UNKNOWN

    def test_unknown_is_not_delegated(self):
        """计划 4.2：「客户不知道」和「客户让 AI 决定」必须区分。"""
        slot, intent = _intent("I don't know")
        assert slot == "viewing_distance"
        assert intent != DELEGATED


class TestDelegatedIntent:

    @pytest.mark.parametrize("message", [
        "You decide",
        "you can decide",
        "You recommend",
        "just recommend something suitable",
        "whatever you recommend",
        "any is fine",
        "你决定吧",
        "你推荐就行",
        "都行",
    ])
    def test_delegated(self, message):
        assert _intent(message)[1] == DELEGATED

    def test_delegated_has_evidence(self):
        item = detect_response_intents("You decide", last_asked_slot="size")[0]
        assert item.evidence
        assert item.confidence > 0.5


class TestDeclinedIntent:

    @pytest.mark.parametrize("message", [
        "I don't want to provide that",
        "that doesn't matter",
        "I don't care about that",
        "no preference",
        "不用问这个",
        "这个不重要",
    ])
    def test_declined(self, message):
        assert _intent(message)[1] == DECLINED


class TestNoIntentForPlainValues:

    @pytest.mark.parametrize("message", [
        "about 5 meters",
        "it's indoors",
        "fixed installation",
        "5m x 3m",
        "P2.5",
        "church",
    ])
    def test_plain_value_is_not_a_decision(self, message):
        """客户直接给值 → Extractor 负责记录；这里不能误判成决策。"""
        assert detect_response_intents(message, last_asked_slot="viewing_distance") == []


class TestCorrection:

    @pytest.mark.parametrize("message", [
        "Actually, it's 5 meters, not 3.",
        "correction, the width is 6m",
        "I meant the height is 3m",
        "更正一下，是 5 米",
    ])
    def test_correction_detected(self, message):
        assert is_customer_correction(message) is True

    def test_plain_answer_is_not_correction(self):
        assert is_customer_correction("about 5 meters") is False


class TestMultiSlotMessages:
    """计划 7.1 / 7.2：一句话可以同时给出多个槽位的决策。"""

    def test_unknown_one_slot_delegated_another(self):
        decisions = slot_decisions(
            "I don't know the viewing distance either, you can decide the pitch.",
            last_asked_slot="viewing_distance",
        )
        assert decisions.get("viewing_distance") == UNKNOWN
        assert decisions.get("pixel_pitch") == DELEGATED

    def test_delegate_size_and_pitch(self):
        decisions = slot_decisions(
            "Indoor church, fixed installation. You decide the size and pitch.",
            last_asked_slot="size",
        )
        assert decisions.get("size") == DELEGATED
        assert decisions.get("pixel_pitch") == DELEGATED

    def test_unknown_width_and_height(self):
        decisions = slot_decisions(
            "I don't know the width and the height.", last_asked_slot="size"
        )
        assert decisions.get("width") == UNKNOWN
        assert decisions.get("height") == UNKNOWN

    def test_later_statement_wins_in_one_message(self):
        """「不知道宽度，你推荐就行」→ 最终是 DELEGATED，不是 UNKNOWN。"""
        decisions = slot_decisions(
            "I don't know the width, just recommend something suitable.",
            last_asked_slot="size",
        )
        assert decisions.get("width") == DELEGATED
