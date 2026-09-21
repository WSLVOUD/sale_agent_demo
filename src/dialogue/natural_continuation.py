"""v2.7 §12（Phase 11）：NaturalContinuation —— 自然地承接客户上一句。

输入：customer_message / speech_act / newly_extracted_information /
      previous_action / momentum / next_required_slot
输出：``natural_action_context``（这一轮的表达上下文）

计划原文的例子：

    客户: 3 by 5 meters.    → 不要 "Thank you. I have recorded your screen size."
                             直接 "What is the typical viewing distance?"
    客户: Around 8 meters.  → "And will it be installed indoors or outdoors?"
    客户: Indoor.           → "Will it be a fixed installation or a rental setup?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .response_density import (
    DETAILED,
    MINIMAL,
    NORMAL,
    decide_response_density,
    short_question_response,
)


@dataclass
class NaturalContinuation:
    """这一轮"怎么接话"的结构化上下文（不是话术模板）。"""

    mode: str = "ask"          # answer / ask / answer_then_ask / recommend / acknowledge
    opening: str = ""
    transition: str = ""
    density: str = NORMAL
    avoid_ack: bool = True
    direct_question: bool = False
    momentum_slot: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "opening": self.opening,
            "transition": self.transition,
            "density": self.density,
            "avoid_ack": self.avoid_ack,
            "direct_question": self.direct_question,
            "momentum_slot": self.momentum_slot,
            "notes": list(self.notes),
        }


_TRANSITIONS = {
    "environment": "And ",
    "installation": "",
    "size": "",
    "pixel_pitch": "",
    "viewing_distance": "And ",
    "purpose": "",
    "price_preference": "",
}


def build_natural_continuation(
    *,
    customer_message: str = "",
    speech_act: Any = None,
    newly_filled_slots: Optional[Iterable[str]] = None,
    newly_extracted_information: Optional[Iterable[str]] = None,
    previous_action: str = "",
    momentum: Any = None,
    next_required_slot: str = "",
    customer_question: bool = False,
    question_kind: str = "",
    conflicts: Optional[Iterable[str]] = None,
    has_recommendation: bool = False,
    has_engineering: bool = False,
    is_first_contact: bool = False,
    is_greeting: bool = False,
) -> NaturalContinuation:
    """算出这一轮的承接方式 + 密度（纯函数）。"""
    filled = [
        str(slot)
        for slot in (newly_filled_slots or newly_extracted_information or [])
        if slot
    ]
    density = decide_response_density(
        customer_question=customer_question,
        question_kind=question_kind,
        newly_filled_slots=filled,
        conflicts=conflicts,
        has_recommendation=has_recommendation,
        has_engineering=has_engineering,
        is_first_contact=is_first_contact,
        is_greeting=is_greeting,
    )
    momentum_slot = str(getattr(momentum, "slot", "") or "")

    if has_recommendation or has_engineering:
        mode = "recommend"
    elif customer_question and next_required_slot:
        mode = "answer_then_ask"
    elif customer_question:
        mode = "answer"
    elif next_required_slot:
        mode = "ask"
    else:
        mode = "acknowledge"

    return NaturalContinuation(
        mode=mode,
        opening="",   # 计划 §16：普通参数不确认、不铺垫
        transition=_TRANSITIONS.get(momentum_slot, "") if mode == "ask" else "",
        density=density,
        avoid_ack=True,
        direct_question=density == MINIMAL and mode in ("ask", "answer_then_ask"),
        momentum_slot=momentum_slot,
        notes=[f"speech_act={getattr(speech_act, 'speech_act', '') or ''}"],
    )


def render_minimal(
    continuation: NaturalContinuation,
    *,
    question: str = "",
    answer: str = "",
) -> str:
    """MINIMAL 密度下的最终文本：要么那一句答案，要么那一个问题。"""
    if continuation.mode == "answer" and answer:
        return short_question_response(answer)
    if question:
        return short_question_response(f"{continuation.transition}{question}")
    return short_question_response(answer)


__all__ = [
    "DETAILED",
    "MINIMAL",
    "NORMAL",
    "NaturalContinuation",
    "build_natural_continuation",
    "render_minimal",
]
