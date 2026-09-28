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


# ── 承接轮（客户说题外话时只接住、不再抛问题）─────────────────────────────
# 从 Orchestrator 迁入（计划《Orchestrator.py 二次瘦身计划》§二/§五）：
# "承接这一句"属于对话表达，不属于编排。
NEUTRAL_CONTINUATIONS: tuple[str, ...] = (
    "That makes sense.",
    "Right, I follow you.",
    "Good to know.",
    "Makes sense so far.",
)


def neutral_continuation(seed: int = 0) -> str:
    """没拿到 LLM 接话时的中性兜底（短、像人、不复读客户原话）。"""
    return NEUTRAL_CONTINUATIONS[int(seed or 0) % len(NEUTRAL_CONTINUATIONS)]


def continuation_only_text(
    *,
    text: str,
    acknowledgement: str = "",
    customer_input: str = "",
) -> str:
    """承接轮：把正文里的问句去掉，只留"接住客户那句话"的内容。

    客户口径：客户没回答时不要再抛问题；先顺着客户的消息聊一句，
    最多一条之后（由 continuation_budget 控制）必须拉回需求。
    """
    from .final_guard import FinalResponseGuard

    guard = FinalResponseGuard()
    stripped = guard.strip_questions(str(text or "")).strip()
    if stripped:
        return stripped
    # 去掉问句后没内容了 → 用这一轮的"接话"（LLM 生成的 acknowledgement）
    ack = str(acknowledgement or "").strip()
    if ack:
        return guard.strip_questions(ack).strip() or ack
    # 兜底：把客户刚说的话接住（不提问）
    customer = " ".join(str(customer_input or "").split())
    if customer:
        # 客户口径：不要"回执腔"复读客户原话（"3*5 — noted." 很僵硬），
        # 用一句简短的人话接住即可（真正的接话由 LLM 的 acknowledgement 负责）。
        return neutral_continuation(seed=len(customer))
    return "Got it."


__all__ = [
    "DETAILED",
    "MINIMAL",
    "NORMAL",
    "NEUTRAL_CONTINUATIONS",
    "NaturalContinuation",
    "build_natural_continuation",
    "continuation_only_text",
    "neutral_continuation",
    "render_minimal",
]
