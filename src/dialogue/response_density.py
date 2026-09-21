"""v2.7 §14/§15（Phase 13）：Response Density —— 回复密度。

    MINIMAL    客户只给了一个参数 → 直接问下一个（不铺垫、不确认）
    NORMAL     需要一点上下文 → 一句承接 + 一个问题
    DETAILED   技术解释 / 推荐 / 工程计算 / 客户要求详细说明

计划 §15 的短回答模式：客户没有主动问题 + 只提供一个参数 + 没有冲突 +
没有推荐结果 → MINIMAL。

    客户: 8 meters.
    AI:   "Will the screen be installed indoors or outdoors?"
"""
from __future__ import annotations

from typing import Iterable, Optional

MINIMAL = "MINIMAL"
NORMAL = "NORMAL"
DETAILED = "DETAILED"

ALL_DENSITIES = (MINIMAL, NORMAL, DETAILED)


def decide_response_density(
    *,
    customer_question: bool = False,
    question_kind: str = "",
    newly_filled_slots: Optional[Iterable[str]] = None,
    conflicts: Optional[Iterable[str]] = None,
    has_recommendation: bool = False,
    has_engineering: bool = False,
    customer_requested_detail: bool = False,
    is_first_contact: bool = False,
    is_greeting: bool = False,
) -> str:
    """算这一轮该说多长（纯函数）。"""
    filled = [str(item) for item in (newly_filled_slots or []) if item]
    conflicts = [str(item) for item in (conflicts or []) if item]
    kind = str(question_kind or "").upper()

    if has_recommendation or has_engineering or customer_requested_detail:
        return DETAILED
    if is_first_contact or is_greeting:
        return NORMAL
    if conflicts:
        return NORMAL
    if customer_question:
        return MINIMAL if kind in ("PRICE_QUESTION", "DELIVERY_QUESTION") else NORMAL
    if len(filled) == 1:
        return MINIMAL
    if filled:
        return NORMAL
    return NORMAL


def short_question_response(question: str) -> str:
    """§15：短回答模式 —— 就是那一个问题本身。"""
    return " ".join(str(question or "").split())


__all__ = [
    "ALL_DENSITIES",
    "DETAILED",
    "MINIMAL",
    "NORMAL",
    "decide_response_density",
    "short_question_response",
]
