"""v2.7 §9/§10（Phase 9）：唯一 Dialogue Decision。

计划 §9：RequirementProfile **不等于** Dialogue Decision。

    RequirementProfile  只说明：客户知道什么 / 不知道什么 / 确认了什么 / 哪里冲突
    Dialogue Policy     才决定：这一轮到底做什么

一个 Turn **只能**选一个 Action：

    ANSWER / ASK / ANSWER_AND_ASK / RECOMMEND / CALCULATE / CLARIFY / WAIT / HANDOFF

计划 §10：``缺失 ≠ 现在必须问``。提问要过一遍

    Missing Slot → Business Relevance → Conversation Context → Customer Intent
                → Required Now? → Question

优先级（§10.1）：P0 客户提问 > P1 刚提供的信息 > P2 纠正 > P3 自然延伸 >
P4 推荐必需 > P5 其它辅助。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

ANSWER = "ANSWER"
ASK = "ASK"
ANSWER_AND_ASK = "ANSWER_AND_ASK"
RECOMMEND = "RECOMMEND"
CALCULATE = "CALCULATE"
CLARIFY = "CLARIFY"
WAIT = "WAIT"
HANDOFF = "HANDOFF"

ALL_ACTIONS = (ANSWER, ASK, ANSWER_AND_ASK, RECOMMEND, CALCULATE, CLARIFY, WAIT, HANDOFF)

P0_CUSTOMER_QUESTION = 0
P1_NEW_INFORMATION = 1
P2_CORRECTION = 2
P3_NATURAL_CONTINUATION = 3
P4_REQUIRED_FOR_RECOMMENDATION = 4
P5_AUXILIARY = 5

PRIORITY_LABELS = {
    P0_CUSTOMER_QUESTION: "customer_question",
    P1_NEW_INFORMATION: "just_provided_information",
    P2_CORRECTION: "correction",
    P3_NATURAL_CONTINUATION: "natural_continuation",
    P4_REQUIRED_FOR_RECOMMENDATION: "required_for_recommendation",
    P5_AUXILIARY: "auxiliary",
}

# 这两类客户提问"答完就结束"，不硬塞需求问题
_ANSWER_ONLY_KINDS = ("PRICE_QUESTION", "DELIVERY_QUESTION")


@dataclass
class TurnAction:
    """这一轮唯一要做的事。"""

    action: str = WAIT
    target_slot: str = ""
    priority: int = P5_AUXILIARY
    reason: str = ""
    question: str = ""
    asks_question: bool = False
    answers_customer: bool = False
    momentum_slot: str = ""
    missing_slots: List[str] = field(default_factory=list)
    blocked_slot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target_slot": self.target_slot,
            "priority": self.priority,
            "priority_label": PRIORITY_LABELS.get(self.priority, ""),
            "reason": self.reason,
            "question": self.question,
            "asks_question": self.asks_question,
            "answers_customer": self.answers_customer,
            "momentum_slot": self.momentum_slot,
            "missing_slots": list(self.missing_slots),
            "blocked_slot": self.blocked_slot,
        }


def decide_turn_action(
    *,
    customer_question: bool = False,
    question_kind: str = "",
    ready_to_recommend: bool = False,
    recommend_requested: bool = False,
    conflicts: Optional[Iterable[str]] = None,
    needs_confirmation: bool = False,
    newly_filled_slots: Optional[Iterable[str]] = None,
    missing_slots: Optional[Iterable[str]] = None,
    question_candidates: Optional[Sequence[str]] = None,
    momentum_slot: str = "",
    previous_question_slot: str = "",
    blocked_slot: str = "",
    hard_gate_slot: str = "",
) -> TurnAction:
    """§9/§10：算这一轮**唯一**的 Action（纯函数，可单测）。"""
    conflicts = [str(item) for item in (conflicts or [])]
    filled = [str(item) for item in (newly_filled_slots or [])]
    missing = [str(item) for item in (missing_slots or [])]
    candidates = [str(item) for item in (question_candidates or [])]
    blocked = str(blocked_slot or "")

    # P2：冲突 / 纠正 / 需要确认 → 先澄清
    if conflicts:
        return TurnAction(
            action=CLARIFY,
            priority=P2_CORRECTION,
            reason="requirement_conflict",
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )
    if needs_confirmation:
        return TurnAction(
            action=CLARIFY,
            priority=P2_CORRECTION,
            reason="vision_confirmation",
            answers_customer=True,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # P0：客户主动提问 → 先回答（§10：即使还缺 viewing_distance 也先答 delivery）
    if customer_question:
        kind = str(question_kind or "").upper()
        if kind in _ANSWER_ONLY_KINDS:
            return TurnAction(
                action=ANSWER,
                priority=P0_CUSTOMER_QUESTION,
                reason="customer_question_answer_only",
                answers_customer=True,
                missing_slots=missing,
                momentum_slot=momentum_slot,
            )
        needed = _required_to_answer(kind, missing)
        if ready_to_recommend:
            return TurnAction(
                action=RECOMMEND,
                priority=P0_CUSTOMER_QUESTION,
                reason="customer_question_gate_ready",
                answers_customer=True,
                missing_slots=missing,
                momentum_slot=momentum_slot,
            )
        if needed and needed != blocked:
            return TurnAction(
                action=ANSWER_AND_ASK,
                target_slot=needed,
                priority=P0_CUSTOMER_QUESTION,
                reason="answer_then_ask_required_field",
                asks_question=True,
                answers_customer=True,
                blocked_slot=blocked,
                missing_slots=missing,
                momentum_slot=momentum_slot,
            )
        return TurnAction(
            action=ANSWER,
            priority=P0_CUSTOMER_QUESTION,
            reason="customer_question_answer_only",
            answers_customer=True,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # 明确要推荐 / Gate 放行
    if recommend_requested or ready_to_recommend:
        return TurnAction(
            action=RECOMMEND,
            priority=P4_REQUIRED_FOR_RECOMMENDATION,
            reason="recommend_requested" if recommend_requested else "gate_ready",
            answers_customer=True,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # P3：顺着客户当前话题的自然延伸
    if momentum_slot and momentum_slot in candidates and momentum_slot != blocked:
        return TurnAction(
            action=ASK,
            target_slot=momentum_slot,
            priority=P3_NATURAL_CONTINUATION,
            reason="natural_continuation",
            asks_question=True,
            blocked_slot=blocked,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # P4：推荐所必需的信息（硬性 Gate 优先）
    required = [slot for slot in missing if slot != blocked]
    if hard_gate_slot and hard_gate_slot in required:
        required = [hard_gate_slot] + [slot for slot in required if slot != hard_gate_slot]
    if not required:
        # 候选里还有没被闸门挡住的 → 按价值问（这里保持候选顺序）
        required = [slot for slot in candidates if slot and slot != blocked]
    if required:
        return TurnAction(
            action=ASK,
            target_slot=required[0],
            priority=P4_REQUIRED_FOR_RECOMMENDATION,
            reason="required_for_recommendation",
            asks_question=True,
            blocked_slot=blocked,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # P1：客户刚提供了信息，但没有必须现在问的 → 只接住
    if filled:
        return TurnAction(
            action=ANSWER,
            priority=P1_NEW_INFORMATION,
            reason="information_recorded",
            answers_customer=True,
            missing_slots=missing,
            momentum_slot=momentum_slot,
        )

    # P5：没有可问的、也没有要答的 → 等客户（不制造问题）
    return TurnAction(
        action=WAIT,
        priority=P5_AUXILIARY,
        reason="nothing_required_now",
        missing_slots=missing,
        momentum_slot=momentum_slot,
    )


def _required_to_answer(question_kind: str, missing: Sequence[str]) -> str:
    """回答这类客户提问**必须**先知道的槽位（没有就返回空）。"""
    if str(question_kind or "").upper() != "PRODUCT_QUESTION":
        return ""
    for slot in ("size", "pixel_pitch", "environment"):
        if slot in missing:
            return slot
    return ""


def next_candidate_slot(
    candidates: Sequence[str],
    *,
    blocked_slot: str = "",
    answered_slots: Optional[Iterable[str]] = None,
) -> str:
    """重复提问被闸门拦下后 → 换下一个合理的问题（§19.1 的补救路径）。"""
    answered = {str(item) for item in (answered_slots or [])}
    for slot in candidates:
        item = str(slot)
        if not item or item == blocked_slot or item in answered:
            continue
        return item
    return ""


__all__ = [
    "ALL_ACTIONS",
    "ANSWER",
    "ANSWER_AND_ASK",
    "ASK",
    "CALCULATE",
    "CLARIFY",
    "HANDOFF",
    "P0_CUSTOMER_QUESTION",
    "P1_NEW_INFORMATION",
    "P2_CORRECTION",
    "P3_NATURAL_CONTINUATION",
    "P4_REQUIRED_FOR_RECOMMENDATION",
    "P5_AUXILIARY",
    "PRIORITY_LABELS",
    "RECOMMEND",
    "TurnAction",
    "WAIT",
    "decide_turn_action",
    "next_candidate_slot",
]
