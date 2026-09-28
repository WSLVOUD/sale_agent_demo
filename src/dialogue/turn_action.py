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


# ── 候选 → 唯一动作（从 Orchestrator 迁入，计划《二次瘦身计划》§二）──────────
def dialogue_action_candidates(sales_result: Any) -> List[Dict[str, Any]]:
    """本轮出现过的**候选** Action（最终只执行一个，计划 §4.4）。

    候选可以有多个（Policy 选的、QuestionFlow 想追问的…），
    但"候选 ≠ 最终"：最终动作由 ``FinalResponse.action`` 唯一确定。
    """
    data = sales_result or {}
    candidates: List[Dict[str, Any]] = []
    decision = data.get("dialogue_action") or {}
    if isinstance(decision, dict) and decision.get("action"):
        candidates.append(dict(decision))
    if data.get("pending_question"):
        candidates.append({
            "action": "ask_only",
            "target_slot": str(data.get("pending_slot") or ""),
            "question": str(data.get("pending_question") or ""),
            "source": "question_flow",
        })
    return candidates


def select_turn_action(result: Any) -> "tuple[Any, List[Any]]":
    """把本轮出现的候选 Action 收成唯一一个（计划 §4.4/§4.5）。

    Returns:
        ``(selected, discarded)``；没有候选时返回 ``(None, [])``。
    """
    data = result if isinstance(result, dict) else {}
    try:
        from .action import (
            QUESTION_PRIORITY_SALES_PREFERENCE,
            DialogueDecision,
            question_priority,
            select_single_action,
        )
    except Exception:  # pragma: no cover - 防御式
        return None, []
    candidates: List[Any] = []
    decision = data.get("dialogue_action") or {}
    if isinstance(decision, dict) and decision.get("action"):
        slot = str(decision.get("target_slot") or "") or str(data.get("pending_slot") or "")
        candidates.append(DialogueDecision(
            action=str(decision.get("action")),
            reason=str(decision.get("reason") or ""),
            question_slot=slot if decision.get("question") else "",
            question_target=slot if decision.get("question") else "",
            question_priority=int(
                decision.get("priority") or QUESTION_PRIORITY_SALES_PREFERENCE
            ),
            question_count=1 if decision.get("question") else 0,
        ))
    # ── Phase 1（架构收口）：DialoguePolicy 定了槽位就以它为准 ──────────────
    # 计划 §16.1/§16.2：QuestionPlanner / QuestionFlow / script_generator
    # 只能"提候选"，不能改最终决定。以前这里无条件把销售层准备的
    # pending_slot 也当成候选，于是它可能压过 Policy 的选择
    # （实测：Policy=ASK(viewing_distance) 最终却问了 pixel_pitch）。
    policy_decided_slot = (
        str((data.get("dialogue_action") or {}).get("target_slot") or "")
        if isinstance(data.get("dialogue_action"), dict)
        else ""
    )
    if data.get("pending_question") and not policy_decided_slot:
        slot = str(data.get("pending_slot") or "")
        # 注意：这里必须用 **Dialogue Policy 的动作词表**（ask_only），
        # 不能混进 DialogueDecision 的 ASK —— 否则日志里同一件事会有两个名字。
        candidates.append(DialogueDecision(
            action="ask_only",
            reason="question_flow",
            question_slot=slot,
            question_target=slot,
            question_priority=question_priority(slot) if slot else 99,
            question_count=1,
        ))
    if not candidates:
        return None, []
    return select_single_action(candidates)


def question_text_for_slot(slot: str) -> str:
    """按槽位取标准问句（Policy 定槽位、模板给句子，措辞仍由 LLM 重写）。

    只在"销售层准备的问句与 Policy 定的槽位不一致"时兜底用。
    """
    target = str(slot or "").strip()
    if not target:
        return ""
    try:
        from ..rag.readiness import question_for

        return str(question_for(target, "en", 0, easier=False) or "")
    except Exception:  # pragma: no cover - 防御式
        import logging

        logging.getLogger(__name__).warning(
            "[ActionConsistency] 取标准问句失败 slot=%s", target, exc_info=True
        )
        return ""


def align_question_with_policy(
    result: Dict[str, Any],
    *,
    selected_action: Any,
    question_slot: str,
    duplicate_check: str = "",
) -> "tuple[str, List[Dict[str, Any]]]":
    """Phase 1（计划 §16.2）：最终问的必须是 DialoguePolicy 定的那一项。

    销售层准备好的问句只能"提候选"：如果 Policy 定的槽位与它不一致，以 Policy
    为准（日志里问的槽位和客户看到的问句必须一致）。
    例外：重复提问闸门刚刚**故意改问别的槽位**（duplicate_question*）——
    那是同一决策层的防重放行，不能反过来被覆盖。

    Returns:
        ``(question_slot, questions)``
    """
    from .final_response import question_candidates_from_result

    policy_slot = str(getattr(selected_action, "question_slot", "") or "")
    firewall_rerouted = str(duplicate_check or "").startswith("duplicate_question")
    if (
        firewall_rerouted
        or not policy_slot
        or not question_slot
        or policy_slot == question_slot
    ):
        return question_slot, question_candidates_from_result(result)

    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        "[ActionConsistency] Policy=ASK(%s) 与销售层准备的问句(%s)不一致 → 以 Policy 为准",
        policy_slot, question_slot,
    )
    overridden_slot = question_slot
    aligned = question_text_for_slot(policy_slot)
    question_slot = policy_slot if aligned else ""
    result["pending_slot"] = question_slot
    result["pending_question"] = aligned
    if aligned:
        # 正文里那句"问错槽位"的问句必须换掉：先去掉旧问句，再补 Policy 槽位的问句
        from .final_guard import FinalResponseGuard

        kept = FinalResponseGuard().strip_questions(
            str(result.get("response") or "")
        ).strip()
        result["response"] = f"{kept} {aligned}".strip() if kept else aligned
    result["action_consistency"] = {
        "policy_slot": policy_slot,
        "overridden_slot": overridden_slot,
        "question_realigned": bool(aligned),
    }
    return question_slot, question_candidates_from_result(result)


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
    "dialogue_action_candidates",
    "align_question_with_policy",
    "next_candidate_slot",
    "question_text_for_slot",
    "select_turn_action",
]
