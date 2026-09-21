"""v2.5 Phase 2：DialogueAction —— 先决定"这一轮做什么"，再决定"怎么说"。

客户口径（2026-09-20）：系统不该每轮都走

    ACK → Connector → ASK

而应该根据客户输入决定本轮行为：

    DIRECT_ANSWER   客户在问具体问题（价格 / 交期 / 质保 / 规格…）→ 先回答
    ASK             客户回答后还缺一个关键硬条件 → 问那一个
    ANSWER_AND_ASK  客户既问了问题、需求又还缺 → 先答再问
    RECOMMEND       需求足够 / 客户明确要推荐 → 推荐
    CLARIFY         信息冲突（含工程比例 / 分辨率冲突）→ 澄清
    CONFIRM         需要客户确认图片识别结果
    ACK_ONLY        客户只是补充了一个信息、没什么好问的 → 只回应，不硬塞问题
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

DIRECT_ANSWER = "DIRECT_ANSWER"
ASK = "ASK"
ANSWER_AND_ASK = "ANSWER_AND_ASK"
RECOMMEND = "RECOMMEND"
CLARIFY = "CLARIFY"
CONFIRM = "CONFIRM"
ACK_ONLY = "ACK_ONLY"

ALL_ACTIONS = (DIRECT_ANSWER, ASK, ANSWER_AND_ASK, RECOMMEND, CLARIFY, CONFIRM, ACK_ONLY)

# 计划 §4.5：一个 Turn 只允许一个 Action（候选可以有多个，最终只能留一个）
ONE_TURN_ONE_ACTION = "ONE_TURN_ONE_ACTION"

# 客户可见回复里最多一个问题（计划 §4.6）
MAX_QUESTIONS_PER_TURN = 1

# "这一轮会问一个问题"的动作（两套词表都认：DialogueDecision 的大写 +
# Dialogue Policy 的小写，避免同一个决策在不同层有两个名字）
_ASKING_ACTIONS = frozenset({
    ASK, ANSWER_AND_ASK, CLARIFY,
    "ask_only", "answer_then_ask", "clarify_only",
})

# 客户在问这些 → 必须先回答
_CUSTOMER_QUESTION_INTENTS = ("product_question", "objection", "others")

# ── v2.5+++（对话决策与输出链路优化 §3.3）：问题优先级 ────────────────────
# 同一轮内部若同时产生多个候选问题，只能留**优先级最高**的那一个，
# 其余问题进入下一轮（顺序：硬性 Gate 缺失 > 工程决策必要 > 推荐优化 > 销售偏好）。
QUESTION_PRIORITY_HARD_GATE = 0
QUESTION_PRIORITY_ENGINEERING = 1
QUESTION_PRIORITY_RECOMMENDATION = 2
QUESTION_PRIORITY_SALES_PREFERENCE = 3

QUESTION_PRIORITY_LABELS = {
    QUESTION_PRIORITY_HARD_GATE: "hard_gate_missing",
    QUESTION_PRIORITY_ENGINEERING: "engineering_required",
    QUESTION_PRIORITY_RECOMMENDATION: "recommendation_optimisation",
    QUESTION_PRIORITY_SALES_PREFERENCE: "sales_preference",
}


def question_priority(slot: str) -> int:
    """某个槽位的问题优先级（数字越小越优先）。判定集中在 field_policy。"""
    try:
        from src.rag.field_policy import policy_for

        policy = policy_for(slot)
    except Exception:  # pragma: no cover - 防御式
        return QUESTION_PRIORITY_SALES_PREFERENCE
    if getattr(policy, "hard_condition", False):
        return QUESTION_PRIORITY_HARD_GATE
    if getattr(policy, "required_for_calculation", False) or getattr(
        policy, "required_for_recommendation", False
    ):
        return QUESTION_PRIORITY_ENGINEERING
    priority = int(getattr(policy, "ask_priority", 90) or 90)
    if priority <= 50:
        return QUESTION_PRIORITY_RECOMMENDATION
    return QUESTION_PRIORITY_SALES_PREFERENCE


def question_priority_label(slot: str) -> str:
    return QUESTION_PRIORITY_LABELS.get(question_priority(slot), "sales_preference")


@dataclass
class DialogueDecision:
    action: str = ASK
    reason: str = ""
    question_slot: str = ""
    # v2.5+++：这一轮的问题目标是哪一项、优先级多少、允许出现几个问题（永远 ≤ 1）
    question_target: str = ""
    question_priority: int = QUESTION_PRIORITY_HARD_GATE
    question_count: int = 0

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "question_slot": self.question_slot,
            "question_target": self.question_target,
            "question_priority": self.question_priority,
            "question_priority_label": question_priority_label(self.question_slot)
            if self.question_slot
            else "",
            "question_count": self.question_count,
        }


def _decision(action: str, reason: str, slot: str = "") -> DialogueDecision:
    """统一构造：带上问题目标 / 优先级 / 问题个数（≤1）。"""
    return DialogueDecision(
        action=action,
        reason=reason,
        question_slot=slot,
        question_target=slot,
        question_priority=question_priority(slot) if slot else QUESTION_PRIORITY_HARD_GATE,
        question_count=1 if slot else 0,
    )


def decide_dialogue_action(
    *,
    intent: str = "",
    customer_question: bool = False,
    has_pending_question: bool = False,
    recommend_requested: bool = False,
    ready_to_recommend: bool = False,
    conflicts: bool = False,
    needs_confirmation: bool = False,
    facts_added: bool = False,
    question_slot: str = "",
) -> DialogueDecision:
    """决定本轮对话行为（纯函数，可单测）。

    优先级（客户口径）：
        冲突/澄清 > 图片确认 > 客户提问（先答） > 推荐 > 追问 > 只回应
    """
    if conflicts:
        return _decision(CLARIFY, "requirement_conflict", question_slot)
    if needs_confirmation:
        return _decision(CONFIRM, "vision_confirmation", "")
    if recommend_requested or ready_to_recommend:
        if customer_question and not ready_to_recommend:
            # 客户问了问题、但需求也够了：先回答，再推荐（由 ResponseContext 承载两件事）
            return _decision(DIRECT_ANSWER, "answer_then_recommend", "")
        return _decision(
            RECOMMEND, "recommend_requested" if recommend_requested else "ready", ""
        )
    if customer_question:
        return _decision(
            ANSWER_AND_ASK if has_pending_question else DIRECT_ANSWER,
            "customer_asked" + ("_and_missing" if has_pending_question else ""),
            question_slot,
        )
    if has_pending_question:
        return _decision(ASK, "missing_hard_condition", question_slot)
    if facts_added:
        return _decision(ACK_ONLY, "just_acknowledge", "")
    if intent in _CUSTOMER_QUESTION_INTENTS:
        return _decision(DIRECT_ANSWER, "intent_question", "")
    return _decision(ACK_ONLY, "nothing_to_ask", "")


def select_single_action(
    candidates: "list[DialogueDecision] | tuple[DialogueDecision, ...]",
) -> "tuple[DialogueDecision, list[DialogueDecision]]":
    """计划 §4.4/§4.5：候选 Action → **唯一**最终 Action。

    多个节点（Sales / Solution / 服务口径 / 图片核对）各自可能想"问一项"，
    这里按优先级挑出唯一要执行的那一个（数字越小越优先，并列保留先出现的），
    其余全部丢弃、留给下一轮。

    Returns:
        ``(selected, discarded)``；候选为空时返回 ``(ACK_ONLY 决策, [])``。
    """
    items = [item for item in (candidates or []) if isinstance(item, DialogueDecision)]
    if not items:
        return _decision(ACK_ONLY, "no_candidate_action", ""), []
    best = items[0]
    for item in items[1:]:
        # 先比业务优先级，再比"是否真的要问问题"（要问的优先于仅供参考的）
        if (item.question_priority, 0 if item.question_count else 1) < (
            best.question_priority,
            0 if best.question_count else 1,
        ):
            best = item
    discarded = [item for item in items if item is not best]
    # 计划 §4.6：answer_then_ask 也最多一个问题
    if best.question_count > MAX_QUESTIONS_PER_TURN:
        best.question_count = MAX_QUESTIONS_PER_TURN
    if best.action not in _ASKING_ACTIONS:
        # 这些动作这一轮不问需求问题 → 问题目标必须清空（不能再冒出一个问题）
        best.question_count = 0
        best.question_target = ""
        best.question_slot = ""
    return best, discarded


__all__ = [
    "ACK_ONLY",
    "ALL_ACTIONS",
    "ANSWER_AND_ASK",
    "ASK",
    "CLARIFY",
    "CONFIRM",
    "DIRECT_ANSWER",
    "DialogueDecision",
    "QUESTION_PRIORITY_ENGINEERING",
    "QUESTION_PRIORITY_HARD_GATE",
    "QUESTION_PRIORITY_LABELS",
    "QUESTION_PRIORITY_RECOMMENDATION",
    "QUESTION_PRIORITY_SALES_PREFERENCE",
    "RECOMMEND",
    "MAX_QUESTIONS_PER_TURN",
    "ONE_TURN_ONE_ACTION",
    "decide_dialogue_action",
    "question_priority",
    "question_priority_label",
    "select_single_action",
]
