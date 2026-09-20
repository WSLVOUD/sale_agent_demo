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

# 客户在问这些 → 必须先回答
_CUSTOMER_QUESTION_INTENTS = ("product_question", "objection", "others")


@dataclass
class DialogueDecision:
    action: str = ASK
    reason: str = ""
    question_slot: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "question_slot": self.question_slot}


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
        return DialogueDecision(CLARIFY, "requirement_conflict", question_slot)
    if needs_confirmation:
        return DialogueDecision(CONFIRM, "vision_confirmation", "")
    if recommend_requested or ready_to_recommend:
        if customer_question and not ready_to_recommend:
            # 客户问了问题、但需求也够了：先回答，再推荐（由 ResponseContext 承载两件事）
            return DialogueDecision(DIRECT_ANSWER, "answer_then_recommend", "")
        return DialogueDecision(RECOMMEND, "recommend_requested" if recommend_requested else "ready", "")
    if customer_question:
        return DialogueDecision(
            ANSWER_AND_ASK if has_pending_question else DIRECT_ANSWER,
            "customer_asked" + ("_and_missing" if has_pending_question else ""),
            question_slot,
        )
    if has_pending_question:
        return DialogueDecision(ASK, "missing_hard_condition", question_slot)
    if facts_added:
        return DialogueDecision(ACK_ONLY, "just_acknowledge", "")
    if intent in _CUSTOMER_QUESTION_INTENTS:
        return DialogueDecision(DIRECT_ANSWER, "intent_question", "")
    return DialogueDecision(ACK_ONLY, "nothing_to_ask", "")


__all__ = [
    "ACK_ONLY",
    "ALL_ACTIONS",
    "ANSWER_AND_ASK",
    "ASK",
    "CLARIFY",
    "CONFIRM",
    "DIRECT_ANSWER",
    "DialogueDecision",
    "RECOMMEND",
    "decide_dialogue_action",
]
