"""客户口径（2026-09-21）：不要连续提问 —— 先承接，最多 3 条，第 4 条必须问需求。

规则原文：

    · 不要让 AI 出现连续询问的情况；
    · 当客户没有回复 AI 的话术时，可以先顺着客户的消息和客户闲谈，
      然后最多三条闲谈就回到需求询问的话题上；
    · 硬性条件也可以跳转（每个问题没得到答案都可以跳），
      只有在最后要推荐产品时才必须满足硬性条件。

落地成一条可测的规则：

    AI 上一句问了需求，客户这一句**没有回答那一项**
        → 这一轮只"承接/闲谈"，**不问问题**
        → 连续承接最多 3 条；第 4 条必须问需求（拉回需求话题）
    客户回答了上一问（或主动回到需求话题）
        → 立刻恢复正常节奏：接住 + 问下一项，承接计数清零
    客户主动问业务问题（价格/交期等）
        → 正常回答，不占承接额度
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# 每一次闲聊最长 3 条
MAX_ACK_STREAK = 3


@dataclass
class ContinuationDecision:
    """这一轮要不要把"需求问题"压下去（改成承接）。"""

    suppress_question: bool = False
    reason: str = ""
    next_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suppress_question": self.suppress_question,
            "reason": self.reason,
            "next_streak": self.next_streak,
        }


def decide_continuation(
    *,
    has_question: bool,
    answered_pending: bool,
    had_pending_question: bool = True,
    ack_streak: int = 0,
    customer_question: bool = False,
    max_ack_streak: int = MAX_ACK_STREAK,
) -> ContinuationDecision:
    """这一轮是"承接（不问）"还是"可以问需求"。纯函数，方便单测。

    Args:
        has_question: 本轮计划里是否有要问客户的需求问题
        answered_pending: 客户这一句是否回答了上一轮 AI 问的那一项
        had_pending_question: 上一轮 AI 是否真的问过一项（第一轮没有上一问）
        ack_streak: 已经连续承接了几条（不含本轮）
        customer_question: 客户这一句是否在问业务问题（价格/交期/规格…）
        max_ack_streak: 承接上限（默认 3）
    """
    streak = max(0, int(ack_streak or 0))
    limit = max(0, int(max_ack_streak if max_ack_streak is not None else MAX_ACK_STREAK))

    if not has_question:
        # 本轮本来就没有需求问题：回答客户问题不算承接；纯承接则累计
        return ContinuationDecision(
            False, "no_question_this_turn", streak if customer_question else streak + 1
        )
    if not had_pending_question:
        # 没有"上一问"（例如对话刚开始 / 上一轮没提问）→ 正常问
        return ContinuationDecision(False, "no_pending_question", 0)
    if answered_pending:
        # 客户回到了需求话题 → 跟着客户话题走，问下一项
        return ContinuationDecision(False, "answered_pending_question", 0)
    if streak >= limit:
        # 承接额度用完 → 这一轮必须拉回需求
        return ContinuationDecision(False, "ack_budget_exhausted", 0)
    # 客户没有回答 → 先承接，不问问题
    return ContinuationDecision(True, "customer_did_not_answer_chat_first", streak + 1)


__all__ = ["ContinuationDecision", "MAX_ACK_STREAK", "decide_continuation"]
