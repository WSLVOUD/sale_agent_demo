"""v2.3 §10：Question Planner —— 提问的唯一出口。

    Gate（决定"问哪个字段"）
        ↓
    Question Planner（决定"这一轮怎么问"：换说法、降门槛、不重复）
        ↓
    Response Planner（决定整句话怎么组织）
        ↓
    Sales LLM（只说人话）

规则：一次只问一个问题；不问已经确认过的字段；客户说过"不知道"的字段降门槛；
同一句话术不连着说第二遍（借助 ConversationState 记录最近问过的话）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .conversation_state import ConversationState, get_conversation_state


@dataclass
class QuestionPlan:
    slot: str = ""
    question: str = ""
    action: str = ""
    easier: bool = False
    reason: str = ""
    reused: bool = False          # 这句话本会话里问过（换了说法）
    why: str = ""                 # v2.4：复问硬性条件时的"为什么需要知道"（英文一句）


def plan_question(
    decision: Any,
    profile: Any = None,
    *,
    language: str = "en",
    seed: int = 0,
    session_id: str = "",
    conversation: Optional[ConversationState] = None,
    slot: Optional[str] = None,
    question: Optional[str] = None,
    easier: Optional[bool] = None,
    reason: str = "",
    why: str = "",
) -> QuestionPlan:
    """把 Gate 的决策（slot + 草稿问句）变成这一轮实际要问的那句话。

    ``slot`` / ``question`` 允许调用方覆盖：调用侧可能已经做过调整
    （例如"这一项刚在图片核对里说过，改问下一项"），以调用侧的为准。
    """
    from src.rag.readiness import question_for

    slot = str(slot or getattr(decision, "next_slot", "") or "")
    draft = str(question or getattr(decision, "next_question", "") or "")
    status = str(getattr(decision, "status", "") or "")
    if not slot and not draft:
        return QuestionPlan()

    conversation = conversation or (get_conversation_state(session_id) if session_id else None)
    if easier is None:
        easier = status.upper() == "CONFLICT" or bool(
            profile is not None and getattr(profile, "is_unknown", lambda *_: False)(slot)
        )

    question = draft
    reused = False
    if conversation is not None and draft and conversation.asked_before(draft):
        # 同一句话术问过了 → 换一种说法（同一槽位的其它问法语义完全一致）
        alternative = None
        for offset in range(1, 8):
            candidate = question_for(slot, language, seed + offset, easier=easier)
            if candidate and not conversation.asked_before(candidate):
                alternative = candidate
                break
        if alternative:
            question = alternative
            reused = True

    if not reason:
        reason = {
            "CONFLICT": "conflict_clarification",
            "CONTINUE_ASKING": "narrow_recommendation_window",
        }.get(status.upper(), "keep_collecting")
    return QuestionPlan(
        slot=slot, question=question, action="ASK", easier=easier,
        reason=reason, reused=reused, why=str(why or ""),
    )


__all__ = ["QuestionPlan", "plan_question"]
