"""v2.3 §20：Conversation State（对话表达状态，不等于 Memory）。

只保存"当前这轮对话怎么表达"需要的信息，避免客户刚回答完 8m，AI 下一轮又问
观看距离；也避免同一句话术连着说两遍。

    conversation_stage          采集 / 推荐 / 冲突澄清
    last_customer_intent        上一轮客户的意图
    last_action                 上一轮系统做了什么（ASK / RECOMMEND / …）
    last_question               上一轮问的那句话
    last_answer                 客户对上一轮的回复
    repeated_question_count     同一句话术连续出现的次数

存储是进程内的（按 session_id），随会话结束自然失效；不写入 Memory 模块。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

STAGE_COLLECTING = "collecting"
STAGE_CONFLICT = "conflict"
STAGE_RECOMMENDING = "recommending"
STAGE_AFTER_RECOMMEND = "after_recommend"


@dataclass
class ConversationState:
    session_id: str = ""
    conversation_stage: str = STAGE_COLLECTING
    last_customer_intent: str = ""
    last_action: str = ""
    last_question: str = ""
    last_slot: str = ""
    last_answer: str = ""
    repeated_question_count: int = 0
    asked_questions: List[str] = field(default_factory=list)
    asked_slots: List[str] = field(default_factory=list)

    # ── 更新 ────────────────────────────────────────────────────────────
    def note_turn(
        self,
        *,
        answer: str = "",
        question: str = "",
        slot: str = "",
        action: str = "",
        intent: str = "",
        stage: str = "",
    ) -> None:
        if answer:
            self.last_answer = str(answer)[:400]
        if intent:
            self.last_customer_intent = intent
        if action:
            self.last_action = action
        if stage:
            self.conversation_stage = stage
        if question:
            normalized = str(question).strip().lower()
            if normalized and normalized == self.last_question.strip().lower():
                self.repeated_question_count += 1
            else:
                self.repeated_question_count = 0
            self.last_question = str(question)
            self.asked_questions.append(str(question)[:200])
            self.asked_questions = self.asked_questions[-12:]
        if slot:
            self.last_slot = str(slot)
            self.asked_slots.append(str(slot))
            self.asked_slots = self.asked_slots[-12:]

    # ── 查询 ────────────────────────────────────────────────────────────
    def asked_before(self, question: str) -> bool:
        normalized = str(question or "").strip().lower()
        return bool(normalized) and normalized in {
            item.strip().lower() for item in self.asked_questions[-6:]
        }

    @property
    def question_is_repeating(self) -> bool:
        """同一句话术连着说了两次以上 → Response Planner 要换一种说法。"""
        return self.repeated_question_count >= 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "conversation_stage": self.conversation_stage,
            "last_customer_intent": self.last_customer_intent,
            "last_action": self.last_action,
            "last_question": self.last_question,
            "last_slot": self.last_slot,
            "last_answer": self.last_answer,
            "repeated_question_count": self.repeated_question_count,
        }


_LOCK = threading.Lock()
_STATES: Dict[str, ConversationState] = {}
_MAX_SESSIONS = 256


def get_conversation_state(session_id: str) -> ConversationState:
    """取（或新建）某个会话的对话状态。"""
    key = str(session_id or "")
    if not key:
        return ConversationState()
    with _LOCK:
        state = _STATES.get(key)
        if state is None:
            state = ConversationState(session_id=key)
            if len(_STATES) >= _MAX_SESSIONS:
                _STATES.pop(next(iter(_STATES)), None)
            _STATES[key] = state
        return state


def update_conversation_state(session_id: str, **kwargs: object) -> ConversationState:
    state = get_conversation_state(session_id)
    state.note_turn(**kwargs)  # type: ignore[arg-type]
    return state


def reset_conversation_state(session_id: str) -> None:
    with _LOCK:
        _STATES.pop(str(session_id or ""), None)


def stage_from_status(status: str) -> str:
    """Gate 状态 → 对话阶段。"""
    mapping = {
        "READY": STAGE_RECOMMENDING,
        "DEGRADED_READY": STAGE_RECOMMENDING,
        "CONFLICT": STAGE_CONFLICT,
    }
    return mapping.get(str(status or "").upper(), STAGE_COLLECTING)


def known_question(question: Optional[str], conversation: ConversationState) -> bool:
    return bool(question) and conversation.asked_before(str(question))


__all__ = [
    "ConversationState",
    "STAGE_AFTER_RECOMMEND",
    "STAGE_COLLECTING",
    "STAGE_CONFLICT",
    "STAGE_RECOMMENDING",
    "get_conversation_state",
    "known_question",
    "reset_conversation_state",
    "stage_from_status",
    "update_conversation_state",
]
