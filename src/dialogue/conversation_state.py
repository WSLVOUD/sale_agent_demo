"""v2.3 §20 + v2.6 §5~§10：Conversation State（对话工作状态，不等于 Memory）。

v2.3 只保存"当前这轮对话怎么表达"需要的信息；v2.6 按计划补上
**"客户现在这句话在回答哪个问题"** 这一层（计划 §5~§10）：

    conversation_stage          采集 / 推荐 / 冲突澄清
    last_customer_intent        上一轮客户的意图
    last_action / last_ai_action 上一轮系统做了什么（ASK / RECOMMEND / …）
    last_question / last_ai_question  上一轮问的那句话
    last_slot / last_question_slot    上一轮问的是哪一项
    last_answer                 客户对上一轮的回复
    last_response               上一轮系统最终发出去的那句话
    current_speech_act          客户这一句是什么类型（v2.6）
    current_answer_slot         客户这一句在回答哪一项（v2.6）
    pending_customer_question   客户这一句里带的业务问题（v2.6）
    pending_customer_answer     还没被消化掉的客户回答（v2.6）
    last_turn_id / current_turn_id  轮次编号，用于日志 / DecisionAudit
    repeated_question_count     同一句话术连续出现的次数
    registry                    AskedQuestionRegistry（§21/§22 问题状态机）

与 RequirementProfile 的分工（计划 §7）：

    RequirementProfile   "我们知道客户什么需求？"（environment=indoor …）
    ConversationState    "客户现在在做什么？"（上一句问了什么、这句答了哪个）

存储是进程内的（按 session_id），随会话结束自然失效；不写入 Memory 模块。
"""
from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .question_registry import AskedQuestionRegistry

STAGE_COLLECTING = "collecting"
STAGE_CONFLICT = "conflict"
STAGE_RECOMMENDING = "recommending"
STAGE_AFTER_RECOMMEND = "after_recommend"

# ── §9 Answer-to-Question Matching 的四种结论 ────────────────────────────
ANSWER_PREVIOUS_QUESTION = "ANSWER_PREVIOUS_QUESTION"
ANSWER_WRONG_SLOT = "ANSWER_WRONG_SLOT"
NEW_REQUIREMENT = "NEW_REQUIREMENT"
NO_REQUIREMENT = "NO_REQUIREMENT"

# 永远不算"在回答某个问题"的槽位（品类默认值，不是客户针对问题给的信息）
_NEVER_ANSWER_SLOTS = frozenset({"display_type"})

# 规则抽取的原始键 → 对话层槽位名（提问池 / Gate / 提问记录用的是后者）
_SLOT_ALIASES = {
    "pixel_pitch_mm": "pixel_pitch",
    "viewing_distance_m": "viewing_distance",
    "target_width_mm": "size",
    "target_height_mm": "size",
    "screen_size_hint_mm": "size",
    "budget_level": "price_preference",
}

# "都行 / both" 这类回答本身没有信息量，只能靠"上一轮问的是哪一项"落地
_BARE_BOTH_RE = re.compile(
    r"^\s*(?:both|either|whatever|any(?: one)?|"
    r"两者都行|两个都行|两边都行|两种都行|两个都可以|都行|都可以|随便|无所谓)\s*[.!。！]?\s*$",
    re.IGNORECASE,
)
_TWO_OPTION_SLOTS = ("price_preference", "content_type")


def _new_turn_id() -> str:
    return uuid.uuid4().hex[:12]


def explicit_slots(message: str) -> Dict[str, Any]:
    """客户这句话里**明确**给出的需求槽位（排除默认推断出来的）。

    规则：
      · 键以 "_" 开头的是元信息（``_default_slots`` / ``_explicit_keys``），不算；
      · 出现在 ``_default_slots`` 里的值（例如 indoor 顺带推出的 installation=fixed）
        是系统默认，不算客户明说；
      · ``display_type`` 是品类默认（LED），不算回答。

    返回的是**抽取器的原始键**（``target_width_mm`` / ``pixel_pitch_mm`` …），
    信息一条不丢；要"这一句覆盖了哪些对话槽位"请用 :func:`canonical_slots`。
    """
    try:
        from src.rag.query_understanding import extract_slots

        raw = dict(extract_slots(message) or {})
    except Exception:  # pragma: no cover - 防御式
        return {}
    defaults = {str(item) for item in (raw.get("_default_slots") or [])}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key)
        if name.startswith("_") or name in defaults or name in _NEVER_ANSWER_SLOTS:
            continue
        if value in (None, "", [], {}):
            continue
        out[name] = value
    return out


def canonical_slot_name(name: str) -> str:
    """抽取器原始键 → 对话层槽位名（``pixel_pitch_mm`` → ``pixel_pitch``）。"""
    return _SLOT_ALIASES.get(str(name or ""), str(name or ""))


def canonical_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    """把一组原始槽位映射成"这一句覆盖了哪些对话槽位"。"""
    out: Dict[str, Any] = {}
    for key, value in (slots or {}).items():
        out[canonical_slot_name(key)] = value
    return out


@dataclass
class AnswerMatch:
    """客户这一句与"上一轮 AI 问的那一项"的匹配结果（计划 §9 / §10）。"""

    kind: str = NO_REQUIREMENT
    slot: str = ""
    expected_slot: str = ""
    # 抽取器原始槽位（``target_width_mm`` …）——信息一条不丢
    slots: Dict[str, Any] = field(default_factory=dict)
    # 这一句覆盖的**对话层槽位名**（``size`` / ``pixel_pitch`` …）
    covered_slots: List[str] = field(default_factory=list)

    @property
    def answers_previous_question(self) -> bool:
        return self.kind == ANSWER_PREVIOUS_QUESTION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "slot": self.slot,
            "expected_slot": self.expected_slot,
            "slots": dict(self.slots),
            "covered_slots": list(self.covered_slots),
        }


def match_answer_to_question(
    message: str,
    *,
    last_question_slot: str = "",
    fallback_slot: str = "",
) -> AnswerMatch:
    """判断这句话在回答哪一项（计划 §9/§10 的核心规则，纯函数）。

    1. 先看这句话**能不能回答上一轮问的那一项** → ``ANSWER_PREVIOUS_QUESTION``；
    2. 答非所问（内容落到别的槽位）→ ``ANSWER_WRONG_SLOT``，
      **信息不能丢**（§10：AI 问 indoor/outdoor、客户答 3×5 → 记成尺寸，环境仍未知）；
    3. 没问过问题但补了需求 → ``NEW_REQUIREMENT``；
    4. 什么都没给 → ``NO_REQUIREMENT``。
    """
    slots = explicit_slots(message)
    expected = str(last_question_slot or fallback_slot or "").strip()
    # "都行 / both" 这类回答：本身没信息量，按"上一轮问的就是这一项"落地
    if expected in _TWO_OPTION_SLOTS and _BARE_BOTH_RE.match(str(message or "")):
        value = "both" if expected == "price_preference" else "mixed"
        return AnswerMatch(
            kind=ANSWER_PREVIOUS_QUESTION,
            slot=expected,
            expected_slot=expected,
            slots={expected: value},
            covered_slots=[expected],
        )
    covered = canonical_slots(slots)
    covered_names = list(covered)
    if not covered:
        return AnswerMatch(kind=NO_REQUIREMENT, expected_slot=expected)
    if expected and expected in covered:
        return AnswerMatch(
            kind=ANSWER_PREVIOUS_QUESTION,
            slot=expected,
            expected_slot=expected,
            slots=slots,
            covered_slots=covered_names,
        )
    first = next(iter(covered))
    if expected:
        # 答非所问：客户给了另一个字段的信息 —— 照样记账，环境/原问题保持未知（§10）
        return AnswerMatch(
            kind=ANSWER_WRONG_SLOT,
            slot=first,
            expected_slot=expected,
            slots=slots,
            covered_slots=covered_names,
        )
    return AnswerMatch(
        kind=NEW_REQUIREMENT,
        slot=first,
        expected_slot="",
        slots=slots,
        covered_slots=covered_names,
    )


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
    # ── v2.6 §6：最小可用字段集 ─────────────────────────────────────────
    last_ai_action: str = ""
    last_ai_question: str = ""
    last_question_slot: str = ""
    current_speech_act: str = ""
    current_answer_slot: str = ""
    pending_customer_question: str = ""
    pending_customer_answer: str = ""
    last_turn_id: str = ""
    current_turn_id: str = ""
    last_response: str = ""
    # v2.7 §18：客户这一句与"上一轮问的项"的匹配结果（Answer Coverage 的输入）
    last_answer_match: Dict[str, Any] = field(default_factory=dict)
    # ── v2.7 §14（Phase 6）：Conversation State 的职责字段 ─────────────────
    last_customer_message: str = ""
    last_ai_response: str = ""
    current_topic: str = ""
    # 客户口径（2026-09-21）：连续"承接/闲谈"的条数（不含本轮问题）
    ack_streak: int = 0
    turn_index: int = 0
    registry: AskedQuestionRegistry = field(default_factory=AskedQuestionRegistry)

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

    # ── v2.6 §8：每次 AI 问问题，都必须记录 ──────────────────────────────
    def note_ai_turn(
        self,
        *,
        action: str = "",
        question: str = "",
        slot: str = "",
        response: str = "",
        turn_id: str = "",
        speech_act: str = "",
    ) -> "ConversationState":
        """记录"这一轮 AI 做了什么 / 问了什么 / 最终说了什么"。"""
        if action:
            self.last_ai_action = str(action)
            self.last_action = str(action)
        if question:
            self.last_ai_question = str(question)
            self.last_question = str(question)
            self.asked_questions.append(str(question)[:200])
            self.asked_questions = self.asked_questions[-12:]
        if slot:
            self.last_question_slot = str(slot)
            self.last_slot = str(slot)
            self.asked_slots.append(str(slot))
            self.asked_slots = self.asked_slots[-12:]
            self.registry.note_asked(str(slot), turn_index=self.turn_index)
        if response:
            self.last_response = str(response)[:1000]
            self.last_ai_response = str(response)[:1000]
        if speech_act:
            self.current_speech_act = str(speech_act)
        self.last_turn_id = str(turn_id or self.current_turn_id or self.last_turn_id)
        return self

    # ── v2.6 §9/§10：客户这一句在做什么 ──────────────────────────────────
    def note_customer_turn(
        self,
        *,
        text: str = "",
        speech_act: str = "",
        answer_slot: str = "",
        customer_question: str = "",
        turn_id: str = "",
    ) -> "ConversationState":
        if text:
            self.last_answer = str(text)[:400]
            self.last_customer_message = str(text)[:400]
        if speech_act:
            self.current_speech_act = str(speech_act)
        self.current_answer_slot = str(answer_slot or "")
        self.pending_customer_question = str(customer_question or "")
        self.pending_customer_answer = str(text or "")[:400] if text else ""
        self.current_turn_id = str(turn_id or _new_turn_id())
        self.turn_index += 1
        self.registry.note_turn(self.turn_index)
        if answer_slot:
            # 客户答过这一项 → 登记簿里记 ANSWERED（§22）
            self.registry.note_answered(str(answer_slot), turn_index=self.turn_index)
        return self

    def answer_to(
        self, message: str, *, fallback_slot: str = ""
    ) -> AnswerMatch:
        """客户这句话在回答哪一项（默认拿上一轮 AI 问的槽位做匹配）。"""
        return match_answer_to_question(
            message,
            last_question_slot=self.last_question_slot or self.last_slot,
            fallback_slot=fallback_slot,
        )

    # ── v2.6 §21/§22：问题登记簿 ────────────────────────────────────────
    def note_asked(self, slot: str) -> None:
        self.registry.note_asked(str(slot), turn_index=self.turn_index)

    def note_answered(self, slot: str) -> None:
        self.registry.note_answered(str(slot), turn_index=self.turn_index)

    def should_skip_question(self, slot: str) -> bool:
        """这个问题还要不要问（答过 / 已有结论 → 跳过）。"""
        return self.registry.should_skip(str(slot))

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
            # ── v2.6 ────────────────────────────────────────────────────
            "last_ai_action": self.last_ai_action,
            "last_ai_question": self.last_ai_question,
            "last_question_slot": self.last_question_slot,
            "current_speech_act": self.current_speech_act,
            "current_answer_slot": self.current_answer_slot,
            "pending_customer_question": self.pending_customer_question,
            "pending_customer_answer": self.pending_customer_answer,
            "last_turn_id": self.last_turn_id,
            "current_turn_id": self.current_turn_id,
            "last_response": self.last_response,
            "last_answer_match": dict(self.last_answer_match or {}),
            "last_customer_message": self.last_customer_message,
            "last_ai_response": self.last_ai_response,
            "current_topic": self.current_topic,
            "ack_streak": self.ack_streak,
            "turn_index": self.turn_index,
            "registry": self.registry.to_dict(),
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


def note_customer_turn(
    session_id: str,
    message: str,
    *,
    speech_act: str = "",
    answer_slot: str = "",
    customer_question: str = "",
    turn_id: str = "",
) -> ConversationState:
    """便捷入口：客户这一轮说了什么（计划 §9/§10）。"""
    state = get_conversation_state(session_id)
    return state.note_customer_turn(
        text=message,
        speech_act=speech_act,
        answer_slot=answer_slot,
        customer_question=customer_question,
        turn_id=turn_id,
    )


def note_ai_turn(
    session_id: str,
    *,
    action: str = "",
    question: str = "",
    slot: str = "",
    response: str = "",
    turn_id: str = "",
    speech_act: str = "",
) -> ConversationState:
    """便捷入口：这一轮 AI 问了什么、最终说了什么（计划 §8）。"""
    state = get_conversation_state(session_id)
    return state.note_ai_turn(
        action=action,
        question=question,
        slot=slot,
        response=response,
        turn_id=turn_id,
        speech_act=speech_act,
    )


__all__ = [
    "ANSWER_PREVIOUS_QUESTION",
    "ANSWER_WRONG_SLOT",
    "AnswerMatch",
    "ConversationState",
    "NEW_REQUIREMENT",
    "NO_REQUIREMENT",
    "STAGE_AFTER_RECOMMEND",
    "STAGE_COLLECTING",
    "STAGE_CONFLICT",
    "STAGE_RECOMMENDING",
    "canonical_slot_name",
    "canonical_slots",
    "explicit_slots",
    "get_conversation_state",
    "known_question",
    "match_answer_to_question",
    "note_ai_turn",
    "note_customer_turn",
    "reset_conversation_state",
    "stage_from_status",
    "update_conversation_state",
]
