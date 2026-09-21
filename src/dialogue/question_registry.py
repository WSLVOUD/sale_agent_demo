"""v2.6 §21/§22：AskedQuestionRegistry —— 当前会话的**问题状态**。

计划 §21 明确：这不是 Memory。它只记录"这次对话里，哪一项问过、客户答没答"，
会话结束就自然失效（与 ConversationState 同一生命周期）。

每个需求字段一条记录，走同一个状态机（计划 §22）：

    UNKNOWN ──▶ ASKED ──▶ ANSWERED ──▶ CONFIRMED
                            │
                            ├──▶ RESOLVED   （客户给了 P 值 → 点间距已定）
                            └──▶ INFERRED   （由场景 / 公式推出来的）

问题选择器（Question Planner / Gate）不能再重复问
**已经 ANSWERED / CONFIRMED / INFERRED / RESOLVED** 的字段。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ── §22 状态机取值 ──────────────────────────────────────────────────────
UNKNOWN = "UNKNOWN"
ASKED = "ASKED"
ANSWERED = "ANSWERED"
CONFIRMED = "CONFIRMED"
INFERRED = "INFERRED"
RESOLVED = "RESOLVED"

ALL_STATES = (UNKNOWN, ASKED, ANSWERED, CONFIRMED, INFERRED, RESOLVED)

# 已经有结论 → 不再问（与 question_flow._SETTLED 同一口径，措辞对齐状态机）
SETTLED_STATES = frozenset({ANSWERED, CONFIRMED, INFERRED, RESOLVED})


@dataclass
class QuestionRecord:
    """一个槽位的问题状态。"""

    slot: str = ""
    asked_at_turn: int = 0
    asked_count: int = 0
    answered: bool = False
    answer_turn: int = 0
    state: str = UNKNOWN

    def note_asked(self, turn_index: int = 0) -> None:
        self.asked_count += 1
        if not self.asked_at_turn:
            self.asked_at_turn = int(turn_index or 0)
        if self.state in (UNKNOWN, ""):
            self.state = ASKED

    def note_answered(self, turn_index: int = 0) -> None:
        self.answered = True
        self.answer_turn = int(turn_index or 0)
        self.state = ANSWERED

    def mark(self, state: str, turn_index: int = 0) -> None:
        state = str(state or "").upper()
        if state not in ALL_STATES:
            return
        self.state = state
        if state in SETTLED_STATES:
            self.answered = True
            if not self.answer_turn:
                self.answer_turn = int(turn_index or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "state": self.state,
            "asked_at_turn": self.asked_at_turn,
            "asked_count": self.asked_count,
            "answered": self.answered,
            "answer_turn": self.answer_turn,
        }


@dataclass
class AskedQuestionRegistry:
    """当前会话的问题登记簿（按槽位）。"""

    records: Dict[str, QuestionRecord] = field(default_factory=dict)
    turn_index: int = 0

    # ── 写入 ────────────────────────────────────────────────────────────
    def note_asked(self, slot: str, *, turn_index: Optional[int] = None) -> QuestionRecord:
        record = self._record(slot)
        record.note_asked(self._turn(turn_index))
        return record

    def note_answered(self, slot: str, *, turn_index: Optional[int] = None) -> QuestionRecord:
        record = self._record(slot)
        record.note_answered(self._turn(turn_index))
        return record

    def mark_state(
        self, slot: str, state: str, *, turn_index: Optional[int] = None
    ) -> QuestionRecord:
        record = self._record(slot)
        record.mark(state, self._turn(turn_index))
        return record

    def note_turn(self, turn_index: Optional[int] = None) -> int:
        """推进会话轮次（每次客户说完一句 + 系统回复完 = 一轮）。"""
        if turn_index is None:
            self.turn_index += 1
        else:
            self.turn_index = int(turn_index)
        return self.turn_index

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, slot: str) -> Optional[QuestionRecord]:
        return self.records.get(str(slot or ""))

    def has_asked(self, slot: str) -> bool:
        record = self.get(slot)
        return bool(record and record.asked_count > 0)

    def is_answered(self, slot: str) -> bool:
        record = self.get(slot)
        return bool(record and (record.answered or record.state in SETTLED_STATES))

    def is_settled(self, slot: str) -> bool:
        """已有结论（答过 / 确认过 / 推出来过）→ 不该再问。"""
        record = self.get(slot)
        return bool(record and record.state in SETTLED_STATES)

    def should_skip(self, slot: str) -> bool:
        """问题选择器的过滤器：问过并答过、或已有结论的槽位直接跳过。"""
        return self.is_settled(slot)

    def asked_slots(self) -> list:
        return [slot for slot, record in self.records.items() if record.asked_count > 0]

    def answered_slots(self) -> list:
        return [slot for slot, record in self.records.items() if record.answered]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "records": {slot: record.to_dict() for slot, record in self.records.items()},
        }

    # ── 内部 ────────────────────────────────────────────────────────────
    def _record(self, slot: str) -> QuestionRecord:
        key = str(slot or "").strip()
        record = self.records.get(key)
        if record is None:
            record = QuestionRecord(slot=key)
            self.records[key] = record
        return record

    def _turn(self, turn_index: Optional[int]) -> int:
        return int(self.turn_index if turn_index is None else turn_index)


__all__ = [
    "ALL_STATES",
    "ANSWERED",
    "ASKED",
    "AskedQuestionRegistry",
    "CONFIRMED",
    "INFERRED",
    "QuestionRecord",
    "RESOLVED",
    "SETTLED_STATES",
    "UNKNOWN",
]
