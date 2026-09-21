"""v2.7 §18（Phase 7）：Answer Coverage —— 客户这一轮到底答了什么。

每一轮都要算清楚：

    asked_slot          这一轮 AI 问的是哪一项
    answered_slots      客户实际给出的槽位
    newly_filled_slots  这一轮新填进档案的槽位
    unresolved_slots    问了但没拿到、且没有别的来源的槽位

计划 §18 的要求：

    AI: What size do you need?
    客户: Viewing distance is 8 meters.

    → asked_slot = size / answered_slots = [viewing_distance] / size 仍 UNKNOWN

    **不能**因为 size 没回答就立刻再问一次 size（这正是重复提问的来源）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class AnswerCoverage:
    """一轮的"问答覆盖"结论。"""

    asked_slot: str = ""
    answered_slots: List[str] = field(default_factory=list)
    newly_filled_slots: List[str] = field(default_factory=list)
    unresolved_slots: List[str] = field(default_factory=list)
    answer_kind: str = ""
    expected_slot: str = ""

    @property
    def answered_asked_slot(self) -> bool:
        return bool(self.asked_slot) and self.asked_slot in self.answered_slots

    @property
    def is_wrong_slot(self) -> bool:
        """答非所问（给了别的信息，问的那一项没拿到）。"""
        return bool(self.asked_slot) and not self.answered_asked_slot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asked_slot": self.asked_slot,
            "answered_slots": list(self.answered_slots),
            "newly_filled_slots": list(self.newly_filled_slots),
            "unresolved_slots": list(self.unresolved_slots),
            "answer_kind": self.answer_kind,
            "expected_slot": self.expected_slot,
            "answered_asked_slot": self.answered_asked_slot,
            "is_wrong_slot": self.is_wrong_slot,
        }


def compute_answer_coverage(
    *,
    asked_slot: str = "",
    match: Optional[Any] = None,
    newly_filled_slots: Optional[Iterable[str]] = None,
    unresolved_slots: Optional[Iterable[str]] = None,
) -> AnswerCoverage:
    """把"上一轮问了什么 + 客户这一轮给了什么"算成覆盖率结论。"""
    answered: List[str] = []
    if match is not None:
        for slot in list(getattr(match, "covered_slots", None) or []):
            if slot and slot not in answered:
                answered.append(str(slot))
        primary = str(getattr(match, "slot", "") or "")
        if primary and primary not in answered:
            answered.append(primary)
    filled = [str(slot) for slot in (newly_filled_slots or []) if slot]
    for slot in filled:
        if slot not in answered:
            answered.append(slot)

    asked = str(asked_slot or "")
    unresolved = [str(slot) for slot in (unresolved_slots or []) if slot]
    if asked and asked not in answered and asked not in unresolved:
        unresolved.append(asked)

    return AnswerCoverage(
        asked_slot=asked,
        answered_slots=answered,
        newly_filled_slots=filled,
        unresolved_slots=unresolved,
        answer_kind=str(getattr(match, "kind", "") or "") if match is not None else "",
        expected_slot=(
            str(getattr(match, "expected_slot", "") or "") if match is not None else ""
        ),
    )


__all__ = ["AnswerCoverage", "compute_answer_coverage"]
