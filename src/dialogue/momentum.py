"""v2.7 §11（Phase 10）：ConversationMomentum —— 客户现在正在聊什么。

计划 §11：下一问应该围绕当前话题自然推进，而不是每轮重新扫描所有 missing slots。
计划 §11.1：**不要**把 momentum 存成客户需求 —— 它只用于当前对话规划。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MOMENTUM_WINDOW = 3

# 槽位 → 自然的下一话题
FOLLOW_UPS: Dict[str, tuple] = {
    "environment": ("installation", "size", "pixel_pitch"),
    "purpose": ("size", "environment", "installation"),
    "installation": ("size", "pixel_pitch", "environment"),
    "size": ("pixel_pitch", "viewing_distance", "installation"),
    "viewing_distance": ("pixel_pitch", "size", "installation"),
    "pixel_pitch": ("viewing_distance", "size", "installation"),
    "price_preference": ("size", "pixel_pitch"),
    "content_type": ("pixel_pitch", "size"),
}


@dataclass
class ConversationMomentum:
    """客户当前在聊什么（会话级、可跨轮累计）。"""

    slot: str = ""
    slots: List[str] = field(default_factory=list)
    source: str = ""

    def note(self, slot: str, *, source: str = "") -> "ConversationMomentum":
        key = str(slot or "")
        if not key:
            return self
        self.slot = key
        if key in self.slots:
            self.slots.remove(key)
        self.slots.insert(0, key)
        self.slots = self.slots[:DEFAULT_MOMENTUM_WINDOW]
        if source:
            self.source = str(source)
        return self

    @property
    def follow_ups(self) -> tuple:
        return FOLLOW_UPS.get(self.slot, ())

    def to_dict(self) -> Dict[str, Any]:
        return {"slot": self.slot, "slots": list(self.slots), "source": self.source}


def compute_momentum(
    *,
    newly_filled_slots: Optional[Iterable[str]] = None,
    answered_slots: Optional[Iterable[str]] = None,
    last_question_slot: str = "",
    recent_slots: Optional[Iterable[str]] = None,
) -> ConversationMomentum:
    """从"这一轮发生了什么"算出当前 momentum（新的优先）。"""
    momentum = ConversationMomentum()
    for slot in (recent_slots or []):
        momentum.note(str(slot), source="recent")
    for slot in (answered_slots or []):
        momentum.note(str(slot), source="answered")
    for slot in (newly_filled_slots or []):
        momentum.note(str(slot), source="newly_filled")
    if not momentum.slot and last_question_slot:
        momentum.note(str(last_question_slot), source="question")
    return momentum


def continuation_candidates(
    momentum: ConversationMomentum, candidates: Iterable[str]
) -> List[str]:
    """把候选按"和 momentum 的贴合度"排序。

    顺序：momentum 的自然延伸（按 ``FOLLOW_UPS`` 定义顺序）→ 其它候选
    → momentum 本身（客户刚说过的项，最后才考虑再问）。
    """
    items = [str(item) for item in candidates if item]
    if momentum is None or not momentum.slot:
        return items
    follow_ups = list(momentum.follow_ups)
    order = {slot: index for index, slot in enumerate(follow_ups)}

    def rank(item: str) -> tuple:
        if item in order:
            return (0, order[item], 0)
        if item == momentum.slot:
            return (2, 0, 0)
        return (1, items.index(item), 0)

    return sorted(items, key=rank)


def momentum_bonus(slot: str, momentum: Optional[ConversationMomentum]) -> float:
    """给"顺着当前话题"的问题加分（用于价值排序，不改变硬性 Gate）。"""
    if momentum is None or not momentum.slot:
        return 0.0
    if str(slot) == momentum.slot:
        return 0.0
    return 0.6 if str(slot) in set(momentum.follow_ups) else 0.0


__all__ = [
    "DEFAULT_MOMENTUM_WINDOW",
    "FOLLOW_UPS",
    "ConversationMomentum",
    "compute_momentum",
    "continuation_candidates",
    "momentum_bonus",
]
