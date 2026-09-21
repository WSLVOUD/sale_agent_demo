"""v2.7 §19/§20（Phase 8）：重复提问闸门 + 回复条数护栏。

发送**之前**的最后一道业务闸门（是 Policy 的复核点，不是"最后加一个 if"）：

    FinalResponse
        ↓
    DuplicateQuestionFirewall
        · 当前 question_slot / 上一轮 question_slot
        · question_state（ASKED / ANSWERED / …）
        · answered_slots / newly_filled_slots
        · current turn_id / previous turn_id

§19.1 绝对禁止：``previous_turn.question_slot == current_turn.question_slot``
且客户**没有提供任何相关新信息** → 不许发送，回去重跑 Dialogue Policy。

§20 Response Count Guard：一个 turn ``response_count <= 1``；已经产出过
FinalResponse 的 turn，再生成一律拒绝。

计划 §27 同时说明：Guard **不是**主要方案 —— 主要方案是
Message ID + Turn ID + Dedup + Sealing + Session Lock + Commit。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

# v2.7 §19.1 的总开关：上一轮问过的项，没有新信息就不再重复问（含 environment）。
# 若要回到"环境每轮必问"的严格口径（v2.6 §12），把这里改成 False。
DEFER_REPEATED_SLOT = True

# 允许"再问一次同一项"的情形：客户这一轮给了该项的新信息（澄清 / 修正 / 补充）
_NEW_INFO_STATES = frozenset({"ANSWERED", "CONFIRMED", "INFERRED", "RESOLVED"})


@dataclass
class FirewallDecision:
    """闸门结论。"""

    allowed: bool = True
    reason: str = ""
    blocked_slot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "blocked_slot": self.blocked_slot,
        }


class DuplicateQuestionFirewall:
    """一问一答的跨轮一致性检查（纯函数，方便单测）。"""

    def check(
        self,
        *,
        current_slot: str = "",
        previous_slot: str = "",
        current_turn_id: str = "",
        previous_turn_id: str = "",
        answered_slots: Optional[Iterable[str]] = None,
        newly_filled_slots: Optional[Iterable[str]] = None,
        question_state: str = "",
        response_count: int = 1,
        defer_repeated_slot: Optional[bool] = None,
    ) -> FirewallDecision:
        slot = str(current_slot or "")
        if not slot:
            return FirewallDecision(True, "no_question")
        if int(response_count or 0) > 1:
            return FirewallDecision(False, "response_count_exceeded", slot)
        if current_turn_id and previous_turn_id and current_turn_id == previous_turn_id:
            return FirewallDecision(False, "same_turn_regeneration", slot)
        if not previous_slot or str(previous_slot) != slot:
            return FirewallDecision(True, "different_slot")
        defer = DEFER_REPEATED_SLOT if defer_repeated_slot is None else bool(defer_repeated_slot)
        if not defer:
            return FirewallDecision(True, "repeat_allowed_by_policy")

        answered = {str(item) for item in (answered_slots or [])}
        filled = {str(item) for item in (newly_filled_slots or [])}
        state = str(question_state or "").upper()
        if slot in answered or slot in filled or state in _NEW_INFO_STATES:
            return FirewallDecision(True, "new_information_about_slot")
        return FirewallDecision(False, "duplicate_question_without_new_info", slot)


@dataclass
class ResponseCountGuard:
    """§20：一个 turn 只能有一条客户可见回复。"""

    limit: int = 1
    _spent: int = 0

    def reset(self) -> None:
        self._spent = 0

    def check(self, *, response_count: int = 1) -> FirewallDecision:
        if int(response_count or 0) > int(self.limit):
            return FirewallDecision(False, "response_count_exceeded", "")
        if self._spent >= self.limit:
            return FirewallDecision(False, "turn_already_responded", "")
        return FirewallDecision(True, "within_budget")

    def spend(self) -> None:
        self._spent += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"limit": self.limit, "spent": self._spent}


FIREWALL = DuplicateQuestionFirewall()


__all__ = [
    "DEFER_REPEATED_SLOT",
    "FIREWALL",
    "DuplicateQuestionFirewall",
    "FirewallDecision",
    "ResponseCountGuard",
]
