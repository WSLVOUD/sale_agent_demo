"""v2.3 §19 / §21 / §23：Response Planner —— 只决定"这一轮怎么说"。

Python 决定做什么（ASK / RECOMMEND / CONFLICT / UNKNOWN），Response Planner 把这些
动作翻译成**表达结构**，再由 Sales LLM 说人话：

    ASK        ACKNOWLEDGE + EXPLAIN_WHY + ASK_ONE_QUESTION
    RECOMMEND  SHORT_SUMMARY + RECOMMEND + KEY_REASON
    CONFLICT   ACKNOWLEDGE + EXPLAIN_CONFLICT + ASK_ONE_CLARIFICATION
    UNKNOWN    REASSURE + OFFER_SIMPLE_ALTERNATIVE + ASK

它不选产品、不算参数、不改字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ACKNOWLEDGE = "ACKNOWLEDGE"
EXPLAIN_WHY = "EXPLAIN_WHY"
EXPLAIN_CONFLICT = "EXPLAIN_CONFLICT"
ASK_ONE_QUESTION = "ASK_ONE_QUESTION"
ASK_ONE_CLARIFICATION = "ASK_ONE_CLARIFICATION"
REASSURE = "REASSURE"
OFFER_SIMPLE_ALTERNATIVE = "OFFER_SIMPLE_ALTERNATIVE"
SHORT_SUMMARY = "SHORT_SUMMARY"
RECOMMEND = "RECOMMEND"
KEY_REASON = "KEY_REASON"

ASK = "ASK"
RECOMMEND_ACTION = "RECOMMEND"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"
RESULT = "RESULT"

# 每个字段"为什么要问"（用于 EXPLAIN_WHY；客户口径：只在有用时解释）
_WHY: Dict[str, str] = {
    "environment": "indoor and outdoor products are quite different",
    "purpose": "the application decides which series fits best",
    "installation": "fixed and rental cabinets are built differently",
    "pixel_pitch": "the pitch decides how sharp the image looks from where people sit",
    "viewing_distance": "the viewing distance decides which pixel pitch is enough",
    "size": "the screen size decides the cabinet and module layout",
    "price_preference": "it decides whether I optimise for cost or for image quality",
}

# 客户说"不知道"时的安抚 + 低门槛替代说法
_REASSURE = "No problem — most customers don't have that number memorised."


@dataclass
class ResponsePlan:
    action: str = ASK
    slot: str = ""
    strategy: str = "ASK"
    blocks: List[str] = field(default_factory=list)
    why: str = ""
    reason: str = ""
    question: str = ""
    one_question: bool = True
    expand_only_when_asked: bool = True
    instructions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "slot": self.slot,
            "strategy": self.strategy,
            "blocks": list(self.blocks),
            "why": self.why,
            "reason": self.reason,
            "question": self.question,
            "one_question": self.one_question,
            "expand_only_when_asked": self.expand_only_when_asked,
        }

    def prompt_lines(self) -> List[str]:
        """给 Sales LLM 的行为提示（§22）。"""
        lines = [
            "Follow the structure below exactly:",
            "- " + " then ".join(self.blocks),
        ]
        if self.why:
            lines.append(f"- Explain briefly why the question matters: {self.why}.")
        lines.extend(
            [
                "- Never ask for information the customer already gave.",
                "- Ask at most ONE question.",
                "- Do not repeat the whole requirement summary every turn.",
                "- Never invent technical specifications and never expose internal field names.",
                "- Sound like a consultant, not a questionnaire; keep it short and natural.",
            ]
        )
        if self.expand_only_when_asked:
            lines.append(
                "- Keep the first recommendation short; expand the technical detail only "
                "if the customer asks why."
            )
        return lines


def plan_response(
    decision: Any = None,
    *,
    profile: Any = None,
    question_plan: Any = None,
    unknown: bool = False,
    session_id: str = "",
) -> ResponsePlan:
    """根据 Gate 决策 + 问题计划，产出这一轮的表达方案。"""
    status = str(getattr(decision, "status", "") or "").upper()
    slot = str(getattr(question_plan, "slot", "") or getattr(decision, "next_slot", "") or "")
    question = str(getattr(question_plan, "question", "") or getattr(decision, "next_question", "") or "")

    if status == "CONFLICT":
        return ResponsePlan(
            action=CONFLICT,
            slot=slot,
            strategy=CONFLICT,
            blocks=[ACKNOWLEDGE, EXPLAIN_CONFLICT, ASK_ONE_CLARIFICATION],
            reason="conflict",
            question=question,
        )

    if status in ("READY", "DEGRADED_READY"):
        return ResponsePlan(
            action=RECOMMEND_ACTION,
            slot="",
            strategy=RECOMMEND,
            blocks=[SHORT_SUMMARY, RECOMMEND, KEY_REASON],
            reason="ready_to_recommend",
            question="",
        )

    if unknown:
        return ResponsePlan(
            action=UNKNOWN,
            slot=slot,
            strategy=UNKNOWN,
            blocks=[REASSURE, OFFER_SIMPLE_ALTERNATIVE, ASK_ONE_QUESTION],
            why=_WHY.get(slot, ""),
            reason="customer_does_not_know",
            question=question,
        )

    return ResponsePlan(
        action=ASK,
        slot=slot,
        strategy=ASK,
        blocks=[ACKNOWLEDGE, EXPLAIN_WHY, ASK_ONE_QUESTION],
        why=_WHY.get(slot, ""),
        reason=str(getattr(question_plan, "reason", "") or "keep_collecting"),
        question=question,
    )


def reassure_line() -> str:
    return _REASSURE


__all__ = [
    "ACKNOWLEDGE",
    "ASK",
    "ASK_ONE_CLARIFICATION",
    "ASK_ONE_QUESTION",
    "CONFLICT",
    "EXPLAIN_CONFLICT",
    "EXPLAIN_WHY",
    "KEY_REASON",
    "OFFER_SIMPLE_ALTERNATIVE",
    "RECOMMEND",
    "RECOMMEND_ACTION",
    "REASSURE",
    "RESULT",
    "ResponsePlan",
    "SHORT_SUMMARY",
    "UNKNOWN",
    "plan_response",
    "reassure_line",
]
