"""DialoguePolicy 动作 → 表达层动作（计划 v2.9 §十六 Phase 8 / §十五）。

背景：`script_generator._natural_reply()` 以前会**根据 answer/question 重新推断**
业务动作（ANSWER_AND_ASK / DIRECT_ANSWER / ASK），这是第二个决策中心。

现在它只做映射：Policy 已经定好的动作（`state["dialogue_action"]`）直接翻译成
表达层动作；只有 Policy 没给动作时，才允许按"有没有 answer / question"兜底。
"""
from __future__ import annotations

from typing import Any, Dict

ASK = "ASK"
ANSWER_AND_ASK = "ANSWER_AND_ASK"
DIRECT_ANSWER = "DIRECT_ANSWER"
RECOMMEND = "RECOMMEND"
CLARIFY = "CLARIFY"
CONFIRM = "CONFIRM"
ACK_ONLY = "ACK_ONLY"

_POLICY_TO_EXPRESSION: Dict[str, str] = {
    "ask_only": ASK,
    "ask": ASK,
    "answer_then_ask": ANSWER_AND_ASK,
    "answer_and_ask": ANSWER_AND_ASK,
    "answer_only": DIRECT_ANSWER,
    "direct_answer": DIRECT_ANSWER,
    "trigger_solution": RECOMMEND,
    "recommend": RECOMMEND,
    "clarify": CLARIFY,
    "confirm": CONFIRM,
    "ack_only": ACK_ONLY,
    "others": DIRECT_ANSWER,
    "product_question": DIRECT_ANSWER,
    "end": ACK_ONLY,
    "closing": ACK_ONLY,
}


def expression_action(state: Any, *, answer: str = "", question: str = "") -> str:
    """这一轮的表达层动作（Policy 定，表达层只翻译，不再自己推断）。"""
    decision: Any = {}
    if isinstance(state, dict):
        decision = state.get("dialogue_action") or {}
    elif state is not None:
        decision = getattr(state, "dialogue_action", {}) or {}
    policy_action = ""
    if isinstance(decision, dict):
        policy_action = str(decision.get("action") or "")
    elif isinstance(decision, str):
        policy_action = decision

    mapped = _POLICY_TO_EXPRESSION.get(policy_action.strip().lower(), "")
    if mapped:
        if mapped in (DIRECT_ANSWER, RECOMMEND, ACK_ONLY):
            return mapped
        # Policy 说"要问"时，如果还有要传达的答案内容（服务口径 / 客户问题），
        # 表达层合并成"答 + 问"：内容不丢，但问哪一项仍由 Policy 决定。
        if mapped == ASK and answer:
            return ANSWER_AND_ASK
        return mapped

    if answer and question:
        return ANSWER_AND_ASK
    if answer:
        return DIRECT_ANSWER
    return ASK


__all__ = [
    "ACK_ONLY",
    "ANSWER_AND_ASK",
    "ASK",
    "CLARIFY",
    "CONFIRM",
    "DIRECT_ANSWER",
    "RECOMMEND",
    "expression_action",
]
