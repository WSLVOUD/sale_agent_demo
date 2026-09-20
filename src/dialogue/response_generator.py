"""v2.5 Phase 2：ResponseGenerator —— 用 ResponseContext 生成自然销售回复。

职责边界（客户口径）：业务决策已经定好（DialogueAction + ResponseContext），
本模块只负责"怎么说"：

    · 有答复 → 先答客户的问题（绝不为继续收集需求而忽略客户问题）
    · 有推荐 → 短总结 + 推荐 + 关键理由（第一轮短，客户追问再展开）
    · 有提问 → 一次只问一个，并可选说明"为什么问"
    · 不强制 ACK / 不堆过渡词 / 不暴露内部术语

LLM 可用时可润色；失败或校验不合格时退回结构化拼装（永远给得出回复）。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from .action import (
    ACK_ONLY,
    ANSWER_AND_ASK,
    ASK,
    CLARIFY,
    CONFIRM,
    DIRECT_ANSWER,
    RECOMMEND,
)
from .response_context import ResponseContext
from .response_validator import validate_response

logger = logging.getLogger(__name__)


def generate_response(
    context: ResponseContext,
    *,
    llm: Optional[Any] = None,
    seed: int = 0,
) -> str:
    """按 ResponseContext 生成回复；有 LLM 就润色，没有就用结构化拼装。"""
    draft = compose_from_context(context, seed=seed)
    if llm is None or not draft:
        return draft
    try:
        prompt = (
            "You are an experienced B2B LED display sales consultant talking to a customer.\n"
            "Rewrite the draft below so it sounds like a real person, keeping every business "
            "decision exactly as it is (do not add specs, prices, warranties or new questions).\n"
            "Do not start with a generic acknowledgement unless the draft clearly does.\n"
            "Ask at most one question, and never use internal system terms.\n\n"
            f"Customer said: {context.customer_message[:200]}\n"
            f"{context.prompt_block()}\n\n"
            f"Draft: {draft}\n\nReply:"
        )
        response = llm.invoke(prompt)
        text = str(getattr(response, "content", response) or "").strip()
        if not text:
            return draft
        check = validate_response(
            text,
            allow_ack=context.allow_ack,
            allow_connector=context.allow_connector,
            one_question=context.one_question,
            customer_question=bool(context.answer),
            answer=context.answer,
        )
        if not check.ok:
            logger.info("[ResponseGenerator] 润色结果不合格 %s → 用结构化拼装", check.issues)
            return draft
        return text
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        logger.warning("Response polish failed: %s", exc)
        return draft


def compose_from_context(context: ResponseContext, *, seed: int = 0) -> str:
    """结构化拼装（不依赖 LLM，保证一定有回复）。"""
    parts: List[str] = []
    if context.answer:
        parts.append(context.answer.strip())
    elif context.action == DIRECT_ANSWER and context.notes:
        parts.append(context.notes[0].strip())

    if context.action == RECOMMEND and context.recommendation:
        model = str(context.recommendation.get("model") or "").strip()
        if model:
            prefix = "Based on what you told me, I'd go with" if not parts else "For that, I'd go with"
            parts.append(f"{prefix} {model}.")
        reasons = context.recommendation.get("reasons") or []
        if reasons:
            parts.append(str(reasons[0]) + ".")
    elif context.action == CLARIFY and context.conflicts:
        parts.append("Before I pick a model I need to clear one thing up: " + context.conflicts[0] + ".")

    if context.question:
        if context.why:
            parts.append(f"{context.why.rstrip('.')}.")
        parts.append(context.question.strip())
    elif context.action == ACK_ONLY and not parts:
        parts.append("Noted, thanks.")
    elif context.action == CONFIRM and not parts and context.notes:
        parts.append(context.notes[0].strip())
    return " ".join(part for part in parts if part).strip()


def build_context(
    *,
    action: str,
    customer_message: str = "",
    question: str = "",
    why: str = "",
    answer: str = "",
    recommendation: Optional[dict] = None,
    engineering_result: Optional[dict] = None,
    newly_confirmed: Optional[dict] = None,
    missing_fields: Optional[List[str]] = None,
    conflicts: Optional[List[str]] = None,
) -> ResponseContext:
    """便捷构造：按 action 决定表达侧的松紧（不要每轮都 ACK + Connector）。"""
    return ResponseContext(
        action=action,
        customer_message=customer_message,
        newly_confirmed_fields=dict(newly_confirmed or {}),
        missing_fields=list(missing_fields or []),
        conflicts=list(conflicts or []),
        recommendation=recommendation,
        engineering_result=engineering_result,
        question=question,
        why=why,
        answer=answer,
        allow_ack=action not in (DIRECT_ANSWER, ANSWER_AND_ASK, RECOMMEND),
        allow_connector=action not in (DIRECT_ANSWER, ANSWER_AND_ASK),
        one_question=True,
    )


__all__ = ["build_context", "compose_from_context", "generate_response"]
