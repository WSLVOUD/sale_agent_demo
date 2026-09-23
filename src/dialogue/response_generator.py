"""v2.5++（僵硬话术优化 · 第二/三阶段）：ResponseGenerator —— **唯一客户文本出口**。

客户口径（《LED_RAG 僵硬话术优化工程计划》§5 / §6 / §16）：

    旧：业务逻辑 → 固定模板 → LLM 润色 → 客户      （Strategy A，僵硬）
    新：业务上下文 → LLM 原生生成 → Validator → 客户（Strategy B，默认）

    · ResponseGenerator 是唯一的客户可见文本出口；
    · Python 决定"说什么"（DialogueAction + ResponseContext），
      LLM 决定"怎么说"（要不要接话、要不要复述、句子长短、直接问还是先铺垫）；
    · LLM 不得改变业务决策、不得编造参数/价格/交期；
    · 没有 LLM 或生成不合格时，退回**结构化拼装**（一定给得出回复），
      旧模板链路作为最后兜底保留（§16：先建新链路，旧链路留作回退）。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

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

# ── 策略开关（§15 A/B Test：两套策略都要能跑起来对比）─────────────────────
STRATEGY_TEMPLATE_POLISH = "A"   # 旧：Template → LLM Polish
STRATEGY_NATIVE = "B"            # 新：ResponseContext → LLM Native Generation
DEFAULT_STRATEGY = STRATEGY_NATIVE
STRATEGY_ENV = "LED_RAG_RESPONSE_STRATEGY"


def get_response_strategy() -> str:
    """当前策略（默认 B）。可用环境变量 ``LED_RAG_RESPONSE_STRATEGY=A|B`` 切换。"""
    value = str(os.getenv(STRATEGY_ENV) or "").strip().upper()
    return value if value in (STRATEGY_TEMPLATE_POLISH, STRATEGY_NATIVE) else DEFAULT_STRATEGY


def set_response_strategy(name: str) -> str:
    """设置策略（A/B Test 用），返回生效后的策略。"""
    value = str(name or "").strip().upper()
    if value in (STRATEGY_TEMPLATE_POLISH, STRATEGY_NATIVE):
        os.environ[STRATEGY_ENV] = value
        return value
    return get_response_strategy()


# ── LLM 熔断（连续失败后短暂停用，避免每轮都等超时）────────────────────
LLM_COOLDOWN_SECONDS = 60.0
_llm_state = {"fails": 0, "until": 0.0}


def llm_available() -> bool:
    """LLM 是否可用（连续失败 2 次后冷却 60s，期间直接走结构化兜底）。"""
    return time.time() >= float(_llm_state["until"])


def _note_llm_failure() -> None:
    _llm_state["fails"] = int(_llm_state["fails"]) + 1
    if int(_llm_state["fails"]) >= 2:
        _llm_state["until"] = time.time() + LLM_COOLDOWN_SECONDS
        logger.warning(
            "[ResponseGenerator] LLM 连续失败 %s 次 → 冷却 %.0fs，期间使用结构化兜底",
            _llm_state["fails"], LLM_COOLDOWN_SECONDS,
        )


def _note_llm_success() -> None:
    _llm_state["fails"] = 0
    _llm_state["until"] = 0.0


def reset_llm_breaker() -> None:
    """清空熔断状态（A/B 对比、测试或人工恢复时使用）。"""
    _note_llm_success()


# ── 第三阶段：原生生成用的系统提示（§6 的约束原样落地）────────────────────
NATIVE_SYSTEM_PROMPT = (
    "You are an experienced B2B LED sales consultant.\n"
    "\n"
    "Respond naturally, like a real salesperson having a normal conversation.\n"
    "\n"
    "The business decision has already been made by the system.\n"
    "Preserve the MEANING of the required question and every provided fact.\n"
    "The WORDING is entirely yours — never copy a canned system line verbatim;\n"
    "say it the way you would actually say it to a customer.\n"
    "\n"
    "You are explicitly allowed to:\n"
    "- write your own short reaction to what the customer just said (your words)\n"
    "- rephrase the required question completely, as long as it still asks for\n"
    "  exactly the same thing\n"
    "- vary sentence order, length and tone from turn to turn\n"
    "- continue the conversation from the recent messages you are given\n"
    "\n"
    "You are NOT required to:\n"
    "- acknowledge every customer message\n"
    "- repeat what the customer just said\n"
    "- use a transition phrase\n"
    "- explain why you are asking\n"
    "- follow a fixed sentence structure\n"
    "\n"
    "Length: normally 3 to 4 short sentences — a natural reaction to what the\n"
    "customer just said, one sentence of useful context or reasoning, and the\n"
    "question. Do not answer with a single bare line.\n"
    "Only shorten this when the customer only needs a one-word confirmation.\n"
    "\n"
    "Do not sound like a questionnaire.\n"
    "Do not sound like a scripted chatbot.\n"
    "Do not use sales filler.\n"
    "Do not force phrases such as:\n"
    "\"Got it\", \"Thanks\", \"Based on that\", \"By the way\", \"So\", \"That said\".\n"
    "Do not open two turns in a row with the same kind of phrasing.\n"
    "Never repeat a question or a fact that is already in the recent conversation.\n"
    "\n"
    "Ask no more than one question.\n"
    "That question must be about the same topic as \"Question to ask\" —\n"
    "only the wording changes.\n"
    "\n"
    "Do not change the business decision.\n"
    "Do not invent specifications, prices, models, lead times,\n"
    "engineering results, or commitments.\n"
    "\n"
    "You may use confirmed, inferred, calculated, retrieved,\n"
    "and recommended facts provided in the business context.\n"
    "You may explain or rephrase these facts naturally.\n"
    "Do not introduce any new business fact that is not present\n"
    "in the provided business context.\n"
    "Do not invent:\n"
    "- product specifications\n"
    "- viewing distance\n"
    "- brightness\n"
    "- installation method\n"
    "- waterproofing\n"
    "- rental features\n"
    "- product suitability\n"
    "- engineering calculations\n"
    "- customer requirements\n"
    "\n"
    "Keep every sentence short and conversational, but do not make the whole reply\n"
    "shorter than 2 sentences.\n"
    "\n"
    "Output language: {language_rule}\n"
    "\n"
    "Reply with the message text only — no JSON, no quotes, no explanations."
)

POLISH_SYSTEM_PROMPT = (
    "You are an experienced B2B LED display sales consultant talking to a customer.\n"
    "Rewrite the draft below so it sounds like a real person, keeping every business "
    "decision exactly as it is (do not add specs, prices, warranties or new questions).\n"
    "Do not start with a generic acknowledgement unless the draft clearly does.\n"
    "Ask at most one question, and never use internal system terms.\n"
    "Output language: {language_rule}\n"
)


def _language_rule(context: ResponseContext) -> str:
    language = str(context.language or "").strip().lower()
    if not language or language in ("en", "en-us", "en-gb", "english"):
        return "ALWAYS use English, regardless of the customer's language."
    return f"Use {language}."


def _fallback_language_rule(context: ResponseContext) -> str:
    return _language_rule(context)


def generate_response(
    context: ResponseContext,
    *,
    llm: Optional[Any] = None,
    seed: int = 0,
    strategy: Optional[str] = None,
) -> str:
    """按 ResponseContext 生成最终客户回复。

    Strategy B（默认）：LLM 直接从结构化上下文原生生成 → 校验 → 不合格退回结构化拼装；
    Strategy A（回退/对比）：结构化拼装草稿 → LLM 润色 → 校验 → 不合格退回草稿。
    """
    fallback = compose_from_context(context, seed=seed)
    if llm is not None and not llm_available():
        llm = None
    if llm is None:
        return fallback
    active = str(strategy or get_response_strategy()).strip().upper()
    text = (
        _generate_native(context, llm)
        if active != STRATEGY_TEMPLATE_POLISH
        else _polish_draft(context, fallback, llm)
    )
    if not text:
        return fallback
    check = _validate(context, text)
    if check.ok:
        return text
    # 只是"口味"问题（通用客套 / 连接词 / 问法重复）→ 不整段回退，直接采用
    hard_issues = [item for item in check.issues if item not in SOFT_ISSUES]
    if not hard_issues:
        logger.info(
            "[ResponseGenerator] %s 仅有措辞提示 %s → 采用 LLM 说法",
            active, check.issues,
        )
        return text
    # 有硬问题 → 让 LLM 自己修一次（比整段丢给模板自然得多）
    repaired = _repair_response(context, text, check.issues, llm)
    if repaired:
        recheck = _validate(context, repaired)
        if not [item for item in recheck.issues if item not in SOFT_ISSUES]:
            logger.info(
                "[ResponseGenerator] 重写一次后通过（原问题：%s）", check.issues
            )
            return repaired
    logger.info("[ResponseGenerator] %s 生成结果不合格 %s → 结构化拼装", active, check.issues)
    return fallback


# 只影响"口味"的问题：不触发整段回退（否则客户永远看到模板腔）
SOFT_ISSUES = frozenset({"generic_ack", "repeated_connector", "repeated_question_phrasing"})


def _validate(context: ResponseContext, text: str):
    """按当前上下文校验一段回复。"""
    # 计划 §7/§16.4：ResponseGenerator 只能改表达方式 ——
    # 校验用的问句槽位一律取自 QuestionSpec（唯一一份"要问什么"），
    # 生成侧不得自行换槽位。
    spec_slot = str(getattr(context.question_spec, "slot", "") or "")
    return validate_response(
        text,
        allow_ack=context.allow_ack,
        allow_connector=context.allow_connector,
        one_question=context.one_question,
        customer_question=bool(context.answer),
        answer=context.answer,
        customer_message=context.customer_message,
        supported_parameters=_supported_parameters(context),
        required_question=(
            str(getattr(context.question_spec, "text", "") or "")
            or context.question
            or context.required_question
        ),
        engineering_result=context.engineering_result,
        grounded_facts=context.grounded_facts,
        pitch_resolution=context.pitch_resolution,
        question_slot=spec_slot or context.question_slot,
        recent_questions=context.recent_questions,
    )


def _repair_response(
    context: ResponseContext, text: str, issues: List[str], llm: Any
) -> str:
    """让 LLM 自己把不合格的那一版重写一遍（保住事实、只改说法）。"""
    if llm is None:
        return ""
    try:
        prompt = (
            "You are the same experienced B2B LED sales consultant.\n"
            "Rewrite your previous reply so it satisfies every rule below.\n"
            "Keep exactly the same business facts and the same single question\n"
            "(same topic, different wording). Never invent specifications, prices,\n"
            "models, lead times or engineering results.\n"
            f"Problems to fix: {', '.join(str(item) for item in issues)}\n"
            "Ask no more than one question. Do not copy any canned line verbatim.\n"
            "Output the rewritten message only.\n"
            "\n\n--- Business context ---\n"
            + context.prompt_block()
            + f"\n\n--- Your previous reply ---\n{text}\n\n--- Rewritten reply ---\n"
        )
        repaired = _invoke(llm, prompt)
        _note_llm_success() if repaired else _note_llm_failure()
        return repaired
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        _note_llm_failure()
        logger.warning("Response repair failed: %s", exc)
        return ""


def _supported_parameters(context: ResponseContext) -> List[str]:
    """本轮的"有依据参数"（型号自带的参数 + 工程结果），其它具体参数都算编造。"""
    supported: List[str] = []
    model = str((context.recommendation or {}).get("model") or "")
    if model:
        supported.append(model)
        for token in model.replace("_", "-").split("-"):
            if token.upper().startswith("P") and len(token) > 1:
                supported.append(token)
    for item in context.engineering_constraints:
        supported.extend(str(item).split())
    fit = (context.engineering_result or {}).get("resolution_fit") or {}
    for key in ("actual", "target"):
        value = fit.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            supported.append(f"{value[0]}x{value[1]}")
    lines = "\n".join(context.notes or []) + "\n" + (context.answer or "")
    return [token for token in set(supported + lines.split()) if token]


def _invoke(llm: Any, prompt: Any) -> str:
    response = llm.invoke(prompt)
    text = str(getattr(response, "content", response) or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    return text.strip('"\'“”').strip()


def _generate_native(context: ResponseContext, llm: Any) -> str:
    """Strategy B：结构化上下文 → LLM 直接生成（没有草稿，也就没有模板腔）。"""
    try:
        prompt = (
            NATIVE_SYSTEM_PROMPT.format(language_rule=_fallback_language_rule(context))
            + "\n\n--- Business context ---\n"
            + context.prompt_block()
            + "\n\n--- Your reply ---\n"
        )
        text = _invoke(llm, prompt)
        _note_llm_success() if text else _note_llm_failure()
        return text
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        _note_llm_failure()
        logger.warning("Native response generation failed: %s", exc)
        return ""


def _polish_draft(context: ResponseContext, draft: str, llm: Any) -> str:
    """Strategy A（旧链路）：草稿 → 润色。仅在对比或回退时使用。"""
    if not draft:
        return ""
    try:
        prompt = (
            POLISH_SYSTEM_PROMPT.format(language_rule=_fallback_language_rule(context))
            + f"\nCustomer said: {context.customer_message[:200]}\n"
            + context.prompt_block()
            + f"\n\nDraft: {draft}\n\nReply:"
        )
        text = _invoke(llm, prompt)
        _note_llm_success() if text else _note_llm_failure()
        return text
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        _note_llm_failure()
        logger.warning("Response polish failed: %s", exc)
        return ""


def compose_from_context(context: ResponseContext, *, seed: int = 0) -> str:
    """结构化拼装（不依赖 LLM，保证一定有回复）。

    注意：这里**只做"内容兜底"**，不追求像人——像人是 LLM 的活儿。
    所以不带"Thanks / Based on that"这类模板腔，只把必须说的内容按顺序排好。
    """
    parts: List[str] = []
    if context.opening:
        parts.append(context.opening.strip())
    if context.answer:
        parts.append(context.answer.strip())
    elif context.action == DIRECT_ANSWER and context.notes:
        parts.append(context.notes[0].strip())

    if context.action == RECOMMEND and context.recommendation:
        model = str(context.recommendation.get("model") or "").strip()
        if model:
            parts.append(f"I'd go with {model}.")
        reasons = context.recommendation.get("reasons") or []
        if reasons:
            parts.append(str(reasons[0]).rstrip(".") + ".")
    elif context.action == CLARIFY and context.conflicts:
        parts.append("Before I pick a model, one thing to clear up: " + context.conflicts[0] + ".")

    if context.question:
        if context.why and not parts:
            parts.append(f"{context.why.rstrip('.')}.")
        parts.append(context.question.strip())
    elif context.action == ACK_ONLY and not parts:
        parts.append("Noted.")
    elif context.action == CONFIRM and not parts and context.notes:
        parts.append(context.notes[0].strip())
    return " ".join(part for part in parts if part).strip()


def build_context(
    *,
    action: str,
    customer_message: str = "",
    question: str = "",
    question_slot: str = "",
    question_intent: str = "",
    recent_questions: Optional[List[str]] = None,
    recent_dialogue: str = "",
    why: str = "",
    answer: str = "",
    recommendation: Optional[dict] = None,
    engineering_result: Optional[dict] = None,
    newly_confirmed: Optional[dict] = None,
    missing_fields: Optional[List[str]] = None,
    conflicts: Optional[List[str]] = None,
    known_facts: Optional[List[str]] = None,
    grounded_facts: Optional[List[Any]] = None,
    pitch_resolution: Optional[Dict[str, Any]] = None,
    opening: str = "",
    business_goal: str = "",
    required_question: str = "",
    engineering_constraints: Optional[List[str]] = None,
    language: str = "en",
    restrictions: Optional[List[str]] = None,
    style: str = "natural_b2b_sales",
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
        question_slot=question_slot,
        question_intent=question_intent,
        recent_questions=list(recent_questions or []),
        recent_dialogue=str(recent_dialogue or ""),
        why=why,
        answer=answer,
        opening=opening,
        known_facts=list(known_facts or []),
        grounded_facts=list(grounded_facts or []),
        pitch_resolution=pitch_resolution,
        missing_facts=list(missing_fields or []),
        business_goal=business_goal,
        required_question=required_question or (missing_fields[0] if missing_fields else ""),
        engineering_constraints=list(engineering_constraints or []),
        language=language,
        restrictions=list(restrictions or []),
        style=style,
        allow_ack=action not in (DIRECT_ANSWER, ANSWER_AND_ASK, RECOMMEND),
        allow_connector=action not in (DIRECT_ANSWER, ANSWER_AND_ASK),
        one_question=True,
    )


__all__ = [
    "DEFAULT_STRATEGY",
    "NATIVE_SYSTEM_PROMPT",
    "POLISH_SYSTEM_PROMPT",
    "STRATEGY_ENV",
    "STRATEGY_NATIVE",
    "STRATEGY_TEMPLATE_POLISH",
    "build_context",
    "compose_from_context",
    "generate_response",
    "get_response_strategy",
    "llm_available",
    "reset_llm_breaker",
    "set_response_strategy",
]
