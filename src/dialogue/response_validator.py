"""v2.5++（僵硬话术优化 · 第七/十阶段）：ResponseValidator + 自然度指标。

客户口径（《LED_RAG 僵硬话术优化工程计划》§10 / §13）：

    Validator 是"**事实、安全、业务边界守门员**"，不是"话术点评员"。

必须检查：客户提问是否被回答、有没有编造参数/型号/价格/交期、有没有改动工程结论、
有没有超过一个问题、有没有泄漏内部字段、有没有违反当前 Action。

**不**检查（也不再强制）：必须 ACK、必须连接词、必须过渡句、必须复述客户 ——
这些恰恰是"僵硬话术"的来源；它们只作为**指标**记录下来，供自然度统计使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .grounded_facts import (
    CLAIM_REQUIREMENTS,
    NUMBER_WITH_UNIT_RE,
    allowed_tokens,
    fact_fields,
)

GENERIC_ACK_RE = re.compile(
    r"^\s*(?:thanks for (?:that|the information|sharing)|got it|understood|"
    r"noted|sure|okay|ok|no problem|thanks)[.!,—\s]",
    re.IGNORECASE,
)
CONNECTOR_RE = re.compile(
    r"\b(?:based on that|based on your|in addition|to better understand|"
    r"now i(?:'d| would) like to ask|that said|with that said|by the way)\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"\b(?:usd|us\$|\$\s?\d|price is|costs? about)\b|报价|价格为", re.IGNORECASE)
LEAD_TIME_RE = re.compile(
    r"\b(\d{1,3})\s*(?:-\s*\d{1,3}\s*)?(?:days?|weeks?|个月|天|周)\b", re.IGNORECASE
)
MODEL_CODE_RE = re.compile(r"\bTW\s*\d{2}\s*-\s*[A-Za-z0-9-]+", re.IGNORECASE)
RESOLUTION_RE = re.compile(r"\b(\d{3,5})\s*[x×*]\s*(\d{3,5})\b", re.IGNORECASE)
INTERNAL_TERMS = (
    "RequirementProfile", "RecommendationGate", "recommendation_gate", "field_decision",
    "pixel_pitch_mm", "viewing_distance_m", "Provenance", "provenance", "ActionPlanner",
    "Action Planner", "Gate", "LangGraph", "slot", "target_width",
)
PRODUCT_PARAM_RE = re.compile(r"\bP\d(?:\.\d+)?\b|\b\d{3,5}\s*(?:nit|nits)\b", re.IGNORECASE)

# 用于"问错槽位"扫描的槽位清单（关键词由 readiness.question_keywords 提供）
_SLOT_KEYWORDS = (
    "environment",
    "purpose",
    "installation",
    "pixel_pitch",
    "viewing_distance",
    "size",
    "brightness",
)

# 客户已说过的事实短语（重复复述会被记为 repeated_known_facts）
_KNOWN_FACT_RES = (
    re.compile(r"\b(?:indoor|outdoor)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:x|by|\*)\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:church|mosque|school|mall|shopping mall|meeting room|classroom)\b", re.IGNORECASE),
)


def _known_fact_phrases(message: str) -> List[str]:
    """客户这句话里已经确认的事实短语（用于检测"机械复述"）。"""
    text = str(message or "")
    found: List[str] = []
    for pattern in _KNOWN_FACT_RES:
        for match in pattern.finditer(text):
            phrase = match.group(0).strip()
            if phrase and phrase.lower() not in {item.lower() for item in found}:
                found.append(phrase)
    return found

# ── 计划 v2.8 §十八：话术块拼接 / 语义完整性 ─────────────────────────────
# "多个独立回复块"：空行分隔、且每一块都自带完整句子结束符 + 各自有动词主语
_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?。！？]\s*$")
# 明显没写完的收尾（模型被截断 / 半句话）
_TRUNCATED_TAIL_RE = re.compile(
    r"(?:\b(?:and|or|with|for|to|the|a|an|of|about|such as|including|which|that)\b"
    r"|[,;:，、]|\bwe can\b|\blet me\b|\bI can\b)\s*$",
    re.IGNORECASE,
)

# 客户已经回答过、却又被重复追问的判定（用于 question_repeat_rate）
REASK_PATTERNS: Dict[str, str] = {
    "viewing_distance": r"how far|how far away|viewing distance",
    "size": r"how (?:wide|big|large)|what size|width and height|dimensions",
    "pixel_pitch": r"pixel pitch|what pitch|\bp\s*\d(?:\.\d+)?\b",
    "environment": r"indoors or outdoors|indoor or outdoor",
    "installation": r"fixed install|permanent install|rental",
}
ANSWER_PATTERNS: Dict[str, str] = {
    "viewing_distance": r"\b\d+(?:\.\d+)?\s*(?:m|meters?|metres?|米)\b",
    "size": r"\d+(?:\.\d+)?\s*(?:x|×|\*)\s*\d+(?:\.\d+)?",
    "pixel_pitch": r"\bp\s*\d(?:\.\d+)?\b",
    "environment": r"\b(?:indoor|outdoor)s?\b|室内|室外",
    "installation": r"\b(?:fixed|permanent|rental)\b|固装|租赁",
}
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "have", "will", "your", "you",
    "are", "was", "our", "its", "it's", "about", "from", "they", "them", "what",
    "which", "there", "here", "very", "just", "need", "want", "like", "into",
})


@dataclass
class ResponseValidation:
    ok: bool = True
    text: str = ""
    issues: List[str] = field(default_factory=list)
    has_generic_ack: bool = False
    has_connector: bool = False
    question_count: int = 0
    internal_terms: List[str] = field(default_factory=list)
    unsupported_parameters: List[str] = field(default_factory=list)
    unsupported_models: List[str] = field(default_factory=list)
    unsupported_lead_times: List[str] = field(default_factory=list)
    changed_engineering_results: List[str] = field(default_factory=list)
    customer_question_answered: bool = True
    customer_echo: bool = False
    echo_ratio: float = 0.0
    questionnaire_pattern: bool = False
    repeated_question_slot: str = ""
    missing_required_question: bool = False
    # v2.5+++（计划 §11.2 / §11.4 / §11.5）：事实 / 数字 / 点间距一致性
    ungrounded_facts: List[str] = field(default_factory=list)
    ungrounded_numbers: List[str] = field(default_factory=list)
    pitch_mismatch_unexplained: bool = False
    # v2.8：话术块拼接 / 重复已知事实 / 语义完整性 / 问错槽位
    response_blocks: int = 1
    repeated_known_facts: List[str] = field(default_factory=list)
    incomplete_tail: bool = False
    wrong_question_slot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": list(self.issues),
            "has_generic_ack": self.has_generic_ack,
            "has_connector": self.has_connector,
            "question_count": self.question_count,
            "internal_terms": list(self.internal_terms),
            "unsupported_parameters": list(self.unsupported_parameters),
            "unsupported_models": list(self.unsupported_models),
            "unsupported_lead_times": list(self.unsupported_lead_times),
            "changed_engineering_results": list(self.changed_engineering_results),
            "customer_question_answered": self.customer_question_answered,
            "customer_echo": self.customer_echo,
            "echo_ratio": round(self.echo_ratio, 4),
            "questionnaire_pattern": self.questionnaire_pattern,
            "repeated_question_slot": self.repeated_question_slot,
            "missing_required_question": self.missing_required_question,
            "ungrounded_facts": list(self.ungrounded_facts),
            "ungrounded_numbers": list(self.ungrounded_numbers),
            "pitch_mismatch_unexplained": self.pitch_mismatch_unexplained,
            "response_blocks": self.response_blocks,
            "repeated_known_facts": list(self.repeated_known_facts),
            "incomplete_tail": self.incomplete_tail,
            "wrong_question_slot": self.wrong_question_slot,
        }


_PITCH_EXPLANATION_RE = re.compile(
    r"closest|nearest|not an exact|isn't an exact|is not an exact|"
    r"instead|rather than|available match|not available|closest available",
    re.IGNORECASE,
)


def _content_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z']{4,}", str(text or "").lower())
    return [word for word in words if word not in _STOPWORDS]


def assertion_text(text: str) -> str:
    """只保留**陈述句**（去掉问句）。

    事实边界只管"系统断言了什么"：问句里出现的点间距示例
    （"…for example P2.5, P3 or P5?"）是提问选项，不是产品参数声明。
    """
    # 句号后面必须跟空白/结尾才算断句（否则 "P2.5" 里的点号会被当成句号）
    sentences = re.split(r"(?<=[.!?。！？])(?=\s|$)", str(text or ""))
    kept = [s for s in sentences if s.strip() and not s.strip().endswith(("?", "？"))]
    return " ".join(kept)


def echo_ratio(reply: str, customer_message: str) -> float:
    """回复里有多少比例是"复述客户刚说的话"（0~1）。"""
    customer = set(_content_tokens(customer_message))
    if not customer:
        return 0.0
    reply_tokens = set(_content_tokens(reply))
    if not reply_tokens:
        return 0.0
    shared = customer & reply_tokens
    return len(shared) / max(1, len(customer))


def repeated_question_slot(reply: str, customer_message: str) -> str:
    """回复是不是在问客户**刚刚已经回答过**的项（返回槽位名，否则空串）。"""
    text = str(reply or "").lower()
    message = str(customer_message or "")
    if "?" not in text and "？" not in text:
        return ""
    for slot, ask_pattern in REASK_PATTERNS.items():
        if re.search(ask_pattern, text) and re.search(ANSWER_PATTERNS[slot], message, re.IGNORECASE):
            return slot
    return ""


def questionnaire_pattern(reply: str) -> bool:
    """像问卷的三种典型特征：泛 ACK + 过渡词 + 问句 / 一次问多个 / 三句以上全是套话。"""
    text = str(reply or "")
    questions = text.count("?") + text.count("？")
    if questions > 1:
        return True
    has_ack = bool(GENERIC_ACK_RE.search(text)) or bool(
        re.search(r"^\s*(?:sure|of course)[,!.]", text, re.IGNORECASE)
    )
    has_connector = bool(CONNECTOR_RE.search(text))
    return bool(questions == 1 and has_ack and has_connector)


def validate_response(
    text: str,
    *,
    allow_ack: bool = True,
    allow_connector: bool = True,
    one_question: bool = True,
    customer_question: bool = False,
    answer: str = "",
    supported_parameters: Optional[Iterable[str]] = None,
    customer_message: str = "",
    required_question: str = "",
    engineering_result: Optional[Dict[str, Any]] = None,
    grounded_facts: Optional[Iterable[Any]] = None,
    pitch_resolution: Optional[Dict[str, Any]] = None,
    allowed_numbers: Optional[Iterable[str]] = None,
    question_slot: str = "",
    recent_questions: Optional[Iterable[str]] = None,
) -> ResponseValidation:
    """校验一轮回复是否越界（不改文本，只报告；调用方据此重写或回退）。"""
    result = ResponseValidation(text=str(text or ""))
    content = str(text or "")
    if not content.strip():
        result.ok = False
        result.issues.append("empty_response")
        return result

    result.has_generic_ack = bool(GENERIC_ACK_RE.search(content))
    result.has_connector = bool(CONNECTOR_RE.search(content))
    result.question_count = content.count("?") + content.count("？")
    result.internal_terms = [term for term in INTERNAL_TERMS if term in content]
    result.customer_echo = bool(customer_message) and echo_ratio(
        content, customer_message
    ) >= 0.5
    result.echo_ratio = echo_ratio(content, customer_message) if customer_message else 0.0
    result.questionnaire_pattern = questionnaire_pattern(content)
    result.repeated_question_slot = repeated_question_slot(content, customer_message)

    # ── 硬边界（越界一律判不合格）────────────────────────────────────────
    if one_question and result.question_count > 1:
        result.issues.append("too_many_questions")
    if result.internal_terms:
        result.issues.append("internal_term_leak")
    if PRICE_RE.search(content):
        result.issues.append("mentions_price")
    if required_question and result.question_count == 0:
        result.missing_required_question = True
        result.issues.append("missing_required_question")
    # v2.7 修订：问句只要"问的是同一件事"即可 —— 措辞交给 LLM 自己组织；
    # 这里只做关键词级的意图校验，避免它把问题问成别的东西。
    if question_slot and result.question_count:
        keywords = _slot_keywords(question_slot)
        if keywords:
            lowered_question = content.lower()
            if not any(word.lower() in lowered_question for word in keywords):
                result.issues.append("question_intent_mismatch")

    # ── 计划 v2.8 §十八：问错槽位 / 多个回复块 / 重复已知事实 / 语义不完整 ──
    # 1) 问错槽位：Python 定 slot=application，AI 却问 viewing distance → fail
    if question_slot and result.question_count:
        others = [
            other for other in _SLOT_KEYWORDS
            if other != question_slot and _slot_keywords(other)
        ]
        lowered = content.lower()
        asked_slot = next(
            (
                other for other in others
                if all(word.lower() in lowered for word in _slot_keywords(other))
            ),
            "",
        )
        if asked_slot:
            result.wrong_question_slot = True
            result.issues.append("wrong_question_slot")

    # 2) 多个独立回复块：空行分隔的块 ≥2，且每块自成完整句子（不是同一段的换行）
    blocks = [
        part.strip() for part in _BLOCK_SPLIT_RE.split(content) if part.strip()
    ]
    result.response_blocks = max(1, len(blocks))
    if len(blocks) >= 3 or (
        len(blocks) == 2
        and all(len(part) > 40 and _SENTENCE_END_RE.search(part) for part in blocks)
    ):
        result.issues.append("multiple_response_blocks")

    # 3) 机械重复已确认事实（客户已说过的信息再原样复述一遍）
    if customer_message:
        for fact in _known_fact_phrases(customer_message):
            if fact.lower() in content.lower():
                result.repeated_known_facts.append(fact)
        # 计划 v2.8 §十八：允许"顺带确认"一两个已确认事实；
        # 只有机械复述一大堆（≥3）才算话术拼接
        if len(result.repeated_known_facts) >= 3:
            result.issues.append("repeated_known_facts")

    # 4) 语义完整性：结尾明显没写完（半句话 / 被截断）
    if _TRUNCATED_TAIL_RE.search(content.strip()) and not _SENTENCE_END_RE.search(
        content.strip()
    ):
        result.incomplete_tail = True
        result.issues.append("incomplete_response")
    # 同一句话换汤不换药地重复问 → 提示换说法（软问题，交给重写）
    recent = [str(item) for item in (recent_questions or []) if str(item).strip()]
    if recent and result.question_count:
        normalized = _normalise_question_sentences(content)
        for previous in recent:
            if previous.strip().lower() in normalized:
                result.issues.append("repeated_question_phrasing")
                break

    if result.has_generic_ack and not allow_ack:
        result.issues.append("generic_ack")
    if result.has_connector and not allow_connector:
        result.issues.append("repeated_connector")

    # 事实检查只看陈述句（问句里的"例如 P2.5"是选项，不是编造参数）
    asserted = assertion_text(content)
    supported = {str(item).lower() for item in (supported_parameters or [])}
    if supported:
        for token in PRODUCT_PARAM_RE.findall(asserted):
            if token.lower() not in supported:
                result.unsupported_parameters.append(token)
        if result.unsupported_parameters:
            result.issues.append("unsupported_parameter")
        for model in MODEL_CODE_RE.findall(asserted):
            if model.lower() not in supported:
                result.unsupported_models.append(model)
        if result.unsupported_models:
            result.issues.append("unsupported_model")
        for match in LEAD_TIME_RE.finditer(asserted):
            if match.group(0).lower() not in supported:
                result.unsupported_lead_times.append(match.group(0))
        if result.unsupported_lead_times:
            result.issues.append("unsupported_lead_time")

    # 工程结论不能被改写（例如系统算的是 3840x2160，回复里不许出现别的分辨率）
    fit = (engineering_result or {}).get("resolution_fit") or {}
    allowed = set()
    for key in ("actual", "target"):
        value = fit.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            allowed.add(f"{int(value[0])}x{int(value[1])}".lower())
    if allowed:
        for width, height in RESOLUTION_RE.findall(asserted):
            if f"{width}x{height}".lower() not in allowed:
                result.changed_engineering_results.append(f"{width}x{height}")
        if result.changed_engineering_results:
            result.issues.append("changed_engineering_result")

    if customer_question and answer:
        # 客户问了问题 → 回复里必须能看到答复的实质内容（取前 12 个字符做包含判断）
        probe = " ".join(answer.split())[:12].lower()
        if probe and probe not in content.lower():
            result.customer_question_answered = False
            result.issues.append("customer_question_not_answered")

    # ── Fact Guard（§11.2）：业务事实必须有来源 ──────────────────────────
    facts = list(grounded_facts or [])
    fields = fact_fields(facts)
    lowered = asserted.lower()
    for phrase, required_field in CLAIM_REQUIREMENTS.items():
        if phrase in lowered and required_field not in fields:
            result.ungrounded_facts.append(phrase)
    if result.ungrounded_facts:
        result.issues.append("ungrounded_business_fact")

    # ── Numeric Guard（§11.5）：数字必须能在事实 / 客户原话里找到 ─────────
    if facts or allowed_numbers:
        allowed = set(str(item) for item in (allowed_numbers or []))
        allowed |= allowed_tokens(facts, [customer_message, answer])
        for number, _unit in NUMBER_WITH_UNIT_RE.findall(asserted):
            normalised = number.replace(",", ".")
            if normalised not in allowed and number not in allowed:
                result.ungrounded_numbers.append(f"{number}")
        if result.ungrounded_numbers:
            result.issues.append("ungrounded_numeric")

    # ── Pitch Guard（§11.4）：requested != resolved 必须解释 ──────────────
    resolution = pitch_resolution or {}
    if resolution.get("needs_explanation"):
        requested = str(resolution.get("requested_pitch_text") or "").strip()
        if requested and requested.lower() in asserted.lower():
            if not _PITCH_EXPLANATION_RE.search(asserted):
                result.pitch_mismatch_unexplained = True
                result.issues.append("pitch_mismatch_not_explained")

    result.ok = not result.issues
    return result


def _slot_keywords(slot: str):
    try:
        from src.rag.readiness import question_keywords

        return question_keywords(slot)
    except Exception:  # pragma: no cover - 防御式
        return ()


def _normalise_question_sentences(text: str) -> str:
    """把文本里所有问句抽出来、归一化，用于"是不是又问了一遍"。"""
    import re as _re

    sentences = _re.split(r"(?<=[.!?。！？])\s*", str(text or ""))
    questions = [item for item in sentences if "?" in item or "？" in item]
    return " ".join(" ".join(item.lower().split()) for item in questions)


def compute_metrics(samples: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """把一批（带校验结果的）样本汇总成自然度指标（计划 §13 的十项）。

    ``samples``：``[{"validation": ResponseValidation, "is_question": bool}, …]``
    """
    items = list(samples)
    total = len(items) or 1

    def rate(predicate) -> float:
        return round(sum(1 for item in items if predicate(item)) / total, 4)

    lengths = [len(str(item["validation"].text or "")) for item in items] or [0]
    return {
        "generic_ack_rate": rate(lambda item: item["validation"].has_generic_ack),
        "question_repeat_rate": rate(
            lambda item: bool(item["validation"].repeated_question_slot)
        ),
        "connector_repeat_rate": rate(lambda item: item["validation"].has_connector),
        "customer_echo_rate": rate(lambda item: item["validation"].customer_echo),
        "questionnaire_pattern_rate": rate(
            lambda item: item["validation"].questionnaire_pattern
        ),
        "customer_question_answer_rate": rate(
            lambda item: item.get("is_question")
            and item["validation"].customer_question_answered
        ),
        "one_question_compliance": round(
            1.0 - rate(lambda item: item["validation"].question_count > 1), 4
        ),
        "unsupported_fact_rate": rate(
            lambda item: bool(
                item["validation"].unsupported_parameters
                or item["validation"].unsupported_models
                or item["validation"].changed_engineering_results
                or item["validation"].ungrounded_facts
                or item["validation"].ungrounded_numbers
            )
        ),
        "internal_term_leak_rate": rate(
            lambda item: bool(item["validation"].internal_terms)
        ),
        "response_length": round(sum(lengths) / len(lengths), 1),
    }


__all__ = [
    "ANSWER_PATTERNS",
    "CONNECTOR_RE",
    "GENERIC_ACK_RE",
    "INTERNAL_TERMS",
    "REASK_PATTERNS",
    "ResponseValidation",
    "assertion_text",
    "compute_metrics",
    "echo_ratio",
    "questionnaire_pattern",
    "repeated_question_slot",
    "validate_response",
]
