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
        }


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

    result.ok = not result.issues
    return result


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
