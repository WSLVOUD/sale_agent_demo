"""v2.5 Phase 2：Response Validator + 话术质量指标。

客户口径（计划 2.6）：话术质量不能只靠主观感觉，要有指标：

    generic_ack_rate          无意义 ACK 比例（Thanks for the information / Got it …）
    question_repeat_rate      重复提问比例
    connector_repeat_rate     重复过渡词比例
    customer_question_answer_rate 客户提问被正面回答的比例
    one_question_compliance   一轮最多一个问句
    internal_term_leak_rate   内部术语泄漏比例（Gate / RequirementProfile / 字段名…）
    unsupported_fact_rate     没有工程结果支撑却出现具体参数的比例
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
    r"now i(?:'d| would) like to ask|that said|with that said)\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"\b(?:usd|us\$|\$\s?\d|price is|costs? about)\b|报价|价格为", re.IGNORECASE)
INTERNAL_TERMS = (
    "RequirementProfile", "RecommendationGate", "recommendation_gate", "field_decision",
    "pixel_pitch_mm", "viewing_distance_m", "Provenance", "provenance", "ActionPlanner",
    "Action Planner", "Gate", "LangGraph", "slot", "target_width",
)
PRODUCT_PARAM_RE = re.compile(r"\bP\d(?:\.\d+)?\b|\b\d{3,5}\s*(?:nit|nits)\b", re.IGNORECASE)


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
    customer_question_answered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": list(self.issues),
            "has_generic_ack": self.has_generic_ack,
            "has_connector": self.has_connector,
            "question_count": self.question_count,
            "internal_terms": list(self.internal_terms),
            "unsupported_parameters": list(self.unsupported_parameters),
            "customer_question_answered": self.customer_question_answered,
        }


def validate_response(
    text: str,
    *,
    allow_ack: bool = True,
    allow_connector: bool = True,
    one_question: bool = True,
    customer_question: bool = False,
    answer: str = "",
    supported_parameters: Optional[Iterable[str]] = None,
) -> ResponseValidation:
    """校验一轮回复是否合格（不改文本，只报告；调用方可据此重写或修正）。"""
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

    if result.has_generic_ack and not allow_ack:
        result.issues.append("generic_ack")
    if result.has_connector and not allow_connector:
        result.issues.append("repeated_connector")
    if one_question and result.question_count > 1:
        result.issues.append("too_many_questions")
    if result.internal_terms:
        result.issues.append("internal_term_leak")
    if PRICE_RE.search(content):
        result.issues.append("mentions_price")

    supported = {str(item).lower() for item in (supported_parameters or [])}
    if supported:
        for token in PRODUCT_PARAM_RE.findall(content):
            if token.lower() not in supported:
                result.unsupported_parameters.append(token)
        if result.unsupported_parameters:
            result.issues.append("unsupported_parameter")

    if customer_question and answer:
        # 客户问了问题 → 回复里必须能看到答复的实质内容（取前 12 个字符做包含判断）
        probe = " ".join(answer.split())[:12].lower()
        if probe and probe not in content.lower():
            result.customer_question_answered = False
            result.issues.append("customer_question_not_answered")

    result.ok = not result.issues
    return result


def compute_metrics(samples: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """把一批（带校验结果的）样本汇总成话术质量指标。

    ``samples``：``[{"validation": ResponseValidation, "is_question": bool}, …]``
    """
    items = list(samples)
    total = len(items) or 1

    def rate(predicate) -> float:
        return round(sum(1 for item in items if predicate(item)) / total, 4)

    return {
        "generic_ack_rate": rate(lambda item: item["validation"].has_generic_ack),
        "connector_repeat_rate": rate(lambda item: item["validation"].has_connector),
        "question_repeat_rate": rate(
            lambda item: "too_many_questions" in item["validation"].issues
        ),
        "one_question_compliance": round(
            1.0 - rate(lambda item: item["validation"].question_count > 1), 4
        ),
        "internal_term_leak_rate": rate(
            lambda item: bool(item["validation"].internal_terms)
        ),
        "unsupported_fact_rate": rate(
            lambda item: bool(item["validation"].unsupported_parameters)
        ),
        "customer_question_answer_rate": rate(
            lambda item: item.get("is_question")
            and item["validation"].customer_question_answered
        ),
    }


__all__ = [
    "CONNECTOR_RE",
    "GENERIC_ACK_RE",
    "INTERNAL_TERMS",
    "ResponseValidation",
    "compute_metrics",
    "validate_response",
]
