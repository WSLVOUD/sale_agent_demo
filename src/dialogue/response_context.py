"""v2.5 Phase 2：ResponseContext —— 交给最终 LLM 的**结构化**上下文。

客户口径：不要让最终 LLM 自己重新判断业务逻辑。它只负责把已经定好的
业务决策用自然的销售语言表达出来。

    ResponseContext(
        action, customer_message, newly_confirmed_fields, missing_fields,
        conflicts, recommendation, engineering_result, question,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResponseContext:
    action: str = ""
    customer_message: str = ""
    newly_confirmed_fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    recommendation: Optional[Dict[str, Any]] = None
    engineering_result: Optional[Dict[str, Any]] = None
    question: str = ""
    why: str = ""
    answer: str = ""
    notes: List[str] = field(default_factory=list)
    # 表达侧要求（由 ResponsePlanner / action 决定）
    allow_ack: bool = True
    allow_connector: bool = True
    one_question: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "customer_message": str(self.customer_message or "")[:300],
            "newly_confirmed_fields": dict(self.newly_confirmed_fields),
            "missing_fields": list(self.missing_fields),
            "conflicts": list(self.conflicts),
            "recommendation": self.recommendation,
            "engineering_result": self.engineering_result,
            "question": self.question,
            "why": self.why,
            "answer": self.answer,
            "notes": list(self.notes),
            "allow_ack": self.allow_ack,
            "allow_connector": self.allow_connector,
            "one_question": self.one_question,
        }

    def prompt_block(self) -> str:
        """给 LLM 的结构化上下文（不含任何"该不该推荐"的判断权）。"""
        lines = [f"Dialogue action: {self.action}"]
        if self.answer:
            lines.append(f"Answer to give first: {self.answer}")
        if self.newly_confirmed_fields:
            lines.append(
                "Newly confirmed by the customer: "
                + ", ".join(f"{k}={v}" for k, v in self.newly_confirmed_fields.items())
            )
        if self.recommendation:
            model = self.recommendation.get("model")
            lines.append(f"Selected model (already decided, do not change): {model}")
        if self.engineering_result:
            fit = (self.engineering_result or {}).get("resolution_fit") or {}
            if fit:
                lines.append(
                    "Resolution check: actual "
                    f"{fit.get('actual')} vs target {fit.get('target')} "
                    f"({fit.get('fit_level')})"
                )
        if self.question:
            lines.append(f"One question to ask: {self.question}")
        if self.why:
            lines.append(f"Why this matters (say it briefly): {self.why}")
        if self.conflicts:
            lines.append("Conflicts to clarify: " + "; ".join(self.conflicts))
        rules = ["Never expose internal field names or system terms."]
        if not self.allow_ack:
            rules.append("Do not open with a generic acknowledgement.")
        if not self.allow_connector:
            rules.append("Do not use formulaic connectors such as 'Based on that'.")
        if self.one_question:
            rules.append("Ask at most one question.")
        lines.append("Rules: " + " ".join(rules))
        return "\n".join(lines)


__all__ = ["ResponseContext"]
