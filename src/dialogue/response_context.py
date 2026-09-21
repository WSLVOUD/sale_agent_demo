"""v2.5++（僵硬话术优化 · 第一阶段）：ResponseContext —— **纯结构化**上下文。

客户口径（《LED_RAG 僵硬话术优化工程计划》§3.1 / §4）：

    Python 决定"**说什么**"，LLM 决定"**怎么说**"。

所以这里**不保存拼好的销售句子**，只保存"业务决策"：

    action                  这一轮做什么（DialogueAction）
    known_facts             客户已经确认的事实
    missing_facts           还缺什么
    business_goal           这一轮的业务目的（为什么问 / 为什么推）
    required_question       必须问的那一项（槽位名，例如 viewing_distance）
    question                这一项的标准问句（**内容**：意思必须保住，措辞交给 LLM）
    answer                  必须转达给客户的答复（**内容**：事实不动，措辞交给 LLM）
    recommendation          已经定好的型号与理由（不许改）
    engineering_constraints 工程结论（分辨率 / 箱体 / 尺寸…）
    restrictions            表达侧硬约束（一轮只问一个 …）
    style                   表达风格

禁止：

    draft = "Thanks for sharing..."          # ❌ 模板句子

正确：

    required_question = "viewing_distance"   # ✅ 结构化决策
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_RESTRICTIONS = (
    "ask_only_one_question",
    "do_not_invent_facts",
    "do_not_repeat_customer_unnecessarily",
)


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
    # 系统已经准备好的"接话"（例如对客户自我介绍的回应）。
    # 客户口径：LLM 可以自己决定**要不要用它**（不强制每轮都接话）；
    # 没有 LLM 时结构化拼装会带上它，保证这份内容不丢。
    opening: str = ""
    notes: List[str] = field(default_factory=list)
    # ── v2.5++：结构化业务上下文（不再只是"该说的那句话"）────────────────
    known_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    business_goal: str = ""
    required_question: str = ""
    engineering_constraints: List[str] = field(default_factory=list)
    language: str = "en"
    restrictions: List[str] = field(default_factory=list)
    style: str = "natural_b2b_sales"
    # 表达侧要求（由 DialogueAction 决定；不是"必须 ACK"，而是"这一轮允不允许"）
    allow_ack: bool = True
    allow_connector: bool = True
    one_question: bool = True

    def effective_restrictions(self) -> List[str]:
        items = list(self.restrictions or DEFAULT_RESTRICTIONS)
        for item in DEFAULT_RESTRICTIONS:
            if item not in items:
                items.append(item)
        if not self.allow_ack and "do_not_open_with_generic_ack" not in items:
            items.append("do_not_open_with_generic_ack")
        if not self.allow_connector and "do_not_use_formulaic_connectors" not in items:
            items.append("do_not_use_formulaic_connectors")
        return items

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
            "opening": self.opening,
            "notes": list(self.notes),
            "known_facts": list(self.known_facts),
            "missing_facts": list(self.missing_facts),
            "business_goal": self.business_goal,
            "required_question": self.required_question,
            "engineering_constraints": list(self.engineering_constraints),
            "language": self.language,
            "restrictions": self.effective_restrictions(),
            "style": self.style,
            "allow_ack": self.allow_ack,
            "allow_connector": self.allow_connector,
            "one_question": self.one_question,
        }

    def prompt_block(self) -> str:
        """给 LLM 的**结构化**业务上下文（不含任何"该不该推荐"的判断权）。"""
        lines = [f"Dialogue action: {self.action}"]
        if self.business_goal:
            lines.append(f"Business goal: {self.business_goal}")
        if self.customer_message:
            lines.append(f"Customer's latest message: {str(self.customer_message)[:300]}")
        if self.known_facts:
            lines.append("Known facts (already confirmed): " + ", ".join(self.known_facts))
        elif self.newly_confirmed_fields:
            lines.append(
                "Newly confirmed by the customer: "
                + ", ".join(f"{k}={v}" for k, v in self.newly_confirmed_fields.items())
            )
        missing = list(self.missing_facts or self.missing_fields)
        if missing:
            lines.append("Still missing: " + ", ".join(missing))
        if self.answer:
            lines.append(f"Answer to give first (keep these facts): {self.answer}")
        if self.opening:
            lines.append(
                "Optional opening line from the system (use only if it fits naturally, "
                f"otherwise skip it): {self.opening}"
            )
        if self.recommendation:
            model = self.recommendation.get("model")
            lines.append(f"Selected model (already decided, do not change): {model}")
            reasons = self.recommendation.get("reasons") or []
            if reasons:
                lines.append("Reasons you may use: " + "; ".join(str(r) for r in reasons[:3]))
        if self.engineering_result:
            fit = (self.engineering_result or {}).get("resolution_fit") or {}
            if fit:
                lines.append(
                    "Resolution check: actual "
                    f"{fit.get('actual')} vs target {fit.get('target')} "
                    f"({fit.get('fit_level')})"
                )
        for constraint in self.engineering_constraints:
            lines.append(f"Engineering constraint: {constraint}")
        if self.required_question:
            lines.append(f"Required question (slot): {self.required_question}")
        if self.question:
            lines.append(
                f"Question to ask — keep the meaning, phrasing is yours: {self.question}"
            )
        if self.why:
            lines.append(f"Why this matters (mention only if it helps): {self.why}")
        if self.conflicts:
            lines.append("Conflicts to clarify: " + "; ".join(self.conflicts))
        if self.notes:
            lines.append("Notes: " + "; ".join(str(note) for note in self.notes[:3]))
        restrictions = self.effective_restrictions()
        if restrictions:
            lines.append("Restrictions: " + ", ".join(restrictions))
        lines.append(f"Style: {self.style}")
        return "\n".join(lines)


__all__ = ["DEFAULT_RESTRICTIONS", "ResponseContext"]
