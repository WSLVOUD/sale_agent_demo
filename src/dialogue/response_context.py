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

from .grounded_facts import (
    SOURCE_CALCULATED,
    SOURCE_CALCULATED_FROM_DIMENSIONS,
    SOURCE_CUSTOMER,
    SOURCE_INFERRED,
    SOURCE_INFERRED_FROM_DISTANCE,
    SOURCE_INFERRED_FROM_PITCH,
    SOURCE_INFERRED_FROM_SCENE,
    SOURCE_RECOMMENDED,
    SOURCE_RECOMMENDED_BY_ENGINE,
    SOURCE_RETRIEVED,
    SOURCE_RETRIEVED_FROM_PRODUCT_KB,
    GroundedFact,
)

DEFAULT_RESTRICTIONS = (
    "ask_only_one_question",
    "do_not_invent_facts",
    "do_not_repeat_customer_unnecessarily",
)


@dataclass
class QuestionSpec:
    """这一轮"要问的那一项"（计划 §6）。

        slot    = Python 决策（DialoguePolicy 定的槽位）
        intent  = Python 决策（问这一项的目的）
        anchor  = **语义表达锚点**（例如 "ask roughly how far viewers will be"），
                  **不是最终客户问句** —— 最终表达由 ResponseGenerator 的 LLM 生成。
        text    = 旧字段，保留兼容（等价于 anchor）

    以前 question / question_slot / question_intent / required_question 四个字段
    各说各话；现在它们都归到这一个对象上（旧字段保留兼容）。
    """

    slot: str = ""
    intent: str = ""
    anchor: str = ""
    text: str = ""  # 兼容别名：没有 anchor 时读它

    def __post_init__(self) -> None:
        # 计划 v2.8 §四：text → anchor（明确"这不是最终话术"）
        if not self.anchor and self.text:
            self.anchor = self.text
        elif not self.text and self.anchor:
            self.text = self.anchor

    @property
    def expression_anchor(self) -> str:
        return self.anchor or self.text

    @property
    def is_ask(self) -> bool:
        return bool(self.slot or self.intent or self.expression_anchor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "intent": self.intent,
            "anchor": self.expression_anchor,
        }


@dataclass
class ResponseShape:
    """这一轮**允许表达什么**（计划 v2.8 §十二）。

    不是 token 限制，而是告诉 LLM："这轮只需要完成一个动作"：

        ResponseShape(action="ASK", allow_answer=False, allow_context=True,
                      allow_question=True, max_questions=1)

    用"语义任务范围"控制长度，而不是用 max_output_tokens 截断（计划 §十/§十一）。
    """

    action: str = ""
    allow_answer: bool = False
    allow_context: bool = True
    allow_question: bool = False
    max_questions: int = 1

    @classmethod
    def for_action(cls, action: str, *, has_question: bool = False) -> "ResponseShape":
        """按 Action 给出默认形状（ASK / ANSWER_AND_ASK / DIRECT_ANSWER / RECOMMEND…）。"""
        name = str(action or "").strip().upper()
        allow_answer = name in ("DIRECT_ANSWER", "ANSWER_AND_ASK", "ANSWER", "RECOMMEND")
        allow_question = bool(has_question) or name in ("ASK", "ANSWER_AND_ASK")
        return cls(
            action=name,
            allow_answer=allow_answer,
            allow_context=True,
            allow_question=allow_question,
            max_questions=1 if allow_question else 0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "allow_answer": self.allow_answer,
            "allow_context": self.allow_context,
            "allow_question": self.allow_question,
            "max_questions": self.max_questions,
        }


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
    # v2.7 修订（客户口径）：问句以"槽位 + 意图"交给 LLM，由它自己组织说法；
    # 系统给的成句只当"意思锚点"，明确要求不要照抄。
    question_slot: str = ""
    question_intent: str = ""
    recent_questions: List[str] = field(default_factory=list)
    # 最近若干轮对话原文（客户口径 2026-09-21：AI 要带上会话记忆说话）
    recent_dialogue: str = ""
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
    # v2.5+++（计划 §5 / §6）：客户可见事实必须带来源
    grounded_facts: List[GroundedFact] = field(default_factory=list)
    pitch_resolution: Optional[Dict[str, Any]] = None
    language: str = "en"
    restrictions: List[str] = field(default_factory=list)
    style: str = "natural_b2b_sales"
    # 表达侧要求（由 DialogueAction 决定；不是"必须 ACK"，而是"这一轮允不允许"）
    allow_ack: bool = True
    allow_connector: bool = True
    one_question: bool = True
    # 计划 §6：要问的那一项（slot / intent / 意思锚点）
    question_spec: QuestionSpec = field(default_factory=QuestionSpec)
    # 计划 v2.8 §十二：这一轮允许表达什么（语义任务范围，不是 token 限制）
    response_shape: ResponseShape = field(default_factory=ResponseShape)

    def __post_init__(self) -> None:
        """旧字段 → QuestionSpec 的一次性归一（保证只有一份"要问什么"）。"""
        if not self.question_spec.is_ask and (
            self.question_slot or self.question or self.question_intent
        ):
            self.question_spec = QuestionSpec(
                slot=str(self.question_slot or ""),
                intent=str(self.question_intent or ""),
                text=str(self.question or ""),
            )
        if not self.response_shape.action:
            self.response_shape = ResponseShape.for_action(
                self.action, has_question=self.question_spec.is_ask
            )

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

    # ── v2.5+++：按来源分类的事实（计划 §10.4）────────────────────────────
    @property
    def customer_facts(self) -> List[GroundedFact]:
        return [f for f in self.grounded_facts if f.source == SOURCE_CUSTOMER]

    @property
    def inferred_facts(self) -> List[GroundedFact]:
        return [
            f for f in self.grounded_facts
            if f.source in (
                SOURCE_INFERRED, SOURCE_INFERRED_FROM_SCENE,
                SOURCE_INFERRED_FROM_PITCH, SOURCE_INFERRED_FROM_DISTANCE,
            )
        ]

    @property
    def calculated_facts(self) -> List[GroundedFact]:
        return [
            f for f in self.grounded_facts
            if f.source in (SOURCE_CALCULATED, SOURCE_CALCULATED_FROM_DIMENSIONS)
        ]

    @property
    def retrieved_facts(self) -> List[GroundedFact]:
        return [
            f for f in self.grounded_facts
            if f.source in (SOURCE_RETRIEVED, SOURCE_RETRIEVED_FROM_PRODUCT_KB)
        ]

    @property
    def recommended_facts(self) -> List[GroundedFact]:
        return [
            f for f in self.grounded_facts
            if f.source in (SOURCE_RECOMMENDED, SOURCE_RECOMMENDED_BY_ENGINE)
        ]

    def fact_lines(self) -> List[str]:
        lines: List[str] = []
        for fact in self.grounded_facts:
            lines.append(f"- {fact.text()} [{fact.source}]")
        return lines

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
            "grounded_facts": [fact.to_dict() for fact in self.grounded_facts],
            "pitch_resolution": self.pitch_resolution,
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
        shape = self.response_shape.to_dict()
        lines.append(
            "Response shape (what this turn allows): "
            f"answer={shape['allow_answer']}, context={shape['allow_context']}, "
            f"question={shape['allow_question']}, max_questions={shape['max_questions']}"
        )
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
        # v2.5+++：带来源的事实清单（LLM 只能使用这里出现过的业务事实）
        fact_lines = self.fact_lines()
        if fact_lines:
            lines.append("Grounded facts (each one carries its source):")
            lines.extend(fact_lines)
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
        if self.pitch_resolution:
            requested = str(self.pitch_resolution.get("requested_pitch_text") or "")
            resolved = str(self.pitch_resolution.get("resolved_pitch_text") or "")
            match_type = str(self.pitch_resolution.get("pitch_match_type") or "")
            if requested or resolved:
                lines.append(
                    "Pitch resolution: "
                    + (f"requested {requested}, " if requested else "")
                    + (f"resolved {resolved}, " if resolved else "")
                    + f"match={match_type}"
                )
            if self.pitch_resolution.get("needs_explanation"):
                lines.append(
                    "You MUST explain that the requested pitch is not an exact available "
                    "option and name the closest available one. Never say the requested "
                    f"pitch is the chosen one. Suggested wording: "
                    f"{self.pitch_resolution.get('pitch_resolution_reason') or ''}"
                )
        for constraint in self.engineering_constraints:
            lines.append(f"Engineering constraint: {constraint}")
        if self.required_question:
            lines.append(f"Required question (slot): {self.required_question}")
        spec = self.question_spec
        if spec.is_ask or self.question_slot or self.question_intent:
            slot = spec.slot or self.question_slot or self.required_question
            intent = spec.intent or self.question_intent or ""
            lines.append(f"Question to ask — slot: {slot}; what it must achieve: {intent}")
            lines.append(
                "Phrasing is entirely yours: ask it in your own natural words "
                "(exactly one question). Sound like a person, not a form."
            )
        anchor = spec.expression_anchor or self.question
        if anchor:
            lines.append(
                "Meaning anchor only (NOT the final wording — do NOT copy it "
                f"verbatim, rewrite it in your own words): {anchor}"
            )
        if self.recent_questions:
            lines.append(
                "You already used these phrasings — do not repeat them, pick a "
                "different natural wording: "
                + " | ".join(str(item)[:120] for item in self.recent_questions[:3])
            )
        if self.recent_dialogue:
            lines.append(
                "Recent conversation (continue naturally from here; never repeat a "
                "question or a fact you already said):\n"
                + str(self.recent_dialogue)[:2000]
            )
        lines.append(
            "You may add ONE short, natural sentence in your own words to react to "
            "what the customer just said (no filler openers such as \"Got it\", "
            "\"Thanks\", \"Understood\")."
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
