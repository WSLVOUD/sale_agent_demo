"""v2.5++++（消息聚合与自然对话链路优化 · 第三阶段）：SpeechAct 理解。

计划 §6：不能再只靠 `recommendation / need_query / others / objection` 这几个意图，
要识别客户**这句话在做什么**：

    ANSWER_REQUIREMENT   在回答我们上一轮问的那一项（"P3" / "indoor"）
    NEW_REQUIREMENT      主动补充需求（"3*5 indoor"）
    CUSTOMER_QUESTION    泛问（"can you help me?"）
    PRICE_QUESTION       问价格
    DELIVERY_QUESTION    问交期 / 档期
    PRODUCT_QUESTION     问产品 / 参数 / 型号
    OBJECTION            异议（太贵 / 为什么）
    CONFIRMATION         确认（yes / ok / 对的）
    CORRECTION           纠正（no / actually / instead）
    CASUAL               闲聊
    MULTI_INTENT         一句话里既是需求又是问题（"P3. Also how long is delivery?"）

纯规则 + 现有解析器（不额外增加 LLM 调用），结果给 Dialogue Policy 用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Dict, List, Optional

ANSWER_REQUIREMENT = "ANSWER_REQUIREMENT"
NEW_REQUIREMENT = "NEW_REQUIREMENT"
CUSTOMER_QUESTION = "CUSTOMER_QUESTION"
PRICE_QUESTION = "PRICE_QUESTION"
DELIVERY_QUESTION = "DELIVERY_QUESTION"
PRODUCT_QUESTION = "PRODUCT_QUESTION"
OBJECTION = "OBJECTION"
CONFIRMATION = "CONFIRMATION"
CORRECTION = "CORRECTION"
CASUAL = "CASUAL"
MULTI_INTENT = "MULTI_INTENT"

ALL_SPEECH_ACTS = (
    ANSWER_REQUIREMENT, NEW_REQUIREMENT, CUSTOMER_QUESTION, PRICE_QUESTION,
    DELIVERY_QUESTION, PRODUCT_QUESTION, OBJECTION, CONFIRMATION, CORRECTION,
    CASUAL, MULTI_INTENT,
)

# 客户"主动提问"这一类（优先级最高：先回答客户）
CUSTOMER_QUESTION_ACTS = frozenset({
    CUSTOMER_QUESTION, PRICE_QUESTION, DELIVERY_QUESTION, PRODUCT_QUESTION, OBJECTION,
})

_CONFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|sure|ok(?:ay)?|correct|right|exactly|agreed|confirm(?:ed)?|"
    r"对|是的|对的|好的|可以|没错|嗯)\s*[.!。！]?\s*$",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"\b(?:actually|instead|not exactly|that's not|it's not|no,|rather)\b|"
    r"其实|不是|改成|应该是|纠正",
    re.IGNORECASE,
)
_OBJECTION_RE = re.compile(
    r"\b(?:too expensive|too much|why so|cheaper|discount|can't afford)\b|"
    r"太贵|为什么这么|便宜点|打折|预算不够",
    re.IGNORECASE,
)

# 客户在问"产品 / 参数 / 型号"时的特征词
_PRODUCT_QUESTION_RE = re.compile(
    r"\b(?:model|series|spec|specs|specification|resolution|4k|8k|1080p|"
    r"pixel pitch|brightness|refresh|cabinet|module|waterproof|ip6[56]|cob|hdr|"
    r"warranty|guarantee|certificat\w*)\b|"
    r"型号|规格|参数|分辨率|点间距|亮度|刷新|箱体|模组|防水|质保|保修|认证",
    re.IGNORECASE,
)

# 哪些槽位算"回答了我们上一轮问的那一项"
_SLOT_ALIASES: Dict[str, tuple] = {
    "environment": ("environment", "indoor", "outdoor", "semi_outdoor"),
    "installation": ("installation", "is_rental"),
    "pixel_pitch": ("pixel_pitch_mm", "pixel_pitch"),
    "viewing_distance": ("viewing_distance_m", "viewing_distance", "distance"),
    "size": ("target_width_mm", "target_height_mm", "size", "screen_size_hint_mm"),
    "purpose": ("purpose",),
    "price_preference": ("price_preference",),
    "budget": ("budget_level",),
    "content_type": ("content_type",),
    "brightness": ("brightness_min", "brightness_min_nit"),
}


def _extract_requirement_slots(message: str) -> Dict[str, Any]:
    try:
        from src.rag.query_understanding import extract_slots

        slots = dict(extract_slots(message) or {})
    except Exception:  # pragma: no cover - 防御式
        return {}
    return {
        key: value for key, value in slots.items()
        if not str(key).startswith("_") and value not in (None, "", [], {})
    }


def _questions(message: str, *, last_asked_slot: str = "") -> List[str]:
    """客户这一轮问了什么（价格 / 交期 / 产品 / 其它）。"""
    kinds: List[str] = []
    try:
        from src.rag.reply_composer import is_price_question_with_context

        # 带上下文：客户在回答"价格 vs 质量"时说的 cost/cheap 是偏好，不是问价
        if is_price_question_with_context(message, last_asked_slot=last_asked_slot):
            kinds.append(PRICE_QUESTION)
    except Exception:  # pragma: no cover - 防御式
        pass
    try:
        from src.rag.delivery_info import is_delivery_question

        if is_delivery_question(message):
            kinds.append(DELIVERY_QUESTION)
    except Exception:  # pragma: no cover - 防御式
        try:
            from src.rag.delivery_info import delivery_answer

            if delivery_answer(message):
                kinds.append(DELIVERY_QUESTION)
        except Exception:  # pragma: no cover - 防御式
            pass
    try:
        from src.rag.company_info import is_company_question

        if is_company_question(message):
            kinds.append(CUSTOMER_QUESTION)
    except Exception:  # pragma: no cover - 防御式
        pass
    return kinds


def detect_speech_act(
    message: str,
    *,
    profile: Any = None,
    last_asked_slot: str = "",
) -> "SpeechActResult":
    """识别客户这一轮"在做什么"（纯规则，不调 LLM）。"""
    text = str(message or "").strip()
    if not text:
        return SpeechActResult(speech_act=CASUAL, confidence=0.0)

    requirements = _extract_requirement_slots(text)
    question_kinds = _questions(text, last_asked_slot=last_asked_slot)
    is_question = False
    try:
        from src.rag.query_understanding import looks_like_question

        is_question = bool(looks_like_question(text))
    except Exception:  # pragma: no cover - 防御式
        is_question = "?" in text

    product_question = bool(is_question and _PRODUCT_QUESTION_RE.search(text))
    objection = bool(_OBJECTION_RE.search(text))
    correction = bool(_CORRECTION_RE.search(text))
    confirmation = bool(_CONFIRM_RE.match(text))

    # ① 客户提问（优先级最高）：价格 / 交期 > 产品 > 泛问
    if question_kinds:
        act = question_kinds[0]
    elif product_question:
        act = PRODUCT_QUESTION
    elif objection:
        act = OBJECTION
    elif is_question:
        act = CUSTOMER_QUESTION
    elif correction:
        act = CORRECTION
    elif confirmation and not requirements:
        act = CONFIRMATION
    else:
        act = ""

    # ② 同时带了需求 → 多意图（先更新状态，再回答客户）
    if requirements and act:
        return SpeechActResult(
            speech_act=MULTI_INTENT,
            requirements=requirements,
            customer_questions=[act],
            field=_answered_field(requirements, last_asked_slot),
            value=_answered_value(requirements),
            confidence=0.85,
            evidence=text[:120],
        )

    # ③ 只有需求：是在回答上一轮，还是客户主动补充
    if requirements:
        field_name = _answered_field(requirements, last_asked_slot)
        act = ANSWER_REQUIREMENT if field_name == str(last_asked_slot or "") else NEW_REQUIREMENT
        return SpeechActResult(
            speech_act=act,
            requirements=requirements,
            field=field_name,
            value=_answered_value(requirements),
            confidence=0.8,
            evidence=text[:120],
        )

    if not act:
        act = CASUAL
    return SpeechActResult(
        speech_act=act,
        customer_questions=[act] if act in CUSTOMER_QUESTION_ACTS else [],
        confidence=0.7 if act != CASUAL else 0.3,
        evidence=text[:120],
    )


def _answered_field(requirements: Dict[str, Any], last_asked_slot: str) -> str:
    """客户给的这些槽位里，哪一项对应"我们上一轮问的那个"。"""
    asked = str(last_asked_slot or "")
    if asked:
        for alias in _SLOT_ALIASES.get(asked, (asked,)):
            if alias in requirements:
                return asked
    for slot, aliases in _SLOT_ALIASES.items():
        if any(alias in requirements for alias in aliases):
            return slot
    return ""


def _answered_value(requirements: Dict[str, Any]) -> Any:
    for key, value in requirements.items():
        if key not in ("display_type",):
            return value
    return next(iter(requirements.values()), None)


@dataclass
class SpeechActResult:
    """一轮的"说话行为"识别结果。"""

    speech_act: str = CASUAL
    field: str = ""
    value: Any = None
    requirements: Dict[str, Any] = _dc_field(default_factory=dict)
    customer_questions: List[str] = _dc_field(default_factory=list)
    confidence: float = 0.0
    evidence: str = ""

    @property
    def is_customer_question(self) -> bool:
        if self.speech_act in CUSTOMER_QUESTION_ACTS:
            return True
        return any(item in CUSTOMER_QUESTION_ACTS for item in self.customer_questions)

    @property
    def is_multi_intent(self) -> bool:
        return self.speech_act == MULTI_INTENT

    def question_kind(self) -> str:
        """客户问的类型（没有则空串）。"""
        if self.speech_act in CUSTOMER_QUESTION_ACTS:
            return self.speech_act
        for item in self.customer_questions:
            if item in CUSTOMER_QUESTION_ACTS:
                return item
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speech_act": self.speech_act,
            "field": self.field,
            "value": self.value,
            "requirements": dict(self.requirements),
            "customer_questions": list(self.customer_questions),
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
        }


__all__ = [
    "ALL_SPEECH_ACTS",
    "ANSWER_REQUIREMENT",
    "CASUAL",
    "CONFIRMATION",
    "CORRECTION",
    "CUSTOMER_QUESTION",
    "CUSTOMER_QUESTION_ACTS",
    "DELIVERY_QUESTION",
    "MULTI_INTENT",
    "NEW_REQUIREMENT",
    "OBJECTION",
    "PRICE_QUESTION",
    "PRODUCT_QUESTION",
    "SpeechActResult",
    "detect_speech_act",
]
