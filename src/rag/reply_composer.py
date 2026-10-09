"""
"先回应客户这句话 + 再追问一个需求" 的组合回复。

背景（客户实测反馈）：
    客户问 "Do u have P 1.2 COB Led"，系统只回了一句
    "…will it be indoors or outdoors?"；客户问 "can I get ur representative in
    Indonesia"，系统又答了一堆通用话术、完全没提需求。销售不能只会问问题。

设计原则：
  1. **先回应** —— 客户这句话本身要被接住：
       - 问"有没有某规格 / 某型号" → 拿产品数据**核实后**正面回答（不编造）
       - 客户给出需求信息（场景 / 室内外 / 视距 / 尺寸…） → 把它复述一遍
       - 别的问题 → 由 RAG 答案负责回答（这里不再硬加一句空话）
  2. **再追问** —— 追问的问题文本由 Ready Gate 决定，**原样保留**，
     这里只负责给不同轮次换不同的引导语，避免每次都同一套固定话术。
  3. 纯规则 / 复用既有解析器，不额外增加 LLM 调用。
"""
# ── v2.5++（僵硬话术优化 §8）：本模块现在的定位是 **Fact / Format Utility** ──
# 保留：产品事实格式化、数值/单位格式化、语言工具、安全清洗、输出辅助。
# 逐步退出：ACK 选择、Connector 选择、Bridge 选择、客户信息 Echo、问题前置话术 ——
#   它们**不再承担"最终话术结构"职责**。客户可见文本的唯一出口是
#   ``src/dialogue/response_generator.py``（ResponseContext → LLM 原生生成 → Validator）；
#   下面的组合函数只在"没有 LLM / LLM 不可用"时作为兜底使用。
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .language_utils import (
    _clean_llm_ack,
    _lang,
    contains_cjk,
    enforce_english,
    reply_language,
)
from .product_fact_formatter import (
    _ACK_ECHO_LEADS,
    _PURPOSE_LABELS,
    ack_conflicts_with_slot,
    purpose_phrase,
    requirement_echo,
    vision_confirmation_items,
    vision_confirmation_sentence,
)
from .commercial_answers import (
    availability_answer,
    is_price_question,
    is_price_question_with_context,
    price_policy_answer,
)
from .fallback_answers import (
    degraded_note,
    has_no_product_phrase,
    is_relaxation_answer,
    missing_impact,
    off_topic_steer_answer,
    product_fallback_answer,
    quote_confirmation_answer,
    relaxation_answer,
)

logger = logging.getLogger(__name__)

# 已有答案为前提，往追问过渡的引导语
_CONNECTORS = {
    "en": (
        "So I can match the right model,",
        "In the meantime,",
        "While we're at it,",
        "And so I can point you to the right one,",
        "To narrow it down,",
        "That said,",
    ),
    "zh": (
        "另外，",
        "顺便问一下，",
        "为了给您匹配更合适的型号，",
        "同时，",
        "再确认一下，",
        "这样我好帮您缩小范围，",
    ),
}

# 实在没有可回应内容时的中性兜底（避免出现"只丢一个问题"的回复）
# 刚"回答完客户的问题"（公司/价格/规格核实）之后再抛需求问题时的过渡语，
# 直接硬接问句会很生硬（实测反馈："…from there. Is this a permanent installation…?"）
_BRIDGES = {
    "en": (
        "By the way —",
        "On that note —",
        "While we're at it —",
        "So I can point you to the right model —",
        "That said —",
        "Now —",
    ),
    "zh": (
        "另外，",
        "顺便问一下，",
        "说到这个，",
        "同时，",
    ),
}


_ACK_GENERIC = {
    "en": (
        "Understood.",
        "Got it.",
        "Noted, thanks.",
        "Alright.",
        "Thanks for that.",
    ),
    "zh": (
        "好的。",
        "明白了。",
        "收到。",
        "了解。",
    ),
}


# 图片确认句 → 追问之间的过渡词（让两句接得上，而不是硬拼）
_VISION_BRIDGES = {
    "en": ("So, ", "Then, ", "Now, ", "Also, ", "So I can match the right model, "),
    "zh": ("那么，", "这样的话，", "那个，", "顺便问一下，"),
}

# 纯客套、没有实质信息的"回应"：已经有图片确认句时就不再叠一遍
_GENERIC_ACK_STARTS = {
    "en": (
        "got it", "thanks", "thank you", "sure", "ok", "okay", "understood",
        "alright", "sounds good", "happy to help", "no problem", "noted",
    ),
    "zh": ("好的", "收到", "明白", "了解", "没问题", "谢谢", "嗯", "可以"),
}


def _is_generic_ack(text: str, language: str) -> bool:
    """这句"回应"是不是纯客套（是的话，图片确认句在场时就不重复了）。"""
    cleaned = str(text or "").strip().lower()
    if not cleaned:
        return False
    if language == "en" and len(cleaned.split()) > 14:
        return False
    return any(cleaned.startswith(prefix) for prefix in _GENERIC_ACK_STARTS.get(language, ()))


def acknowledge(
    message: str,
    *,
    requirement: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    seed: int = 0,
    data_dir: Optional[str] = None,
    llm_ack: str = "",
    slot: str = "",
) -> str:
    """生成"接住客户这句话"的一小段回应（没有可接的内容则返回空串）。

    优先级：
      1. 用产品数据**核实过**的可用性回答（"有没有 P1.2 COB" → 不能靠猜）
      2. LLM 依据客户原话生成的口语回应（自我介绍 / 提问 / 要报价等）
      3. 规则复述客户刚说的需求（场景 / 视距 / 尺寸…）
      4. 中性兜底（"Got it."）—— 保证销售永远不会只丢一个问题过去

    ``slot`` 是接下来要继续追问的那一项：如果回应里已经确认了这一项，
    就退回到中性兜底，避免"刚确认完又问同一件事"。
    """
    lang = _lang(language or reply_language(message))
    candidates: List[str] = []
    availability = availability_answer(message, language=lang, data_dir=data_dir)
    if availability:
        candidates.append(availability)
    # 公司 / 办事处 / 地址类问题：按 data/company_profile.txt 照实回答
    from src.rag.company_info import company_answer

    company = company_answer(message, language=lang, seed=seed)
    if company:
        candidates.append(company)
    llm_cleaned = _clean_llm_ack(llm_ack, lang)
    if llm_cleaned:
        candidates.append(llm_cleaned)
    echo = requirement_echo(message, requirement, language=lang, exclude_slot=slot)
    if echo:
        candidates.append(_ACK_ECHO_LEADS[lang][seed % len(_ACK_ECHO_LEADS[lang])].format(echo=echo))

    for candidate in candidates:
        if not ack_conflicts_with_slot(candidate, slot, lang):
            return candidate
    generic = _ACK_GENERIC[lang]
    return generic[seed % len(generic)]


def _already_asks(answer: str, slot: str, language: str) -> bool:
    """答案里是否已经在问同一个槽位（避免同一轮把同一个问题问两遍）。"""
    if not answer or not slot:
        return False
    try:
        from src.rag.readiness import QUESTION_VARIANTS

        for variant in (QUESTION_VARIANTS.get(slot) or {}).get(_lang(language)) or ():
            if variant.strip() and variant.strip() in answer:
                return True
    except Exception:  # pragma: no cover - 防御式
        return False
    return False


# 问句模板里的"固定铺垫"（纯过渡话术，没有实际信息）——接了过渡语之后就该去掉；
# 而 "Fixed installation or rental —" 这种是**问句内容**，必须保留。
_CANNED_PREAMBLES = (
    "that's okay", "that is okay", "that's fine", "that is fine", "no problem",
    "just so i quote", "just so i use it correctly", "just so i match", "just so i plan",
    "just so i can point you", "quick check", "quick one", "one quick question",
    "while we're at it", "if you're not sure", "if you are not sure",
    "so i plan this properly", "in the meantime",
    "没关系", "不确定也没关系", "顺便", "另外",
)


def _tail_question(question: str) -> str:
    """取问句本体：只去掉"固定铺垫"，保留问句本身的内容。

    例：
      "Just so I match the right models — is this a fixed install or a rental?"
        → "is this a fixed install or a rental?"          （铺垫是固定话术 → 去掉）
      "Fixed installation or rental — which one is it for you?"
        → 原样保留（"Fixed installation or rental" 是问句内容，不能砍）
    """
    text = str(question or "").strip()
    for dash in ("—", "–"):
        if dash not in text:
            continue
        head, tail = text.rsplit(dash, 1)
        head_clean = head.strip().lower()
        tail = tail.strip()
        if not tail:
            continue
        if head_clean.startswith(_CANNED_PREAMBLES) or len(head_clean.split()) <= 3:
            return tail
    return text


def _lower_first_word(text: str) -> str:
    """引导语（以逗号结尾）后面的问句首字母小写，读起来才自然。

    例："…how many cabinets would fit best. In the meantime, Will it be an
    indoor or outdoor setup?" → 末尾问句改成小写 w。
    """
    text = str(text or "")
    if not text:
        return text
    head = text.split(" ", 1)[0]
    if head in ("I", "I'd", "I'll", "I'm") or head.isupper():
        return text
    return text[0].lower() + text[1:]


def _ack_is_customer_answer(message: str, data_dir: Optional[str] = None) -> bool:
    """这一轮的"回应"是不是在**回答客户提出的问题**（公司 / 价格 / 规格核实）。

    这类回应之后再抛需求问题需要过渡语；而"复述客户刚说的需求"（Got it — a
    church.）直接接问句是自然的。
    """
    text = str(message or "")
    if not text:
        return False
    if is_price_question(text):
        return True
    try:
        from src.rag.company_info import is_company_question

        if is_company_question(text):
            return True
    except Exception:  # pragma: no cover - 防御式
        pass
    return bool(availability_answer(text, data_dir=data_dir))


# 【Legacy · 仅作兜底】v2.5++ 起主链路不再用它"拼"客户话术（见文件头说明）：
# 只在 ResponseGenerator 拿不到 LLM / 生成失败时，由 sales 节点回退到这里。
def compose_requirement_reply(
    *,
    answer: str = "",
    question: str = "",
    slot: str = "",
    message: str = "",
    language: Optional[str] = None,
    seed: int = 0,
    requirement: Optional[Dict[str, Any]] = None,
    include_ack: bool = True,
    data_dir: Optional[str] = None,
    llm_ack: str = "",
    vision_confirmation: str = "",
) -> str:
    """把"回应"与"追问"合成一句自然的销售回复。

    - ``answer`` 为空（Sales 自己收需求）：``回应 + 追问``
    - ``answer`` 已经是同一个追问（Solution 的 clarify）：只补一句回应
    - ``answer`` 是别的答复（RAG 答客户问题）：``答复 + 引导语 + 追问``
    """
    answer = str(answer or "").strip()
    question = str(question or "").strip()
    vision_confirmation = str(vision_confirmation or "").strip()

    lang = _lang(language or reply_language(message))
    ack = (
        acknowledge(
            message,
            requirement=requirement,
            language=lang,
            seed=seed,
            data_dir=data_dir,
            llm_ack=llm_ack,
            slot=slot,
        )
        if include_ack
        else ""
    )
    # 图片确认句本身就包含了"接住客户这句话"（照片收到 + 我看到什么），
    # 所以纯客套的 ack（"Got it — happy to help you find a display like that."）
    # 不再叠上去 —— 否则就是三句话各说各的，读起来很生硬。
    if vision_confirmation and _is_generic_ack(ack, lang):
        ack = ""
    # 【实测 bug】图片确认句里刚说了"看起来是固定安装"，紧接着又问"是固装还是租赁" → 自相矛盾。
    # 图片确认本身就是"跟客户核对这一项"，所以这一项本轮不再追问。
    if question and vision_confirmation and slot and ack_conflicts_with_slot(
        vision_confirmation, slot, lang
    ):
        logger.info(
            "Dropping question for slot=%s — already covered by the vision confirmation",
            slot,
        )
        question = ""
    # 图片识别结果先跟客户确认：放在"回应"之后、追问之前
    lead = " ".join(part for part in (ack, vision_confirmation) if part).strip()

    if not question:
        # 没有待问项时保持原行为（不要把 ack 硬拼上来）
        if not vision_confirmation:
            return answer
        return " ".join(part for part in (answer, vision_confirmation) if part).strip()

    if answer and _already_asks(answer, slot, lang):
        return f"{lead} {answer}".strip() if lead else answer
    if answer:
        connectors = _CONNECTORS[lang]
        tail = _tail_question(question)
        if lang == "en":
            tail = _lower_first_word(tail)
        body = f"{answer} {connectors[seed % len(connectors)]} {tail}".strip()
        return f"{lead} {body}".strip() if lead else body
    if lead:
        # 有图片确认句时：用过渡词把"确认"和"追问"接起来，并用完整问句
        # （"…looks like an indoor LED screen for a conference room, correct me if I've misread it.
        #   So, is this a permanent install, or is it for rental/events?"）
        if vision_confirmation:
            bridges = _VISION_BRIDGES[lang]
            full_question = question
            if lang == "en":
                full_question = _lower_first_word(full_question)
            # 中文不加空格，英文加空格
            separator = "" if lang == "zh" else " "
            return f"{lead}{separator}{bridges[seed % len(bridges)]}{full_question}".strip()
        # 【客户口径】只要既有"接话"又有"追问"，就必须有一个过渡把它们连起来，
        # 不能两句硬拼（"…enjoy a good walk. Where's it going to be used?"）。
        # 过渡语按轮次轮换，英文还会把问句首字母小写，读起来是一段话。
        bridges = _BRIDGES[lang]
        tail = _tail_question(question)
        if lang == "en":
            tail = _lower_first_word(tail)
        separator = "" if lang == "zh" else " "
        return f"{lead}{separator}{bridges[seed % len(bridges)]} {tail}".strip()
    return question


__all__ = [
    "ack_conflicts_with_slot",
    "acknowledge",
    "availability_answer",
    "compose_requirement_reply",
    "contains_cjk",
    "degraded_note",
    "enforce_english",
    "has_no_product_phrase",
    "is_price_question",
    "missing_impact",
    "price_policy_answer",
    "relaxation_answer",
    "product_fallback_answer",
    "reply_language",
    "requirement_echo",
    "vision_confirmation_items",
    "vision_confirmation_sentence",
]
