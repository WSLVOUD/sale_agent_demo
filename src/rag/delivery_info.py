"""
交付时间问答（客户口径）：

  - 客户问交期 → 从下单付款开始计算，常规交付时间约 **15–30 天**；
  - 客户要求加快 → 可以走**空运**（能提前交付，但**成本会增加**）；
  - 客户说"想 11 月安装"这类档期 → 先接住他的话，再给交期，
    让他自己判断来不来得及（不承诺具体日期）。

和公司信息/价格口径一样：多种说法轮换，避免每次同一句死板话术。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

LEAD_TIME_DAYS = (15, 30)

# 交付 / 交期类提问
_DELIVERY_QUESTION_RE = re.compile(
    r"交期|交货期|交货时间|交付时间|发货时间|到货时间|多久(?:能)?(?:发货|到货|交付|交付|交货|生产)|"
    r"什么时候(?:能)?(?:发货|到货|交付|交货)|生产周期|备货时间|货期|"
    r"\b(?:lead time|delivery time|delivery date|delivery lead|shipping time|time to deliver|"
    r"when (?:can|will|do) you (?:ship|deliver)|how long (?:does|will|would|to)|how soon)\b",
    re.IGNORECASE,
)

# 要求加快 / 加急
_FASTER_RE = re.compile(
    r"加快|加急|尽快|越快越好|能不能快|可以快|能快一点|急用|急着|赶时间|赶工期|提前交货|提前发货|"
    r"\b(?:faster|sooner|asap|as soon as possible|urgent|urgently|expedite|rush(?:ed)?)\b",
    re.IGNORECASE,
)

# 客户提到安装 / 交付档期（"11月安装"、"need it by December"）
_MONTH_TOKEN = (
    r"(?:\d{1,2}\s*月|[一二三四五六七八九十]{1,3}月|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"下个?月|这个月|年底|年前|春节前|节前|next month|this month|end of the year)"
)
_INSTALL_TIMING_RE = re.compile(
    r"(?:安装|装好|交付|交货|发货|送到|deliver|install|set up|needs? it by|need it by|before|by)"
    r"[^。.!?？]{0,20}?" + _MONTH_TOKEN
    + r"|" + _MONTH_TOKEN + r"[^。.!?？]{0,20}?(?:安装|交付|交货|发货|deliver|install)",
    re.IGNORECASE,
)

_LEAD_TIME_ANSWERS = {
    "en": (
        "Counting from when the order is placed and paid, our usual delivery time is about 15–30 days — I'll confirm the exact schedule once the model and quantity are fixed.",
        "From order and payment, delivery normally takes around 15–30 days, depending on the model and quantity.",
        "Our standard lead time is roughly 15–30 days from order and payment; I'll pin down the dates once we settle the configuration.",
        "Once the order and payment are in, we usually need about 15–30 days for delivery — the final date follows the model and quantity.",
    ),
    "zh": (
        "从下单付款开始计算，我们正常的交付时间大约是 15–30 天；具体日期会在型号和数量确认后跟您敲定。",
        "下单并付款之后，一般 15–30 天可以交付，实际还要看型号和数量，确认后我把准数给您。",
        "我们的常规交期是下单付款后约 15–30 天，配置定下来我再跟您确认具体时间。",
    ),
}

_FASTER_ANSWERS = {
    "en": (
        "If you need it sooner, we can ship by air — that shortens the delivery time, but it does add to the shipping cost, and I'll include it in the quotation.",
        "For a faster schedule we'd switch to air freight: it costs more, but it brings the delivery date forward — I can price that option for you.",
    ),
    "zh": (
        "如果时间紧，我们可以走空运：空运能缩短交付时间，但成本会增加，这部分我会在报价里一并给您。",
        "需要加快的话可以改走空运，交期能提前，不过运费成本会高一些，我可以帮您把这部分一起算进去。",
    ),
}

_INSTALL_TIMING_NOTES = {
    "en": (
        "Noted — you're aiming to install it in {when}. Counting from order and payment we normally "
        "need about 15–30 days, so confirming the order with enough lead time keeps that date on track.",
        "Got it, {when} is your target for installation. From order and payment our delivery usually "
        "takes about 15–30 days, so we'd want the order confirmed in good time.",
    ),
    "zh": (
        "明白，您是计划在{when}安装。我们的交期是下单付款后约 15–30 天，提前下单就能赶上这个时间。",
        "收到，{when}安装这个时间点我记下了。下单付款后交付大约需要 15–30 天，早一点确认订单就能对上您的档期。",
    ),
}


def _lang(language: Optional[str]) -> str:
    return "zh" if str(language or "").lower().startswith("zh") else "en"


def is_delivery_question(message: str) -> bool:
    return bool(_DELIVERY_QUESTION_RE.search(str(message or "")))


def wants_faster_delivery(message: str) -> bool:
    return bool(_FASTER_RE.search(str(message or "")))


def extract_timing_phrase(message: str) -> str:
    """取客户说的档期原词（"11月" / "November" / "下个月"）。"""
    match = _INSTALL_TIMING_RE.search(str(message or ""))
    if not match:
        return ""
    month = re.search(_MONTH_TOKEN, match.group(0), re.IGNORECASE)
    return month.group(0).strip() if month else ""


def mentions_install_timing(message: str) -> bool:
    return bool(_INSTALL_TIMING_RE.search(str(message or "")))


def delivery_answer(message: str, *, language: Optional[str] = None, seed: int = 0) -> Optional[str]:
    """交付 / 交期问题的回答；不是这类问题（也没要求加快）时返回 None。"""
    text = str(message or "")
    is_question = is_delivery_question(text)
    faster = wants_faster_delivery(text)
    if not (is_question or faster):
        return None
    lang = _lang(language)
    parts = []
    if is_question or mentions_install_timing(text):
        variants = _LEAD_TIME_ANSWERS[lang]
        parts.append(variants[seed % len(variants)])
    if faster:
        faster_variants = _FASTER_ANSWERS[lang]
        parts.append(faster_variants[seed % len(faster_variants)])
    return " ".join(parts) if parts else None


def install_timing_note(message: str, *, language: Optional[str] = None, seed: int = 0) -> Optional[str]:
    """客户说"想 11 月安装"这类档期 → 先接住 + 给出交期；没有档期信息时返回 None。"""
    when = extract_timing_phrase(message)
    if not when:
        return None
    lang = _lang(language)
    templates = _INSTALL_TIMING_NOTES[lang]
    note = templates[seed % len(templates)].format(when=when)
    if wants_faster_delivery(message):
        faster_variants = _FASTER_ANSWERS[lang]
        note = f"{note} {faster_variants[seed % len(faster_variants)]}"
    return note


__all__ = [
    "LEAD_TIME_DAYS",
    "delivery_answer",
    "extract_timing_phrase",
    "install_timing_note",
    "is_delivery_question",
    "mentions_install_timing",
    "wants_faster_delivery",
]
