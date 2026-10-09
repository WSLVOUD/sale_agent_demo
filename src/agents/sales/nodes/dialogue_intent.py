"""Sales-turn intent helpers for distinguishing business questions from off-topic chat."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

_PRODUCT_TERMS_RE = re.compile(
    r"\bTW\s*\d{2}\b|\bP\s*\d(?:\.\d+)?\b|\b(?:led|lcd|ifp|cob|gob|hdr|ip6[56])\b|"
    r"pixel\s*pitch|brightness|refresh|resolution|\bnit\b|specs?\b|models?\b|series\b|"
    r"warrant(?:y|ies)|guarantee|certificat(?:e|ion)|"
    r"显示屏|屏幕|大屏|单色屏|点间距|亮度|刷新率|分辨率|型号|规格|参数|防水|像素|屏体|屏",
    re.IGNORECASE,
)

_ZH_BUSINESS_TERMS_RE = re.compile(
    r"定制|订制|定做|订做|改尺寸|加工|OEM|ODM|"
    r"质保|保修|售后|维修|认证|证书|检测报告|"
    r"代理|经销商|分销|工厂|厂家|"
    r"付款|定金|订金|预付|发票|运费|发货|到货|交付|交货|工期|生产周期|"
    r"亮度|分辨率|刷新|点间距|安装方式|"
    r"能(?:不)?能做|可以(?:不)?可以|支持吗|有.{0,4}吗",
    re.IGNORECASE,
)


def is_product_or_business_question(
    message: str, *, last_asked_slot: str = ""
) -> bool:
    """Whether a message asks about products, business, pricing, delivery, or company facts."""
    text = str(message or "")
    if _PRODUCT_TERMS_RE.search(text) or _ZH_BUSINESS_TERMS_RE.search(text):
        return True
    try:
        from ....rag.company_info import is_company_question
        from ....rag.delivery_info import is_delivery_question
        from ....rag.reply_composer import is_price_question_with_context
    except Exception:  # pragma: no cover - optional response helpers
        return False
    try:
        return bool(
            is_company_question(text)
            or is_delivery_question(text)
            or is_price_question_with_context(text, last_asked_slot=last_asked_slot)
        )
    except Exception:  # pragma: no cover - classifier must not block requirement mining
        return False


def is_offtopic_message(
    message: str,
    *,
    rule_slots: Optional[Dict[str, Any]] = None,
    semantic_payload: Optional[Dict[str, Any]] = None,
    usage: str = "",
    last_asked_slot: str = "",
) -> bool:
    """Whether a turn is unrelated to requirements and should be handled as small talk."""
    if rule_slots or semantic_payload or usage:
        return False
    return not is_product_or_business_question(
        str(message or ""), last_asked_slot=last_asked_slot
    )


_is_product_or_business_question = is_product_or_business_question

__all__ = [
    "is_offtopic_message",
    "is_product_or_business_question",
    "_is_product_or_business_question",
]
