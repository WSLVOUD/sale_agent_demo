"""Evidence-based answers for product availability and commercial questions."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .language_utils import _lang, reply_language
from .product_fact_formatter import _format_number, _slot_tokens

logger = logging.getLogger(__name__)



# ── 客户问"有没有某规格 / 某型号" → 用产品数据核实后正面回答 ────────────────
_AVAILABILITY_PATTERNS = (
    r"\bdo (?:you|u|ya) (?:have|carry|stock|sell|make|offer|produce)\b",
    r"\bhave (?:you|u) got\b",
    r"\b(?:can|could) (?:i|we) (?:get|have|order|buy)\b",
    r"\bis there (?:a|an|any)\b",
    r"\bare there any\b",
    r"有没有",
    r"你们有",
    r"有(?:没有)?(?:这种|那种)",
    r"能(?:做|提供|生产)(?:吗|么)",
    r"可以(?:做|提供)",
)


_AVAILABILITY_TEMPLATES = {
    "en": {
        "yes": "Yes — we do carry {spec}.",
        "yes_named": "Yes — we do have {spec}.",
        "unsure": "Let me double-check that spec for you.",
    },
    "zh": {
        "yes": "有的，我们有 {spec} 的 LED 屏。",
        "yes_named": "有的，我们有 {spec}。",
        "unsure": "这个规格我帮您确认一下。",
    },
}


# 客户写 "P 1.2" / "p1.2" 时的兜底点间距识别（只用于"正面回应"，不参与选型 Gate）
_LOOSE_PITCH_RE = re.compile(r"(?<![a-z0-9])p\s*(\d{1,2}(?:\.\d{1,2})?)", re.IGNORECASE)

@lru_cache(maxsize=4)
def _catalog_models(data_dir: str) -> Tuple[Any, ...]:
    """加载（并缓存）Model 级产品数据，用于核实"有没有这个规格"。"""
    try:
        from src.rag.json_loader import load_canonical_models

        return tuple(load_canonical_models(data_dir))
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Reply composer catalog load failed: %s", exc)
        return ()



def _default_data_dir() -> str:
    try:
        from src.config import config

        return str(getattr(config, "DATA_DIR", "data"))
    except Exception:  # pragma: no cover - 防御式
        return "data"



def _pitch_matches(model_pitch: float, asked: float) -> bool:
    """点间距比对：既接受数值接近（1.2 ≈ 1.25），也接受型号命名口径一致。"""
    if model_pitch is None or asked is None:
        return False
    if abs(float(model_pitch) - float(asked)) <= 0.15:
        return True
    return abs(round(float(model_pitch), 1) - round(float(asked), 1)) < 1e-6



def _asked_pitch(slots: Dict[str, Any], message: str = "") -> Optional[float]:
    """客户问的点间距：优先解析器结果，退而求其次用 "P 1.2" 这种宽松写法。"""
    pitch = slots.get("pixel_pitch_mm") or slots.get("pixel_pitch")
    if pitch:
        return float(pitch)
    match = _LOOSE_PITCH_RE.search(str(message or ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:  # pragma: no cover - 防御式
            return None
    return None



def _spec_text(slots: Dict[str, Any], lang: str, message: str = "") -> Optional[str]:
    """把客户问的规格整理成一句话里的名词短语。"""
    pitch = _asked_pitch(slots, message)
    pieces: List[str] = []
    if pitch:
        pieces.append(f"{_format_number(pitch)} mm" if lang == "en" else f"{_format_number(pitch)}mm")
    if slots.get("cob"):
        pieces.append("COB")
    if slots.get("hdr"):
        pieces.append("HDR")
    if slots.get("waterproof"):
        pieces.append("waterproof" if lang == "en" else "防水")
    display_type = str(slots.get("display_type") or "").strip()
    if not pieces and not display_type and not (slots.get("series_id") or slots.get("model")):
        return None
    if lang == "en":
        head = " ".join(pieces)
        # LED / LCD / IFP 走 "an LED"，其余走 "a 1.2 mm COB LED"
        if head and display_type:
            text = f"{head} {display_type}"
        elif head:
            text = head
        else:
            text = display_type or "that"
        article = "an" if text.split()[0][:1].lower() in ("l", "i", "a", "e", "o") else "a"
        return f"{article} {text}"
    head = " ".join(pieces)
    return (head + display_type).strip() or display_type or "该规格"



def availability_answer(
    message: str,
    *,
    language: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    """客户问"你们有没有某规格"时，用产品数据核实后正面回答。"""
    text = str(message or "")
    if not text or not any(re.search(p, text, re.IGNORECASE) for p in _AVAILABILITY_PATTERNS):
        return None

    # 公司 / 办事处 / 地址类问题不是"有没有某规格"，交给 company_info 照实回答
    # （实测 bug：客户问"western region 有没有你们的人"，这里回了一句
    #  "Yes — we do carry an LED."）
    from src.rag.company_info import is_company_question

    if is_company_question(text):
        return None

    slots = _slot_tokens(text)
    lang = _lang(language or reply_language(text))
    models = _catalog_models(str(data_dir or _default_data_dir()))
    templates = _AVAILABILITY_TEMPLATES[lang]

    # 1) 客户点名型号 / 系列 → 直接查目录
    named = str(slots.get("model") or slots.get("series_id") or "").strip()
    if named:
        for model in models:
            if named.lower() in {str(getattr(model, "model", "")).lower(),
                                 str(getattr(model, "series_id", "")).lower(),
                                 str(getattr(model, "series", "")).lower()}:
                return templates["yes_named"].format(spec=named)
        return templates["unsure"]

    # 2) 客户问的是规格（点间距 / COB / HDR / 防水）
    #    必须"有具体规格"才算可用性提问：只问"有没有 LED 屏"这种泛问，
    #    回一句 "Yes — we do carry an LED." 没有信息量（实测 bug）
    pitch = _asked_pitch(slots, text)
    has_specific_spec = bool(
        pitch
        or slots.get("cob")
        or slots.get("hdr")
        or slots.get("waterproof")
        or slots.get("brightness_min")
    )
    if not has_specific_spec:
        return None

    spec = _spec_text(slots, lang, text)
    if not spec:
        return None
    matched = False
    for model in models:
        if pitch and not _pitch_matches(getattr(model, "pixel_pitch_mm", None), pitch):
            continue
        if slots.get("cob") and not getattr(model, "cob", False):
            continue
        if slots.get("hdr") and not getattr(model, "hdr", False):
            continue
        if slots.get("waterproof") and not getattr(model, "waterproof", False):
            continue
        matched = True
        break

    if matched:
        return templates["yes"].format(spec=spec)
    # 没查到就不硬说"有"，也不要编造：如实说需要确认
    if pitch or slots.get("cob") or slots.get("hdr") or slots.get("waterproof"):
        return templates["unsure"]
    return None



# ── 价格 / 报价类提问：需求没问清之前，先说"要确认产品和配置才能报价" ────────
_PRICE_QUESTION_RE = re.compile(
    r"(?<![a-z])(?:price|prices|pricing|cost|costs|quote|quotation|"
    r"how much|discount|cheaper|expensive|budget)\b|"
    r"价格|报价|多少钱|价钱|优惠|折扣|便宜|贵",
    re.IGNORECASE,
)


_PRICE_POLICY_ANSWERS = {
    "en": (
        "Pricing depends on the exact model and cabinet configuration, so I need to confirm the product first — then I'll put together a quotation for you.",
        "I can quote you properly once the model and cabinet layout are confirmed — let me pin those down first.",
        "Prices are per configuration, so I'll confirm the right model first and follow up with the quotation.",
        "The quotation depends on which model and configuration we land on — I'll confirm those details first, then price it up.",
    ),
    "zh": (
        "价格要按具体型号和箱体配置来算，确认好产品之后我马上给您报价。",
        "不同型号和配置价格差别比较大，我先把型号确认下来，随后给您准确报价。",
        "报价需要先确定型号和箱体排布，我们先把需求确认清楚。",
    ),
}



def is_price_question(message: str) -> bool:
    """客户这句话是不是在**问**价格 / 报价。

    客户口径（2026-09-21）实测 bug："cost down the priority"（客户在回答
    "更看重价格还是质量"）里带了 cost，被旧实现判成"问价格" → AI 回了报价口径，
    客户的偏好却没被记下来。

    现在的判定：
      · 明确要报价（quote / quotation / 报价 / how much / 多少钱）→ 是问价
      · 其它价格词（price / cost / budget / cheaper…）→ **必须有疑问语气**
        （问号，或"价格是…/什么价"）才算问价；
      · 客户正在回答"价格 vs 质量"这一问时，一律按**偏好答案**处理。
    """
    return is_price_question_with_context(message, last_asked_slot="")



def is_price_question_with_context(message: str, *, last_asked_slot: str = "") -> bool:
    """带上下文的问价判定（见 :func:`is_price_question` 的说明）。"""
    text = str(message or "")
    if not _PRICE_QUESTION_RE.search(text):
        return False
    if _PRICE_STRONG_RE.search(text):
        return True
    if "?" in text or "？" in text:
        return True
    if _QUESTION_SHAPE_RE.search(text):
        return True
    # 是疑问句（"what is the price of…" / "价格能便宜点吗"）→ 仍然是问价
    try:
        from src.rag.query_understanding import looks_like_question

        if looks_like_question(text):
            return True
    except Exception:  # pragma: no cover - 防御式
        pass
    if str(last_asked_slot or "") == "price_preference":
        # 客户在回答"价格／质量" → 这是偏好，不是问价
        return False
    # 没有疑问语气、也不是明确要报价 → 不当成问价（例如 "cost down"）
    return False



# 明确"要报价"的信号（这些即使没有问号也算问价）
_PRICE_STRONG_RE = re.compile(
    r"quote|quotation|price list|how much|"
    r"报价|多少钱|价格表|什么价|单价",
    re.IGNORECASE,
)


# 疑问句形态（客户在问，而不是在给偏好）——"what is the price…" 这类
# 没有问号也算问句；中文的"…吗 / …呢 / 多少"同理。
_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:what|what's|whats|which|how much|how about|can you|could you|"
    r"do you|does|is there|are there|tell me|give me|send me)\b|"
    r"[?？]|吗|呢|多少|怎么|什么",
    re.IGNORECASE,
)



def price_policy_answer(language: str = "en", seed: int = 0) -> str:
    """"要先确认产品才能报价"的标准应答（多种说法轮换）。"""
    variants = _PRICE_POLICY_ANSWERS.get(_lang(language)) or _PRICE_POLICY_ANSWERS["en"]
    return variants[seed % len(variants)]

__all__ = ['_AVAILABILITY_PATTERNS', '_AVAILABILITY_TEMPLATES', '_LOOSE_PITCH_RE', '_PRICE_POLICY_ANSWERS', '_PRICE_QUESTION_RE', '_PRICE_STRONG_RE', '_QUESTION_SHAPE_RE', '_asked_pitch', '_catalog_models', '_default_data_dir', '_pitch_matches', '_spec_text', 'availability_answer', 'is_price_question', 'is_price_question_with_context', 'price_policy_answer']
