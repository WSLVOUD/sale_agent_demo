"""Language selection and customer-facing text safety helpers."""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)



# ── 语言策略 ───────────────────────────────────────────────────────────────
def reply_language(message: str = "") -> str:
    """回复语言：`.env` 里是 auto 时跟随客户语言，否则统一英语。"""
    try:
        from src.config import config

        if getattr(config, "RESPONSE_LANGUAGE_POLICY", "en") != "auto":
            return "en"
        from src.rag.query_understanding import detect_language

        return detect_language(message)
    except Exception:  # pragma: no cover - 防御式
        return "en"



def _lang(language: str) -> str:
    return "zh" if str(language or "").lower().startswith("zh") else "en"



# ── 回复语言兜底（客户口径：策略=en 时绝不能出现中文）────────────────────────
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


_ENFORCE_EN_PROMPT = """Rewrite the customer reply below in English ONLY.

Rules:
1. Keep every fact, number, model code and question exactly as it is.
2. Output English only — absolutely no Chinese characters.
3. Keep the same conversational sales tone and roughly the same length.
4. Do not add, remove or answer anything; just say the same thing in English.
5. Output the rewritten reply only — no quotes, no explanation, no markdown.

Reply:
{text}"""



def contains_cjk(text: str) -> bool:
    """文本里是否含中日韩字符（用于"必须是英文"的语言护栏）。"""
    return bool(_CJK_RE.search(str(text or "")))



def enforce_english(text: str, *, message: str = "") -> str:
    """语言护栏：策略为 en 时，保证发给客户的文本里没有中文。

    - 没有中文 → 原样返回（不增加任何 LLM 调用）；
    - 有中文 → 用**一次** LLM 调用重写成英文；
    - 重写失败或结果仍是中文 → 返回空串，由调用方退回英文兜底话术，
      绝不让中文发给客户。

    策略为 ``auto`` 时不做任何处理（跟随客户语言是预期行为）。
    """
    text = str(text or "")
    if not text or not contains_cjk(text):
        return text
    if reply_language(message) != "en":
        return text
    logger.warning("CJK reply detected under policy=en — rewriting to English: %r", text[:80])
    try:
        from src.core.llm import get_llm

        response = get_llm(temperature=0).invoke(_ENFORCE_EN_PROMPT.format(text=text[:1500]))
        rewritten = str(getattr(response, "content", response) or "").strip()
        if rewritten and not contains_cjk(rewritten):
            return rewritten
        logger.warning("English rewrite still contains CJK — dropping it")
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        logger.warning("English rewrite failed: %s", exc)
    return ""



_ACK_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# 只有一句空泛客套、完全没有客户内容的"接话"（客户口径：这种不要发）
_CLICHE_ACK_RE = re.compile(
    r"^(?:(?:got it|ok|okay|understood|sure|noted|thanks|thank you|thanks for that|alright|fine|"
    r"noted thanks|好的|收到|了解|明白|知道了|嗯|行|可以)[.。！!,，、\s]*)+$",
    re.IGNORECASE,
)



def _clean_llm_ack(text: str, language: str = "en") -> str:
    """清洗 LLM 生成的回应：去换行 / 去引号 / 丢弃含问句的内容 / 语言纠偏。"""
    ack = " ".join(str(text or "").split()).strip()
    if not ack:
        return ""
    ack = ack.strip('"\'“”')
    if any(mark in ack for mark in ("?", "？")):
        # 追问由系统另外拼接，避免一句话里出现两个问题
        return ""
    if _lang(language) == "en" and _ACK_CJK_RE.search(ack):
        # 策略要求全英文时，模型偶尔仍会用中文回一句
        # （实测 "Sello 你好，很高兴认识你。" + 英文追问），这里直接丢弃
        return ""
    if _CLICHE_ACK_RE.match(ack):
        # 只有 "Got it / 好的" 这种没有任何客户内容的空话 → 丢弃，
        # 交给带客户内容的 echo 兜底（客户口径：接话要顺着客户说）
        return ""
    return ack[:240].rstrip()

__all__ = ['_ACK_CJK_RE', '_CJK_RE', '_CLICHE_ACK_RE', '_ENFORCE_EN_PROMPT', '_clean_llm_ack', '_lang', 'contains_cjk', 'enforce_english', 'reply_language']
