"""轮次类型与产品域（计划 v2.9.1 §八/§九/§十）。

两件事必须解耦（计划原文）：

    Intent                   负责判断"这句话主要在做什么"
    Requirement Extraction   负责判断"这句话有没有业务信息"

所以这里给出两个互相独立、纯函数式的判定：

    detect_product_domain(message, profile)   → LED / LCD / IFP / UNKNOWN / MULTI
    classify_turn_kind(message, profile)      → PURE_CONVERSATION /
                                                CONVERSATION_WITH_BUSINESS_SIGNAL /
                                                REQUIREMENT_COLLECTION / FREE_QUESTION

注意（计划 §八）：本阶段**只建立字段和接口**，不实现 LCD/IFP 的需求采集策略。
"""
from __future__ import annotations

import re
from typing import Any

# ── 产品域 ────────────────────────────────────────────────────────────────
LED = "LED"
LCD = "LCD"
IFP = "IFP"
UNKNOWN = "UNKNOWN"
MULTI = "MULTI"

ALL_PRODUCT_DOMAINS = (LED, LCD, IFP, UNKNOWN, MULTI)

_LED_RE = re.compile(r"\b(?:led)\b|LED屏|点间距|箱体|模组", re.IGNORECASE)
_LCD_RE = re.compile(
    r"\blcd\b|video ?wall|splicing|拼接|touch monitor|触摸显示器|拼缝|bezel", re.IGNORECASE
)
_IFP_RE = re.compile(
    r"\bifp\b|interactive flat panel|interactive display|smart ?board|whiteboard|"
    r"会议平板|交互平板|触摸一体机",
    re.IGNORECASE,
)

# 纯寒暄（哪怕带问号也算闲聊，计划 §五：这些不该进 Free Question）
_CHIT_CHAT_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|how are you|how's it going|"
    r"nice to meet you|thanks|thank you|have a nice day|"
    r"你好|您好|在吗|早上好|下午好|晚上好|最近怎么样|谢谢|感谢|辛苦了)[\s!.?,，。！？]*$",
    re.IGNORECASE,
)

# 泛指的"屏 / screen"（没有点名品类时，默认按主力产品域 LED 处理，计划 §八 示例）
_GENERIC_SCREEN_RE = re.compile(r"\b(?:screen|display|panel|wall)\b|屏|大屏|显示器", re.IGNORECASE)

# 以寒暄开场、后面跟着业务信息（"Hi, we're planning..." / "How are you? By the way..."）
_GREETING_OPENER_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|how are you|how's it going|"
    r"nice to meet you|你好|您好|在吗|早上好|下午好|晚上好|最近怎么样)[\s,，!.。！?？]*",
    re.IGNORECASE,
)

# ── 轮次类型 ──────────────────────────────────────────────────────────────
PURE_CONVERSATION = "PURE_CONVERSATION"
CONVERSATION_WITH_BUSINESS_SIGNAL = "CONVERSATION_WITH_BUSINESS_SIGNAL"
REQUIREMENT_COLLECTION = "REQUIREMENT_COLLECTION"
FREE_QUESTION = "FREE_QUESTION"

ALL_TURN_KINDS = (
    PURE_CONVERSATION,
    CONVERSATION_WITH_BUSINESS_SIGNAL,
    REQUIREMENT_COLLECTION,
    FREE_QUESTION,
)


def _profile_value(profile: Any, name: str) -> str:
    if profile is None:
        return ""
    if isinstance(profile, dict):
        return str(profile.get(name) or "")
    return str(getattr(profile, name, "") or "")


def detect_product_domain(message: str, profile: Any = None) -> str:
    """这句话（或当前档案）属于哪个产品域。

    客户口径（整改计划 §八"IFP 判断必须统一"）：**IFP 不再由关键词判成独立产品域** ——
    IFP 是 LCD 的子系列，最终只由 LCD Decision Center（``lcd_decision.is_ifp_requirement``）
    决定。所以这里命中 "interactive flat panel / whiteboard" 这类词时归 **LCD**，
    不再产出 IFP 域（否则就是第二个 IFP 判定中心）。
    """
    text = str(message or "")
    hits = set()
    if _LED_RE.search(text):
        hits.add(LED)
    if _LCD_RE.search(text):
        hits.add(LCD)
    if _IFP_RE.search(text):
        # 交互平板 → LCD 域（子类型由 LCD 决策中心在需求链里判）
        hits.add(LCD)
    if len(hits) > 1:
        return MULTI
    if len(hits) == 1:
        return next(iter(hits))
    # ── 客户**指名型号** → 型号本身就确定了产品域（客户口径 2026-10）──────────
    # 实测：客户说 "i need a TW11-OD-P4"，消息里没有 "LED" 字样 → 域判成 UNKNOWN
    # → 系统反问 "are you after an LED display, or an LCD video wall?" —— 型号白说了。
    # 目录里该型号属于哪个域，就以它为准（LED / LCD；IFP 是 LCD 的子类型，归 LCD）。
    try:
        from ..rag.model_mention import resolve_mentioned_model

        _resolved = resolve_mentioned_model(text)
    except Exception:  # pragma: no cover - 防御式
        _resolved = None
    if _resolved:
        _dtype = str(_resolved.get("display_type") or "").upper()
        if _dtype == IFP:
            _dtype = LCD
        if _dtype in (LED, LCD):
            return _dtype
    stored = _profile_value(profile, "display_type").upper()
    if stored == IFP:
        # 档案里是 IFP（LCD 决策中心已经判过）→ 产品域仍然是 LCD
        return LCD
    if stored in (LED, LCD):
        return stored
    # 计划 v2.9.3 §五：**不再"没点名就默认 LED"** —— 只说 "screen/display" 就是 UNKNOWN，
    # 由 Product Type Router 推断（要确认）或问客户；LED 只在最终兜底时才用。
    return UNKNOWN


def business_signals(message: str) -> dict:
    """这句话里的业务信息（需求抽取层的职责，与 Intent 无关）。"""
    try:
        from src.rag.query_understanding import extract_slots

        slots = dict(extract_slots(message) or {})
    except Exception:  # pragma: no cover - 防御式
        return {}
    return {
        key: value
        for key, value in slots.items()
        if not str(key).startswith("_") and key != "display_type" and value not in (None, "", [], {})
    }


def has_greeting_opener(message: str) -> bool:
    """这句话是不是以打招呼开场（"Hi" / "Hi, I need a display" / "你好，我想问…"）。"""
    text = str(message or "")
    return bool(_GREETING_OPENER_RE.match(text)) or bool(_CHIT_CHAT_RE.match(text))


def classify_turn_kind(message: str, profile: Any = None) -> str:
    """这句话是纯闲聊 / 闲聊带业务信号 / 需求采集 / 自由问题。"""
    from src.dialogue.speech_act import CASUAL, detect_speech_act

    speech = detect_speech_act(message, profile=profile)
    text = str(message or "")
    casual = speech.speech_act == CASUAL or bool(_CHIT_CHAT_RE.match(text))
    # 以寒暄开场的短句（"Hi, how are you?"）本身也是闲聊，不该进 Free Question
    greeting_opener = bool(_GREETING_OPENER_RE.match(text))
    signals = business_signals(message)
    if signals:
        # 寒暄开场 + 业务信息（计划 §十 的 "How are you? By the way, we're looking for..."）
        chitchat_opener = greeting_opener and not _CHIT_CHAT_RE.match(text)
        return (
            CONVERSATION_WITH_BUSINESS_SIGNAL
            if (casual or chitchat_opener)
            else REQUIREMENT_COLLECTION
        )
    if casual or greeting_opener:
        return PURE_CONVERSATION
    return FREE_QUESTION


__all__ = [
    "ALL_PRODUCT_DOMAINS",
    "ALL_TURN_KINDS",
    "CONVERSATION_WITH_BUSINESS_SIGNAL",
    "FREE_QUESTION",
    "IFP",
    "LCD",
    "LED",
    "MULTI",
    "PURE_CONVERSATION",
    "REQUIREMENT_COLLECTION",
    "UNKNOWN",
    "business_signals",
    "classify_turn_kind",
    "detect_product_domain",
    "has_greeting_opener",
]
