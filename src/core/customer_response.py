"""Phase 1（v2.1）：客户回答意图识别 —— Customer Response Intent。

解决的问题（《客户决策状态与灵活追问优化实施计划》第 5 节）：
    现在系统只知道"字段有没有值"，分不清客户这句话到底是

        I don't know.                  → 客户不知道（可以降低门槛再问一次）
        You decide / You recommend.    → **客户把决定权交给 AI**（不再追问，允许确定性推导）
        I don't want to provide that.  → 客户明确不提供 / 不在乎（不再追问）
        Around 5 meters.               → 客户直接给了值
        Actually, it's 5m, not 3m.     → 客户是在纠正之前的信息

    这些如果都存成 ``value = None``，后续 Gate 就无法判断客户的真实意图，
    于是出现"客户说 You decide，AI 还一直追问同一个字段"。

本模块是**纯规则**（零 LLM、可单测）：输入客户原话 + 上一轮问的槽位，
输出客户对各个槽位的决策意图。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 意图取值（与 RequirementProfile.field_decisions 一致，小写）──────────────
CONFIRMED = "confirmed"
UNKNOWN = "unknown"
DELEGATED = "delegated"
DECLINED = "declined"
CORRECTION = "correction"

# ── 客户"让 AI 决定"（授权）────────────────────────────────────────────────
DELEGATED_RE = re.compile(
    r"\byou (?:can |could |may )?(?:decide|choose|pick|recommend|suggest|select)\b|"
    r"\b(?:up to you|your call|your choice|whatever you (?:recommend|suggest|think|choose)|"
    r"whatever (?:you think )?(?:is )?(?:best|suitable|works)|"
    r"just recommend|recommend something|pick one for me|any is fine|"
    r"whichever (?:you|is))(?!\s+(?:not|no))\b|"
    r"由你决定|你来决定|你决定|你帮我决定|你推荐就行|你推荐|你选|你帮我选|帮我选一个|"
    r"听你的|看你的|你看着办|随便(?:你|就行|啦)?|都行|都可以|都随意|哪款都行|"
    r"你定|你看着定",
    re.IGNORECASE,
)

# ── 客户"不知道"────────────────────────────────────────────────────────────
UNKNOWN_RE = re.compile(
    r"\bi (?:really )?(?:do ?n[o']t|don'?t) know\b|"
    r"\b(?:do ?n[o']t|don'?t) know (?:yet|for sure|exactly)\b|"
    r"\bno idea\b|\bhave no idea\b|\bnot sure\b|\bi'?m not sure\b|\bunsure\b|"
    r"\bi (?:do ?n[o']t|don'?t) have (?:the )?(?:measurements?|information)\b|"
    r"\bcan(?:'?t| not) (?:tell|say|estimate|guess)\b|\bhard to (?:say|tell)\b|"
    r"不知道|不清楚|不太清楚|不确定|不太确定|没量过|没测量|没有这个信息|"
    r"无法确定|没法估计|不好估计|说不好|说不准|还不确定|不太了解",
    re.IGNORECASE,
)

# ── 客户"不愿提供 / 不在乎"（DECLINED）────────────────────────────────────
DECLINED_RE = re.compile(
    r"\bi (?:do ?n[o']t|don'?t) (?:want|need|care|wish) to (?:provide|specify|share|tell|answer)\b|"
    r"\b(?:that|it) does ?n[o']t matter\b|\bi do ?n[o']t care(?: about (?:that|it))?\b|"
    r"\bno preference\b|\bi have no preference\b|\bskip (?:that|it|this)\b|"
    r"\bdo ?n[o']t ask\b|"
    r"不想提供|不想说|不方便说|不用问|不用管|这个不用|这个不重要|不重要|无所谓|没有偏好|"
    r"不用了|先跳过|跳过|免了",
    re.IGNORECASE,
)

# ── 客户在纠正之前的信息（CORRECTION）──────────────────────────────────────
CORRECTION_RE = re.compile(
    r"\bactually\b|\bi meant\b|\bcorrection\b|\bnot\s+\d+\s*,?\s*(?:but|it'?s)\b|"
    r"更正|改成|改为|不是.{0,6}是|应该是|说错了|记错了",
    re.IGNORECASE,
)

# ── 客户"又/还"式补充（either / also / 也）→ 说明同一句里覆盖了多个槽位 ────
ALSO_RE = re.compile(r"\beither\b|\balso\b|\btoo\b|也|另外|还有", re.IGNORECASE)

# ── 槽位提及（客户原话里点名了哪一个字段）─────────────────────────────────
SLOT_MENTION_PATTERNS: tuple[tuple[str, str], ...] = (
    # 先匹配更具体的，避免 "width" 命中 "viewing distance" 之类
    ("viewing_distance", r"viewing distance|view distance|视距|观看距离|可视距离|多远|distance"),
    ("pixel_pitch", r"pixel pitch|pitch|点间距|间距|p\s?\d"),
    ("size", r"screen size|size|dimension|尺寸|大小|多大"),
    ("width", r"width|宽度|多宽|宽"),
    ("height", r"height|高度|多高|高"),
    ("environment", r"indoor|outdoor|室内|室外|户外"),
    ("installation", r"installation|install|fixed|rental|固装|固定|租赁|安装方式"),
    ("purpose", r"purpose|scenario|use case|场景|用途|做什么用"),
    ("content_type", r"video|image|图片|视频|内容"),
    ("price_preference", r"price|quality|价格|质量|预算|budget"),
)

_SLOT_MENTION_RE: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (slot, re.compile(pattern, re.IGNORECASE)) for slot, pattern in SLOT_MENTION_PATTERNS
)

# 从句切分：逗号 / 分号 / 句号 / and / but / 另外 …
# 注意：句号要排除小数点（"P2.5" 不能被切开）。
_CLAUSE_SPLIT_RE = re.compile(
    r"[,，;；!?！？\n]|(?<!\d)\.(?!\d)|\band\b|\bbut\b|\balso\b|\beither\b|、|另外|还有|不过",
    re.IGNORECASE,
)
# 句子切分（第二遍回填用）：一句话里"意图 + 多个槽位"
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？\n]+")


@dataclass
class CustomerResponseIntent:
    """客户对某个槽位的回答意图（计划 5.3 的输出结构）。"""

    slot: str = ""
    intent: str = ""
    evidence: str = ""
    confidence: float = 0.0
    value: Any = None
    # 额外：客户是否明确"也/还"(either/also)，用于日志与排查
    also: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "intent": self.intent,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "value": self.value,
        }


def _clauses(text: str) -> List[str]:
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(str(text or "")) if part.strip()]


def _match_phrase(pattern: "re.Pattern[str]", text: str) -> Optional[str]:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _intent_of_clause(clause: str) -> Optional[str]:
    """一个从句表达的是哪一种决策意图（都没有则 None）。"""
    if _match_phrase(DECLINED_RE, clause):
        return DECLINED
    if _match_phrase(DELEGATED_RE, clause):
        return DELEGATED
    if _match_phrase(UNKNOWN_RE, clause):
        return UNKNOWN
    return None


def _slots_in_clause(clause: str) -> List[str]:
    slots: List[str] = []
    for slot, pattern in _SLOT_MENTION_RE:
        if pattern.search(clause) and slot not in slots:
            slots.append(slot)
    # "width" 与 "height" 若同时出现在 "尺寸" 里，保留更具体的那个
    return slots


def detect_response_intents(
    message: str,
    last_asked_slot: str = "",
    conversation_context: str = "",
) -> List[CustomerResponseIntent]:
    """识别客户这句话对各个槽位的回答意图（可一句话覆盖多个槽位）。

    Args:
        message: 客户原话
        last_asked_slot: 上一轮 AI 问的槽位（客户没说清是哪一项时归到它）
        conversation_context: 最近对话（保留参数，规则实现暂不需要）

    Returns:
        ``CustomerResponseIntent`` 列表，按出现顺序；一句话可以返回多条
        （计划 7.1 / 7.2：「I don't know the viewing distance either, you can
        decide the pitch.」→ viewing_distance=UNKNOWN + pixel_pitch=DELEGATED）。
    """
    text = str(message or "").strip()
    if not text:
        return []

    intents: List[CustomerResponseIntent] = []
    seen: Dict[str, int] = {}          # slot -> intents 下标（后出现的覆盖先出现的）
    # 从句里"有意图但没点名槽位"（"just recommend something suitable."）时，
    # 先记下来，处理完整句再回填到"同一句里提到过的槽位"，否则才用 last_asked_slot。
    pending: List[tuple[str, str, float, str]] = []   # (intent, evidence, conf, sentence)

    def _put(slot: str, intent: str, evidence: str, confidence: float) -> None:
        item = CustomerResponseIntent(
            slot=slot,
            intent=intent,
            evidence=evidence[:80],
            confidence=confidence,
            also=bool(ALSO_RE.search(text)),
        )
        if slot and slot in seen:
            # 同一句话里后面的说法覆盖前面的（"I don't know the width, just recommend
            # something suitable." → 最终 width=DELEGATED）
            intents[seen[slot]] = item
        else:
            seen[slot] = len(intents)
            intents.append(item)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    def _sentence_of(clause: str) -> str:
        for sentence in sentences:
            if clause and clause in sentence:
                return sentence
        return text

    for clause in _clauses(text):
        intent = _intent_of_clause(clause)
        if not intent:
            continue
        slots = _slots_in_clause(clause)
        evidence = (
            _match_phrase(DECLINED_RE, clause)
            or _match_phrase(DELEGATED_RE, clause)
            or _match_phrase(UNKNOWN_RE, clause)
            or clause
        )
        confidence = {
            DECLINED: 0.95, DELEGATED: 0.98, UNKNOWN: 0.97,
        }.get(intent, 0.8)
        if slots:
            for slot in slots:
                _put(slot, intent, evidence, confidence)
        else:
            pending.append((intent, evidence, confidence, _sentence_of(clause)))

    # 回填：先看"同一句里点过名的槽位"，再退到 last_asked_slot
    for intent, evidence, confidence, sentence in pending:
        slots = _slots_in_clause(sentence)
        if not slots and last_asked_slot:
            # 客户没说哪一项，但这句话是在回答上一轮问的那一项
            slots = [str(last_asked_slot)]
        for slot in slots:
            _put(slot, intent, evidence, confidence)

    # 第二遍：句子级回填 —— 处理"一句话里给了意图 + 多个槽位"的说法，
    # 例如 "You decide the size and pitch."（计划第 18 节 Case 4）。
    # 只补那些**还没有决策**、但出现在同一句里的槽位。
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not sentence.strip():
            continue
        intent = _intent_of_clause(sentence)
        if not intent:
            continue
        missing_slots = [slot for slot in _slots_in_clause(sentence) if slot not in seen]
        if not missing_slots:
            continue
        evidence = (
            _match_phrase(DECLINED_RE, sentence)
            or _match_phrase(DELEGATED_RE, sentence)
            or _match_phrase(UNKNOWN_RE, sentence)
            or sentence.strip()
        )
        confidence = {DECLINED: 0.9, DELEGATED: 0.92, UNKNOWN: 0.9}.get(intent, 0.8)
        for slot in missing_slots:
            _put(slot, intent, evidence, confidence)

    if intents:
        logger.info(
            "[CustomerResponse] %r (last_asked=%s) → %s",
            text[:60], last_asked_slot or "-",
            [(item.slot, item.intent) for item in intents],
        )
    return intents


def slot_decisions(message: str, last_asked_slot: str = "") -> Dict[str, str]:
    """把意图压缩成 ``{slot: delegated|declined|unknown}``（供 Profile 直接落状态）。"""
    decisions: Dict[str, str] = {}
    for item in detect_response_intents(message, last_asked_slot=last_asked_slot):
        if item.intent in (DELEGATED, DECLINED, UNKNOWN):
            decisions[item.slot] = item.intent
    return decisions


def is_customer_correction(message: str) -> bool:
    """客户是不是在纠正/修改之前说过的信息（计划 5.4 CORRECTION）。"""
    return bool(CORRECTION_RE.search(str(message or "")))


__all__ = [
    "CONFIRMED",
    "CORRECTION",
    "DECLINED",
    "DELEGATED",
    "UNKNOWN",
    "CustomerResponseIntent",
    "detect_response_intents",
    "is_customer_correction",
    "slot_decisions",
]
