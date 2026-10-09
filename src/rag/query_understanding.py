"""
Phase 4：Query Understanding + Query Rewrite。

目标（来自计划文档）：
    不要直接使用客户原始自然语言进行 RAG。

    "I need a screen for a conference room around 5m viewing distance."
        ↓ 结构化
    {"display_type": "LED", "environment": "indoor", "purpose": "conference",
     "installation": "fixed", "viewing_distance_m": 5}
        ↓ 标准化检索式
    "indoor fixed installation LED display conference room viewing distance 5m"

设计要点：
  - 纯规则实现（确定性 + 零延迟 + 可测试），LLM 只作为可选补充（``use_llm=True``）
  - 支持中英文与常见多语言关键词（西/法/德/葡/俄/日），为 Phase 13 打底
  - 输出的 slots 直接对应 Phase 6 的 Requirement Profile 与 Phase 8 的推荐评分
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .language import (
    LANGUAGE_NAMES,
    detect_language,
    language_name,
    response_language_rule,
)
from .retrieval_query import (
    build_retrieval_query as _build_retrieval_query,
    merge_slots,
)
from .slot_extractor import (
    _DISPLAY_TYPE_KEYWORDS,
    _FIXED_KEYWORDS,
    _INDOOR_KEYWORDS,
    _OUTDOOR_KEYWORDS,
    _RENTAL_KEYWORDS,
    _SEMI_OUTDOOR_KEYWORDS,
    _detect_purpose,
    environment_from_purpose,
    extract_slots,
    looks_like_question,
    purpose_english,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LANGUAGE_NAMES",
    "QueryUnderstanding",
    "build_retrieval_query",
    "detect_language",
    "environment_from_purpose",
    "extract_slots",
    "language_name",
    "looks_like_question",
    "merge_slots",
    "purpose_english",
    "response_language_rule",
    "understand_query",
    "_DISPLAY_TYPE_KEYWORDS",
    "_FIXED_KEYWORDS",
    "_INDOOR_KEYWORDS",
    "_OUTDOOR_KEYWORDS",
    "_RENTAL_KEYWORDS",
    "_SEMI_OUTDOOR_KEYWORDS",
    "_detect_purpose",
]


# ── 结果结构 ────────────────────────────────────────────────────────────────
@dataclass
class QueryUnderstanding:
    """Query 理解结果：结构化槽位 + 标准化检索式。"""
    raw_query: str
    language: str = "en"
    slots: Dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""
    method: str = "rule"
    # Phase 6：结构化需求档案（历史 + 本轮合并）
    profile: Any = None

    # 便捷访问
    @property
    def display_type(self) -> Optional[str]:
        return self.slots.get("display_type")

    @property
    def environment(self) -> Optional[str]:
        return self.slots.get("environment")

    @property
    def installation(self) -> Optional[str]:
        return self.slots.get("installation")

    @property
    def purpose(self) -> Optional[str]:
        return self.slots.get("purpose")

    @property
    def viewing_distance_m(self) -> Optional[float]:
        return self.slots.get("viewing_distance_m")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "language": self.language,
            "slots": dict(self.slots),
            "retrieval_query": self.retrieval_query,
            "method": self.method,
            "profile": self.profile.model_dump() if self.profile is not None else None,
        }


def build_retrieval_query(slots: Dict[str, Any], fallback: str = "") -> str:
    """Keep the existing query-understanding API for normalized retrieval queries."""
    return _build_retrieval_query(
        slots,
        fallback,
        purpose_english=purpose_english,
    )


def understand_query(
    message: str,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    llm_slots: Optional[Dict[str, Any]] = None,
    profile: Any = None,
) -> QueryUnderstanding:
    """Query 理解主入口。

    Args:
        message: 本轮用户消息
        history: 对话历史（用于补全多轮槽位）
        llm_slots: 可选，由 LLM 提取的槽位（LLM 只做"提取事实"，不做技术参数推断）
    """
    from src.models.requirement import RequirementProfile, merge_profiles

    text = str(message or "").strip()
    slots: Dict[str, Any] = {}
    history_slots: Dict[str, Any] = {}
    for turn in history or []:
        role = turn.get("role") or turn.get("type")
        if role in ("user", "human"):
            history_slots = merge_slots(history_slots, extract_slots(str(turn.get("content", ""))))
    slots = merge_slots({}, history_slots)

    current_slots = extract_slots(text)
    slots = merge_slots(slots, current_slots)
    if llm_slots:
        # LLM 只补充它独有的信息，不覆盖规则已确定的事实
        for key, value in llm_slots.items():
            if value in (None, "", [], {}):
                continue
            slots.setdefault(key, value)

    # Phase 6：历史事实标记为 inferred，本轮明确说出的标记为 explicit
    base = profile if isinstance(profile, RequirementProfile) else None
    if base is None and history_slots:
        base = RequirementProfile.from_slots(history_slots)
    merged_profile = merge_profiles(base, current_slots, explicit_keys=set(current_slots))
    if llm_slots:
        merged_profile = merge_profiles(merged_profile, llm_slots)

    return QueryUnderstanding(
        raw_query=text,
        language=detect_language(text) if text else "en",
        slots=slots,
        retrieval_query=build_retrieval_query(slots, fallback=text),
        method="llm+rule" if llm_slots else "rule",
        profile=merged_profile,
    )
