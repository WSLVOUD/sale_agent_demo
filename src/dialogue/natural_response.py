"""v2.7 §16（Phase 14）：取消机械确认话术。

禁止列表（计划原文）：

    Got it. / Thanks. / Understood. / Perfect. / Great. / That helps.
    Based on that... / I understand. / Thank you for providing...

例外：客户主动提供大量复杂信息、确实需要自然回应时才允许（``allow=True``）。
普通参数（P3 / 8 meters / indoor）→ **不确认**，直接继续。
"""
from __future__ import annotations

import re
from typing import List, Tuple

BANNED_PREFIXES: Tuple[str, ...] = (
    "got it",
    "thanks",
    "thank you",
    "okay",
    "ok",
    "understood",
    "perfect",
    "great",
    "that helps",
    "based on that",
    "i understand",
    "thank you for providing",
    "thanks for providing",
    "thanks for the information",
    "thank you for the information",
    "noted",
    "no problem",
    "no worries",
)

_SPLIT_RE = re.compile(
    r"^(?P<prefix>[^.!?。！？]{1,80}?)(?P<sep>[.!?。！？,，—-]\s*)(?P<rest>.*)$", re.S
)


def _is_banned(head: str) -> bool:
    lowered = str(head or "").strip().lower()
    return any(
        lowered == item
        or lowered.startswith(item + " ")
        or lowered.startswith(item + ",")
        for item in BANNED_PREFIXES
    )


def find_mechanical_prefixes(text: str) -> List[str]:
    """找出文本里用到的机械确认开头（用于指标统计）。"""
    found: List[str] = []
    body = str(text or "").strip()
    for _ in range(3):
        match = _SPLIT_RE.match(body)
        if not match:
            break
        head = match.group("prefix").strip()
        if not _is_banned(head):
            break
        found.append(head)
        body = match.group("rest").strip()
    return found


def strip_mechanical_phrases(text: str, *, allow: bool = False) -> Tuple[str, List[str]]:
    """去掉机械确认开头，返回 ``(新文本, 被去掉的开头)``。"""
    body = str(text or "").strip()
    if allow or not body:
        return body, []
    removed: List[str] = []
    for _ in range(3):
        match = _SPLIT_RE.match(body)
        if not match:
            break
        head = match.group("prefix").strip()
        if not _is_banned(head):
            break
        removed.append(head)
        body = match.group("rest").strip()
    if removed and not body:
        return "", removed
    if removed and body:
        body = body[0].upper() + body[1:]
    return body, removed


def is_mechanical(text: str) -> bool:
    return bool(find_mechanical_prefixes(text))


__all__ = [
    "BANNED_PREFIXES",
    "find_mechanical_prefixes",
    "is_mechanical",
    "strip_mechanical_phrases",
]
