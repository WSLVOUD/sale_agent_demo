"""会话记忆窗口（客户口径 2026-09-21）：把最近 50 条以内的对话交给 LLM。

问题背景：之前只有"当前这一句"进 LLM（意图分类）或只给 6 条（需求采集），
于是客户换个说法、或者前面已经说过的事实，AI 都"看不见"。

这个模块是**唯一**取对话窗口的地方（条数 / 单条长度 / 总量都在这里控）：

    get_dialogue_window(session_id, limit=50, per_message_chars=300, total_chars=6000)

返回按时间正序的 ``[{"role": "user"|"assistant", "content": str, "ts": float}]``。

配置（.env）：

    LED_RAG_HISTORY_LIMIT=50                # 最多取多少条
    LED_RAG_HISTORY_PER_MESSAGE_CHARS=300   # 单条截断
    LED_RAG_HISTORY_TOTAL_CHARS=6000        # 总字符预算（真正限制上下文大小的是它）
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_LIMIT = 50
DEFAULT_PER_MESSAGE_CHARS = 300
DEFAULT_TOTAL_CHARS = 6000

_USER_ROLES = frozenset({"user", "human"})
_ASSISTANT_ROLES = frozenset({"assistant", "ai"})


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default)) or default))
    except Exception:  # pragma: no cover - 防御式
        return default


def history_limit() -> int:
    return _env_int("LED_RAG_HISTORY_LIMIT", DEFAULT_HISTORY_LIMIT)


def per_message_chars() -> int:
    return _env_int("LED_RAG_HISTORY_PER_MESSAGE_CHARS", DEFAULT_PER_MESSAGE_CHARS)


def total_chars_budget() -> int:
    return _env_int("LED_RAG_HISTORY_TOTAL_CHARS", DEFAULT_TOTAL_CHARS)


def _normalise(item: Any) -> Optional[Dict[str, Any]]:
    """把 memory 里的一条消息统一成 ``{role, content, ts}``（不是对话就返回 None）。"""
    if isinstance(item, dict):
        role = str(item.get("role") or item.get("type") or "").strip().lower()
        content = str(item.get("content") or item.get("text") or "").strip()
        ts = item.get("ts") or item.get("timestamp") or item.get("received_at") or 0
    else:  # pragma: no cover - LangChain 消息对象
        role = str(getattr(item, "type", "") or getattr(item, "role", "")).strip().lower()
        content = str(getattr(item, "content", "") or "").strip()
        ts = getattr(item, "timestamp", 0)
    if role in _USER_ROLES:
        role = "user"
    elif role in _ASSISTANT_ROLES:
        role = "assistant"
    else:
        return None
    if not content:
        return None
    return {"role": role, "content": content, "ts": float(ts or 0)}


def get_dialogue_window(
    session_id: str,
    *,
    limit: Optional[int] = None,
    per_message: Optional[int] = None,
    total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """取最近 ``limit`` 条真实对话（按时间正序，带长度预算）。"""
    limit = int(limit or history_limit())
    per_message = int(per_message or per_message_chars())
    total = int(total or total_chars_budget())
    try:
        from .store import memory

        raw = memory.get_history(str(session_id or "")) or []
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("history window: load failed: %s", exc)
        return []

    items: List[Dict[str, Any]] = []
    for entry in raw:
        normalised = _normalise(entry)
        if normalised is not None:
            items.append(normalised)
    items = items[-limit:]

    # 从最新往回按预算收（保证最近的内容一定在）
    kept: List[Dict[str, Any]] = []
    budget = max(0, total)
    for item in reversed(items):
        content = item["content"]
        if len(content) > per_message:
            content = content[:per_message].rstrip() + "…"
        if budget - len(content) < 0 and kept:
            break
        budget -= len(content)
        kept.append({**item, "content": content})
    kept.reverse()
    return kept


def render_dialogue_window(items: List[Dict[str, Any]], *, max_items: Optional[int] = None) -> str:
    """把窗口渲染成给 LLM 看的对话文本。"""
    selected = list(items or [])
    if max_items:
        selected = selected[-int(max_items):]
    lines: List[str] = []
    for item in selected:
        role = "客户" if item.get("role") == "user" else "AI"
        lines.append(f"{role}: {item.get('content')}")
    return "\n".join(lines)


def dialogue_window_text(session_id: str, *, limit: Optional[int] = None, max_items: Optional[int] = None) -> str:
    """便捷入口：直接拿到可以直接塞进 prompt 的对话文本。"""
    return render_dialogue_window(
        get_dialogue_window(session_id, limit=limit), max_items=max_items
    )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_PER_MESSAGE_CHARS",
    "DEFAULT_TOTAL_CHARS",
    "dialogue_window_text",
    "get_dialogue_window",
    "history_limit",
    "per_message_chars",
    "render_dialogue_window",
    "total_chars_budget",
]
