"""v2.7 Phase 2（§5.2）：MessageStore —— 消息/幂等状态的持久化接口。

计划 §5.2：不允许只使用内存 Dict。内存 `_seen_ids` 只能当单进程辅助机制，
必须有一个**持久化存储接口**，以后切换 SQLite / Redis / PostgreSQL 时
**业务接口不变**。

本文件提供两个实现（接口一致）：

    InMemoryMessageStore   默认（进程内，带 TTL 清理）
    SQLiteMessageStore     设置 LED_RAG_MESSAGE_STORE_PATH 时启用（写穿，重启有效）
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

KIND_MESSAGE_ID = "message_id"
KIND_FINGERPRINT = "fingerprint"


class MessageStore:
    """幂等状态存储接口（业务只依赖这四个方法）。"""

    def is_seen(self, key: str, *, kind: str = KIND_MESSAGE_ID, ttl: float = 0.0) -> bool:
        raise NotImplementedError

    def add(self, key: str, *, kind: str = KIND_MESSAGE_ID, session_id: str = "") -> bool:
        """登记一个 key；返回 True 表示这次是新登记的（之前没见过）。"""
        raise NotImplementedError

    def forget(self, key: str, *, kind: str = KIND_MESSAGE_ID) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class InMemoryMessageStore(MessageStore):
    """进程内实现（单实例够用；多实例部署请换 SQLite/Redis）。"""

    def __init__(self, *, max_entries: int = 8192):
        self._lock = threading.RLock()
        self._seen: Dict[Tuple[str, str], float] = {}
        self._max_entries = max(100, int(max_entries))

    def is_seen(self, key: str, *, kind: str = KIND_MESSAGE_ID, ttl: float = 0.0) -> bool:
        item = (str(kind), str(key))
        with self._lock:
            seen_at = self._seen.get(item)
            if seen_at is None:
                return False
            if ttl > 0 and (time.time() - seen_at) > ttl:
                self._seen.pop(item, None)
                return False
            return True

    def add(self, key: str, *, kind: str = KIND_MESSAGE_ID, session_id: str = "") -> bool:
        item = (str(kind), str(key))
        with self._lock:
            existed = item in self._seen
            self._seen[item] = time.time()
            if len(self._seen) > self._max_entries:
                for old in list(self._seen)[: len(self._seen) - self._max_entries]:
                    self._seen.pop(old, None)
            return not existed

    def forget(self, key: str, *, kind: str = KIND_MESSAGE_ID) -> None:
        with self._lock:
            self._seen.pop((str(kind), str(key)), None)

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


class SQLiteMessageStore(MessageStore):
    """写穿 SQLite（重启后仍然记得"这条消息处理过"）。"""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen ("
            "kind TEXT, key TEXT, session_id TEXT, seen_at REAL, PRIMARY KEY (kind, key))"
        )
        self._conn.commit()

    def is_seen(self, key: str, *, kind: str = KIND_MESSAGE_ID, ttl: float = 0.0) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT seen_at FROM seen WHERE kind = ? AND key = ?", (str(kind), str(key))
            ).fetchone()
        if not row:
            return False
        if ttl > 0 and (time.time() - float(row[0])) > ttl:
            self.forget(key, kind=kind)
            return False
        return True

    def add(self, key: str, *, kind: str = KIND_MESSAGE_ID, session_id: str = "") -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO seen (kind, key, session_id, seen_at) VALUES (?, ?, ?, ?)",
                (str(kind), str(key), str(session_id or ""), time.time()),
            )
            self._conn.commit()
            return bool(cursor.rowcount)

    def forget(self, key: str, *, kind: str = KIND_MESSAGE_ID) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM seen WHERE kind = ? AND key = ?", (str(kind), str(key))
            )
            self._conn.commit()

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM seen")
            self._conn.commit()


def message_store_path() -> str:
    return str(os.environ.get("LED_RAG_MESSAGE_STORE_PATH", "") or "")


def build_message_store(path: str = "") -> MessageStore:
    """按配置建一个 MessageStore（建不起来就退回内存实现）。"""
    target = str(path or message_store_path())
    if not target:
        return InMemoryMessageStore()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        return SQLiteMessageStore(target)
    except Exception as exc:  # pragma: no cover - 持久化失败不影响业务
        logger.warning("MessageStore sqlite unavailable (%s); using in-memory", exc)
        return InMemoryMessageStore()


STORE: MessageStore = build_message_store()


def get_message_store() -> MessageStore:
    return STORE


__all__ = [
    "KIND_FINGERPRINT",
    "KIND_MESSAGE_ID",
    "InMemoryMessageStore",
    "MessageStore",
    "SQLiteMessageStore",
    "STORE",
    "build_message_store",
    "get_message_store",
    "message_store_path",
]
