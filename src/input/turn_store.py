"""v2.7 Phase 3：Turn Store —— Turn 的持久化与幂等。

计划 §7：保存 turn_id / session_id / message_ids / status / action /
question_slot / final_response / response_count / created_at / sealed_at /
committed_at；状态 OPEN → SEALED → PROCESSING → DECIDED → RESPONDED →
COMMITTED（或 FAILED）。

计划 §7.1：``turn_id`` 已是 COMMITTED → **不再运行 Agent**，直接返回上次的
``final_response``。这是 HTTP retry / n8n retry / Webhook retry / 前端重复请求
的根本方案。

存储默认进程内；设置 ``LED_RAG_TURN_STORE_PATH``（.sqlite）时写穿持久化，
重启后仍能挡住重放。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OPEN = "OPEN"
SEALED = "SEALED"
PROCESSING = "PROCESSING"
DECIDED = "DECIDED"
RESPONDED = "RESPONDED"
COMMITTED = "COMMITTED"
FAILED = "FAILED"

ALL_TURN_STATUSES = (OPEN, SEALED, PROCESSING, DECIDED, RESPONDED, COMMITTED, FAILED)
TERMINAL_TURN_STATUSES = frozenset({COMMITTED, FAILED})


def new_turn_id() -> str:
    return f"turn_{uuid.uuid4().hex[:12]}"


@dataclass
class TurnRecord:
    """一个客户 Turn（系统真正的执行单位）。"""

    turn_id: str = field(default_factory=new_turn_id)
    session_id: str = ""
    message_ids: List[str] = field(default_factory=list)
    status: str = OPEN
    text: str = ""
    images: List[Any] = field(default_factory=list)
    source: str = "api"
    action: str = ""
    question_slot: str = ""
    final_response: str = ""
    response_count: int = 0
    created_at: float = field(default_factory=time.time)
    sealed_at: Optional[float] = None
    processed_at: Optional[float] = None
    decided_at: Optional[float] = None
    responded_at: Optional[float] = None
    committed_at: Optional[float] = None
    duplicate_of: str = ""
    error: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_committed(self) -> bool:
        return self.status == COMMITTED

    @property
    def response(self) -> str:
        """计划 §10 的字段名（与 final_response 同义）。"""
        return self.final_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "message_ids": list(self.message_ids),
            "status": self.status,
            "action": self.action,
            "question_slot": self.question_slot,
            "final_response": self.final_response,
            "response": self.final_response,
            "response_count": self.response_count,
            "created_at": self.created_at,
            "sealed_at": self.sealed_at,
            "processed_at": self.processed_at,
            "decided_at": self.decided_at,
            "responded_at": self.responded_at,
            "committed_at": self.committed_at,
            "duplicate_of": self.duplicate_of,
            "error": self.error,
        }


class TurnStore:
    """Turn 登记簿（线程安全 + 可选 sqlite 写穿）。"""

    def __init__(self, path: str = "", *, max_turns: int = 4000):
        self._lock = threading.RLock()
        self._turns: Dict[str, TurnRecord] = {}
        self._by_message: Dict[str, str] = {}
        self._session_order: Dict[str, List[str]] = {}
        self._max_turns = max(100, int(max_turns))
        self._path = str(path or "")
        self._conn: Optional[sqlite3.Connection] = None
        if self._path:
            self._init_sqlite()

    # ── 生命周期 ────────────────────────────────────────────────────────
    def create(
        self,
        session_id: str,
        *,
        message_ids: Optional[List[str]] = None,
        text: str = "",
        images: Optional[List[Any]] = None,
        source: str = "api",
        turn_id: str = "",
    ) -> TurnRecord:
        record = TurnRecord(
            turn_id=str(turn_id or "").strip() or new_turn_id(),
            session_id=str(session_id or ""),
            message_ids=list(message_ids or []),
            text=str(text or ""),
            images=list(images or []),
            source=str(source or "api"),
        )
        with self._lock:
            self._turns[record.turn_id] = record
            self._session_order.setdefault(record.session_id, []).append(record.turn_id)
            for message_id in record.message_ids:
                self._by_message[str(message_id)] = record.turn_id
            self._prune()
        self._persist(record)
        return record

    def update(self, turn_id: str, **fields: Any) -> Optional[TurnRecord]:
        with self._lock:
            record = self._turns.get(str(turn_id or ""))
            if record is None:
                return None
            for key, value in fields.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            if fields.get("message_ids"):
                for message_id in record.message_ids:
                    self._by_message[str(message_id)] = record.turn_id
        self._persist(record)
        return record

    def mark(self, turn_id: str, status: str) -> Optional[TurnRecord]:
        fields: Dict[str, Any] = {"status": str(status)}
        stamp = time.time()
        if status == SEALED:
            fields["sealed_at"] = time.time()
        elif status == PROCESSING:
            fields["processed_at"] = stamp
        elif status == DECIDED:
            fields["decided_at"] = stamp
        elif status == RESPONDED:
            fields["responded_at"] = stamp
        return self.update(turn_id, **fields)

    def commit(
        self,
        turn_id: str,
        *,
        response: str = "",
        action: str = "",
        question_slot: str = "",
        response_count: int = 1,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Optional[TurnRecord]:
        """唯一的提交入口（幂等：已提交的 turn 不会被覆盖成别结果）。"""
        with self._lock:
            record = self._turns.get(str(turn_id or ""))
            if record is None:
                return None
            if record.status == COMMITTED and record.final_response:
                return record
            record.status = COMMITTED
            record.final_response = str(response or "")
            record.action = str(action or "")
            record.question_slot = str(question_slot or "")
            record.response_count = int(response_count or 0)
            record.committed_at = time.time()
            if trace:
                record.trace = dict(trace)
        self._persist(record)
        return record

    def fail(self, turn_id: str, error: str = "") -> Optional[TurnRecord]:
        return self.update(turn_id, status=FAILED, error=str(error or "")[:400])

    # ── 查询（幂等的依据）───────────────────────────────────────────────
    def get(self, turn_id: str) -> Optional[TurnRecord]:
        key = str(turn_id or "")
        with self._lock:
            record = self._turns.get(key)
        if record is not None:
            return record
        return self._load(key)

    def owner_of_message(self, message_id: str) -> Optional[TurnRecord]:
        key = str(message_id or "")
        with self._lock:
            turn_id = self._by_message.get(key)
            return self._turns.get(turn_id) if turn_id else None

    def latest_committed(self, session_id: str) -> Optional[TurnRecord]:
        with self._lock:
            for turn_id in reversed(self._session_order.get(str(session_id or ""), [])):
                record = self._turns.get(turn_id)
                if record is not None and record.status == COMMITTED:
                    return record
        return None

    def previous_question(self, session_id: str) -> Dict[str, str]:
        """上一轮的 ``(turn_id, question_slot)``（给 Duplicate Question Firewall 用）。"""
        record = self.latest_committed(session_id)
        if record is None:
            return {"turn_id": "", "question_slot": "", "question": ""}
        return {
            "turn_id": record.turn_id,
            "question_slot": str(record.question_slot or ""),
            "question": str(record.final_response or ""),
        }

    def reset(self, session_id: str = "") -> None:
        with self._lock:
            if not session_id:
                self._turns.clear()
                self._by_message.clear()
                self._session_order.clear()
                return
            for turn_id in self._session_order.pop(str(session_id), []):
                record = self._turns.pop(turn_id, None)
                if record:
                    for message_id in record.message_ids:
                        self._by_message.pop(str(message_id), None)

    # ── sqlite 写穿（可选）─────────────────────────────────────────────
    def _init_sqlite(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "turn_id TEXT PRIMARY KEY, session_id TEXT, status TEXT, payload TEXT)"
            )
            self._conn.commit()
        except Exception as exc:  # pragma: no cover - 持久化失败不影响业务
            logger.warning("TurnStore sqlite unavailable (%s); running in-memory", exc)
            self._conn = None

    def _persist(self, record: TurnRecord) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO turns (turn_id, session_id, status, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(turn_id) DO UPDATE SET status=excluded.status, payload=excluded.payload",
                (
                    record.turn_id,
                    record.session_id,
                    record.status,
                    json.dumps(record.to_dict(), ensure_ascii=False),
                ),
            )
            self._conn.commit()
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("TurnStore persist failed: %s", exc)

    def _load(self, turn_id: str) -> Optional[TurnRecord]:
        if self._conn is None or not turn_id:
            return None
        try:
            row = self._conn.execute(
                "SELECT payload FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        except Exception:  # pragma: no cover - 防御式
            return None
        if not row:
            return None
        try:
            payload = json.loads(row[0])
        except Exception:  # pragma: no cover - 防御式
            return None
        record = TurnRecord(
            turn_id=str(payload.get("turn_id") or turn_id),
            session_id=str(payload.get("session_id") or ""),
            message_ids=list(payload.get("message_ids") or []),
            status=str(payload.get("status") or ""),
            action=str(payload.get("action") or ""),
            question_slot=str(payload.get("question_slot") or ""),
            final_response=str(payload.get("final_response") or ""),
            response_count=int(payload.get("response_count") or 0),
            created_at=float(payload.get("created_at") or time.time()),
            duplicate_of=str(payload.get("duplicate_of") or ""),
        )
        with self._lock:
            self._turns.setdefault(record.turn_id, record)
        return record

    def _prune(self) -> None:
        while len(self._turns) > self._max_turns:
            oldest_turn = None
            oldest_time = None
            for turn_id, record in self._turns.items():
                if record.status not in TERMINAL_TURN_STATUSES:
                    continue
                if oldest_time is None or record.created_at < oldest_time:
                    oldest_turn, oldest_time = turn_id, record.created_at
            if oldest_turn is None:
                return
            record = self._turns.pop(oldest_turn, None)
            if record:
                for message_id in record.message_ids:
                    self._by_message.pop(str(message_id), None)


def turn_store_path() -> str:
    return str(os.environ.get("LED_RAG_TURN_STORE_PATH", "") or "")


STORE = TurnStore(turn_store_path())


def get_turn_store() -> TurnStore:
    return STORE


__all__ = [
    "ALL_TURN_STATUSES",
    "COMMITTED",
    "DECIDED",
    "FAILED",
    "OPEN",
    "PROCESSING",
    "RESPONDED",
    "SEALED",
    "STORE",
    "TERMINAL_TURN_STATUSES",
    "TurnRecord",
    "TurnStore",
    "get_turn_store",
    "new_turn_id",
    "turn_store_path",
]
