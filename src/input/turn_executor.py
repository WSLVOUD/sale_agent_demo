"""v2.7 Phase 4/5/6：TurnExecutor —— 会话级并发控制 + Turn 生命周期 + 幂等。

计划 §3 第一原则：

    一个 turn_id = 一次业务决策 = 一个 DialogueAction
                = 一个 FinalResponse = 一次 Commit

计划 §6：**同一个 session 同一时间只能有一个 Processing Turn**；
锁必须位于 Agent 之外 —— 所以放在这一层，而不是 MessageAggregator 的
一个 processing bool（计划 §27 明确否定了那种做法）。

计划 §6/§7.1 的根本方案：

    Message ID + Turn ID + Message Deduplication + Turn Sealing
    + Session Lock + Turn State Machine + Idempotent Commit

于是：连续消息聚合成一个 Turn（Case 1/2）；同消息/同 Turn 重试不重跑（Case 3/4）；
并发请求被会话锁串行化（Case 5）；n8n/webhook 重放只有一个业务结果（Case 6）。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .message_deduplicator import MessageDeduplicator, get_message_deduplicator
from .message import CustomerMessage
from .message_inbox import (
    ASSIGNED_TO_TURN,
    BUFFERED,
    COMMITTED,
    DEDUPLICATED,
    FAILED,
    MessageInbox,
    get_message_inbox,
    new_message_id,
)
from .turn_manager import TurnBuilder, TurnManager
from .turn_store import (
    DECIDED,
    PROCESSING,
    RESPONDED,
    SEALED,
    TERMINAL_TURN_STATUSES,
    TurnRecord,
    TurnStore,
    get_turn_store,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_SECONDS = 120.0


@dataclass
class TurnOutcome:
    """一次 Turn 的最终结果（调用方只看这个）。"""

    turn_id: str = ""
    session_id: str = ""
    status: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    action: str = ""
    question_slot: str = ""
    response_count: int = 0
    message_ids: List[str] = field(default_factory=list)
    aggregated: bool = False
    duplicate: bool = False
    joined: bool = False
    waited_ms: float = 0.0
    error: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and self.status == COMMITTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "status": self.status,
            "response": self.response,
            "action": self.action,
            "question_slot": self.question_slot,
            "response_count": self.response_count,
            "message_ids": list(self.message_ids),
            "aggregated": self.aggregated,
            "duplicate": self.duplicate,
            "joined": self.joined,
            "waited_ms": self.waited_ms,
            "error": self.error,
        }


class TurnExecutor:
    """Turn 的执行单位（线程安全，多入口可同时调用）。"""

    def __init__(
        self,
        runner: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        store: Optional[TurnStore] = None,
        builder: Optional[TurnManager] = None,
        inbox: Optional[MessageInbox] = None,
        deduplicator: Optional[MessageDeduplicator] = None,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        state_snapshot: Optional[Callable[[str], Any]] = None,
        state_restore: Optional[Callable[[str, Any], None]] = None,
    ):
        self.runner = runner
        self.store = store or get_turn_store()
        self.builder = builder or TurnManager(self.store)
        self.manager = self.builder  # §6 的正式叫法
        self.inbox = inbox or get_message_inbox()
        self.dedup = deduplicator or get_message_deduplicator()
        self.wait_seconds = max(1.0, float(wait_seconds))
        # 生成期间客户又发消息 → 这一版回复作废并重跑；重跑前必须把
        # "这一版留下的状态"（问了哪一项 / 问过几次 / 已答）退回去，
        # 否则重跑会以为那一项已经问过，转而去问另一个问题（= 连续两问的根因）。
        self._state_snapshot = state_snapshot
        self._state_restore = state_restore
        self._lock = threading.RLock()
        self._events: Dict[str, threading.Event] = {}
        self._outcomes: Dict[str, TurnOutcome] = {}
        self._session_locks: Dict[str, threading.Lock] = {}
        self._fingerprint_turns: Dict[str, str] = {}
        self._turn_fingerprints: Dict[str, str] = {}
        # §9.2 的补充（实测）：客户在"上一轮还在生成"时又发了消息 → 并入同一轮，
        # 否则客户会连着收到两条回复（2026-09-21 真实日志的"连续询问"现象）。
        self._running: Dict[str, str] = {}
        self._follow_ups: Dict[str, List[CustomerMessage]] = {}

    # ── 主入口 ──────────────────────────────────────────────────────────
    def submit(
        self,
        *,
        session_id: str,
        text: str = "",
        images: Optional[List[Any]] = None,
        message_ids: Optional[List[str]] = None,
        messages: Optional[List[CustomerMessage]] = None,
        source: str = "api",
        turn_id: str = "",
    ) -> TurnOutcome:
        """所有入口的**唯一**入口：发消息 → 拿到本轮唯一结果。"""
        started = time.time()
        session = str(session_id or "")
        # §4.3：一条消息 = 一个对象；老调用（散装 text/message_ids）也兼容
        items: List[CustomerMessage] = list(messages or [])
        if not items:
            ids = [str(item).strip() for item in (message_ids or []) if str(item).strip()]
            if not ids:
                ids = [new_message_id()]
            items = [
                CustomerMessage(
                    message_id=ids[0],
                    session_id=session,
                    text=str(text or ""),
                    images=list(images or []),
                    source=str(source or "api"),
                )
            ] + [
                CustomerMessage(message_id=extra, session_id=session, source=str(source or "api"))
                for extra in ids[1:]
            ]
        for message in items:
            if not message.session_id:
                message.session_id = session
        ids = [message.message_id for message in items]
        text = text or "\n".join(message.text for message in items if message.text)
        images = list(images or [])
        for message in items:
            for image in message.images or []:
                if image not in images:
                    images.append(image)

        # Phase 6：Turn 幂等（同一个 turn_id 重放 → 不跑 Agent）
        if turn_id:
            existing = self.store.get(turn_id)
            if existing is not None and existing.status == COMMITTED:
                logger.info("[TurnExecutor] turn=%s 已提交 → 直接返回（不跑 Agent）", turn_id)
                return self._outcome_from_record(existing, duplicate=True, started=started)

        # Phase 1：消息级去重（失败过的可以重试）
        for message_id in ids:
            owner = self.store.owner_of_message(message_id)
            if owner is not None and owner.status == FAILED:
                owner = None
            if owner is not None:
                return self._repeat_outcome(owner, session, message_id, started)
        decision = self.dedup.check(session, message_id=ids[0], text=text, images=images)
        if decision.is_duplicate:
            owner_id = self._fingerprint_turns.get(decision.fingerprint, "")
            owner = self.store.get(owner_id) if owner_id else None
            if owner is not None and owner.status == FAILED:
                owner = None
            if owner is not None:
                return self._repeat_outcome(
                    owner, session, ids[0], started, reason=decision.reason
                )

        # Phase 1/2：登记消息 + 放进当前 OPEN Turn
        for index, message_id in enumerate(ids):
            self.inbox.receive(
                session_id=session,
                text=text if index == 0 else "",
                images=images if index == 0 else [],
                message_id=message_id,
                source=source,
                status=DEDUPLICATED,
            )
        self.dedup.remember(
            session,
            message_id=ids[0],
            text=text,
            images=images,
            fingerprint=decision.fingerprint,
        )
        for extra in items[1:]:
            self.dedup.remember(
                session,
                message_id=extra.message_id,
                text=extra.text,
                images=list(extra.images or []),
            )

        # ── 上一轮还没回复完 → 并入那一轮（客户只应该收到一条回复）──────────
        running_id = self._running_turn(session)
        if running_id:
            self._add_follow_ups(running_id, items)
            for message in items:
                self.inbox.mark(message.message_id, BUFFERED, turn_id=running_id)
            logger.info(
                "[TurnExecutor] session=%s 上一轮还没生成完 → %d 条消息并入 %s（只回一条）",
                session, len(items), running_id,
            )
            joined = self._wait_for_turn(running_id, started=started, joined=True)
            if joined is not None:
                return joined

        turn, created = self.builder.push_message(items[0], turn_id=turn_id)
        if decision.fingerprint:
            self._fingerprint_turns[decision.fingerprint] = turn.turn_id
            self._turn_fingerprints[turn.turn_id] = decision.fingerprint
        for message_id in ids:
            self.inbox.mark(message_id, BUFFERED, turn_id=turn.turn_id)
        for message_id in ids[1:]:
            turn.message_ids.append(message_id)
            self.inbox.mark(message_id, BUFFERED, turn_id=turn.turn_id)
        for message in items[1:]:
            # §8：同一轮里后到的消息进同一个 Turn（对象化，不丢对应关系）
            if self.builder.current(session) is not None:
                self.builder.push_message(message)
        if len(turn.message_ids) > 1:
            self.store.update(turn.turn_id, message_ids=list(turn.message_ids))

        if not created:
            outcome = self._wait_for_turn(turn.turn_id, started=started, joined=True)
            if outcome is not None:
                return outcome
            logger.warning(
                "[TurnExecutor] turn=%s 等待超时 → 自己执行（保证有回复）", turn.turn_id
            )

        return self._execute(turn, started=started)

    # ── 执行 ────────────────────────────────────────────────────────────
    def _execute(self, turn: TurnRecord, *, started: float) -> TurnOutcome:
        session = turn.session_id
        self._event(turn.turn_id)
        try:
            # Phase 2：等这一轮"说完了"（grace / max window）
            buffer = self.builder.buffer_of_turn(turn.turn_id)
            # 注意：这里要拿**引用**而不是拷贝 —— grace 窗口内新到的消息会继续
            # 追加到同一个列表，封口后组装 payload 时必须包含它们（Case 1/2）。
            collected: List[CustomerMessage] = buffer.messages if buffer else []
            if buffer is not None:
                turn = self.builder.wait_until_sealed(buffer)
            # 标记"这一轮正在生成"必须在**封口之后**：grace 窗口内的消息应该照常
            # 聚合进本轮（§8），只有封口后、生成期间补发的消息才走"合并重跑"。
            self._mark_running(session, turn.turn_id)
            for _attempt in range(5):
                snapshot = self._snapshot_state(session)
                payload = self.builder.payload(turn, collected)
                # 计划 §19：把"上一轮问的是哪一项"带进这一轮（重复提问闸门要用）
                payload["previous_question"] = self.store.previous_question(session)
                for message_id in payload["message_ids"]:
                    self.inbox.mark(message_id, ASSIGNED_TO_TURN, turn_id=turn.turn_id)

                # Phase 4：会话锁 —— 同 session 不允许并发处理
                lock = self._session_lock(session)
                if not lock.acquire(timeout=self.wait_seconds):  # pragma: no cover
                    raise TimeoutError(f"session lock timeout: {session}")
                try:
                    self.store.mark(turn.turn_id, PROCESSING)
                    result = self.runner(payload) or {}
                    # §9 Turn 生命周期：PROCESSING → DECIDED → RESPONDED → COMMITTED
                    self.store.mark(turn.turn_id, DECIDED)
                    self.store.mark(turn.turn_id, RESPONDED)
                finally:
                    lock.release()

                # 生成期间客户又发了消息 → 丢掉这一版回复，带着新消息重跑
                follow_ups = self._take_follow_ups(turn.turn_id)
                if not follow_ups:
                    break
                self._restore_state(session, snapshot)
                collected.extend(follow_ups)
                self._merge_follow_ups(turn, follow_ups)
                logger.info(
                    "[TurnExecutor] turn=%s 生成期间又收到 %d 条消息 → 合并重跑（只回一条）",
                    turn.turn_id, len(follow_ups),
                )

            outcome = self._outcome_from_result(turn, result, started=started, payload=payload)
            self._commit(turn, outcome)
            return outcome
        except Exception as exc:
            logger.warning("[TurnExecutor] turn=%s 执行失败：%s", turn.turn_id, exc)
            self.store.fail(turn.turn_id, str(exc))
            for message_id in list(turn.message_ids):
                self.inbox.mark(message_id, FAILED, turn_id=turn.turn_id)
                self.dedup.forget(message_id=message_id)
            fingerprint = self._turn_fingerprints.pop(turn.turn_id, "")
            if fingerprint:
                self.dedup.forget(fingerprint=fingerprint)
                self._fingerprint_turns.pop(fingerprint, None)
            outcome = TurnOutcome(
                turn_id=turn.turn_id,
                session_id=session,
                status=FAILED,
                error=str(exc),
                message_ids=list(turn.message_ids),
                waited_ms=round((time.time() - started) * 1000, 1),
            )
            self._finish(turn.turn_id, outcome)
            return outcome
        finally:
            self._clear_running(session, turn.turn_id)
            self._drop_follow_ups(turn.turn_id)

    def _commit(self, turn: TurnRecord, outcome: TurnOutcome) -> None:
        record = self.store.commit(
            turn.turn_id,
            response=outcome.response,
            action=outcome.action,
            question_slot=outcome.question_slot,
            response_count=outcome.response_count,
            trace=outcome.trace,
        )
        outcome.status = record.status if record else COMMITTED
        for message_id in outcome.message_ids:
            self.inbox.mark(message_id, COMMITTED, turn_id=turn.turn_id)
        self._log_turn_trace(outcome)
        self._finish(turn.turn_id, outcome)

    def _finish(self, turn_id: str, outcome: TurnOutcome) -> None:
        with self._lock:
            self._outcomes[turn_id] = outcome
            event = self._events.get(turn_id)
        if event is not None:
            event.set()

    # ── 结果装配 ────────────────────────────────────────────────────────
    def _outcome_from_result(
        self,
        turn: TurnRecord,
        result: Dict[str, Any],
        *,
        started: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TurnOutcome:
        final_response = result.get("final_response") or {}
        response = str(result.get("response") or final_response.get("text") or "")
        trace = self._build_trace(
            turn, result, response=response, started=started, payload=payload or {}
        )
        return TurnOutcome(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            status=COMMITTED,
            result=dict(result or {}),
            response=response,
            action=str(result.get("action") or ""),
            question_slot=str(result.get("question_slot") or ""),
            response_count=int(result.get("response_count") or (1 if response else 0)),
            message_ids=list(turn.message_ids),
            aggregated=len(turn.message_ids) > 1,
            waited_ms=round((time.time() - started) * 1000, 1),
            trace=trace,
        )

    def _outcome_from_record(
        self, record: TurnRecord, *, duplicate: bool, started: float
    ) -> TurnOutcome:
        return TurnOutcome(
            turn_id=record.turn_id,
            session_id=record.session_id,
            status=record.status,
            result={},
            response=str(record.final_response or ""),
            action=str(record.action or ""),
            question_slot=str(record.question_slot or ""),
            response_count=int(record.response_count or 0),
            message_ids=list(record.message_ids),
            aggregated=len(record.message_ids) > 1,
            duplicate=duplicate,
            waited_ms=round((time.time() - started) * 1000, 1),
            trace=dict(record.trace or {}),
        )

    def _repeat_outcome(
        self,
        owner: TurnRecord,
        session_id: str,
        message_id: str,
        started: float,
        *,
        reason: str = "duplicate_message_id",
    ) -> TurnOutcome:
        logger.info(
            "[TurnExecutor] session=%s message=%s 重复（%s）→ 复用 turn=%s",
            session_id, message_id, reason, owner.turn_id,
        )
        if owner.status == COMMITTED:
            return self._outcome_from_record(owner, duplicate=True, started=started)
        waited = self._wait_for_turn(owner.turn_id, started=started, joined=True)
        if waited is not None:
            return waited
        return self._outcome_from_record(owner, duplicate=True, started=started)

    def _build_trace(
        self,
        turn: TurnRecord,
        result: Dict[str, Any],
        *,
        response: str,
        started: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """计划 §22：Turn Trace（能回答"为什么又问了一次"）。"""
        final_response = result.get("final_response") or {}
        previous = payload.get("previous_question") or self.store.previous_question(
            turn.session_id
        )
        turn_stats = result.get("_turn") or {}
        coverage = result.get("answer_coverage") or {}
        return {
            # §43：TurnTrace 的字段（能回答"为什么问了这个问题 / 为什么又问一次"）
            "session_id": turn.session_id,
            "turn_id": turn.turn_id,
            "message_ids": list(turn.message_ids),
            "message_count": max(1, len(turn.message_ids)),
            "input_message_count": max(1, len(turn.message_ids)),
            "turn_status": COMMITTED,
            "previous_state": dict(result.get("conversation_state_before") or {}),
            "new_state": dict(result.get("conversation_state") or {}),
            "extracted_slots": list((result.get("requirements") or {}).keys())[:12],
            "answered_slots": list(coverage.get("answered_slots") or []),
            "newly_filled_slots": list(coverage.get("newly_filled_slots") or []),
            "missing_slots": list(result.get("missing_slots") or [])[:12],
            "previous_question": {
                "turn_id": str(previous.get("turn_id") or ""),
                "question_slot": str(previous.get("question_slot") or ""),
            },
            "current_question": {"question_slot": str(result.get("question_slot") or "")},
            "selected_action": str(result.get("action") or ""),
            "response_mode": str(result.get("response_mode") or "NATURAL"),
            "response_count": int(result.get("response_count") or (1 if response else 0)),
            "duplicate_check": str(result.get("duplicate_check") or "pass"),
            "commit_status": COMMITTED,
            "total_latency_ms": round((time.time() - started) * 1000, 1),
            "latency_ms": round((time.time() - started) * 1000, 1),
            "llm_call_count": int(turn_stats.get("llm_calls") or 0),
            "speech_act": str(turn_stats.get("speech_act") or ""),
            "response_density": str(result.get("response_density") or ""),
            "validation": str(final_response.get("validation_result") or ""),
        }

    def _log_turn_trace(self, outcome: TurnOutcome) -> None:
        """计划 §22：每个 Turn 一条完整链路日志。"""
        try:
            import json

            logger.info("[TurnTrace] %s", json.dumps(outcome.trace, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover - 日志失败不影响业务
            logger.warning("[TurnTrace] emit failed: %s", exc)

    # ── 并发原语 ────────────────────────────────────────────────────────
    def _snapshot_state(self, session_id: str) -> Any:
        if self._state_snapshot is None:
            return None
        try:
            return self._state_snapshot(session_id)
        except Exception as exc:  # pragma: no cover - 快照失败不影响业务
            logger.warning("[TurnExecutor] state snapshot failed: %s", exc)
            return None

    def _restore_state(self, session_id: str, snapshot: Any) -> None:
        if snapshot is None or self._state_restore is None:
            return
        try:
            self._state_restore(session_id, snapshot)
            logger.info(
                "[TurnExecutor] session=%s 作废的那一版状态已回滚（避免「已经问过」污染）",
                session_id,
            )
        except Exception as exc:  # pragma: no cover - 回滚失败不影响业务
            logger.warning("[TurnExecutor] state restore failed: %s", exc)

    def _mark_running(self, session_id: str, turn_id: str) -> None:
        with self._lock:
            self._running[str(session_id or "")] = str(turn_id)

    def _clear_running(self, session_id: str, turn_id: str) -> None:
        with self._lock:
            current = self._running.get(str(session_id or ""))
            if current == str(turn_id):
                self._running.pop(str(session_id or ""), None)

    def _running_turn(self, session_id: str) -> str:
        """这一轮还没提交的 turn（客户此时补发的消息要并进来）。"""
        key = str(session_id or "")
        with self._lock:
            turn_id = self._running.get(key, "")
        if not turn_id:
            return ""
        record = self.store.get(turn_id)
        if record is None or record.status in TERMINAL_TURN_STATUSES:
            with self._lock:
                self._running.pop(key, None)
            return ""
        return turn_id

    def _add_follow_ups(self, turn_id: str, messages: List[CustomerMessage]) -> None:
        with self._lock:
            self._follow_ups.setdefault(str(turn_id), []).extend(list(messages or []))

    def _take_follow_ups(self, turn_id: str) -> List[CustomerMessage]:
        with self._lock:
            return list(self._follow_ups.pop(str(turn_id), []) or [])

    def _drop_follow_ups(self, turn_id: str) -> None:
        with self._lock:
            self._follow_ups.pop(str(turn_id), None)

    def _merge_follow_ups(
        self, turn: TurnRecord, follow_ups: List[CustomerMessage]
    ) -> None:
        """把生成期间补发的消息并进这一轮（消息 id / 文本 / 图片都要留痕）。"""
        for message in follow_ups:
            if message.message_id and message.message_id not in turn.message_ids:
                turn.message_ids.append(message.message_id)
            for image in message.images or []:
                if image not in turn.images:
                    turn.images.append(image)
            self.inbox.mark(message.message_id, ASSIGNED_TO_TURN, turn_id=turn.turn_id)
        self.store.update(
            turn.turn_id,
            message_ids=list(turn.message_ids),
            images=list(turn.images),
            text=(f"{turn.text}\n" + "\n".join(m.text for m in follow_ups)).strip(),
        )

    def _event(self, turn_id: str) -> threading.Event:
        with self._lock:
            event = self._events.get(turn_id)
            if event is None:
                event = threading.Event()
                self._events[turn_id] = event
            return event

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _wait_for_turn(
        self, turn_id: str, *, started: float, joined: bool
    ) -> Optional[TurnOutcome]:
        event = self._event(turn_id)
        finished = event.wait(timeout=self.wait_seconds)
        with self._lock:
            outcome = self._outcomes.get(turn_id)
        if not finished or outcome is None:
            return None
        return TurnOutcome(
            **{
                **outcome.__dict__,
                "joined": joined,
                "waited_ms": round((time.time() - started) * 1000, 1),
            }
        )

    # ── 调试/测试辅助 ───────────────────────────────────────────────────
    def reset(self, session_id: str = "") -> None:
        with self._lock:
            if session_id:
                self._session_locks.pop(str(session_id), None)
            else:
                self._events.clear()
                self._outcomes.clear()
                self._session_locks.clear()
                self._fingerprint_turns.clear()
                self._turn_fingerprints.clear()
        self.builder.release(session_id)


__all__ = ["DEFAULT_WAIT_SECONDS", "TurnExecutor", "TurnOutcome"]
