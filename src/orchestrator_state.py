"""Conversation history, decision audit, and per-turn telemetry for orchestration."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class TurnStateService:
    def __init__(
        self,
        *,
        memory_store: Any,
        profile_lookup: Callable[[str], Any],
    ) -> None:
        self.memory_store = memory_store
        self.profile_lookup = profile_lookup

    def note_customer_turn(
        self, session_id: str, message: str, *, turn_id: str = ""
    ) -> None:
        """Record the customer's answer and any explicitly covered requirement slots."""
        if not str(message or "").strip():
            return
        try:
            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            match = state.answer_to(message)
            state.last_answer_match = match.to_dict()
            state.note_customer_turn(
                text=message,
                answer_slot=(
                    match.slot
                    if match.kind in ("ANSWER_PREVIOUS_QUESTION", "ANSWER_WRONG_SLOT")
                    else ""
                ),
                turn_id=turn_id,
            )
            for slot in match.covered_slots:
                if slot == match.slot:
                    continue
                state.note_answered(str(slot))
            if match.kind in ("ANSWER_PREVIOUS_QUESTION", "ANSWER_WRONG_SLOT"):
                logger.info(
                    "[ConversationState] turn=%s %s answer_slot=%s expected=%s slots=%s",
                    turn_id,
                    match.kind,
                    match.slot,
                    match.expected_slot,
                    list(match.slots),
                )
        except Exception as exc:  # pragma: no cover - recording must not block the turn
            logger.warning("[ConversationState] note_customer_turn failed: %s", exc)

    def note_ai_turn(self, result: Dict[str, Any], session_id: str, final: Any) -> None:
        try:
            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            state.note_ai_turn(
                action=final.action,
                question=str(result.get("pending_question") or ""),
                slot=final.question_slot or "",
                response=final.text,
                turn_id=final.turn_id,
                speech_act=str((result.get("speech_act") or {}).get("speech_act") or ""),
            )
            result["conversation_state"] = state.to_dict()
        except Exception as exc:  # pragma: no cover - recording must not block the turn
            logger.warning("[ConversationState] note_ai_turn failed: %s", exc)
        self._emit_decision_audit(result, session_id, final)

    def replace_placeholder_history(
        self, result: Dict[str, Any], session_id: str, text: str
    ) -> None:
        """Replace the Sales placeholder with the Solution response in conversation history."""
        if not str(result.get("agent") or "").startswith("solution"):
            return
        if not text or not self.memory_store:
            return
        replace = getattr(self.memory_store, "replace_last_assistant", None)
        if not callable(replace):  # pragma: no cover - another store may omit this method
            return
        try:
            if replace(session_id, text):
                logger.info(
                    "[History] 用最终答复替换占位符（session=%s，%d 字）",
                    session_id,
                    len(text),
                )
        except Exception as exc:  # pragma: no cover - history failure must not block the turn
            logger.warning("[History] 替换占位符失败：%s", exc)

    def _emit_decision_audit(
        self, result: Dict[str, Any], session_id: str, final: Any
    ) -> None:
        """Record why this turn chose its action and question without affecting behavior."""
        try:
            import json

            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            payload = {
                "turn_id": final.turn_id,
                "session_id": session_id,
                "input": str(result.get("customer_input") or "")[:200],
                "speech_act": state.current_speech_act,
                "requirement_state_before": self._requirement_summary(session_id),
                "understanding": result.get("understanding") or {},
                "conversation_state_before": (
                    result.get("conversation_state_before") or state.to_dict()
                ),
                "conversation_state_after": state.to_dict(),
                "candidate_actions": list(result.get("action_candidates") or []),
                "selected_action": final.action,
                "discarded_actions": list(result.get("discarded_actions") or []),
                "final_response": final.text[:400],
                "question_count": final.question_count,
                "question_slot": final.question_slot,
                "validation_result": final.validation_result,
            }
            logger.info("[DecisionAudit] %s", json.dumps(payload, ensure_ascii=False))
            result["decision_audit"] = payload
        except Exception as exc:  # pragma: no cover - audit failure must not block the turn
            logger.warning("[DecisionAudit] emit failed: %s", exc)

    def _requirement_summary(self, session_id: str) -> Dict[str, Any]:
        try:
            profile = self.profile_lookup(session_id)
            if profile is None:
                return {}
            facts = profile.to_facts() if hasattr(profile, "to_facts") else {}
            return {str(key): value for key, value in dict(facts or {}).items()}
        except Exception:  # pragma: no cover - audit failure must not block the turn
            return {}

    def finish_llm_turn(self, result: Dict[str, Any], session_id: str) -> None:
        """End telemetry accounting and attach the per-turn metrics to the result."""
        try:
            import time as _time

            from .observability.llm_tracker import get_llm_tracker

            tracker = get_llm_tracker()
            context = tracker.current_turn()
            stats = tracker.end_turn()
            if stats is None:
                return
            total_ms = ((_time.time() - context.started_at) * 1000) if context else 0.0
            action = str(result.get("action") or "")
            question_slot = str(result.get("question_slot") or "")
            response_count = int(result.get("response_count") or 1)
            question_count = int(result.get("question_count") or 0)
            final_response = result.get("final_response") or {}
            speech_act = ""
            try:
                from .dialogue import get_conversation_state

                speech_act = str(get_conversation_state(session_id).current_speech_act or "")
            except Exception:  # pragma: no cover - telemetry must not block the turn
                speech_act = ""
            logger.info(
                "[Turn] session=%s messages=%s aggregated=%s llm_calls=%s "
                "llm_latency_ms=%s llm_tokens=%s total_ms=%s "
                "turn_id=%s action=%s speech_act=%s question_slot=%s "
                "response_count=%s question_count=%s validation=%s",
                session_id,
                context.message_count if context else 1,
                context.aggregated if context else False,
                stats.calls,
                round(stats.latency_ms, 1),
                stats.total_tokens,
                round(total_ms, 1),
                result.get("turn_id") or (context.turn_id if context else "-"),
                action or "-",
                speech_act or "-",
                question_slot or "-",
                response_count,
                question_count,
                str(final_response.get("validation_result") or "-"),
            )
            result["_llm_stats"] = stats.to_dict()
            perf_summary = result.get("_perf")
            if isinstance(perf_summary, dict):
                perf_summary.update(
                    {
                        "turn_id": result.get("turn_id") or (context.turn_id if context else ""),
                        "action": action,
                        "speech_act": speech_act,
                        "question_slot": question_slot,
                        "response_count": response_count,
                        "question_count": question_count,
                    }
                )
            if context is not None:
                result["_turn"] = {
                    "message_count": context.message_count,
                    "aggregated": context.aggregated,
                    "turn_id": context.turn_id,
                    "action": action,
                    "speech_act": speech_act,
                    "question_slot": question_slot,
                    "response_count": response_count,
                    "question_count": question_count,
                    "llm_calls": stats.calls,
                    "llm_latency_ms": round(stats.latency_ms, 1),
                    "total_latency_ms": round(total_ms, 1),
                }
        except Exception as exc:  # pragma: no cover - telemetry failure must not block the turn
            logger.warning("LLM tracker end_turn failed: %s", exc)
