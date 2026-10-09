"""Adapter between Sales turn state and the LCD/IFP requirement decision."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def decide_lcd_requirement(
    state: Any,
    profile: Any,
    message: str,
    *,
    lcd_domain: Callable[[Any, Any], str],
) -> Any:
    """Delegate LCD/IFP decisions to the existing LCD decision entry point."""
    if not lcd_domain(state, profile):
        return None
    try:
        from ....dialogue.lcd_decision import lcd_turn
        from ....dialogue.lcd_category_understanding import understand_lcd_category

        try:
            from ....dialogue.lcd_decision import reset_lcd_requirement_facts
            from ....rag.project_items import detect_new_item

            is_new_item, why = detect_new_item(
                str(message or ""),
                profile,
                already_recommended=bool(state.get("already_recommended")),
            )
            if is_new_item:
                cleared = reset_lcd_requirement_facts(profile)
                state["lcd_new_item"] = {"reason": why, "cleared": cleared}
                logger.info(
                    "[LCD] 客户开始说新的一块屏（%s）→ 不继承上一块的需求，"
                    "已清 %d 个字段",
                    why,
                    len(cleared),
                )
        except Exception as exc:  # pragma: no cover - optional new-item detection
            logger.warning("[LCD] new-item check failed: %s", exc)

        last_question = ""
        conversation = ""
        try:
            from ....dialogue import get_conversation_state
            from ....memory.history_window import dialogue_window_text

            session_id = str(state.get("session_id") or "")
            if session_id:
                last_question = str(
                    getattr(get_conversation_state(session_id), "last_ai_question", "") or ""
                )
                conversation = dialogue_window_text(session_id)
        except Exception:  # pragma: no cover - continue with the existing fallback
            last_question = ""

        category_signal: dict = {}
        try:
            session_id = str(state.get("session_id") or "")
            category_signal = understand_lcd_category(
                str(message or ""),
                session_id=session_id,
                conversation=conversation,
                asked_question=last_question,
                profile=profile,
            )
            if category_signal:
                logger.info(
                    "[LCD] turn understanding: category=%s confidence=%s facts=%s reason=%s",
                    category_signal.get("category"),
                    category_signal.get("confidence"),
                    category_signal.get("facts"),
                    category_signal.get("reason"),
                )
        except Exception as exc:  # pragma: no cover - preserve deterministic LCD fallback
            logger.warning("[LCD] category understanding failed: %s", exc)
            category_signal = {}

        _, action = lcd_turn(
            profile,
            str(message or ""),
            last_question=last_question,
            category_signal=category_signal,
        )
        return action
    except Exception as exc:  # pragma: no cover - preserve the existing Sales fallback
        logger.warning("[LCD] requirement decision failed: %s", exc)
        return None


def lcd_dialogue_action(action: Any, question: str) -> Dict[str, Any]:
    slot = str(getattr(action, "question_slot", "") or "")
    if bool(getattr(action, "confirmed", False)):
        return {
            "action": "recommend_only",
            "target_slot": "",
            "question": "",
            "reason": "lcd_requirement_complete",
            "priority": 0,
            "priority_label": "lcd_requirement_complete",
            "source": "lcd_chain",
        }
    if not slot or not question:
        return {}
    return {
        "action": "ask_only",
        "target_slot": slot,
        "question": question,
        "reason": "lcd_requirement_chain",
        "priority": 0,
        "priority_label": "lcd_hard_gate_missing",
        "source": "lcd_chain",
    }


def apply_lcd_requirement_result(state: Any, profile: Any, action: Any) -> Any:
    state["requirement_profile"] = profile
    state["lcd_action"] = action.to_dict()
    slot = str(action.question_slot or "")
    question = str(action.question or "") if slot else ""
    state["pending_question"] = question
    state["pending_slot"] = slot
    profile.last_asked_slot = slot
    if slot:
        profile.record_ask(slot)
    lcd_action_payload = lcd_dialogue_action(action, question)
    if lcd_action_payload:
        state["dialogue_action"] = lcd_action_payload
        logger.info(
            "[LCD] 本轮动作由 LCD 决策层决定：action=%s slot=%s",
            lcd_action_payload["action"],
            slot or "-",
        )
    state["recommendation_gate"] = {
        "ready": bool(action.confirmed),
        "gate": "lcd_requirement",
        "missing": list(action.missing_fields),
        "reason": "LCD 需求链（LCD_IFP 整改计划 Phase 5/6）",
        "next_question": question or None,
        "lcd_category": action.lcd_category,
    }
    state["should_generate_solution"] = bool(action.confirmed)
    logger.info(
        "[LCD] category=%s next=%s missing=%s question=%r",
        action.lcd_category,
        action.next_action,
        action.missing_fields,
        question,
    )
    return state


_lcd_requirement_action = decide_lcd_requirement
_lcd_dialogue_action = lcd_dialogue_action
_lcd_requirement_result = apply_lcd_requirement_result

__all__ = [
    "apply_lcd_requirement_result",
    "decide_lcd_requirement",
    "lcd_dialogue_action",
    "_lcd_dialogue_action",
    "_lcd_requirement_action",
    "_lcd_requirement_result",
]
