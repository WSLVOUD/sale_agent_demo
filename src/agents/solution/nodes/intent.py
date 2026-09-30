"""Current-turn intent recognition and conversation routing helpers."""
import logging
import re
from typing import Any, Dict

from ..state import SolutionState
from .requirement import detect_intent

logger = logging.getLogger(__name__)


def _last_assistant_message(state: SolutionState) -> str:
    for message in reversed(state.get("messages", [])):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role == "assistant":
            return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    return ""


def _last_user_message(state: SolutionState) -> str:
    for message in reversed(state.get("messages", [])):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in ("user", "human"):
            return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    return ""


def _lcd_requirements_are_complete(state: SolutionState) -> bool:
    """LCD / IFP 需求是不是已经采集齐全（可以给推荐了）。

    客户口径（2026-09-30）实测：需求问完之后，客户最后一句往往只是回答我们的问题
    （"no" / "personal" 这种光秃秃的答复）。按字面判意图会落到 others → 走自由问答 →
    型号永远出不来，最后掉进 LED 口径的"放宽条件"兜底话术。
    所以只要档案显示 LCD/IFP 的需求已经齐了，这一轮就是"给推荐"。
    """
    profile = state.get("requirement_profile")
    if profile is None:
        return False
    display_type = str(getattr(profile, "display_type", "") or "").upper()
    if display_type not in ("LCD", "IFP"):
        return False
    try:
        from ....dialogue.lcd_decision import decide_lcd_next_action

        return bool(decide_lcd_next_action(profile).confirmed)
    except Exception:  # pragma: no cover - 防御式
        return False


def intent_node(state: SolutionState) -> SolutionState:
    """Classify the turn unless the caller already established recommendation intent."""
    message = _last_user_message(state)
    caller_intent = str(state.get("intent") or "")
    if caller_intent in ("product_question", "conversation"):
        # 客户真在问问题 / 闲聊 → 先回答，不要拿推荐打断
        intent = caller_intent
        logger.info("Preserving caller-established intent: %s", intent)
    elif _lcd_requirements_are_complete(state):
        # 需求齐全 + 客户没在提问 → 这一轮必须给推荐
        intent = "recommendation"
        logger.info(
            "LCD/IFP requirements complete → intent=recommendation (message=%r)",
            message[:60],
        )
    elif caller_intent in ("recommendation", "product_question", "conversation", "others"):
        # 上游（Sales / Orchestrator）已经定好意图就别再判一遍：
        # 否则"客户在问别的事"会被重新判成 recommendation，又走推荐/反问。
        intent = caller_intent
        logger.info("Preserving caller-established intent: %s", intent)
    else:
        intent = detect_intent(message, state=state)
    logger.info("Current-turn intent=%s message=%r", intent, message[:120])
    return {**state, "intent": intent, "current_message": message}


def conversation_node(state: SolutionState) -> SolutionState:
    """Answer ordinary conversation without entering retrieval or recommendation."""
    message = state.get("current_message", "")
    if re.search(r"(hello|hi|hey|你好|您好)", message.lower()):
        answer = "Hello! I can help you look up display specs or recommend the right product based on your use case."
    elif re.search(r"(thank|thanks|谢谢|感谢)", message):
        answer = "You're welcome! If you have a specific model or use case in mind, feel free to share it."
    else:
        answer = "I can help you look up product specs or recommend the right display based on your needs. What would you like to know or discuss?"
    return {**state, "recommendation": answer, "next_action": "end"}


def route_by_intent(state: SolutionState) -> str:
    """Map the recognized current-turn intent to exactly one graph branch."""
    return {
        "recommendation": "recommendation",
        "product_question": "product_question",
        "conversation": "conversation",
        "others": "others",
    }.get(state.get("intent"), "others")


def route_by_turn_kind(state: SolutionState) -> str:
    """计划 v2.9.1 §五：普通闲聊不进 Others（free_question）分支。

        PURE_CONVERSATION                  → conversation
        其它（含 CONVERSATION_WITH_BUSINESS_SIGNAL / FREE_QUESTION）→ 按 intent 的老路由

    也就是说：只有"确实归不进闲聊、也不是产品问题"的才落到 Others（Free Question）。
    """
    from ....dialogue.turn_kind import CONVERSATION_WITH_BUSINESS_SIGNAL, PURE_CONVERSATION

    kind = str(state.get("turn_kind") or "")
    if kind == PURE_CONVERSATION:
        return "conversation"
    if kind == CONVERSATION_WITH_BUSINESS_SIGNAL:
        # 闲聊 + 业务信号：先接住，业务信息已经在需求抽取层入档；仍走老路由
        return route_by_intent(state)
    return route_by_intent(state)
