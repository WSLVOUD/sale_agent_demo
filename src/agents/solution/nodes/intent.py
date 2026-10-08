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
    elif state.get("already_recommended"):
        # 客户口径（2026-09-30）：**推荐过一次之后不要再一直推荐**。
        # 需求齐了不等于这一轮要推荐 —— 客户问安装 / 交期 / 天气，就该照着答
        # （实测：推荐完 DS-O-75 后，客户问"你们包安装吗 / 交付日期多久 / 今天天气"
        #  三轮都被当成推荐，回的都是同一段产品介绍）。
        # 这一支不强制 recommendation，按正常意图判定走（问题→回答，闲聊→承接）。
        intent = detect_intent(message, state=state)
        logger.info(
            "Already recommended → 不强制推荐，按当前意图处理：intent=%s", intent
        )
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


def _llm_chat_reply(state: SolutionState, message: str) -> str:
    """让模型按语境回一句人话（失败返回空串，调用方退回模板）。

    客户口径（2026-09-30）："没有回答客户说的话，就一直说自己的话，而且就算僵硬的一种
    话术" —— 以前这一支是**三条写死的模板**（hello / thanks / 其它），客户说什么都
    只回同一句"D I can help you look up product specs…"。现在交给模型读上下文自己说，
    但必须**接住客户这句话**、不许编造内容、且话术对我方有利。
    """
    try:
        from ....dialogue.chat_reply import generate_chat_reply

        return generate_chat_reply(message, state=state)
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Chat reply helper unavailable: %s", exc)
        return ""


def conversation_node(state: SolutionState) -> SolutionState:
    """Answer ordinary conversation without entering retrieval or recommendation.

    LCD / IFP 会话：交给模型按语境说一句人话（必须接住客户这句话，不编内容，话术有利）；
    模型不可用时才退回模板。LED 侧保持原样（那三条模板不动）。
    """
    message = state.get("current_message", "")
    try:
        from ....utils.product_family import product_family_of

        family = product_family_of(state.get("requirement_profile"))
    except Exception:  # pragma: no cover - 防御式
        family = ""
    if family in ("lcd", "ifp"):
        reply = _llm_chat_reply(state, message)
        if reply:
            return {**state, "recommendation": reply, "next_action": "end"}
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
