"""语境化地判断"客户是怎么回应产品类型问题的"（计划 v2.9.4 §五/§六/§九）。

客户口径原文（2026-09-24）：

    「不要设置任何关键字去触发 AI 的某个动作，这个太局限了，
      一定是要结合上下文的语境去让触发 AI。」

所以这里**不做关键词匹配**，而是让模型读三样东西再下结论：

    1. 我们刚刚问客户的那句（例如 "Are you looking for an LED display, or an LCD…?"）
    2. 最近的对话（越靠下越新）
    3. 客户最新一句

输出是结构化信号：

    chose           客户已经说了要哪一种（display_type = LED / LCD）
    rejects         否认了我们给的方向，但没说清要哪一种
    does_not_know   表示不知道 / 选不出来（→ 先解释 LED 与 LCD 的区别，再让他选）
    asks_meaning    在问 LED / LCD 是什么、有什么区别（→ 同上）
    delegates       让 AI 直接替他决定
    unrelated       这句话跟"选哪种屏"没关系

`product_type_router` 里的正则只在模型不可用时兜底（降级路径）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import config

logger = logging.getLogger(__name__)

REPLY_CHOSE = "chose"
REPLY_REJECTS = "rejects"
REPLY_DOES_NOT_KNOW = "does_not_know"
REPLY_ASKS_MEANING = "asks_meaning"
REPLY_DELEGATES = "delegates"
REPLY_UNRELATED = "unrelated"

ALL_REPLIES = (
    REPLY_CHOSE,
    REPLY_REJECTS,
    REPLY_DOES_NOT_KNOW,
    REPLY_ASKS_MEANING,
    REPLY_DELEGATES,
    REPLY_UNRELATED,
)

_JUDGE_PROMPT = """你是销售对话的语境理解器。

业务背景：我们正在问客户"要 LED 显示屏，还是 LCD（拼接屏 / 交互平板）"，
现在要判断**客户最新这一句话**在这个语境里到底是什么意思。

返回 JSON：
{
  "reply": "chose" | "rejects" | "does_not_know" | "asks_meaning" | "delegates" | "unrelated",
  "display_type": "LED" | "LCD" | "",
  "reason": "一句话理由"
}

判定口径：
- chose：客户已经明确要哪一种（说了 LED / LCD / 拼接屏 / 视频墙 / 交互平板 / 触控一体机，
  或"就按这个方向来"）
- rejects：客户否认我们给的方向，但没说清要哪一种（"不对" / "no" / "不是这个"）
- does_not_know：客户表示不知道、不确定、选不出来（"I don't know" / "我不懂"）
  —— 不要把它当成 chose，也不要当成 unrelated
- asks_meaning：客户在问 LED / LCD 是什么、两者有什么区别
- delegates：客户让 AI 直接替他决定（"你帮我选" / "you decide"）
- unrelated：客户这句话跟"选哪种屏"没关系（在聊别的、在补充别的需求信息、在问价格交期等）

注意：
1. 只根据客户的话和这段语境判断，不要替客户假设他想用哪种屏
2. display_type 只在 chose / rejects 里说得出具体类型时才填，其余留空
3. 只返回 JSON，不要任何解释"""

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LIMIT = 64


def clear_cache() -> None:
    """清空语境判断缓存（测试与需求重置时用）。"""
    _CACHE.clear()


def _cache_key(session_id: str, message: str) -> str:
    return f"{session_id}::{message}"


def _parse(content: Any) -> Dict[str, Any]:
    """模型输出 → 结构化信号（不合法一律当"没判断"，交给调用方兜底）。"""
    raw = str(content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning("Product type reply signal is not JSON: %r", raw[:120])
        return {}
    if not isinstance(payload, dict):
        return {}
    reply = str(payload.get("reply") or "").strip().lower()
    if reply not in ALL_REPLIES:
        return {}
    product_type = str(payload.get("display_type") or "").strip().upper()
    return {
        "reply": reply,
        "display_type": product_type if product_type in ("LED", "LCD") else "",
        "reason": str(payload.get("reason") or "")[:200],
    }


def understand_product_type_reply(
    message: str,
    *,
    session_id: str = "",
    conversation: str = "",
    asked_question: str = "",
    decision: Optional[Any] = None,
) -> Dict[str, Any]:
    """结合语境判断客户这句话在回应产品类型问题时是什么意思。

    Args:
        message: 客户最新一句
        session_id: 会话 id（缓存键 + 日志）
        conversation: 最近的对话文本（越靠下越新）
        asked_question: 我们上一轮问客户的那句话
        decision: 上一轮的产品类型判断（有的话一并作为语境）

    Returns:
        {"reply": ..., "display_type": ..., "reason": ...}；判断不了时返回 ``{}``
        （调用方会退回到规则解析）。
    """
    text = str(message or "").strip()
    if not text:
        return {}
    key = _cache_key(session_id, text)
    cached = _CACHE.get(key)
    if cached is not None:
        return dict(cached)

    context: list[str] = []
    if conversation:
        context.append(f"最近的对话（越靠下越新）：\n{conversation}")
    if asked_question:
        context.append(f"我们刚问客户的那句：{asked_question}")
    if decision is not None:
        try:
            status = str(getattr(decision, "status", "") or "")
            product_type = str(getattr(decision, "display_type", "") or "")
            if product_type:
                context.append(
                    f"我们上一轮给客户的判断：{product_type}（状态 {status or '未知'}）"
                )
        except Exception:  # pragma: no cover - 防御式
            pass
    context.append(f"客户最新一句：{text}")

    try:
        llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=0,
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        response = llm.invoke(
            [SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content="\n\n".join(context))]
        )
        signal = _parse(getattr(response, "content", response))
    except Exception as exc:
        logger.warning("[%s] Product type reply understanding failed: %s", session_id, exc)
        return {}

    if signal:
        _CACHE[key] = dict(signal)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
    return dict(signal)


__all__ = [
    "ALL_REPLIES",
    "REPLY_ASKS_MEANING",
    "REPLY_CHOSE",
    "REPLY_DELEGATES",
    "REPLY_DOES_NOT_KNOW",
    "REPLY_REJECTS",
    "REPLY_UNRELATED",
    "clear_cache",
    "understand_product_type_reply",
]
