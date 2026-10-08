"""闲聊 / 非业务话术的"按语境回一句"（客户口径 2026-09-30）。

客户原话：

    「没有回答客户说的话，就一直说自己的话，而且就算僵硬的一种话术，
      这个需要优化，让 ai 自己根据语境上下文去回答，而且必须回答客户的话，
      也要回答有利于自己的话术。」

以前闲聊走的是**写死的三条模板**（hello / thanks / 其它），客户说什么都只回同一句
"I can help you look up product specs…" —— 既没接住客户，又僵硬。

这里给两条闲聊路径（Solution 的 conversation 分支、销售侧的 offtopic 承接）共用
同一个"按语境生成"的入口：

    · 必须接住客户这句话（用自己的话说，不许无视）；
    · 不许编造我们没提供的事实（不承诺现场安装 / 交期 / 价格 / 参数）；
    · 话术保持对我方有利，能自然带回需求或报价；
    · 模型不可用时返回空串，调用方退回模板（绝不空回复）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROMPT = """You are Mike, a sales engineer at iSEMC (commercial LED / LCD displays).
The customer just said something that is NOT a product question and NOT a requirement
answer — small talk, a greeting, a joke, a random remark.

Customer's message: {message}

Recent conversation (newest at the bottom):
{recent}

Write ONE short reply (1-2 sentences). Rules:
1. **Answer what the customer actually said.** Acknowledge their remark in your own words.
   Never ignore it and never fall back on a generic "I can help you with displays" line.
1b. **Always bring it back to our product.** Whatever they said (news, weather, shoes, TV,
   a joke), acknowledge it briefly and then steer the conversation back to their screen
   requirement or the quotation in the same breath. An acknowledgement that just ends —
   "noted", "got it", "sounds like a busy day" — is NOT acceptable. Two sentences is
   usually enough: one to take their point, one to come back to the display.
1c. **Stay consistent with the conversation above.** Never introduce a product type,
   indoor/outdoor setting, size or use case that contradicts what the customer already
   said, and never invent a model or a price to sound helpful.
2. Stay warm and human, like a real salesperson chatting between work items.
3. Keep it favourable to us: professional, no over-promising.
4. Never invent facts. Do not promise on-site installation, delivery dates, prices,
   warranties or specs that were not given to you.
5. Do not repeat product specifications or re-pitch the model.
6. Reply in the customer's language (English unless the customer wrote Chinese).
7. Plain text only, no markdown, no bullet points, no emoji.

Reply:"""


_QUOTE_PROMPT = """You are Mike, a sales engineer at iSEMC (commercial LED / LCD displays).
The customer has just given you the go-ahead: they want the quotation for the screen you
already recommended. They may have been chatting about unrelated things for a few turns
before this, and now they are back on topic.

Customer's message: {message}

Recent conversation (newest at the bottom):
{recent}

Write ONE short reply (1-2 sentences). Rules:
1. Take the go-ahead naturally, in your own words, and make it clear you are putting the
   quotation together and will send it over. The customer's message may be a bare "yes" —
   read the conversation above to know **which** screen they are approving.
2. Do **not** re-list the model or the tiling figures — they just approved them. Referring
   to the project in a few words ("your outdoor church screen") is good: it shows you know
   what they mean.
3. Never invent facts. No price, no discount, no delivery date or lead time, no
   certification, no stock, no warranty length. Only that it is being prepared and will
   follow shortly.
4. Warm and human, like a real salesperson — and **do not sound like a template**. Vary
   the wording; never repeat a phrasing you have already used above.
5. Reply in the customer's language (English unless the customer wrote Chinese).
6. Plain text only, no markdown, no bullet points, no emoji.

Reply:"""


def generate_chat_reply(
    message: str,
    *,
    session_id: str = "",
    recent: Optional[str] = None,
    state: Optional[Any] = None,
    purpose: str = "chitchat",
) -> str:
    """按语境生成一句闲聊回复；失败返回空串（调用方退回模板）。

    ``purpose``：
      · ``"chitchat"``（默认）—— 跑题/闲聊，接住 + 委婉拉回产品；
      · ``"quote_confirmation"`` —— 客户对已推荐的屏幕说了 "yes / ok / go ahead"，
        回一句"我去准备报价单，稍等"（客户口径 2026-10）。
    """
    text = str(message or "").strip()
    if not text:
        return ""
    if recent is None:
        try:
            from ..memory.history_window import dialogue_window_text

            sid = session_id or str((state or {}).get("session_id") or "")
            recent = dialogue_window_text(sid, max_items=8) if sid else ""
        except Exception:  # pragma: no cover - 防御式
            recent = ""
    template = _QUOTE_PROMPT if str(purpose or "") == "quote_confirmation" else _PROMPT
    try:
        from ..core.llm import get_llm

        prompt = template.format(
            message=text[:300], recent=str(recent or "")[:1200] or "(none)"
        )
        response = get_llm(temperature=0.7).invoke(prompt)
        reply = (response.content if hasattr(response, "content") else str(response)) or ""
        reply = " ".join(str(reply).split()).strip()
        if not reply or len(reply) > 400:
            return ""
        return reply
    except Exception as exc:  # pragma: no cover - 模型不可用
        logger.warning("Chat reply generation failed: %s", exc)
        return ""


__all__ = ["generate_chat_reply"]
