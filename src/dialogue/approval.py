"""语义判定：客户这句话是不是在**同意推进**（例如同意出报价单）。

客户口径（2026-10）：

    系统在销售层用关键词/长度去猜"客户是不是确认了"，客户明确否掉：
    「不能只是用关键词判断客户说的话，来确认是否推进，而是需要了解客户说的
      一整句话的意思结合上下文去理解」

所以这里**不靠关键词、不靠词数**：把最近对话 + 客户这一整句话交给模型，
让它结合上文理解"这句话在做什么"。典型场景：

    AI  : The TW11-OD-P4 is the right fit. … Shall I prepare the quotation for you?
    客户: yes                                   ← 同意出报价
    客户: which one is cheaper?                 ← 不是同意，是提问
    客户: 我不着急，先看看                      ← 不是同意
    客户: ok，但先把安装方式说一下              ← 同意里带条件，也算"在推进"

判定不出来（模型不可用 / 返回不合法）时返回 ``None``，调用方据此**保守处理**
（不当作同意），绝不猜。
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_APPROVAL_PROMPT = """你在判断客户**这一句话**在做什么。

最近的对话（越靠下越新）：
{recent}

客户最新一句：{message}

问题：结合上面的上下文，客户这句话是不是在**同意推进**？
也就是：对已经给出的推荐/方案表示认可，让我们继续往下走（例如同意出报价单、同意按这个方案办）。

请只回答一个字：
- 是 —— 客户在同意推进
- 否 —— 其他任何情况（在问新问题、在提条件、在说需求、在闲聊、在拒绝、看不出来）

要求：
- 必须结合上下文。只发一个 "yes" 时，要看上一条我们问的是什么才能判断。
- 看不出来就回答"否"，不要猜。
"""


def understand_approval(
    message: str,
    *,
    session_id: str = "",
    recent: Optional[str] = None,
) -> Optional[bool]:
    """结合上下文语义判断"客户是否在同意推进"。

    返回 ``True`` / ``False``；模型不可用或判定不了时返回 ``None``（调用方保守处理）。
    """
    text = str(message or "").strip()
    if not text:
        return None
    if recent is None:
        try:
            from ..memory.history_window import dialogue_window_text

            recent = dialogue_window_text(session_id, max_items=8) if session_id else ""
        except Exception:  # pragma: no cover - 防御式
            recent = ""
    try:
        from ..core.llm import get_llm

        prompt = _APPROVAL_PROMPT.format(
            message=text[:300], recent=str(recent or "")[:1200] or "(none)"
        )
        response = get_llm(temperature=0.0).invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)) or ""
        answer = "".join(str(raw).split()).lower()
        if not answer:
            return None
        # 只认明确的"是/否"，其余一律当成"判不出来"
        if answer.startswith("是") or answer in ("yes", "y", "true"):
            return True
        if answer.startswith("否") or answer in ("no", "n", "false"):
            return False
        return None
    except Exception as exc:  # pragma: no cover - 模型不可用
        logger.warning("Approval understanding failed: %s", exc)
        return None


__all__ = ["understand_approval"]
