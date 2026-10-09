"""Generation helpers for acknowledging off-topic customer turns."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..state import SalesState
from ....config import config

logger = logging.getLogger(__name__)

_ACK_PROMPT = """你是 LED/LCD 显示屏产品的销售，正在微信上和客户聊天。
客户刚说的这句话**与产品需求无关**（寒暄、闲聊、感叹、题外话、随口一提，
甚至是新闻、天气、"你有鞋子吗"这类完全跑题的话）。

请像真人销售一样自然接一句（最多两句，口语化），**并且一定要把话头带回我们的产品**。

硬性规则：
1. 先"接住"这句话：表示听到了 / 表示理解 / 顺着他的话头轻轻应一句 —— 但不许敷衍到
   只剩客套（"Got it" / "noted" 这种单独成句就是敷衍）。
2. **必须收在"回到客户的屏幕需求"上**：用一句自然的过渡把话题拉回来，
   例如回到他已经说过的场景 / 尺寸 / 用途，或提一句"选屏"这件事本身。
   不要把话题停在闲聊上就结束。
3. **不要问两个问题**：如果系统会给这一轮另外接一个需求问题，你只负责接话 + 过渡；
   自己不要重复问一遍需求。
4. **绝对不要**回答任何知识性、技术性问题（新闻、时事、鞋子、电视剧……一律不展开、
   不评论、不给建议）；不要给参数、型号、价格、方案。
5. **不许和上下文冲突**：客户之前说过的情况（室内/户外、LED/LCD、尺寸、用途）必须一致，
   不许出现相反或无关的产品说法；也不要凭空冒出型号或报价。
6. 不要复述客户整句话；不要客套话堆砌；不要说"作为AI / 作为助手"。
7. 每次换一种说法，不要固定句式，可以让语气自然一点。
8. 语言要求：{language_rule}
9. 直接输出这一句回应，不要 JSON、不要引号、不要解释。"""

_ACK_STYLE_RULES = """

【接话补充规则 · 客户口径】
1. **禁止**用这些已经用烂的开头：Got it / Okay / OK / Understood / Sure / Thanks for that /
   好的 / 收到 / 了解 / 明白。第一句就直接顺着客户的话说。
2. **顺着客户这句话的具体内容说**：他提到场地就说场地、提到距离就说距离、
   提到用途 / 担忧 / 问题就接那个点；不要只回一句空泛的"收到 / 明白了"。
3. **每次换一种说法**：不能和下面这些最近已经发出去的接话雷同（开头、句式都要换）。
最近已发出的接话：
{recent}
"""


def ack_temperature() -> float:
    from ....config import config as current_config

    try:
        return float(getattr(current_config, "ACK_TEMPERATURE", 0.7))
    except (TypeError, ValueError):  # pragma: no cover - defensive config fallback
        return 0.7


def recent_ack_hints(state: SalesState, limit: int = 3) -> str:
    """Return recent assistant turns as context for varied acknowledgements."""
    lines: list[str] = []
    for item in reversed(state.get("messages") or []):
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "")
            content = str(item.get("content") or "")
        else:  # pragma: no cover - LangChain message objects
            role = str(getattr(item, "type", "") or "")
            content = str(getattr(item, "content", "") or "")
        if role not in ("assistant", "ai"):
            continue
        text = " ".join(content.split())
        if not text:
            continue
        lines.append(f"- {text[:160]}")
        if len(lines) >= limit:
            break
    return "\n".join(reversed(lines)) or "（暂无）"


def generate_offtopic_ack(
    message: str,
    *,
    requirement: str = "",
    language: str = "en",
    llm_factory: Optional[Callable[..., Any]] = None,
) -> str:
    """Generate a short, safe acknowledgement without answering unrelated questions."""
    try:
        from ....rag.query_understanding import response_language_rule
        from ....rag.reply_composer import _clean_llm_ack
    except Exception:  # pragma: no cover - optional response helpers
        return ""
    try:
        llm = (llm_factory or ChatOpenAI)(
            model=config.MODEL_NAME,
            temperature=ack_temperature(),
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        system = SystemMessage(
            content=_ACK_PROMPT.format(language_rule=response_language_rule(language))
        )
        human = HumanMessage(
            content=(
                f"客户这句话：{message}\n"
                f"（已知需求，仅供判断语气，不要复述）：{requirement or '{}'}"
            )
        )
        response = llm.invoke([system, human])
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = _clean_llm_ack(text, language)
        if not cleaned or "{" in cleaned or "}" in cleaned or cleaned.startswith(("[", "(")):
            logger.info("Off-topic ack dropped (not a natural sentence): %r", cleaned[:60])
            return ""
        return cleaned
    except Exception as exc:
        logger.warning("Off-topic ack generation failed: %s", exc)
        return ""


_ack_temperature = ack_temperature
_recent_ack_hints = recent_ack_hints
_generate_offtopic_ack = generate_offtopic_ack

__all__ = [
    "_ACK_PROMPT",
    "_ACK_STYLE_RULES",
    "_ack_temperature",
    "_generate_offtopic_ack",
    "_recent_ack_hints",
]
