"""v2.3 §13：把 Orchestrator 里的"回复组装"抽出来。

Orchestrator 只负责流程编排；这里负责"这一轮最终对客户说什么"：

    业务回复（Sales / Solution 已经生成好的那句话）
        + 售后口径回答（客户问到才加）
        + 图片识别结果核对（带图的那一轮）

抽出来之后，Orchestrator 不再直接碰这些业务细节，也方便单测。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResponseCoordinator:
    """回复的最后两道加工（顺序固定：售后口径 → 图片核对）。"""

    def __init__(self, store: Any = None, profile_lookup: Any = None):
        self.store = store
        self.profile_lookup = profile_lookup

    # ── 售后口径 ────────────────────────────────────────────────────────
    def attach_service_faq(self, response: str, message: str) -> str:
        """客户问到售后口径时**必须回答**（不主动提，尤其质保）。"""
        try:
            from src.rag.reply_composer import reply_language
            from src.rag.service_faq import (
                detect_service_faq,
                sanitize_service_reply,
                service_faq_reply,
            )

            answer = service_faq_reply(message, language=reply_language(message))
            # 与标准口径矛盾的句子（实测："Yes, installation is included."）先清掉
            response = sanitize_service_reply(response, detect_service_faq(message))
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ServiceFAQ] failed: %s", exc)
            return response
        if not answer or answer in str(response or ""):
            return response
        return f"{answer} {response}".strip() if response else answer

    # ── 图片核对 ────────────────────────────────────────────────────────
    def attach_vision_confirmation(self, response: str, session_id: str, message: str) -> str:
        """带图的那一轮：先把"图片里看到什么"跟客户核一遍，再继续问需求。"""
        try:
            profile = self._profile(session_id)
            if profile is None or not getattr(profile, "vision_confirmation_pending", None):
                return response
            from src.rag.reply_composer import (
                reply_language,
                vision_confirmation_items,
                vision_confirmation_sentence,
            )

            language = reply_language(message)
            items = [item for item in vision_confirmation_items(profile, language) if item]
            text = str(response or "")
            if items and all(item in text for item in items):
                return response
            sentence = vision_confirmation_sentence(profile, language, 0)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[Vision] confirmation sentence failed: %s", exc)
            return response
        if not sentence or sentence in str(response or ""):
            return response
        return f"{sentence} {response}".strip() if response else sentence

    def finalize(self, response: str, *, session_id: str = "", message: str = "") -> str:
        """固定顺序：售后口径 → 图片核对。"""
        text = self.attach_service_faq(str(response or ""), message)
        text = self.attach_vision_confirmation(text, session_id, message)
        return text

    # ── 图片核对句（v2.3.1 从 Orchestrator 迁入）────────────────────────────
    def vision_confirmation_sentence(self, session_id: str, message: str) -> str:
        """图片识别结果的"跟客户核对"那一句（没有待确认字段时返回空串）。"""
        if not session_id:
            return ""
        try:
            profile = self._profile(session_id)
            if profile is None or not getattr(profile, "vision_confirmation_pending", None):
                return ""
            from src.rag.reply_composer import reply_language, vision_confirmation_sentence

            return vision_confirmation_sentence(profile, reply_language(message), 0)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[Vision] confirmation sentence failed: %s", exc)
            return ""

    # ── 答复 + 继续追问（v2.3.1 从 Orchestrator 迁入）───────────────────────
    def compose_with_requirement_question(
        self,
        *,
        answer: str,
        sales_result: Any,
        message: str,
        seed: int = 0,
        session_id: str = "",
    ) -> str:
        """把"答复客户"与"继续追问需求"合成一句自然的销售回复。

        需求还没问清时（Ready Gate 未放行），客户的问题照样要答 —— 但答完必须
        把还缺的那个关键问题接上，否则销售就变成"只会问问题"或"答完就断线"。
        """
        sales_result = sales_result or {}
        question = str(sales_result.get("pending_question") or "")
        if not question:
            return answer
        try:
            from src.rag.reply_composer import compose_requirement_reply, reply_language

            return compose_requirement_reply(
                answer=answer,
                question=question,
                slot=str(sales_result.get("pending_slot") or ""),
                message=message,
                language=reply_language(message),
                seed=seed,
                requirement=sales_result.get("requirements") or {},
                include_ack=not sales_result.get("requirements_reset", False),
                llm_ack=str(sales_result.get("acknowledgement") or ""),
                # 带图的那一轮：图片识别结果要跟客户核对（而且不能一边核对一边又问同一项）
                vision_confirmation=self.vision_confirmation_sentence(session_id, message),
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Compose requirement reply failed: %s", exc)
            return answer or question

    # ── 内部 ────────────────────────────────────────────────────────────
    def _profile(self, session_id: str) -> Optional[Any]:
        if not session_id:
            return None
        if callable(self.profile_lookup):
            try:
                return self.profile_lookup(session_id)
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[ResponseCoordinator] profile_lookup failed: %s", exc)
                return None
        try:
            if self.store is not None and hasattr(self.store, "get_requirement_profile"):
                return self.store.get_requirement_profile(session_id)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ResponseCoordinator] profile lookup failed: %s", exc)
        return None


__all__ = ["ResponseCoordinator"]
