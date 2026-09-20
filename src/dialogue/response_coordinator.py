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
