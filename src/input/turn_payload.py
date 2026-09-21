"""v2.5 Phase 1：把"一次请求里的多条消息"合并成一个 UserTurn 负载。

前端会把客户连续发的多条消息放在同一个请求里（`messages: [...]`），
老的客户端仍然只发 `question` + `images`。两种写法都在这里统一：

    parts → { text, images, message_ids }

- 文字按到达顺序拼接（换行分隔，保留原话）；
- 图片保持顺序、去重；
- `message_id` 幂等：同一个 id 重复提交（前端重试 / 网络重发）只算一次。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .message_aggregator import MessageAggregator

_AGGREGATOR = MessageAggregator(debounce_seconds=0.0, max_window_seconds=0.0)


@dataclass
class TurnPayload:
    text: str = ""
    images: List[Any] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    text_parts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "text_parts": list(self.text_parts),
            "image_count": len(self.images),
            "message_ids": list(self.message_ids),
        }


def merge_message_parts(parts: Optional[Iterable[Any]]) -> TurnPayload:
    """把 `messages` 里的各个部分合并（不涉及幂等，纯合并）。"""
    payload = TurnPayload()
    for part in parts or []:
        if isinstance(part, dict):
            text = str(part.get("text") or part.get("question") or "").strip()
            images = list(part.get("images") or [])
            message_id = str(part.get("message_id") or part.get("id") or "")
        else:  # pragma: no cover - 防御式
            text = str(getattr(part, "text", "") or "").strip()
            images = list(getattr(part, "images", None) or [])
            message_id = str(getattr(part, "message_id", "") or "")
        if text:
            payload.text_parts.append(text)
        for image in images:
            if image not in payload.images:
                payload.images.append(image)
        if message_id:
            payload.message_ids.append(message_id)
    payload.text = "\n".join(payload.text_parts)
    return payload


def merge_request_payload(
    session_id: str,
    *,
    question: str = "",
    images: Optional[Sequence[Any]] = None,
    messages: Optional[Iterable[Any]] = None,
    message_ids: Optional[Sequence[str]] = None,
    dedup: bool = True,
) -> TurnPayload:
    """统一的请求负载：`messages` 优先，兼容老的 `question` + `images`。

    ``dedup=True``：同一个 message_id 第二次出现会被忽略（旧口径）。
    v2.7 起幂等交给 TurnExecutor（Message Dedup + Turn Store），
    所以 API 层用 ``dedup=False`` 拿到完整 message_ids 再交给引擎。
    """
    payload = merge_message_parts(messages)
    legacy_images = [image for image in (images or []) if image]
    if str(question or "").strip() or legacy_images:
        legacy_id = str((message_ids or [""])[0] or "")
        legacy = merge_message_parts([
            {"text": str(question or "").strip(), "images": legacy_images, "message_id": legacy_id}
        ])
        payload.text_parts.extend(legacy.text_parts)
        for image in legacy.images:
            if image not in payload.images:
                payload.images.append(image)
        payload.message_ids.extend(legacy.message_ids)
        payload.text = "\n".join(payload.text_parts)

    if not payload.message_ids and not payload.text and not payload.images:
        return payload
    if not dedup:
        return payload

    # 幂等：同一个 message_id 第二次提交（前端重试 / 网络重发）不再重复处理
    accepted = TurnPayload()
    for index, text in enumerate(payload.text_parts):
        message_id = payload.message_ids[index] if index < len(payload.message_ids) else ""
        if message_id and _AGGREGATOR.mark_seen(session_id, message_id):
            continue
        accepted.text_parts.append(text)
        if message_id:
            accepted.message_ids.append(message_id)
    accepted.images = list(payload.images)
    accepted.text = "\n".join(accepted.text_parts)
    return accepted


__all__ = ["TurnPayload", "merge_message_parts", "merge_request_payload"]
