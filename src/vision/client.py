"""
智谱（GLM）视觉模型客户端（计划「第二阶段 4.1 / 第三阶段」）。

职责边界（严格）：
    只负责"把一张图片 + 提示词发给智谱视觉模型，拿回原始文本"。
    这里**不**处理 RequirementProfile、不推荐、不追问、不检索。

支持传入：
    - bytes（图片二进制）
    - base64 字符串
    - data URL（data:image/jpeg;base64,...）
    - 本地文件路径
    - http(s) URL（直接交给模型服务端拉取）
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from src.config import config

logger = logging.getLogger(__name__)

# 只允许常见图片格式（计划第二十一阶段：安全控制）
ALLOWED_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

DEFAULT_MODEL = "glm-4v-plus"
# 配置里的模型名不可用时，按顺序回退（不同账号开通的视觉模型可能不同）
FALLBACK_MODELS = ("glm-4v-plus", "glm-4.5v", "glm-4v")


class VisionError(RuntimeError):
    """视觉链路可预期的失败（调用方必须降级，不能让主流程崩）。"""

    def __init__(self, message: str, *, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


def _guess_mime(image: Any, mime_type: str = "") -> str:
    if mime_type:
        return mime_type.split(";")[0].strip().lower()
    if isinstance(image, dict):
        nested = str(image.get("mime_type") or image.get("mimeType") or "")
        if nested:
            return nested.split(";")[0].strip().lower()
    elif not isinstance(image, (str, bytes)):
        nested = str(getattr(image, "mime_type", "") or "")
        if nested:
            return nested.split(";")[0].strip().lower()
    if isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
        guessed, _ = mimetypes.guess_type(image)
        if guessed:
            return guessed.lower()
    return "image/jpeg"


def _payload_parts(image: Any) -> Optional[tuple]:
    """把 dict / 对象形式的图片负载拆成 ``(url, data, mime_type)``。"""
    if isinstance(image, dict):
        getter = image.get
    elif hasattr(image, "url") or hasattr(image, "data"):
        getter = lambda key, default=None: getattr(image, key, default)  # noqa: E731
    else:
        return None
    return (
        str(getter("url", "") or "").strip(),
        str(getter("data", "") or "").strip(),
        str(getter("mime_type", "") or getter("mimeType", "") or "").strip(),
    )


def _to_data_url(image: Any, mime_type: str = "") -> str:
    """把图片统一成 data URL（http(s) URL 原样返回）。"""
    if isinstance(image, bytes):
        mime = _guess_mime(image, mime_type)
        return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")

    # 前端/第三方把图片发成对象：{url} / {data, mime_type}（实测 bug：
    # 这条链路只认字符串 → 视觉模型根本没被调用，日志 "unsupported image payload"）
    parts = _payload_parts(image)
    if parts is not None:
        url, data, nested_mime = parts
        if url:
            return _to_data_url(url, nested_mime or mime_type)
        if data:
            return _to_data_url(data, nested_mime or mime_type)
        raise VisionError("unsupported image payload (empty url/data)", kind="bad_request")

    if isinstance(image, str):
        text = image.strip()
        if text.startswith("data:"):
            return text
        if text.startswith(("http://", "https://")):
            return text
        # 前端可能带 "data:image/png;base64," 前缀被截断的情况 → 容错补前缀
        if text.startswith("image/") and ";base64," in text:
            return "data:" + text
        if os.path.exists(text):
            with open(text, "rb") as handle:
                raw = handle.read()
            mime = _guess_mime(text, mime_type)
            return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        # 当作裸 base64
        mime = _guess_mime("", mime_type)
        return f"data:{mime};base64," + text

    raise VisionError("unsupported image payload", kind="bad_request")


def check_image_payload(image: Any, *, mime_type: str = "", max_mb: float = 0) -> Dict[str, Any]:
    """计划第二十一阶段：进入模型前的格式 / 大小检查。"""
    parts = _payload_parts(image)
    if parts is not None:
        url, data, nested_mime = parts
        if url:
            return check_image_payload(url, mime_type=nested_mime or mime_type, max_mb=max_mb)
        if data:
            return check_image_payload(data, mime_type=nested_mime or mime_type, max_mb=max_mb)
        raise VisionError("unsupported image payload (empty url/data)", kind="bad_request")
    size_mb = 0.0
    if isinstance(image, bytes):
        size_mb = len(image) / (1024 * 1024)
        mime = _guess_mime(image, mime_type)
    elif isinstance(image, str) and not image.startswith(("http://", "https://")):
        if image.startswith("data:"):
            mime = image[5:].split(";")[0].lower()
            b64 = image.split(",", 1)[1] if "," in image else ""
            size_mb = (len(b64) * 3 / 4) / (1024 * 1024)
        elif os.path.exists(image):
            size_mb = os.path.getsize(image) / (1024 * 1024)
            mime = _guess_mime(image, mime_type)
        else:
            mime = _guess_mime("", mime_type)
            size_mb = (len(image) * 3 / 4) / (1024 * 1024)
    else:
        mime = _guess_mime(image, mime_type)

    if mime and mime not in ALLOWED_MIME:
        raise VisionError(f"unsupported mime type: {mime}", kind="bad_request")
    limit = max_mb or getattr(config, "VISION_MAX_IMAGE_MB", 5.0)
    if limit and size_mb > limit:
        raise VisionError(
            f"image too large: {size_mb:.1f}MB > {limit}MB", kind="bad_request"
        )
    return {"mime_type": mime, "size_mb": round(size_mb, 3)}


class ZhipuVisionClient:
    """智谱视觉模型（GLM-4V 系列）客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else getattr(config, "GLM_API_KEY", "")
        self.model = model or getattr(config, "GLM_VISION_MODEL", "") or DEFAULT_MODEL
        self.api_base = (api_base or getattr(config, "GLM_API_BASE", "") or
                         "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
        self.timeout = float(
            timeout if timeout is not None else getattr(config, "VISION_TIMEOUT_SECS", 40)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else getattr(config, "VISION_MAX_RETRIES", 1)
        )

    # ── 对外接口 ────────────────────────────────────────────────────────
    def analyze_image(
        self,
        image: Any,
        prompt: str,
        system_prompt: Optional[str] = None,
        mime_type: str = "",
    ) -> str:
        """把图片交给视觉模型，返回原始文本响应。"""
        if not self.api_key:
            raise VisionError("GLM_API_KEY is not configured", kind="config")

        check_image_payload(image, mime_type=mime_type)
        url = _to_data_url(image, mime_type)

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        )

        last_error: Optional[Exception] = None
        for model in self._models():
            try:
                return self._post(messages, model)
            except VisionError as error:
                last_error = error
                if error.kind == "model_not_found":
                    logger.warning("Vision model %s unavailable, trying next: %s", model, error)
                    continue
                raise
        raise last_error or VisionError("vision call failed", kind="unknown")

    async def analyze_image_async(self, image: Any, prompt: str, **kwargs: Any) -> str:
        """异步版本（内部走线程，避免阻塞事件循环）。"""
        import asyncio

        return await asyncio.to_thread(self.analyze_image, image, prompt, **kwargs)

    # ── 内部 ────────────────────────────────────────────────────────────
    def _models(self) -> List[str]:
        models = [self.model]
        for name in FALLBACK_MODELS:
            if name not in models:
                models.append(name)
        return models

    def _post(self, messages: List[Dict[str, Any]], model: str) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.api_base}/chat/completions"

        attempt = 0
        while True:
            attempt += 1
            started = time.time()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(endpoint, json=payload, headers=headers)
            except httpx.TimeoutException as error:
                if attempt > self.max_retries:
                    raise VisionError(f"vision timeout: {error}", kind="timeout") from error
                logger.warning("Vision call timeout (attempt %d), retrying", attempt)
                continue
            except Exception as error:  # 网络 / DNS / 连接错误
                if attempt > self.max_retries:
                    raise VisionError(f"vision request failed: {error}", kind="network") from error
                logger.warning("Vision call failed (attempt %d): %s", attempt, error)
                continue

            latency_ms = int((time.time() - started) * 1000)
            if response.status_code >= 400:
                detail = response.text[:300]
                kind = "api_error"
                if response.status_code == 401:
                    kind = "config"
                elif "model" in detail.lower() and ("not" in detail.lower() or "invalid" in detail.lower()):
                    kind = "model_not_found"
                if response.status_code >= 500 and attempt <= self.max_retries:
                    logger.warning("Vision API %s (attempt %d), retrying", response.status_code, attempt)
                    continue
                raise VisionError(
                    f"vision api error {response.status_code}: {detail}", kind=kind
                )

            try:
                data = response.json()
            except Exception as error:
                raise VisionError(f"vision response not json: {error}", kind="bad_response") from error

            content = self._extract_content(data)
            if not content:
                raise VisionError("vision response has no content", kind="bad_response")
            logger.info(
                "Vision call ok: model=%s latency_ms=%d chars=%d",
                model, latency_ms, len(content),
            )
            return content

    @staticmethod
    def _extract_content(data: Dict[str, Any]) -> str:
        """兼容 OpenAI 风格 content 字符串 / GLM 分片列表两种返回。"""
        try:
            message = (data.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content")
        except Exception:  # pragma: no cover - 防御式
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "").strip()


__all__ = [
    "ALLOWED_MIME",
    "DEFAULT_MODEL",
    "VisionError",
    "ZhipuVisionClient",
    "check_image_payload",
]
