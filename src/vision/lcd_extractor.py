"""LCD 视觉提取器（新增一路）—— 图片 → 产品类型（LED/LCD）+ LCD 两个候选事实。

设计约束（客户口径 2026-09-30）：

  · **不改** LED 那一路：`VisionExtractor` / `VISION_SYSTEM_PROMPT` 一行都不动，
    这里只是继承它、复用它的客户端与解析工具，但用自己的提示词和自己的缓存。
  · 类型只产出 LED / LCD —— IFP 一律归到 LCD（"不要直接识别出 ifp"）。
  · 只取两个 LCD 事实：是否拼接、有没有摄像头。
  · **冲突规则**：拼接和摄像头不会同时成立（客户口径）——两个都识别成 true 时，
    丢掉摄像头，只留"是否拼接"。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from ..config import config
from .client import VisionError, ZhipuVisionClient
from .extractor import VisionExtractor, parse_vision_json
from .lcd_prompt import LCD_VISION_SYSTEM_PROMPT, build_lcd_user_prompt
from .schema import VISION_INFERRED, VisionField, VisionRequirement

logger = logging.getLogger(__name__)

# 这一路只认这三个字段
LCD_VISION_FIELDS: tuple[str, ...] = ("display_type", "is_splicing", "camera_observed")

# LCD 候选事实 → 档案字段名（"看见" ≠ "客户需要"，摄像头走 observed）
LCD_FACT_TO_PROFILE: Dict[str, str] = {
    "is_splicing": "lcd_is_splicing",
    "camera_observed": "lcd_camera_observed",
}

_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "是", "有", "有摄像头", "拼接"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "否", "没有", "无", "单体", "单屏"})


class LcdVisionExtractor(VisionExtractor):
    """LCD 专用提取器：自己的提示词、自己的缓存，其余全部复用基类。"""

    # 独立缓存（基类的 _cache 是类属性，子类这里必须自己声明一份，否则和 LED 串味）
    _cache: Dict[str, VisionRequirement] = {}
    _CACHE_LIMIT = 128

    def __init__(self, client: Optional[ZhipuVisionClient] = None) -> None:
        super().__init__(client)

    # ── 缓存键加 "lcd::" 前缀，和 LED 那一路彻底分开 ─────────────────────
    @classmethod
    def _cache_key(cls, session_id: str, image_hash: str) -> str:
        base = super()._cache_key(session_id, image_hash)
        return f"lcd::{base}" if base else ""

    # ── 主入口（与基类同形，只换提示词）────────────────────────────────
    def extract(
        self,
        image: Any,
        *,
        session_id: str = "",
        customer_text: str = "",
        mime_type: str = "",
        use_cache: bool = True,
    ) -> VisionRequirement:
        """单张图片 → VisionRequirement（只含 LED/LCD + 那两个 LCD 事实）。"""
        digest = self.image_hash(image)
        if use_cache:
            cached = self.cache_lookup(session_id, digest)
            if cached is not None:
                logger.info("LCD vision cache hit: session=%s hash=%s", session_id, digest[:12])
                return cached

        raw = self.client.analyze_image(
            image,
            build_lcd_user_prompt(customer_text),
            system_prompt=LCD_VISION_SYSTEM_PROMPT,
            mime_type=mime_type,
        )
        result = self.from_payload(parse_vision_json(raw))
        result.image_hash = digest
        result.model = getattr(self.client, "model", "")
        if use_cache:
            self.cache_store(session_id, digest, result)
        return result

    # ── 解析：只认三个字段，且强制 IFP → LCD ────────────────────────────
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> VisionRequirement:
        data: Dict[str, Any] = {}
        display_type = cls._display_type(payload.get("display_type"))
        if display_type:
            data["display_type"] = display_type

        splicing = cls._coerce_field(payload.get("is_splicing"), cls._bool)
        camera = cls._coerce_field(payload.get("camera_observed"), cls._bool)

        # 冲突规则（客户口径）：拼接和摄像头不会同时成立 → 只留"是否拼接"
        if _is_true(splicing) and _is_true(camera):
            logger.info(
                "LCD vision conflict: is_splicing=True 且 camera_observed=True "
                "→ 只保留 is_splicing（按客户口径）"
            )
            camera = None

        if splicing is not None:
            data["is_splicing"] = splicing
        if camera is not None:
            data["camera_observed"] = camera

        notes = payload.get("notes")
        if isinstance(notes, str) and notes.strip():
            data["notes"] = notes.strip()[:300]
        return VisionRequirement(**data)

    # ── 归一化工具 ──────────────────────────────────────────────────────
    @staticmethod
    def _display_type(value: Any) -> Optional[VisionField]:
        """模型输出 → 类型字段。**只产出 LED / LCD**（IFP 归 LCD）。"""
        if value in (None, "", [], {}):
            return None
        raw = value
        if isinstance(value, dict):
            inner = value.get("value")
            if inner in (None, ""):
                return None
            text = str(inner).strip().upper()
        else:
            text = str(value).strip().upper()
        if not text:
            return None
        # IFP / 一体机 / 交互平板 → 归 LCD（客户口径：不要直接识别出 IFP）
        if (
            "LCD" in text
            or "IFP" in text
            or "一体机" in text
            or "交互平板" in text
            or "触摸屏" in text
        ):
            canonical = "LCD"
        elif "LED" in text:
            canonical = "LED"
        else:
            return None
        if isinstance(raw, dict):
            return VisionExtractor._coerce_field(raw, lambda _value: canonical)
        return VisionField(
            value=canonical, confidence=0.5, source=VISION_INFERRED, evidence=""
        )

    @staticmethod
    def _bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        return None


def _is_true(field: Optional[VisionField]) -> bool:
    return isinstance(field, VisionField) and field.value is True


_lcd_extractor: Optional[LcdVisionExtractor] = None


def get_lcd_vision_extractor() -> LcdVisionExtractor:
    """进程内复用同一个 LCD 提取器（含图片缓存）。"""
    global _lcd_extractor
    if _lcd_extractor is None:
        _lcd_extractor = LcdVisionExtractor()
    return _lcd_extractor


def extract_lcd_vision(
    images: Iterable[Any],
    *,
    session_id: str = "",
    customer_text: str = "",
) -> List[VisionRequirement]:
    """多张图片走 LCD 那一路；失败返回空列表（图片永远不能拖垮主流程）。"""
    image_list = [img for img in (images or []) if img]
    if not image_list:
        return []
    try:
        return get_lcd_vision_extractor().extract_many(
            image_list, session_id=session_id, customer_text=customer_text
        )
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("LCD vision extraction failed: %s", exc)
        return []


__all__ = [
    "LCD_FACT_TO_PROFILE",
    "LCD_VISION_FIELDS",
    "LcdVisionExtractor",
    "extract_lcd_vision",
    "get_lcd_vision_extractor",
]
