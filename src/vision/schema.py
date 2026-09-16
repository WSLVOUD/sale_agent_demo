"""
视觉需求提取的数据结构（计划「第三阶段：设计 Vision Schema」/「第五阶段」）。

只解决一件事：把"智谱视觉模型看到了什么"表示成**带来源**的结构化字段。

来源只有两种（计划第六章：严格区分"看见"和"猜测"）：

  vision_explicit  图片能够比较明确地支持该信息（明显的室内会议室 / 明显的 LED 屏）
  vision_inferred  模型根据图片做的推断（大概尺寸 / 可能固定安装）

``vision_inferred`` **永远不能**被当成客户确认的事实：
它可以用来提问、用来提示，但不能直接进入推荐 Gate 的"已确认"集合，
也不能直接进入工程计算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

VISION_EXPLICIT = "vision_explicit"
VISION_INFERRED = "vision_inferred"

VisionSource = Literal["vision_explicit", "vision_inferred"]

# 计划 5.1：第一版只提取与 LED 销售流程直接相关的字段
CORE_FIELDS: tuple[str, ...] = (
    "display_type",
    "environment",
    "purpose",
    "installation",
    "target_width_m",
    "target_height_m",
    "viewing_distance_m",
    "pixel_pitch_mm",
    "brightness_min_nit",
    "brightness_max_nit",
)

# 计划 5.1：可扩展字段（第一版只记录，不参与决策）
EXTRA_FIELDS: tuple[str, ...] = (
    "screen_shape",
    "cabinet_type_hint",
    "technology_hint",
    "rental_hint",
    "curved_hint",
    "transparent_hint",
)


class VisionField(BaseModel):
    """单个视觉字段：值 + 置信度 + 来源 + 证据。"""

    value: Any = None
    confidence: float = 0.0
    source: VisionSource = VISION_EXPLICIT
    evidence: str = ""

    @property
    def is_explicit(self) -> bool:
        return self.source == VISION_EXPLICIT

    @property
    def is_inferred(self) -> bool:
        return self.source == VISION_INFERRED


class VisionRequirement(BaseModel):
    """一张图片（或多张图片合并后）的视觉需求结果。"""

    # ── 核心字段 ────────────────────────────────────────────────────────
    display_type: Optional[VisionField] = None
    environment: Optional[VisionField] = None
    purpose: Optional[VisionField] = None
    installation: Optional[VisionField] = None
    target_width_m: Optional[VisionField] = None
    target_height_m: Optional[VisionField] = None
    viewing_distance_m: Optional[VisionField] = None
    pixel_pitch_mm: Optional[VisionField] = None
    brightness_min_nit: Optional[VisionField] = None
    brightness_max_nit: Optional[VisionField] = None

    # ── 特殊要求（防水 / COB / HDR …）────────────────────────────────────
    special_requirements: List[str] = Field(default_factory=list)

    # ── 可扩展字段（第一版只记录，不参与任何判断）────────────────────────
    screen_shape: Optional[VisionField] = None
    cabinet_type_hint: Optional[VisionField] = None
    technology_hint: Optional[VisionField] = None
    rental_hint: Optional[VisionField] = None
    curved_hint: Optional[VisionField] = None
    transparent_hint: Optional[VisionField] = None

    # ── 元信息（只用于日志与排查）───────────────────────────────────────
    notes: str = ""
    image_hash: str = ""
    model: str = ""

    # ── 便捷方法 ────────────────────────────────────────────────────────
    def items(self, source: Optional[str] = None, core_only: bool = True) -> Dict[str, VisionField]:
        """返回有值的字段；``source`` 可筛选 explicit / inferred。"""
        names = CORE_FIELDS if core_only else CORE_FIELDS + EXTRA_FIELDS
        result: Dict[str, VisionField] = {}
        for name in names:
            field = getattr(self, name, None)
            if not isinstance(field, VisionField) or field.value in (None, "", [], {}):
                continue
            if source and field.source != source:
                continue
            result[name] = field
        return result

    def explicit_items(self) -> Dict[str, VisionField]:
        return self.items(source=VISION_EXPLICIT)

    def inferred_items(self) -> Dict[str, VisionField]:
        return self.items(source=VISION_INFERRED)

    def has_any(self) -> bool:
        return bool(self.explicit_items() or self.inferred_items() or self.special_requirements)

    def metrics(self) -> Dict[str, Any]:
        """计划第二十二阶段：只记录"提取了几个字段"，不记录图片内容。"""
        return {
            "fields_extracted": len(self.explicit_items()),
            "fields_inferred": len(self.inferred_items()),
            "fields_null": len(CORE_FIELDS) - len(self.items()),
            "special_requirements": len(self.special_requirements),
            "image_hash": self.image_hash,
            "model": self.model,
        }


__all__ = [
    "CORE_FIELDS",
    "EXTRA_FIELDS",
    "VISION_EXPLICIT",
    "VISION_INFERRED",
    "VisionField",
    "VisionRequirement",
    "VisionSource",
]
