"""
Vision → RequirementProfile 的合并（计划「第七 / 八 / 九 / 十八 阶段」）。

核心规则：

1. **不新增第二套需求状态** —— 图片结果直接并进现有 RequirementProfile。
2. **来源优先级**：客户明说 > 图片明确可见 > 场景判定 > 图片推测 > 算法估算 > 默认。
   客户的明确说法永远不会被图片覆盖。
3. **冲突要记下来**：客户说 outdoor、图片像 indoor → 保留客户的值，
   记录冲突并让 Sales Agent 就这一项向客户确认（不覆盖、不擅自改）。
4. **尺寸只能当提示**：图片估计的宽高进入 `vision_size_hint_mm`，
   不写进 target_width_m / target_height_m —— 于是**不可能**进入箱体/模组计算。
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.models.requirement import (
    CONFIRMED_SOURCES,
    _EXPLICIT_STRENGTH,
    RequirementProfile,
)
from src.vision.extractor import get_vision_extractor, merge_vision_results
from src.vision.schema import (
    CORE_FIELDS,
    VISION_EXPLICIT,
    VISION_INFERRED,
    VisionField,
    VisionRequirement,
)

logger = logging.getLogger(__name__)

# 图片尺寸类字段 → 只进 hint
_SIZE_FIELDS = ("target_width_m", "target_height_m")


def _strength(source: str) -> int:
    return _EXPLICIT_STRENGTH.get(str(source or ""), 0)


def _conflict_slot(field_name: str) -> str:
    """档案字段名 → 槽位名（用于追问与登记冲突）。"""
    from src.models.requirement import SLOT_TO_FIELD

    for slot, field in SLOT_TO_FIELD.items():
        if field == field_name:
            return slot
    return field_name


def apply_vision_to_profile(
    profile: Optional[RequirementProfile],
    vision: Optional[VisionRequirement],
) -> Tuple[RequirementProfile, Dict[str, Any]]:
    """把视觉结果并进需求档案，返回（新档案, 合并统计）。"""
    stats = {"merged_fields": 0, "conflict_count": 0, "size_hint": False}
    if profile is None:
        profile = RequirementProfile()
    if vision is None or not vision.has_any():
        return profile, stats

    sources = dict(profile.sources or {})
    conflicts: List[str] = list(profile.conflicts or [])
    conflict_slots: List[str] = list(profile.conflict_slots or [])

    for name, field in vision.items().items():
        if name in _SIZE_FIELDS:
            continue
        if not isinstance(field, VisionField) or field.value in (None, "", [], {}):
            continue

        current = getattr(profile, name, None)
        current_source = str(sources.get(name, "") or "")

        if current in (None, "", [], {}):
            setattr(profile, name, field.value)
            sources[name] = field.source
            stats["merged_fields"] += 1
            continue

        if field.value == current:
            # 图片与现有信息一致 → 只把来源升级（图片明确可见时）
            if _strength(field.source) > _strength(current_source):
                sources[name] = field.source
            continue

        if current_source in CONFIRMED_SOURCES:
            # 客户已经明说的 → 保留客户的值，记录冲突（第八 / 九阶段）
            slot = _conflict_slot(name)
            message = f"customer said {current}, image suggests {field.value} ({slot})"
            if message not in conflicts:
                conflicts.append(message)
            if slot not in conflict_slots:
                conflict_slots.append(slot)
            stats["conflict_count"] += 1
            logger.info("Vision conflict on %s: keep customer=%s, image=%s", name, current, field.value)
            continue

        if _strength(field.source) > _strength(current_source):
            setattr(profile, name, field.value)
            sources[name] = field.source
            stats["merged_fields"] += 1

    # ── 尺寸：只作为提示（第十八 / 二十阶段）─────────────────────────────
    width_field = vision.target_width_m
    height_field = vision.target_height_m
    width_m = width_field.value if isinstance(width_field, VisionField) else None
    height_m = height_field.value if isinstance(height_field, VisionField) else None
    if width_m or height_m:
        profile.vision_size_hint_mm = [
            float(width_m or 0) * 1000,
            float(height_m or 0) * 1000,
        ]
        stats["size_hint"] = True
        logger.info(
            "Vision size hint (needs customer confirmation): %s x %s m",
            width_m, height_m,
        )

    # ── 特殊要求（防水 / COB …）──────────────────────────────────────────
    for token in vision.special_requirements:
        if token not in profile.special_requirements:
            profile.special_requirements.append(token)
            stats["merged_fields"] += 1

    # ── 备注（点间距 / 亮度等提示）──────────────────────────────────────
    for name in ("pixel_pitch_mm", "brightness_min_nit", "brightness_max_nit", "viewing_distance_m"):
        field = getattr(vision, name, None)
        if isinstance(field, VisionField) and field.value not in (None, "", [], {}):
            note = f"image {field.source}: {name}={field.value}"
            if note not in profile.vision_notes:
                profile.vision_notes.append(note)
    if vision.notes:
        note = f"image notes: {vision.notes}"
        if note not in profile.vision_notes:
            profile.vision_notes.append(note[:300])

    profile.sources = sources
    profile.conflicts = conflicts
    profile.conflict_slots = conflict_slots

    # ── 待客户确认：图片识别出的字段要跟客户核一遍（客户口径）───────────────
    # 只登记"客户还没确认过"的字段；范围限定在图片确实能看出来的那几项，
    # 尺寸走 vision_size_hint_mm 的追问，不在这里重复问。
    pending: List[str] = []
    assertions: Dict[str, Any] = dict(profile.vision_assertions or {})
    for name in ("display_type", "environment", "purpose", "installation"):
        source = str(sources.get(name) or "")
        if source.startswith("vision") and getattr(profile, name, None) not in (None, "", [], {}):
            pending.append(name)
            assertions[name] = getattr(profile, name)
    if pending:
        profile.vision_confirmation_pending = sorted(set(pending))
        profile.vision_assertions = assertions
        stats["pending_confirmation"] = list(profile.vision_confirmation_pending)

    return profile, stats


_AFFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|right|exactly|sure|ok(?:ay)?|"
    r"that'?s right|that is right|it is|it'?s right|perfect|exact)\b|"
    r"对的|是的|没错|正确|就是这样|没问题|可以的|对的啊|嗯|是的呀",
    re.IGNORECASE,
)


def resolve_vision_confirmation(profile, message: str) -> Dict[str, Any]:
    """客户对"图片识别结果"的回应落地：确认 → 标记为客户确认；纠正 → 记下客户的值。

    设计（客户口径）：
      - 图片识别出的字段会先跟客户核一遍（见 ``vision_confirmation_sentence``）；
      - 客户说"对/是的" → 这些字段升级为**客户确认**（sources 从 vision_explicit → confirmed）；
      - 客户给了不同的值（"不是，是室外的"）→ 客户的值本来就已经按"客户优先"合并进档案，
        这里额外记一条纠正记录，便于回溯"图片说的 vs 客户说的"；
      - 客户没回应就跳过，不再重复追问同一件事。
    """
    stats: Dict[str, Any] = {"confirmed": [], "corrected": [], "skipped": False}
    if profile is None:
        return stats
    pending = list(getattr(profile, "vision_confirmation_pending", None) or [])
    if not pending:
        return stats

    text = str(message or "")
    affirmed = bool(_AFFIRM_RE.search(text))
    sources = dict(profile.sources or {})
    assertions = dict(getattr(profile, "vision_assertions", None) or {})
    corrections = list(getattr(profile, "vision_corrections", None) or [])

    for field in pending:
        value = getattr(profile, field, None)
        asserted = assertions.get(field)
        source = str(sources.get(field) or "")

        if source in CONFIRMED_SOURCES:
            # 客户已经用**自己的话**给了值
            if asserted not in (None, "") and value != asserted:
                note = f"image said {asserted}, customer said {value} ({field})"
                if note not in corrections:
                    corrections.append(note)
                stats["corrected"].append(field)
                logger.info("Vision correction on %s: image=%s → customer=%s", field, asserted, value)
            elif affirmed and source == "vision_explicit":
                sources[field] = "confirmed"
                stats["confirmed"].append(field)
            continue

        if affirmed and source == "vision_explicit":
            sources[field] = "confirmed"
            stats["confirmed"].append(field)

    if not stats["confirmed"] and not stats["corrected"]:
        stats["skipped"] = True

    profile.sources = sources
    profile.vision_corrections = corrections
    # 核对过一轮就不再重复问同一件事
    profile.vision_confirmation_pending = []
    profile.vision_assertions = {}
    return stats


def extract_vision_for_turn(
    images: Optional[Iterable[Any]],
    session_id: str,
    customer_text: str = "",
) -> Tuple[List[VisionRequirement], Dict[str, Any]]:
    """把这一轮的图片转成视觉需求结果；**任何失败都不抛出**。

    返回（结果列表, metrics）。metrics 用于日志（计划第二十二阶段）。
    """
    metrics: Dict[str, Any] = {
        "vision_success": False,
        "vision_latency_ms": 0,
        "images": 0,
        "fields_extracted": 0,
        "fields_inferred": 0,
        "fields_null": 0,
        "special_requirements": 0,
        "parse_success": False,
        "merge_success": False,
        "conflict_count": 0,
        "error": "",
    }
    image_list = [img for img in (images or []) if img]
    if not image_list:
        return [], metrics

    metrics["images"] = len(image_list)
    started = time.time()
    try:
        extractor = get_vision_extractor()
        results = extractor.extract_many(
            image_list, session_id=session_id, customer_text=customer_text
        )
    except Exception as error:  # pragma: no cover - 防御式
        logger.warning("Vision extraction failed: %s", error)
        metrics["error"] = str(error)
        metrics["vision_latency_ms"] = int((time.time() - started) * 1000)
        return [], metrics

    metrics["vision_latency_ms"] = int((time.time() - started) * 1000)
    if not results:
        metrics["error"] = "no vision result"
        return [], metrics

    metrics["vision_success"] = True
    metrics["parse_success"] = True
    merged = merge_vision_results(results)
    explicit = merged.explicit_items()
    inferred = merged.inferred_items()
    metrics["fields_extracted"] = len(explicit)
    metrics["fields_inferred"] = len(inferred)
    metrics["fields_null"] = len(CORE_FIELDS) - len(merged.items())
    metrics["special_requirements"] = len(merged.special_requirements)

    logger.info(
        "Vision metrics: images=%d latency_ms=%d explicit=%s inferred=%s null=%d",
        metrics["images"], metrics["vision_latency_ms"],
        sorted(explicit), sorted(inferred), metrics["fields_null"],
    )
    return results, metrics


__all__ = [
    "apply_vision_to_profile",
    "extract_vision_for_turn",
    "resolve_vision_confirmation",
]
