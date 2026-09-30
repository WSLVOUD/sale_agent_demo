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

# LCD 那一路提取的候选事实：识别字段名 → 档案字段名（"看见" ≠ "客户需要"）
_LCD_VISION_FACT_FIELD: Dict[str, str] = {
    "is_splicing": "lcd_is_splicing",
    "camera_observed": "lcd_camera_observed",
}


def _strength(source: str) -> int:
    return _EXPLICIT_STRENGTH.get(str(source or ""), 0)


def _vision_display_type(vision: Any) -> str:
    field = getattr(vision, "display_type", None)
    return str(getattr(field, "value", "") or "").strip().upper()


def _merge_lcd_vision_facts(
    profile: Any,
    vision: Any,
    sources: Dict[str, str],
    stats: Dict[str, Any],
    conflicts: List[str],
) -> List[str]:
    """把 LCD 那一路识别出的"是否拼接 / 有没有摄像头"并进档案。

    规则（客户口径 2026-09-30）：

      · 只在图片判成 LCD 时生效（LED 的结果没有这两个字段，行为不变）；
      · **拼接和摄像头不会同时成立** —— 两个都是 true 时丢掉摄像头，只留拼接；
      · 写进去的值仍然只是"图片看到的"（vision_*），要请客户确认才算客户事实。
    """
    if _vision_display_type(vision) not in ("LCD", "IFP"):
        return []

    observed: Dict[str, Any] = {}
    for name in ("is_splicing", "camera_observed"):
        field = getattr(vision, name, None)
        if isinstance(field, VisionField) and field.value is not None:
            observed[name] = field

    # 冲突：拼接 + 摄像头不会同时成立 → 只问拼接（宁可少问一项，也不给客户两个矛盾的问题）
    splicing_field = observed.get("is_splicing")
    camera_field = observed.get("camera_observed")
    splicing_true = isinstance(splicing_field, VisionField) and splicing_field.value is True
    camera_true = isinstance(camera_field, VisionField) and camera_field.value is True
    already_splicing = getattr(profile, "lcd_is_splicing", None) is True
    if camera_field is not None and camera_true and (splicing_true or already_splicing):
        observed.pop("camera_observed", None)
        logger.info(
            "LCD vision conflict: 拼接与摄像头同时成立 → 按客户口径只保留 is_splicing"
        )

    pending: List[str] = []
    assertions: Dict[str, Any] = dict(getattr(profile, "vision_assertions", None) or {})
    for name, field in observed.items():
        field_name = _LCD_VISION_FACT_FIELD.get(name, name)
        value = bool(field.value)
        current = getattr(profile, field_name, None)
        current_source = str(sources.get(field_name) or "")
        if current in (None, "", [], {}):
            setattr(profile, field_name, value)
            sources[field_name] = field.source
            stats["merged_fields"] += 1
        elif current == value:
            if _strength(field.source) > _strength(current_source):
                sources[field_name] = field.source
        elif current_source in CONFIRMED_SOURCES:
            # 客户已经说过的 → 留客户的值，只记一条冲突
            note = f"customer said {current}, image suggests {value} ({name})"
            if note not in conflicts:
                conflicts.append(note)
            stats["conflict_count"] += 1
            continue
        else:
            setattr(profile, field_name, value)
            sources[field_name] = field.source
            stats["merged_fields"] += 1
        pending.append(field_name)
        assertions[field_name] = value

    if pending:
        profile.vision_assertions = assertions
        stats.setdefault("lcd_fields", []).extend(pending)
    return pending


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

        # ── v2.3 §14：Vision 输出先过统一 Validation（字段 / 单位 / 范围）──────
        # 非法值（例如荒谬的点间距 / 尺寸）不入档；这里不做"覆盖客户值"的判断，
        # 那部分仍由下面的冲突逻辑处理（客户说过的值永远优先，并记录冲突）。
        from ..rag.fact_validation import validate_incoming_facts

        validated = validate_incoming_facts({name: field.value}, source="vision")
        if name not in validated.accepted:
            logger.info(
                "Vision field %s=%r rejected by validation: %s",
                name, field.value, validated.rejected.get(name),
            )
            continue
        field_value = validated.accepted[name]

        current = getattr(profile, name, None)
        current_source = str(sources.get(name, "") or "")

        if current in (None, "", [], {}):
            setattr(profile, name, field_value)
            sources[name] = field.source
            stats["merged_fields"] += 1
            continue

        if field_value == current:
            # 图片与现有信息一致 → 只把来源升级（图片明确可见时）
            if _strength(field.source) > _strength(current_source):
                sources[name] = field.source
            continue

        if current_source in CONFIRMED_SOURCES:
            # 客户已经明说的 → 保留客户的值，记录冲突（第八 / 九阶段）
            slot = _conflict_slot(name)
            message = f"customer said {current}, image suggests {field_value} ({slot})"
            if message not in conflicts:
                conflicts.append(message)
            if slot not in conflict_slots:
                conflict_slots.append(slot)
            stats["conflict_count"] += 1
            logger.info("Vision conflict on %s: keep customer=%s, image=%s", name, current, field_value)
            continue

        if _strength(field.source) > _strength(current_source):
            setattr(profile, name, field_value)
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

    # ── LCD 候选事实（客户口径 2026-09-30：LCD 那一路只取这两项）──────────
    # 只在"图片判成 LCD"时生效；LED 的结果没有这两个字段 → 行为完全不变。
    lcd_pending = _merge_lcd_vision_facts(profile, vision, sources, stats, conflicts)

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
    # LCD 那两项（是否拼接 / 有没有摄像头）同样要请客户核一遍
    pending.extend(lcd_pending)
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

# 客户"否认图片识别结果"的说法（客户口径 2026-09-30：客户说不对 ≠ 没反对）
# 只在**句首**认英文否定词：像 "actually, we also need…" 这种是补充信息，
# 不是"你识别错了"；中文的"不对/不是"歧义小，允许出现在句中。
_DENY_RE = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect|not right|not correct|that'?s not|that is not)\b"
    r"|不是|不对|错了|搞错|弄错",
    re.IGNORECASE,
)

# 客户把决定权交给我们的说法（"你定 / 让 AI 定"）→ 直接采用识别结果
_DELEGATE_RE = re.compile(
    r"\b(?:you (?:decide|choose|pick)|your call|up to you|whatever you think)\b|"
    r"你(?:来)?(?:定|决定|选)|听你的|你看着办|都行|随便",
    re.IGNORECASE,
)


def resolve_vision_confirmation(profile, message: str) -> Dict[str, Any]:
    """客户对"图片识别结果"的回应落地：确认 → 标记为客户确认；纠正 → 记下客户的值。

    设计（客户口径）：
      - 图片识别出的字段会先跟客户核一遍（见 ``vision_confirmation_sentence``）；
      - 客户说"对/是的" → 这些字段升级为**客户确认**（sources 从 vision_explicit → confirmed）；
      - 客户给了不同的值（"不是，是室外的"）→ 客户的值本来就已经按"客户优先"合并进档案，
        这里额外记一条纠正记录，便于回溯"图片说的 vs 客户说的"；
      - 客户没纠正（没提这一项）→ **视为接受**（`accepted`）：因为识别结果已经摆在上一轮的
        回复里请他核对了，再问一次就是重复 —— 实测客户反馈："你都说识别出是固定安装了，
        为什么还问我固装还是租赁？"
    """
    stats: Dict[str, Any] = {
        "confirmed": [], "accepted": [], "corrected": [], "delegated": [],
        "denied": [], "skipped": False,
    }
    if profile is None:
        return stats
    pending = list(getattr(profile, "vision_confirmation_pending", None) or [])
    if not pending:
        return stats

    text = str(message or "")
    affirmed = bool(_AFFIRM_RE.search(text))
    denied = bool(_DENY_RE.search(text)) and not affirmed
    delegated = bool(_DELEGATE_RE.search(text)) and not denied
    sources = dict(profile.sources or {})
    assertions = dict(getattr(profile, "vision_assertions", None) or {})
    corrections = list(getattr(profile, "vision_corrections", None) or [])
    # 记下"图片给的原始来源"，客户说"不对"又没指出哪里不对时要撤回原样
    original_sources = {field: str(sources.get(field) or "") for field in pending}
    unresolved: List[str] = []

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
            elif affirmed:
                sources[field] = "confirmed"
                stats["confirmed"].append(field)
            continue

        if source.startswith("vision"):
            # 客户说"对" → 客户确认；客户没提这一项 → 视为"已核对过、没反对"（vision_accepted）。
            # 两种都不再重复追问同一件事。
            if affirmed:
                sources[field] = "confirmed"
                stats["confirmed"].append(field)
            else:
                # "你定" 也属于"客户把决定权交给我们" → 采用识别结果
                if delegated:
                    stats["delegated"].append(field)
                sources[field] = "vision_accepted"
                stats["accepted"].append(field)

    # 客户说"不对"、却**没有指出哪里不对**（整句里一项都没纠正）→ 不能当成"没反对"：
    # 撤回到待确认，下一轮请他指出来哪里不对（客户口径 2026-09-30）。
    # 注意：客户一边说"no"一边给了正确的值（"no, it is outdoor"）时，纠正已经记下，
    # 其余没被反对的项仍然算"已核对过"，不重复问。
    if denied and not stats["corrected"]:
        for field in list(stats["accepted"]):
            stats["accepted"].remove(field)
            sources[field] = original_sources.get(field) or "vision_explicit"
            unresolved.append(field)
            stats["denied"].append(field)

    if (
        not stats["confirmed"] and not stats["accepted"]
        and not stats["corrected"] and not stats["denied"]
    ):
        stats["skipped"] = True

    profile.sources = sources
    profile.vision_corrections = corrections
    # 核对过一轮就不再重复问同一件事；但客户说"不对"的那几项要留着，下一轮问他哪里不对。
    profile.vision_confirmation_pending = unresolved
    profile.vision_confirmation_denied = unresolved
    if not unresolved:
        profile.vision_assertions = {}
    return stats


def _settled_display_type(session_id: str) -> str:
    """会话里**已经确认/锁定**的产品类型（LED / LCD）；还没定就返回空串。

    客户口径（2026-09-30）：类型一旦确认，客户再发图就**直接按这个类型**提取，
    不需要重新判类型。
    """
    session = str(session_id or "").strip()
    if not session:
        return ""
    try:
        from ..memory.store import memory

        decision: Dict[str, Any] = {}
        if hasattr(memory, "get_display_type_decision"):
            decision = memory.get_display_type_decision(session) or {}
        display_type = str(decision.get("display_type") or "").upper()
        status = str(decision.get("status") or "").upper()
        if display_type in ("LED", "LCD") and (
            bool(decision.get("locked")) or status == "CONFIRMED"
        ):
            return display_type

        stored = None
        if hasattr(memory, "get_requirement_profile"):
            stored = memory.get_requirement_profile(session)
        if isinstance(stored, dict):
            profile_type = str(stored.get("display_type") or "").upper()
            source = str((stored.get("sources") or {}).get("display_type") or "")
        else:
            profile_type = str(getattr(stored, "display_type", "") or "").upper()
            source = str(
                (getattr(stored, "sources", None) or {}).get("display_type") or ""
            )
        if profile_type in ("LED", "LCD") and source in (
            "explicit", "confirmed", "customer_explicit", "customer_confirmed",
        ):
            return profile_type
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Vision settled-type lookup failed: %s", exc)
    return ""


def _merged_display_type(results: List[Any]) -> str:
    """多张图合并后的类型（LED / LCD / IFP / 空）。"""
    if not results:
        return ""
    try:
        from .extractor import merge_vision_results

        merged = merge_vision_results(results)
    except Exception:  # pragma: no cover - 防御式
        return ""
    return _vision_display_type(merged)


def extract_vision_for_turn(
    images: Optional[Iterable[Any]],
    session_id: str,
    customer_text: str = "",
) -> Tuple[List[VisionRequirement], Dict[str, Any]]:
    """把这一轮的图片转成视觉需求结果；**任何失败都不抛出**。

    返回（结果列表, metrics）。metrics 用于日志（计划第二十二阶段）。

    ── 类型分流（客户口径 2026-09-30）──────────────────────────────────────

        · 会话里已经确认是 LCD → 只走 LCD 那一路（只取"是否拼接 / 有没有摄像头"）；
        · 会话里已经确认是 LED → 只走现有 LED 那一路（行为完全不变）；
        · 类型还没定 → **先用带判定标准的提示词判类型**（LED 与 LCD 怎么区分写在
          LCD 那路提示词里），判成 LED 再走现有 LED 提取器取 LED 需求，
          判成 LCD 就用 LCD 那一路的结果。类型统一成 LED / LCD（不产出 IFP）。

    为什么"判类型"要单独用带标准的提示词（实测 2026-09-30）：客户发的图里有
    明显拼缝、明显边框（一眼 LCD），只因为 LED 那份提示词里只写了
    "display_type LED / LCD / IFP（图片里看到的是哪一类屏）"、一句判定标准都没有，
    模型就把它判成了 LED → 整轮按 LED 口径走，客户纠正后才回到 LCD。
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
        "vision_route": "",
        "error": "",
    }
    image_list = [img for img in (images or []) if img]
    if not image_list:
        return [], metrics

    metrics["images"] = len(image_list)
    started = time.time()
    settled = _settled_display_type(session_id)
    try:
        from .lcd_extractor import extract_lcd_vision

        def _extract_led() -> List[Any]:
            return get_vision_extractor().extract_many(
                image_list, session_id=session_id, customer_text=customer_text
            )

        if settled == "LED":
            results = _extract_led()
            metrics["vision_route"] = "led(confirmed)"
        elif settled == "LCD":
            results = extract_lcd_vision(
                image_list, session_id=session_id, customer_text=customer_text
            )
            metrics["vision_route"] = "lcd(confirmed)"
        else:
            # 类型还没定 → 先用带判定标准的提示词判类型
            probe = extract_lcd_vision(
                image_list, session_id=session_id, customer_text=customer_text
            )
            probe_type = _merged_display_type(probe)
            if probe_type == "LCD":
                results = probe
                metrics["vision_route"] = "lcd(auto)"
            else:
                # 判成 LED，或类型判断拿不准/失败 → 走现有 LED 提取器（行为不变）
                results = _extract_led()
                metrics["vision_route"] = (
                    "led(auto)" if probe_type == "LED" else "led(auto,type-unknown)"
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
        "Vision metrics: images=%d route=%s latency_ms=%d explicit=%s inferred=%s null=%d",
        metrics["images"], metrics.get("vision_route") or "-", metrics["vision_latency_ms"],
        sorted(explicit), sorted(inferred), metrics["fields_null"],
    )
    return results, metrics


__all__ = [
    "apply_vision_to_profile",
    "extract_vision_for_turn",
    "resolve_vision_confirmation",
]
