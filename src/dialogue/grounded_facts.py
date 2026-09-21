"""v2.5+++（对话决策与输出链路优化 · Phase 4）：GroundedFact。

计划 §5：**不是**禁止系统推断/计算/检索/推荐出来的事实 —— 那些都是合法事实；
真正禁止的是 **LLM 自己造一个 ResponseContext 里根本没有的业务事实**。

所以客户可见的每一条业务事实都必须带来源：

    customer                  客户说的
    inferred_from_scene       由场景推断（church → indoor）
    inferred_from_pitch       由点间距推断（P2.9 → 约 5m 视距）
    inferred_from_distance    由视距推断（5m → P2.5~P3 窗口）
    calculated_from_dimensions 由尺寸算出（箱体数 / 实际尺寸 / 实际分辨率）
    retrieved_from_product_kb 产品库里的固有参数（箱体 / 模组 / 亮度 / 刷新率）
    recommended_by_engine     推荐引擎选出来的型号与点间距
    unknown                   没有来源（不允许进入客户可见事实）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

SOURCE_CUSTOMER = "customer"
SOURCE_INFERRED = "inferred"
SOURCE_CALCULATED = "calculated"
SOURCE_RETRIEVED = "retrieved"
SOURCE_RECOMMENDED = "recommended"
SOURCE_UNKNOWN = "unknown"

SOURCE_INFERRED_FROM_SCENE = "inferred_from_scene"
SOURCE_INFERRED_FROM_PITCH = "inferred_from_pitch"
SOURCE_INFERRED_FROM_DISTANCE = "inferred_from_distance"
SOURCE_CALCULATED_FROM_DIMENSIONS = "calculated_from_dimensions"
SOURCE_RETRIEVED_FROM_PRODUCT_KB = "retrieved_from_product_kb"
SOURCE_RECOMMENDED_BY_ENGINE = "recommended_by_engine"

# 合法来源（都允许 LLM 使用；只有 unknown / 空来源才禁止）
ALLOWED_SOURCES: frozenset = frozenset({
    SOURCE_CUSTOMER,
    SOURCE_INFERRED,
    SOURCE_CALCULATED,
    SOURCE_RETRIEVED,
    SOURCE_RECOMMENDED,
    SOURCE_INFERRED_FROM_SCENE,
    SOURCE_INFERRED_FROM_PITCH,
    SOURCE_INFERRED_FROM_DISTANCE,
    SOURCE_CALCULATED_FROM_DIMENSIONS,
    SOURCE_RETRIEVED_FROM_PRODUCT_KB,
    SOURCE_RECOMMENDED_BY_ENGINE,
})

# 场景推断出来的环境（客户没说 indoor/outdoor，是场景定的）
_SCENE_SOURCES = {"scenario_derived", "vision_explicit", "vision_accepted"}

# 客户口头确认过的字段（这些算 customer）
_CUSTOMER_SOURCES = {"explicit", "confirmed"}

# 需要"有依据才能说"的业务断言（计划 §12 Case 7）
CLAIM_REQUIREMENTS: Dict[str, str] = {
    "rental": "installation",
    "quick install": "installation",
    "quick to install": "installation",
    "fast install": "installation",
    "waterproof": "waterproof",
    "ip65": "waterproof",
    "ip66": "waterproof",
    "hdr": "hdr",
    "cob": "cob",
    "flexible": "flexible",
}

NUMBER_WITH_UNIT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(mm|cm|m|metres?|meters?|inch(?:es)?|in|nit|nits|kg|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)


@dataclass
class GroundedFact:
    """一条有来源的业务事实。"""

    field: str
    value: Any
    source: str
    detail: str = ""

    @property
    def grounded(self) -> bool:
        return str(self.source or "") in ALLOWED_SOURCES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value if not isinstance(self.value, (list, tuple)) else list(self.value),
            "source": self.source,
            "detail": self.detail,
        }

    def text(self) -> str:
        value = self.value
        if isinstance(value, (list, tuple)):
            value = " x ".join(str(item) for item in value)
        return f"{self.field}={value}" + (f" ({self.detail})" if self.detail else "")


def _source_for(profile: Any, field_name: str, default: str = SOURCE_CUSTOMER) -> str:
    try:
        source = str((getattr(profile, "sources", None) or {}).get(field_name) or "")
    except Exception:  # pragma: no cover - 防御式
        source = ""
    if source in _CUSTOMER_SOURCES:
        return SOURCE_CUSTOMER
    if source in _SCENE_SOURCES:
        return SOURCE_INFERRED_FROM_SCENE
    if source in ("inferred", "default"):
        return SOURCE_INFERRED
    return default


def build_grounded_facts(
    *,
    profile: Any = None,
    recommendation: Optional[Dict[str, Any]] = None,
    calculations: Optional[Dict[str, Any]] = None,
    retrieved: Optional[Dict[str, Any]] = None,
    facts: Optional[Iterable[Any]] = None,
) -> List[GroundedFact]:
    """把这一轮允许 LLM 使用的业务事实整理成带来源的清单。"""
    items: List[GroundedFact] = []
    for fact in facts or []:
        if isinstance(fact, GroundedFact):
            items.append(fact)
        elif isinstance(fact, dict):
            items.append(GroundedFact(
                field=str(fact.get("field") or ""),
                value=fact.get("value"),
                source=str(fact.get("source") or SOURCE_UNKNOWN),
                detail=str(fact.get("detail") or ""),
            ))

    if profile is not None:
        for field_name, label in (
            ("display_type", "display_type"),
            ("environment", "environment"),
            ("purpose", "purpose"),
            ("installation", "installation"),
            ("viewing_distance_m", "viewing_distance"),
            ("pixel_pitch_mm", "pixel_pitch"),
            ("budget_level", "budget_level"),
        ):
            value = getattr(profile, field_name, None)
            if value in (None, "", [], {}):
                continue
            source = _source_for(profile, field_name)
            if field_name == "environment" and source == SOURCE_INFERRED_FROM_SCENE:
                detail = "由场景推断"
            elif field_name == "viewing_distance_m" and source != SOURCE_CUSTOMER:
                source, detail = SOURCE_INFERRED_FROM_PITCH, "由点间距推断"
            elif field_name == "pixel_pitch_mm" and source != SOURCE_CUSTOMER:
                source, detail = SOURCE_INFERRED_FROM_DISTANCE, "由观看距离推断"
            else:
                detail = ""
            items.append(GroundedFact(field=label, value=value, source=source, detail=detail))
        if getattr(profile, "target_width_m", None) and getattr(profile, "target_height_m", None):
            items.append(GroundedFact(
                field="target_size",
                value=[profile.target_width_m, profile.target_height_m],
                source=_source_for(profile, "target_width_m"),
                detail="单位：米",
            ))
        requirement = getattr(profile, "resolution_requirement", None) or {}
        if isinstance(requirement, dict) and requirement.get("target_width"):
            items.append(GroundedFact(
                field="resolution_target",
                value=[requirement.get("target_width"), requirement.get("target_height")],
                source=SOURCE_CUSTOMER,
            ))

    if recommendation:
        model = str(recommendation.get("model") or "")
        if model:
            items.append(GroundedFact(
                field="recommended_model", value=model,
                source=SOURCE_RECOMMENDED_BY_ENGINE,
            ))
        pitch_resolution = recommendation.get("pitch_resolution") or {}
        if pitch_resolution.get("resolved_pitch"):
            items.append(GroundedFact(
                field="pixel_pitch",
                value=pitch_resolution.get("resolved_pitch_text")
                or pitch_resolution.get("resolved_pitch"),
                source=SOURCE_RECOMMENDED_BY_ENGINE,
                detail=str(pitch_resolution.get("pitch_resolution_reason") or ""),
            ))
        if pitch_resolution.get("requested_pitch"):
            items.append(GroundedFact(
                field="requested_pixel_pitch",
                value=pitch_resolution.get("requested_pitch_text")
                or pitch_resolution.get("requested_pitch"),
                source=SOURCE_CUSTOMER,
                detail=str(pitch_resolution.get("pitch_match_type") or ""),
            ))
        for reason in (recommendation.get("reasons") or [])[:4]:
            items.append(GroundedFact(
                field="recommendation_reason", value=str(reason),
                source=SOURCE_RECOMMENDED_BY_ENGINE,
            ))

    if calculations:
        for key, label in (
            ("actual_width_mm", "actual_screen_width_mm"),
            ("actual_height_mm", "actual_screen_height_mm"),
            ("cabinet_count", "cabinet_count"),
            ("module_count", "module_count"),
            ("columns", "cabinet_columns"),
            ("rows", "cabinet_rows"),
        ):
            value = calculations.get(key)
            if value not in (None, "", [], {}):
                items.append(GroundedFact(
                    field=label, value=value,
                    source=SOURCE_CALCULATED_FROM_DIMENSIONS,
                ))
        resolution = calculations.get("actual_resolution") or calculations.get("resolution")
        if resolution:
            items.append(GroundedFact(
                field="actual_resolution", value=resolution,
                source=SOURCE_CALCULATED_FROM_DIMENSIONS,
            ))

    for key, value in (retrieved or {}).items():
        if value in (None, "", [], {}):
            continue
        items.append(GroundedFact(
            field=str(key), value=value, source=SOURCE_RETRIEVED_FROM_PRODUCT_KB,
        ))

    return _dedupe(items)


def _dedupe(items: Sequence[GroundedFact]) -> List[GroundedFact]:
    seen: Set[str] = set()
    result: List[GroundedFact] = []
    for item in items:
        key = f"{item.field}|{item.value}|{item.source}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def fact_fields(facts: Sequence[Any]) -> Set[str]:
    return {str(getattr(fact, "field", "") or (fact or {}).get("field", "")) for fact in facts or []}


def allowed_tokens(facts: Sequence[Any], extra_texts: Iterable[str] = ()) -> Set[str]:
    """允许出现在客户可见文本里的"带数值的 token"（事实 + 客户原话）。"""
    tokens: Set[str] = set()

    def add(text: Any) -> None:
        for token in re.findall(r"\d+(?:[.,]\d+)?", str(text or "")):
            tokens.add(token.replace(",", "."))
        for token in re.findall(r"\bP\d+(?:\.\d+)?\b", str(text or ""), re.IGNORECASE):
            tokens.add(token.upper())
        for token in re.findall(r"\bTW\s*\d{2}\s*-\s*[A-Za-z0-9-]+", str(text or ""), re.IGNORECASE):
            tokens.add(token.upper().replace(" ", ""))

    for fact in facts or []:
        if isinstance(fact, dict):
            add(fact.get("value"))
            add(fact.get("detail"))
        else:
            add(getattr(fact, "value", None))
            add(getattr(fact, "detail", ""))
    for text in extra_texts:
        add(text)
    return tokens


__all__ = [
    "ALLOWED_SOURCES",
    "CLAIM_REQUIREMENTS",
    "GroundedFact",
    "NUMBER_WITH_UNIT_RE",
    "SOURCE_CALCULATED",
    "SOURCE_CALCULATED_FROM_DIMENSIONS",
    "SOURCE_CUSTOMER",
    "SOURCE_INFERRED",
    "SOURCE_INFERRED_FROM_DISTANCE",
    "SOURCE_INFERRED_FROM_PITCH",
    "SOURCE_INFERRED_FROM_SCENE",
    "SOURCE_RECOMMENDED",
    "SOURCE_RECOMMENDED_BY_ENGINE",
    "SOURCE_RETRIEVED",
    "SOURCE_RETRIEVED_FROM_PRODUCT_KB",
    "SOURCE_UNKNOWN",
    "allowed_tokens",
    "build_grounded_facts",
    "fact_fields",
]
