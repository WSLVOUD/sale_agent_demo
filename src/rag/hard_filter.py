"""
Phase 3：硬约束 Metadata Filter。

目标（来自计划文档）：
    让明确的硬条件在 RAG 检索**之前**完成过滤，禁止出现
    "客户明确要求 outdoor → 召回 indoor → LLM 自己解释为什么也可以"。

硬约束集合（Hard Constraint）：
    display_type、indoor/outdoor、fixed/rental、客户明确指定的型号/点间距、
    客户明确要求的特殊功能（防水 / COB / HDR）

软条件（Soft Constraint，用于打分，不在此过滤）：
    观看距离、推荐点间距区间、预算、亮度偏好、产品等级、保修

用法::

    constraints = build_hard_constraints(requirement)
    results = hybrid.search(query, top_k=20,
                            filters=constraints.chroma_where(),
                            **constraints.search_kwargs())
    results = constraints.apply(results)      # 兜底：绝不让违规条目进入推荐
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PITCH_TOLERANCE = 0.5


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


@dataclass(frozen=True)
class HardConstraints:
    """不可违反的检索约束集合。"""

    display_type: Optional[str] = None
    environment: Optional[str] = None          # indoor / outdoor / semi_outdoor
    installation: Optional[str] = None         # fixed / rental
    pixel_pitch_min: Optional[float] = None
    pixel_pitch_max: Optional[float] = None
    brightness_min: Optional[int] = None
    brightness_max: Optional[int] = None
    waterproof: Optional[bool] = None
    cob: Optional[bool] = None
    hdr: Optional[bool] = None
    gob: Optional[bool] = None
    flexible: Optional[bool] = None
    price_tier: Optional[str] = None
    series_id: Optional[str] = None
    model: Optional[str] = None
    exclude_ifp: bool = False
    sources: tuple[str, ...] = field(default_factory=tuple)

    # ── 序列化 / 描述 ───────────────────────────────────────────────────
    def is_empty(self) -> bool:
        return not any([
            self.display_type, self.environment, self.installation,
            self.pixel_pitch_min, self.pixel_pitch_max,
            self.brightness_min, self.brightness_max,
            self.waterproof, self.cob, self.hdr,
            self.gob,
            self.flexible,
            self.price_tier, self.series_id, self.model, self.exclude_ifp,
        ])

    def describe(self) -> str:
        if self.is_empty():
            return "无硬约束"
        parts: List[str] = []
        if self.display_type:
            parts.append(f"display_type={self.display_type}")
        if self.environment:
            parts.append(f"environment={self.environment}")
        if self.installation:
            parts.append(f"installation={self.installation}")
        if self.pixel_pitch_min is not None or self.pixel_pitch_max is not None:
            parts.append(
                f"pitch={self.pixel_pitch_min or '-'}~{self.pixel_pitch_max or '-'}mm"
            )
        if self.brightness_min is not None:
            parts.append(f"brightness>={self.brightness_min}nit")
        if self.brightness_max is not None:
            parts.append(f"brightness<={self.brightness_max}nit")
        for name in ("waterproof", "cob", "hdr", "gob", "flexible"):
            if getattr(self, name):
                parts.append(name)
        if self.price_tier:
            parts.append(f"price_tier={self.price_tier}")
        if self.model:
            parts.append(f"model={self.model}")
        elif self.series_id:
            parts.append(f"series={self.series_id}")
        if self.exclude_ifp:
            parts.append("exclude_ifp")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and not k.startswith("_")}

    # ── 检索过滤 ────────────────────────────────────────────────────────
    def chroma_where(self) -> Dict[str, Any]:
        """Chroma / HybridSearch 的 metadata 过滤条件（相等条件）。"""
        where: Dict[str, Any] = {}
        if self.display_type:
            where["display_type"] = self.display_type
        if self.environment == "indoor":
            where["indoor"] = True
        elif self.environment == "outdoor":
            where["outdoor"] = True
        elif self.environment == "semi_outdoor":
            where["environment"] = "semi_outdoor"
        if self.installation == "fixed":
            where["is_rental"] = False
        elif self.installation == "rental":
            where["is_rental"] = True
        if self.waterproof:
            where["waterproof"] = True
        if self.cob:
            where["cob"] = True
        if self.hdr:
            where["hdr"] = True
        if self.gob:
            where["gob"] = True
        if self.flexible:
            where["flexible"] = True
        if self.price_tier:
            where["price_tier"] = self.price_tier
        if self.model:
            where["model"] = self.model
        elif self.series_id:
            where["series_id"] = self.series_id
        return where

    def search_kwargs(self) -> Dict[str, Any]:
        """HybridSearch 支持的数值区间参数。"""
        kwargs: Dict[str, Any] = {}
        if self.brightness_min is not None:
            kwargs["brightness_min"] = self.brightness_min
        if self.brightness_max is not None:
            kwargs["brightness_max"] = self.brightness_max
        if self.pixel_pitch_min is not None:
            kwargs["pitch_min"] = self.pixel_pitch_min
        if self.pixel_pitch_max is not None:
            kwargs["pitch_max"] = self.pixel_pitch_max
        return kwargs

    # ── 兜底校验 ────────────────────────────────────────────────────────
    def violations(self, items: Sequence[Any]) -> List[str]:
        """检查结果是否违反硬约束（用于事后兜底与 Phase 11 校验）。"""
        violations: List[str] = []
        for index, item in enumerate(items or [], start=1):
            meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
            label = meta.get("model") or f"#{index}"

            if self.display_type and meta.get("display_type") and meta["display_type"] != self.display_type:
                violations.append(f"[display_type] {label}: {meta['display_type']} != {self.display_type}")
            if self.environment == "indoor" and meta.get("indoor") is False:
                violations.append(f"[environment] {label}: 非室内产品")
            if self.environment == "outdoor" and meta.get("outdoor") is False:
                violations.append(f"[environment] {label}: 非户外产品")
            if self.installation and meta.get("is_rental") is not None:
                expected_rental = self.installation == "rental"
                if bool(meta["is_rental"]) != expected_rental:
                    violations.append(f"[installation] {label}: 非 {self.installation} 产品")
            pitch = _first_not_none(meta.get("pixel_pitch_mm"), meta.get("pixel_pitch_min_mm"))
            if pitch is not None:
                if self.pixel_pitch_min is not None and pitch < self.pixel_pitch_min - 1e-6:
                    violations.append(
                        f"[pitch] {label}: {pitch}mm < 下限 {self.pixel_pitch_min}mm"
                    )
                if self.pixel_pitch_max is not None and pitch > self.pixel_pitch_max + 1e-6:
                    violations.append(
                        f"[pitch] {label}: {pitch}mm > 上限 {self.pixel_pitch_max}mm"
                    )
            brightness = _first_not_none(meta.get("brightness_nit"), meta.get("brightness_max_cd"))
            if brightness is not None:
                if self.brightness_min is not None and brightness < self.brightness_min:
                    violations.append(
                        f"[brightness] {label}: {brightness}nit < {self.brightness_min}nit"
                    )
                if self.brightness_max is not None and brightness > self.brightness_max:
                    violations.append(
                        f"[brightness] {label}: {brightness}nit > {self.brightness_max}nit"
                    )
            for name in ("waterproof", "cob", "hdr", "gob", "flexible"):
                if getattr(self, name) and meta.get(name) is False:
                    violations.append(f"[{name}] {label}: 不满足 {name}")
            if self.model and meta.get("model") and meta["model"] != self.model:
                violations.append(f"[model] {label}: {meta['model']} != {self.model}")
            if self.series_id and meta.get("series_id") and meta["series_id"] != self.series_id:
                violations.append(f"[series] {label}: 非 {self.series_id}")
        return violations

    def apply(self, items: Sequence[Any]) -> List[Any]:
        """兜底过滤：丢弃任何违反硬约束的条目，并记录日志。"""
        if self.is_empty():
            return list(items or [])
        kept: List[Any] = []
        for item in items or []:
            if not self.violations([item]):
                kept.append(item)
        dropped = len(items or []) - len(kept)
        if dropped:
            logger.warning(
                "硬约束兜底过滤丢弃 %d/%d 条结果（%s）", dropped, len(items or []), self.describe()
            )
        return kept


def build_hard_constraints(
    requirement: Optional[Mapping[str, Any]] = None,
    message: str = "",
    *,
    explicit: Optional[Mapping[str, Any]] = None,
) -> HardConstraints:
    """从需求字典 / 自然语言消息构造硬约束。

    兼容两套词汇：
      - 规则提取器（ParameterInference）：indoor / outdoor / is_rental / pixel_pitch / brightness_min ...
      - Requirement Profile（Phase 6）：environment / installation / pixel_pitch_mm / brightness_min_nit ...
    """
    # Phase 6：允许直接传 RequirementProfile
    if requirement is not None and hasattr(requirement, "to_facts"):
        requirement = requirement.to_facts()  # type: ignore[assignment]
    req: Dict[str, Any] = dict(requirement or {})
    if explicit:
        if hasattr(explicit, "to_facts"):
            explicit = explicit.to_facts()  # type: ignore[assignment]
        req.update(explicit)

    sources: List[str] = []
    if requirement:
        sources.append("requirement")
    if message:
        sources.append("message")
    if explicit:
        sources.append("explicit")

    # ── display_type ──
    display_type = req.get("display_type")
    if display_type in ("BOTH", "both", "", None):
        display_type = None

    # ── environment ──
    environment = req.get("environment")
    if environment in (None, "", "unknown"):
        if req.get("outdoor") is True:
            environment = "outdoor"
        elif req.get("indoor") is True:
            environment = "indoor"
        elif req.get("semi_outdoor") is True:
            environment = "semi_outdoor"

    # ── installation ──
    installation = req.get("installation")
    if installation not in ("fixed", "rental"):
        is_rental = req.get("is_rental")
        if is_rental is True:
            installation = "rental"
        elif is_rental is False:
            installation = "fixed"
        else:
            installation = None

    # ── pixel pitch ──
    pitch_min = _first_not_none(req.get("pixel_pitch_min"), req.get("pixel_pitch_min_mm"))
    pitch_max = _first_not_none(req.get("pixel_pitch_max"), req.get("pixel_pitch_max_mm"))
    exact_pitch = _first_not_none(req.get("pixel_pitch"), req.get("pixel_pitch_mm"))
    if exact_pitch is not None:
        tolerance = float(req.get("pixel_pitch_tolerance") or DEFAULT_PITCH_TOLERANCE)
        pitch_min = float(exact_pitch) - tolerance
        pitch_max = float(exact_pitch) + tolerance
    if pitch_min is not None:
        pitch_min = float(pitch_min)
    if pitch_max is not None:
        pitch_max = float(pitch_max)

    # 注：客户没点名点间距时，"环境 + 观看距离"的业务区间（室内 P2.5/P3、
    # 室外 P4/P5/P10）由 parameter_inference 给出，并在引擎的 _violation 里
    # 作为硬约束执行（那里能同时看到 technical 参数）。
    # 这里不再单独设"室外 P6 下限"——那会把室外 P4/P5 档全部挡掉。

    # ── brightness ──
    brightness_min = _first_not_none(req.get("brightness_min"), req.get("brightness_min_nit"))
    brightness_max = _first_not_none(req.get("brightness_max"), req.get("brightness_max_nit"))
    brightness_min = int(brightness_min) if brightness_min is not None else None
    brightness_max = int(brightness_max) if brightness_max is not None else None

    return HardConstraints(
        display_type=display_type,
        environment=environment,
        installation=installation,
        pixel_pitch_min=pitch_min,
        pixel_pitch_max=pitch_max,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        waterproof=bool(req.get("waterproof")) or None,
        cob=bool(req.get("cob")) or None,
        hdr=bool(req.get("hdr")) or None,
        gob=bool(req.get("gob")) or None,
        flexible=bool(req.get("flexible")) or None,
        price_tier=req.get("price_tier"),
        series_id=_first_not_none(req.get("series_id"), req.get("series")),
        model=req.get("model"),
        exclude_ifp=bool(req.get("exclude_ifp")),
        sources=tuple(sources),
    )


def filter_products(
    items: Sequence[Any],
    constraints: HardConstraints,
) -> List[Any]:
    """便捷函数：对结果列表应用硬约束兜底过滤。"""
    return constraints.apply(items)
