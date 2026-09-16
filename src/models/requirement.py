"""
Phase 6：Requirement Profile（结构化客户需求档案）。

设计目标（来自计划文档「六、Phase 6」）：
    把"客户语言"沉淀成**结构化事实**，让后续所有环节（技术参数推断、硬约束过滤、
    推荐打分、工程计算、Reflection 校验）都读同一份档案，而不是各自解析一遍 Query。

关键约定：
  - 只存**事实**，不存推断出来的工程参数结果（工程参数由 Phase 5 的 Python 规则产出）
  - 记录每个字段的来源（explicit = 客户明确说出；inferred = 规则/上下文推断）
  - 合并时"客户明确值"优先，弱来源不得覆盖强来源
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DisplayType = Literal["LED", "LCD", "IFP"]
Environment = Literal["indoor", "outdoor", "semi_outdoor"]
Installation = Literal["fixed", "rental"]
BudgetLevel = Literal["low", "medium", "high"]

# 需求采集顺序（Phase 7 的追问顺序，也是 missing_slots 的排序依据）
SLOT_ORDER: tuple[str, ...] = (
    "display_type",
    "environment",
    "purpose",
    "installation",
    "viewing_distance_m",
    "target_size",
    "budget_level",
    "special_requirements",
)

# 判定"信息是否足够进入推荐"的最小集合
REQUIRED_FOR_RECOMMENDATION: tuple[str, ...] = ("environment", "purpose")

# 槽位名（对话/追问用）↔ 档案字段名（Phase 1 / Phase 17）
SLOT_TO_FIELD: Dict[str, str] = {
    "display_type": "display_type",
    "environment": "environment",
    "purpose": "purpose",
    "installation": "installation",
    "viewing_distance": "viewing_distance_m",
    "size": "target_size",          # 派生属性（宽+高）
    "width": "target_width_m",
    "height": "target_height_m",
    "pixel_pitch": "pixel_pitch_mm",
    "brightness": "brightness_min_nit",
    "budget": "budget_level",
}

# 同一字段最多主动询问次数（Phase 5：超过就不再问）
MAX_ASKS_PER_SLOT = 2

# 槽位优先级（Phase 17）：HIGH 先问，LOW 最后
SLOT_PRIORITY: Dict[str, str] = {
    "environment": "HIGH",
    "purpose": "HIGH",
    "installation": "HIGH",
    "size": "HIGH",
    "display_type": "HIGH",
    "pixel_pitch": "MEDIUM",
    "viewing_distance": "MEDIUM",
    "brightness": "MEDIUM",
    "width": "MEDIUM",
    "height": "MEDIUM",
    "budget": "LOW",
}

# 来源强度（M2 四态）：合并时强度高者胜，同强度时新值覆盖旧值。
#   explicit/confirmed（客户明说） > vision_explicit（图片明确可见）
#   > scenario_derived（场景直接判定） > vision_inferred（图片推测）
#   > inferred（算法估算） > default（系统默认）
_EXPLICIT_STRENGTH = {
    "explicit": 7,
    "confirmed": 7,
    "vision_explicit": 5,
    "scenario_derived": 4,
    "vision_inferred": 3,
    "inferred": 2,
    "default": 1,
}

# 能算作"客户确认"的来源（可用于打开 Ready Gate）
CONFIRMED_SOURCES: frozenset[str] = frozenset({"explicit", "confirmed", "scenario_derived"})

# 图片"明确可见"（vision_explicit）能作为选型依据的字段。
# 这几项图片确实能看出来（明显的室内会议室 / 明显的 LED 屏 / 明显的租赁箱体），
# 所以 Gate 不必再问一遍；但它**不是客户确认**，status 仍报 inferred。
# 观看距离 / 尺寸 / 亮度等图片推不准的字段不在此列 —— 那些继续问客户。
VISION_TRUSTED_FIELDS: frozenset[str] = frozenset(
    {"display_type", "environment", "purpose", "installation"}
)

# 档案字段 ↔ 槽位键的对应关系（用于判定"客户是否明确说出该字段"）
_SLOT_ALIASES: Dict[str, tuple[str, ...]] = {
    "display_type": ("display_type",),
    "environment": ("environment", "indoor", "outdoor", "semi_outdoor"),
    "purpose": ("purpose",),
    "installation": ("installation", "is_rental"),
    "viewing_distance_m": ("viewing_distance_m", "viewing_distance", "distance"),
    "target_width_m": ("target_width_mm", "target_width", "size"),
    "target_height_m": ("target_height_mm", "target_height", "size"),
    "screen_size_hint_mm": ("screen_size_hint_mm",),
    "size_axis": ("size_axis",),
    "budget_level": ("budget_level",),
    "pixel_pitch_mm": ("pixel_pitch_mm", "pixel_pitch"),
    "brightness_min_nit": ("brightness_min", "brightness_min_nit"),
    "brightness_max_nit": ("brightness_max", "brightness_max_nit"),
    "series_id": ("series_id",),
    "model": ("model",),
}


def _is_explicit_field(field: str, explicit: set[str]) -> bool:
    return any(alias in explicit for alias in _SLOT_ALIASES.get(field, (field,)))


# 旧结构里被规则估算的字段名 → 档案字段名（用于把 inferred 溯源准确落到档案上）
_LEGACY_INFERRED_TO_FIELD: Dict[str, str] = {
    "viewing_distance": "viewing_distance_m",
    "distance": "viewing_distance_m",
    "size": "size",
    "brightness": "brightness_min_nit",
    "resolution": "resolution",
}


# Sales Agent 的 usage 取值 → Phase 4 规范 purpose token
_USAGE_ALIASES: Dict[str, str] = {
    "会议室": "conference", "会议": "conference",
    "教室": "classroom", "培训": "classroom",
    "商场": "retail", "商业零售": "retail", "店铺": "retail",
    "展厅": "showroom", "展览": "showroom", "展会": "exhibition",
    "医疗": "hospital",
    "监控指挥": "control_room", "监控": "control_room",
    "舞台演出": "stage", "舞台": "stage", "演出": "stage", "演唱会": "concert",
    "体育场馆": "stadium", "体育": "stadium",
    "广告": "advertising", "广告传媒": "advertising", "建筑幕墙": "advertising", "幕墙": "advertising",
    "租赁活动": "rental", "租赁": "rental",
    "机场": "airport", "酒店": "hotel", "银行": "bank", "餐厅": "restaurant",
    "办公室": "office", "博物馆": "museum", "大厅": "hall",
}


class RequirementProfile(BaseModel):
    """结构化客户需求档案。"""

    # ── 硬条件事实 ──────────────────────────────────────────────────────
    display_type: Optional[DisplayType] = None
    environment: Optional[Environment] = None
    purpose: Optional[str] = None
    installation: Optional[Installation] = None

    # ── 尺寸 / 视距事实 ─────────────────────────────────────────────────
    viewing_distance_m: Optional[float] = Field(None, gt=0, le=200)
    target_width_m: Optional[float] = Field(None, gt=0, le=200)
    target_height_m: Optional[float] = Field(None, gt=0, le=200)
    # 客户只报了一个长度（如 "129,2cm"）时的线索：是宽是高等客户确认，不替他猜
    screen_size_hint_mm: Optional[float] = Field(None, gt=0, le=200000)
    size_axis: Optional[str] = None

    # ── 软条件事实 ──────────────────────────────────────────────────────
    budget_level: Optional[BudgetLevel] = None
    pixel_pitch_mm: Optional[float] = Field(None, gt=0, le=20)
    brightness_min_nit: Optional[int] = Field(None, ge=1, le=20000)
    brightness_max_nit: Optional[int] = Field(None, ge=1, le=20000)
    special_requirements: List[str] = Field(default_factory=list)

    # ── 明确点名的产品 ──────────────────────────────────────────────────
    series_id: Optional[str] = None
    model: Optional[str] = None

    # ── 来源标记 ────────────────────────────────────────────────────────
    sources: Dict[str, str] = Field(default_factory=dict)

    # ── 冲突（Phase 18）：客户明确说的与规则/语义推断互相矛盾时记录 ──────
    # 有冲突时 Recommendation Ready Gate 一律不放行
    conflicts: List[str] = Field(default_factory=list)
    # 冲突对应的字段名（用于"就问那一项"，例如图片说室内、客户说室外）
    conflict_slots: List[str] = Field(default_factory=list)

    # ── Unknown 容错（《全量需求捕获与 Unknown 容错优化》Phase 1）──────────
    # 字段级"已主动询问次数"：slot -> 次数（最多问到 2 次）
    ask_counts: Dict[str, int] = Field(default_factory=dict)
    # 字段级 unknown 原因：slot -> customer_does_not_know / customer_skip
    unknown_reasons: Dict[str, str] = Field(default_factory=dict)
    # 上一轮主动问的是哪个槽位（客户答非所问时要能对上"这一项我没答"）
    last_asked_slot: str = ""

    # ── 视觉需求（《智谱视觉需求提取接入实施计划》第七/十八/二十阶段）─────
    # 图片给出的尺寸只作为**提示**：[宽度mm, 高度mm]。用来追问客户确认，
    # 绝不写进 target_width_m / target_height_m，也就不可能进入箱体计算。
    vision_size_hint_mm: Optional[List[float]] = None
    # 图片给出的点间距 / 亮度 / 其它说明（提示与排查用，不参与 Gate 放行）
    vision_notes: List[str] = Field(default_factory=list)

    @field_validator("special_requirements", mode="before")
    @classmethod
    def _coerce_special(cls, value):
        if value in (None, ""):
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    # ── 构造 ────────────────────────────────────────────────────────────
    @classmethod
    def from_slots(
        cls,
        slots: Dict[str, Any],
        explicit_keys: Iterable[str] | None = None,
    ) -> "RequirementProfile":
        """从 Query Understanding 的槽位构造档案。

        Args:
            slots: ``extract_slots()`` 的输出
            explicit_keys: 客户在本轮**明确说出**的槽位；其余标记为 inferred
        """
        slots = dict(slots or {})
        # ── M2：四态来源 ────────────────────────────────────────────────────
        #   explicit / confirmed   客户明确说过的
        #   scenario_derived       客户原话场景直接判定（会议室→室内）
        #   default                系统业务默认值（场景默认固装）
        #   inferred               算法估算（不能放行 Gate）
        scenario_derived = {str(x) for x in (slots.get("_scenario_derived") or [])}
        default_slots = {str(x) for x in (slots.get("_default_slots") or [])}
        explicit = (
            set(explicit_keys or ())
            - {str(x) for x in (slots.get("_inferred_slots") or [])}
            - scenario_derived
            - default_slots
        )

        data: Dict[str, Any] = {}
        sources: Dict[str, str] = {}

        def put(key: str, value: Any):
            if value in (None, "", [], {}):
                return
            data[key] = value
            # 来源优先级与 _EXPLICIT_STRENGTH 一致：explicit > scenario_derived > default
            if _is_explicit_field(key, explicit):
                sources[key] = "explicit"
            elif _is_explicit_field(key, scenario_derived):
                sources[key] = "scenario_derived"
            elif _is_explicit_field(key, default_slots):
                sources[key] = "default"
            else:
                sources[key] = "inferred"

        put("display_type", slots.get("display_type"))
        put("environment", slots.get("environment"))
        put("purpose", slots.get("purpose"))
        put("installation", slots.get("installation"))
        put("viewing_distance_m", _coerce_distance(slots.get("viewing_distance_m") or slots.get("distance")))
        put("budget_level", _normalize_budget(slots.get("budget_level")))
        put("pixel_pitch_mm", slots.get("pixel_pitch_mm") or slots.get("pixel_pitch"))
        put("brightness_min_nit", slots.get("brightness_min") or slots.get("brightness_min_nit"))
        put("brightness_max_nit", slots.get("brightness_max") or slots.get("brightness_max_nit"))
        put("series_id", slots.get("series_id"))
        put("model", slots.get("model"))

        # 目标尺寸：槽位用 mm，档案用米
        width_mm = slots.get("target_width_mm") or slots.get("target_width")
        height_mm = slots.get("target_height_mm") or slots.get("target_height")
        if width_mm:
            put("target_width_m", _to_meters(width_mm))
        if height_mm:
            put("target_height_m", _to_meters(height_mm))

        # 只报了一个长度的线索 + 客户指认的方向（宽 / 高 / 对角线）
        put("screen_size_hint_mm", slots.get("screen_size_hint_mm"))
        put("size_axis", slots.get("size_axis"))

        specials = _collect_specials(slots)
        if specials:
            data["special_requirements"] = specials
            sources["special_requirements"] = "explicit"

        return cls(**data, sources=sources)

    # ── 派生属性 ────────────────────────────────────────────────────────
    @classmethod
    def from_legacy(cls, requirement: Dict[str, Any] | None) -> "RequirementProfile":
        """兼容旧键名（indoor / outdoor / is_rental / distance / pixel_pitch …）。

        Sales Agent 与 Solution Agent 的 ``understand`` 节点至今仍产出旧键名，
        这里做一次统一映射，避免在两处各写一套转换。

        溯源：旧结构里的 ``_inferred_slots`` 记录"规则估算出来的字段"，
        这些字段会被标记为 inferred，从而不参与 Recommendation Ready Gate。
        """
        req = dict(requirement or {})
        inferred_slots = {str(item) for item in (req.get("_inferred_slots") or [])}
        # 把旧字段名（viewing_distance / distance / size …）换算成档案字段名，
        # 这样从 slots 里"扣掉"的才是同一个键，溯源才真正生效。
        inferred_fields = {
            _LEGACY_INFERRED_TO_FIELD.get(name, name) for name in inferred_slots
        }
        inferred_slot_keys = set(inferred_fields)
        slots: Dict[str, Any] = {}

        if req.get("environment") in ("indoor", "outdoor", "semi_outdoor"):
            slots["environment"] = req["environment"]
        elif req.get("semi_outdoor"):
            slots["environment"] = "semi_outdoor"
        elif req.get("outdoor") is True or str(req.get("outdoor")).lower() == "true":
            slots["environment"] = "outdoor"
        elif req.get("indoor") is True or str(req.get("indoor")).lower() == "true":
            slots["environment"] = "indoor"
        else:
            # Sales Agent 使用 location_type（"室内" / "室外"）
            location = str(req.get("location_type") or "").strip().lower()
            if location:
                if any(kw in location for kw in ("室外", "户外", "露天", "outdoor")):
                    slots["environment"] = "outdoor"
                elif any(kw in location for kw in ("室内", "户内", "indoor")):
                    slots["environment"] = "indoor"

        if req.get("installation") in ("fixed", "rental"):
            slots["installation"] = req["installation"]
        elif req.get("is_rental") is not None:
            slots["installation"] = "rental" if req["is_rental"] else "fixed"

        if req.get("display_type") in ("LED", "LCD", "IFP"):
            slots["display_type"] = req["display_type"]
        purpose_raw = req.get("purpose") or req.get("usage")
        if purpose_raw:
            # 场景名统一成 Phase 4 的规范 token（会议室 → conference）
            purpose_text = str(purpose_raw)
            try:
                from src.rag.query_understanding import _detect_purpose

                canonical = _detect_purpose(purpose_text.lower())
                if not canonical:
                    canonical = next(
                        (v for k, v in _USAGE_ALIASES.items() if k in purpose_text), None
                    )
                slots["purpose"] = canonical or purpose_text
            except Exception:  # pragma: no cover - 防御式
                slots["purpose"] = purpose_text
        if req.get("pixel_pitch") is not None:
            slots["pixel_pitch"] = req["pixel_pitch"]
        if req.get("brightness_min") is not None:
            slots["brightness_min"] = req["brightness_min"]
        if req.get("series_id"):
            slots["series_id"] = req["series_id"]
        if req.get("model"):
            slots["model"] = req["model"]

        # 视距：既可能是数字（米），也可能是 "4米" / "4m" 文本
        distance = req.get(
            "viewing_distance_m",
            req.get("viewing_distance", req.get("distance")),
        )
        if distance not in (None, ""):
            slots["viewing_distance_m"] = distance

        # 只报了一个长度的线索 + 客户指认的方向（跨轮传递）
        if req.get("screen_size_hint_mm") is not None:
            slots["screen_size_hint_mm"] = req["screen_size_hint_mm"]
        if req.get("size_axis"):
            slots["size_axis"] = req["size_axis"]

        # 环境缺失时用场景推断（会议室/教室 → 室内，体育场/广告 → 室外）
        if "environment" not in slots and slots.get("purpose"):
            try:
                from src.rag.query_understanding import environment_from_purpose

                # 只有"一眼室内/室外"的场景（会议室、教堂、户外广告、体育场…）
                # 才会推出环境；这类场景直接算已确定，不再追问室内外。
                # 舞台 / 演唱会 / 租赁室内外都可能 → 推不出，仍由 Gate 询问。
                derived = environment_from_purpose(slots["purpose"])
                if derived:
                    slots["environment"] = derived
            except Exception:  # pragma: no cover - 防御式
                pass

        return cls.from_slots(slots, explicit_keys=set(slots) - inferred_fields)

    @property
    def target_width_mm(self) -> Optional[float]:
        return self.target_width_m * 1000 if self.target_width_m else None

    @property
    def target_height_mm(self) -> Optional[float]:
        return self.target_height_m * 1000 if self.target_height_m else None

    @property
    def has_target_size(self) -> bool:
        return bool(self.target_width_m or self.target_height_m)

    def to_facts(self) -> Dict[str, Any]:
        """转成 Phase 5 ``infer_technical_parameters`` 需要的事实字典。"""
        facts: Dict[str, Any] = {
            "display_type": self.display_type,
            "purpose": self.purpose,
        }
        if self.environment:
            facts["environment"] = self.environment
            facts["indoor"] = self.environment == "indoor"
            facts["outdoor"] = self.environment in ("outdoor", "semi_outdoor")
        if self.installation:
            facts["installation"] = self.installation
            facts["is_rental"] = self.installation == "rental"
        if self.viewing_distance_m is not None:
            facts["viewing_distance_m"] = self.viewing_distance_m
        if self.target_width_mm is not None:
            facts["target_width_mm"] = self.target_width_mm
        if self.target_height_mm is not None:
            facts["target_height_mm"] = self.target_height_mm
        if self.pixel_pitch_mm is not None:
            facts["pixel_pitch_mm"] = self.pixel_pitch_mm
        if self.brightness_min_nit is not None:
            facts["brightness_min"] = self.brightness_min_nit
        if self.brightness_max_nit is not None:
            facts["brightness_max"] = self.brightness_max_nit
        for name, slot in (
            ("waterproof", "waterproof"),
            ("cob", "cob"),
            ("hdr", "hdr"),
            ("gob", "gob"),
            ("flexible", "flexible"),
        ):
            if name in self.special_requirements:
                facts[slot] = True
        if self.series_id:
            facts["series_id"] = self.series_id
        if self.model:
            facts["model"] = self.model
        return {k: v for k, v in facts.items() if v is not None}

    def to_slots(self) -> Dict[str, Any]:
        """转回 Query Understanding 的槽位字典（保留 mm 尺寸键）。"""
        slots = {k: v for k, v in self.model_dump().items() if v not in (None, "", [], {})}
        slots.pop("sources", None)
        if self.target_width_mm:
            slots["target_width_mm"] = self.target_width_mm
        if self.target_height_mm:
            slots["target_height_mm"] = self.target_height_mm
        return slots

    # ── 完整性 / 合并 ───────────────────────────────────────────────────
    def missing_slots(self) -> List[str]:
        """按采集顺序返回仍缺失的槽位。"""
        missing: List[str] = []
        for slot in SLOT_ORDER:
            if slot == "target_size":
                if not self.has_target_size:
                    missing.append(slot)
            elif slot == "special_requirements":
                continue  # 特殊要求是可选加分项，不参与"缺失"判定
            elif getattr(self, slot, None) in (None, [], ""):
                missing.append(slot)
        return missing

    def is_sufficient(self) -> bool:
        """是否具备进入推荐的最小信息量。"""
        return all(getattr(self, slot, None) not in (None, [], "") for slot in REQUIRED_FOR_RECOMMENDATION)

    def is_recommendation_ready(self) -> bool:
        """是否具备可靠选型条件（v2.0 Recommendation Ready Gate）。

        判定逻辑集中在 ``src/rag/readiness.py``，此处仅做委托：
          - 客户点名型号 / 系列                → True
          - 客户明确给出点间距或亮度            → True（v2.0 Case 4）
          - 室内外 + 场景 + （安装方式 或 观看距离）→ True（v2.0 Case 3）
          - 其余                                → False
        """
        from src.rag.readiness import check_recommendation_ready

        return check_recommendation_ready(self).ready

    # ── v2.0 Phase 2：来源三态与置信度 ──────────────────────────────────
    @property
    def status(self) -> Dict[str, str]:
        """每个字段的来源三态：confirmed / inferred / unknown（v2.0 口径）。"""
        result: Dict[str, str] = {}
        for field_name in _SLOT_ALIASES:
            value = getattr(self, field_name, None)
            if field_name == "target_width_m":
                value = self.target_width_mm
            elif field_name == "target_height_m":
                value = self.target_height_mm
            if value in (None, "", [], {}):
                result[field_name] = "unknown"
                continue
            source = self.sources.get(field_name, "inferred")
            # scenario_derived（客户原话场景直接判定）与客户明说一样算"已确认"；
            # default（系统默认）与 inferred（算法估算）都只能算"未确认"。
            result[field_name] = "confirmed" if source in CONFIRMED_SOURCES else "inferred"
        return result

    @property
    def confidence(self) -> Dict[str, float]:
        """来源三态对应的置信度（confirmed=1.0 / inferred=0.7 / unknown=0.0）。"""
        weights = {"confirmed": 1.0, "inferred": 0.7, "unknown": 0.0}
        return {key: weights[value] for key, value in self.status.items()}

    def unknown_fields(self) -> List[str]:
        """仍未知的字段列表（v2.0 Phase 2 要求的 unknown 语义）。"""
        return [key for key, value in self.status.items() if value == "unknown"]

    def describe(self) -> str:
        """输出诊断用的结构化快照（confirmed / inferred / unknown）。"""
        status = self.status
        lines = ["========== REQUIREMENT PROFILE =========="]
        labels = {
            "display_type": "display_type",
            "environment": "environment",
            "purpose": "scene",
            "installation": "installation",
            "viewing_distance_m": "viewing_distance",
            "pixel_pitch_mm": "pitch",
            "brightness_min_nit": "brightness",
            "target_width_m": "width",
            "target_height_m": "height",
            "series_id": "series",
            "model": "model",
        }
        values = {
            "display_type": self.display_type,
            "environment": self.environment,
            "purpose": self.purpose,
            "installation": self.installation,
            "viewing_distance_m": self.viewing_distance_m,
            "pixel_pitch_mm": self.pixel_pitch_mm,
            "brightness_min_nit": self.brightness_min_nit,
            "target_width_m": self.target_width_m,
            "target_height_m": self.target_height_m,
            "series_id": self.series_id,
            "model": self.model,
        }
        for field, label in labels.items():
            value = values.get(field)
            source = status.get(field, "unknown")
            if value in (None, "", [], {}):
                lines.append(f"{label}: None [{source}]")
            else:
                lines.append(f"{label}: {value} [{source}]")
        lines.append("=========================================")
        return "\n".join(lines)

    def as_v2_dict(self) -> Dict[str, Any]:
        """按 v2.0 文档的字段命名输出档案（对外契约用）。

        v2.0 命名：``display_type / environment / purpose / installation /
        viewing_distance / budget / pixel_pitch / brightness / width / height /
        special_requirements`` + 每个字段的 ``status`` 与 ``confidence``。
        """
        status = self.status
        confidence = self.confidence
        payload: Dict[str, Any] = {
            "display_type": self.display_type,
            "environment": self.environment,
            "purpose": self.purpose,
            "installation": self.installation,
            "viewing_distance": self.viewing_distance_m,
            "budget": self.budget_level,
            "pixel_pitch": self.pixel_pitch_mm,
            "brightness": self.brightness_min_nit,
            "width": self.target_width_m,
            "height": self.target_height_m,
            "special_requirements": list(self.special_requirements),
            "model": self.model,
            "series_id": self.series_id,
            "status": status,
            "confidence": confidence,
            "unknown": self.unknown_fields(),
        }
        return payload

    def completeness(self) -> float:
        """档案完整度（0~1），用于评估与埋点。"""
        tracked = [s for s in SLOT_ORDER if s != "special_requirements"]
        filled = 0
        for slot in tracked:
            if slot == "target_size":
                filled += 1 if self.has_target_size else 0
            elif getattr(self, slot, None) not in (None, [], ""):
                filled += 1
        return round(filled / len(tracked), 4) if tracked else 0.0

    # ── Unknown 容错：字段级状态机（Phase 1 / 5 / 10 / 17）────────────────
    def slot_value_present(self, slot: str) -> bool:
        """该槽位是否已经拿到了值（不管是客户说的还是推断的）。"""
        if slot == "size":
            return self.has_target_size
        if slot == "viewing_distance":
            return self.viewing_distance_m is not None
        field = SLOT_TO_FIELD.get(slot, slot)
        return getattr(self, field, None) not in (None, "", [], {})

    def slot_source(self, slot: str) -> str:
        """该槽位的来源（explicit/confirmed/scenario_derived/default/inferred）。"""
        if slot == "size":
            return self.sources.get("target_width_m") or self.sources.get("target_height_m") or "unknown"
        field = SLOT_TO_FIELD.get(slot, slot)
        return self.sources.get(field, "unknown")

    def slot_is_confirmed(self, slot: str) -> bool:
        """该槽位是否是"客户侧确认过"的（客户明说 / 场景直接判定）。

        注意与 ``slot_value_present`` 的区别：场景默认值（例如"教堂默认固装"）
        也算"有值"，但**不算客户确认** —— 客户说"我不知道"时它仍然要能
        走 unknown 流程，否则会出现同一问题被无限追问。
        """
        if not self.slot_value_present(slot):
            return False
        return self.slot_source(slot) in CONFIRMED_SOURCES

    def ask_count(self, slot: str) -> int:
        return int(self.ask_counts.get(slot, 0) or 0)

    def record_ask(self, slot: str) -> int:
        """记录"又问了客户一次"，返回累计次数。"""
        self.ask_counts[slot] = self.ask_count(slot) + 1
        return self.ask_counts[slot]

    def mark_unknown(self, slot: str, reason: str = "customer_does_not_know") -> None:
        """把槽位标记为 unknown（客户不知道 / 明确跳过）。"""
        self.unknown_reasons[slot] = reason
        # 明确跳过时不用问第二次
        if reason == "customer_skip":
            self.ask_counts[slot] = max(self.ask_count(slot), MAX_ASKS_PER_SLOT)

    def is_unknown(self, slot: str) -> bool:
        if self.slot_is_confirmed(slot):
            return False   # Phase 10：客户后来补上了 → unknown 自动解除
        # 客户明确"跳过这一项" → 不用再问第二次，直接算 unknown
        if self.unknown_reasons.get(slot) == "customer_skip":
            return True
        # 客户第一次答"不知道"只是 unknown_pending：还允许按"降低门槛"的方式再问一次，
        # 只有问满 MAX_ASKS_PER_SLOT 次仍无值，才真正锁定为 unknown。
        return self.ask_count(slot) >= MAX_ASKS_PER_SLOT

    def unknown_slots(self) -> List[str]:
        """已经"问过两次/客户明确跳过"且仍然没有值的槽位。"""
        return [slot for slot in SLOT_TO_FIELD if self.ask_count(slot) >= 1 and self.is_unknown(slot)]

    def exhausted_slots(self) -> List[str]:
        """问满两次仍然没有值的槽位（Gate 允许降级推荐）。"""
        return [
            slot for slot in SLOT_TO_FIELD
            if not self.slot_is_confirmed(slot) and self.ask_count(slot) >= MAX_ASKS_PER_SLOT
        ]

    def pending_unknown_slots(self) -> List[str]:
        """已经问过一次、但客户还没给出值也没有明确说不知道的槽位（可以再问一次）。"""
        return [
            slot for slot in SLOT_TO_FIELD
            if not self.slot_is_confirmed(slot)
            and self.ask_count(slot) == 1
            and slot not in self.unknown_reasons
        ]

    def slot_status(self, slot: str) -> str:
        """槽位状态：confirmed / inferred / unknown / unknown_pending / missing。"""
        if self.is_unknown(slot):
            return "unknown"
        if self.slot_value_present(slot):
            source = self.slot_source(slot)
            return "confirmed" if source in CONFIRMED_SOURCES else "inferred"
        if self.ask_count(slot) >= 1:
            return "unknown_pending"
        return "missing"

    def requirement_basis(self) -> Dict[str, List[str]]:
        """Phase 14：推荐依据（哪些字段是客户确认的 / 推断的 / 客户不知道的）。

        派生槽位（``size`` 由宽+高合成）不单独列出，避免与 width/height 重复。
        """
        confirmed: List[str] = []
        inferred: List[str] = []
        unknown: List[str] = []
        for slot in SLOT_TO_FIELD:
            if slot == "size":
                continue
            if self.is_unknown(slot):
                unknown.append(slot)
            elif self.slot_value_present(slot):
                if self.slot_source(slot) in CONFIRMED_SOURCES:
                    confirmed.append(slot)
                else:
                    inferred.append(slot)
        return {"confirmed": confirmed, "inferred": inferred, "unknown": unknown}

    def merge(self, other: "RequirementProfile | Dict[str, Any] | None") -> "RequirementProfile":
        """合并新事实：强来源覆盖弱来源，同强度时新值覆盖旧值。"""
        if other is None:
            return self
        incoming = other if isinstance(other, RequirementProfile) else RequirementProfile.from_slots(other)

        data = self.model_dump()
        sources = dict(self.sources)
        for key, value in incoming.model_dump().items():
            if key == "sources":
                continue
            if value in (None, "", [], {}):
                continue
            current = data.get(key)
            current_strength = _EXPLICIT_STRENGTH.get(sources.get(key, "default"), 1)
            incoming_strength = _EXPLICIT_STRENGTH.get(incoming.sources.get(key, "default"), 1)

            if key in ("special_requirements", "vision_notes"):
                merged = list(dict.fromkeys(list(current or []) + list(value)))
                data[key] = merged
                sources[key] = max(
                    (sources.get(key, "default"), incoming.sources.get(key, "default")),
                    key=lambda s: _EXPLICIT_STRENGTH.get(s, 1),
                )
                continue

            if current in (None, "", [], {}) or incoming_strength >= current_strength:
                data[key] = value
                sources[key] = incoming.sources.get(key, "inferred")

        data["sources"] = sources
        # 冲突合并：两边的冲突都保留（谁检测到都算数）；
        # 真正消解冲突后由 Extractor 用「重新检测」的结果覆盖（见 requirement_extractor）
        data["conflicts"] = list(
            dict.fromkeys(list(self.conflicts or []) + list(incoming.conflicts or []))
        )
        data["conflict_slots"] = list(
            dict.fromkeys(list(self.conflict_slots or []) + list(incoming.conflict_slots or []))
        )
        # ── Unknown 容错状态随档案一起合并（Phase 1 / 10）──────────────────
        ask_counts = dict(self.ask_counts or {})
        for slot, count in (incoming.ask_counts or {}).items():
            ask_counts[slot] = max(int(ask_counts.get(slot, 0) or 0), int(count or 0))
        data["ask_counts"] = ask_counts

        unknown_reasons = dict(self.unknown_reasons or {})
        unknown_reasons.update(incoming.unknown_reasons or {})
        merged_profile = RequirementProfile(**data)
        # 客户后来明确补上了 → 撤销 unknown 标记（场景默认值不算"补上"）
        unknown_reasons = {
            slot: reason for slot, reason in unknown_reasons.items()
            if not merged_profile.slot_is_confirmed(slot)
        }
        merged_profile.unknown_reasons = unknown_reasons
        return merged_profile


# ── 辅助 ────────────────────────────────────────────────────────────────────
def _to_meters(value: Any) -> Optional[float]:
    """槽位里的尺寸是毫米；档案用米。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number / 1000 if number > 100 else number


def _coerce_distance(value: Any) -> Optional[float]:
    """观看距离统一成米（支持 "4米" / "4m" / "4000mm" / 4）。"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000 if number > 100 else (number or None)
    text = str(value).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = float(match.group(1))
    if "mm" in text or "毫米" in text or number > 100:
        number = number / 1000
    return number or None


def _normalize_budget(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    mapping = {
        "low": "low", "cheap": "low", "budget": "low", "economical": "low", "低": "low",
        "mid": "medium", "medium": "medium", "middle": "medium", "中": "medium",
        "high": "high", "premium": "high", "flagship": "high", "高": "high",
    }
    return mapping.get(text, text if text in ("low", "medium", "high") else None)


def _collect_specials(slots: Dict[str, Any]) -> List[str]:
    specials: List[str] = []
    for name in ("waterproof", "cob", "hdr", "gob", "flexible", "interaction"):
        if slots.get(name):
            specials.append(name)
    return specials


def merge_profiles(
    base: RequirementProfile | None,
    slots: Dict[str, Any],
    explicit_keys: Iterable[str] | None = None,
) -> RequirementProfile:
    """便捷函数：把新槽位并入已有档案。"""
    incoming = RequirementProfile.from_slots(slots, explicit_keys=explicit_keys)
    if base is None:
        return incoming
    return base.merge(incoming)


__all__ = [
    "REQUIRED_FOR_RECOMMENDATION",
    "SLOT_ORDER",
    "RequirementProfile",
    "merge_profiles",
]
