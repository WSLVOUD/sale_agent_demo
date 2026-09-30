"""产品域路由 + ProductPolicy 统一接口（计划 v2.9.2 §七/§十一/§十二）。

原则（计划原文）：

    "现在只把 LCD / IFP 的门留好，不提前建完整房间；LED 房间保持现状继续使用。"

所以这里**不做**任何 LED/LCD/IFP 业务判断：只把 Unified Understanding 的
`product_domain` 翻译成一个"入口名"，并把三种产品的策略接口固定下来。

    LED    → LEDPolicy      （现有链路，薄封装）
    LCD    → LCDPolicy      （接口 + Placeholder）
    IFP    → IFPPolicy      （接口 + Placeholder）
    UNKNOWN→ PRODUCT_SELECTION（问客户要哪种屏）
    MULTI  → MULTI_REQUIREMENT_ENTRY / PRODUCT_COMPARISON_ENTRY
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .turn_kind import IFP, LCD, LED, MULTI, UNKNOWN

# ── 入口名 ────────────────────────────────────────────────────────────────
LED_ENTRY = "LED_ENTRY"
LCD_ENTRY = "LCD_ENTRY"
IFP_ENTRY = "IFP_ENTRY"
PRODUCT_SELECTION = "PRODUCT_SELECTION"
MULTI_REQUIREMENT_ENTRY = "MULTI_REQUIREMENT_ENTRY"
PRODUCT_COMPARISON_ENTRY = "PRODUCT_COMPARISON_ENTRY"

ALL_ENTRIES = (
    LED_ENTRY,
    LCD_ENTRY,
    IFP_ENTRY,
    PRODUCT_SELECTION,
    MULTI_REQUIREMENT_ENTRY,
    PRODUCT_COMPARISON_ENTRY,
)

# 客户没给品类时，问这一句（计划 §十六）
PRODUCT_SELECTION_QUESTION = (
    "Are you looking for an LED display, an LCD video wall, or an interactive flat panel?"
)


def route_product_domain(product_domain: str, *, comparison: bool = False) -> str:
    """产品域 → 入口名（薄路由：只做映射，不做业务判断）。"""
    domain = str(product_domain or "").strip().upper()
    if domain == LED:
        return LED_ENTRY
    if domain == LCD:
        return LCD_ENTRY
    if domain == IFP:
        return IFP_ENTRY
    if domain == MULTI:
        return PRODUCT_COMPARISON_ENTRY if comparison else MULTI_REQUIREMENT_ENTRY
    return PRODUCT_SELECTION


# ── ProductPolicy 统一接口 ────────────────────────────────────────────────
@dataclass
class ProductPolicy:
    """三种产品共用的接口（计划 §十二）。

    LED 用现有实现；LCD / IFP 现在只定义接口（`implemented = False`），
    以后直接填 `get_missing_requirements() / get_next_question()`，不必动 Greeting。
    """

    product_domain: str = UNKNOWN
    implemented: bool = False
    notes: str = ""

    # ── 接口（默认 Placeholder：不抛异常，返回"未实现"语义）──
    def get_requirement_profile(self) -> Dict[str, Any]:
        return {}

    def get_missing_requirements(self, profile: Any = None) -> List[str]:
        return []

    def get_next_question(self, profile: Any = None) -> Optional[str]:
        return None

    def can_recommend(self, profile: Any = None) -> bool:
        return False

    def recommend(self, profile: Any = None) -> List[Dict[str, Any]]:
        return []

    def validate(self, response: str, profile: Any = None) -> List[str]:
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_domain": self.product_domain,
            "implemented": self.implemented,
            "notes": self.notes,
        }


class LEDPolicy(ProductPolicy):
    """LED：现有链路（RequirementProfile / Gate / Engine / Calculator / Reflection）。"""

    def __init__(self) -> None:
        super().__init__(
            product_domain=LED,
            implemented=True,
            notes="现有完整链路：requirement → gate → engine → RAG → calculator → reflection",
        )

    def get_missing_requirements(self, profile: Any = None) -> List[str]:
        if profile is None:
            return []
        try:
            return list(profile.missing_slots())
        except Exception:  # pragma: no cover - 防御式
            return []

    def can_recommend(self, profile: Any = None) -> bool:
        if profile is None:
            return False
        try:
            from src.rag.readiness import check_recommendation_ready

            return bool(check_recommendation_ready(profile).ready)
        except Exception:  # pragma: no cover - 防御式
            return False

    def get_next_question(self, profile: Any = None) -> Optional[str]:
        if profile is None:
            return None
        try:
            from src.agents.sales.question_planner import plan_next_question

            plan = plan_next_question(profile) or {}
            return str(plan.get("question") or "") or None
        except Exception:  # pragma: no cover - 防御式
            return None


class LCDPolicy(ProductPolicy):
    """LCD：需求链入口 —— 决策交给 `lcd_decision`（唯一入口，计划 Phase 5）。"""

    def __init__(self) -> None:
        super().__init__(
            product_domain=LCD,
            implemented=True,
            notes="需求链：lcd_decision.decide_lcd_next_action（场景分支 + 分辨率规则 + 唯一 Next Action）",
        )

    def get_requirement_profile(self, profile: Any = None) -> Dict[str, Any]:
        from .lcd_decision import decide_lcd_next_action

        if profile is None:
            # 没有档案 → 不产生业务判断（接口契约：返回空）
            return {}
        action = decide_lcd_next_action(profile)
        return action.to_dict()

    def get_missing_requirements(self, profile: Any = None) -> List[str]:
        from .lcd_decision import decide_lcd_next_action

        return list(decide_lcd_next_action(profile).missing_fields)

    def get_next_question(self, profile: Any = None) -> Optional[str]:
        from .lcd_decision import decide_lcd_next_action

        question = str(decide_lcd_next_action(profile).question or "")
        return question or None

    def can_recommend(self, profile: Any = None) -> bool:
        from .lcd_decision import decide_lcd_next_action

        return bool(decide_lcd_next_action(profile).confirmed)


class IFPPolicy(ProductPolicy):
    """IFP：LCD 子类型 —— 同一套决策，额外覆盖 Tender / OPS / Camera（计划 §十六）。"""

    def __init__(self) -> None:
        super().__init__(
            product_domain=IFP,
            implemented=True,
            notes="需求链：lcd_decision（会议/教育分支 + Tender / OPS / Camera）",
        )

    def get_requirement_profile(self, profile: Any = None) -> Dict[str, Any]:
        return LCDPolicy().get_requirement_profile(profile)

    def get_missing_requirements(self, profile: Any = None) -> List[str]:
        return LCDPolicy().get_missing_requirements(profile)

    def get_next_question(self, profile: Any = None) -> Optional[str]:
        return LCDPolicy().get_next_question(profile)

    def can_recommend(self, profile: Any = None) -> bool:
        return LCDPolicy().can_recommend(profile)


_POLICIES = {
    LED: LEDPolicy,
    LCD: LCDPolicy,
    IFP: IFPPolicy,
}


def policy_for(product_domain: str) -> ProductPolicy:
    """取产品策略（未知域给一个未实现的通用 Placeholder）。"""
    domain = str(product_domain or "").strip().upper()
    factory = _POLICIES.get(domain)
    if factory is not None:
        return factory()
    return ProductPolicy(product_domain=domain or UNKNOWN, implemented=False, notes="待实现")


__all__ = [
    "ALL_ENTRIES",
    "IFP_ENTRY",
    "LCD_ENTRY",
    "LED_ENTRY",
    "MULTI_REQUIREMENT_ENTRY",
    "PRODUCT_COMPARISON_ENTRY",
    "PRODUCT_SELECTION",
    "PRODUCT_SELECTION_QUESTION",
    "IFPPolicy",
    "LCDPolicy",
    "LEDPolicy",
    "ProductPolicy",
    "policy_for",
    "route_product_domain",
]
