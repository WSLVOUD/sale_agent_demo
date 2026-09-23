"""统一回合理解 + 多产品需求簿（计划 v2.9.2 §四 / §十三~§十七）。

一个客户回合理解成：

    conversation_type + intent + product_domain + business_signal
    + requirement_updates + active_requirement + switch_product
    + multi_product + comparison

需求本身统一成 `Requirement`（一个产品需求一条），而不是给 LCD/IFP 另建三套 Profile：

    Requirement(product_domain=LED, status=active,  slots=…)
    Requirement(product_domain=LCD, status=pending, slots=…)

注意（计划 §十三）：这里**只做登记与切换**，不实现任何产品专属的需求问题顺序。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .turn_kind import (
    ALL_PRODUCT_DOMAINS,
    CONVERSATION_WITH_BUSINESS_SIGNAL,
    FREE_QUESTION,
    MULTI,
    PURE_CONVERSATION,
    REQUIREMENT_COLLECTION,
    UNKNOWN,
    business_signals,
    classify_turn_kind,
    detect_product_domain,
    has_greeting_opener,
)

STATUS_ACTIVE = "active"
STATUS_PENDING = "pending"


@dataclass
class Requirement:
    """一个产品需求（计划 §十三）。"""

    product_domain: str = UNKNOWN
    status: str = STATUS_PENDING
    slots: Dict[str, Any] = field(default_factory=dict)

    def update(self, slots: Dict[str, Any]) -> None:
        for key, value in (slots or {}).items():
            if value not in (None, "", [], {}):
                self.slots[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_domain": self.product_domain,
            "status": self.status,
            "slots": dict(self.slots),
        }


class RequirementBook:
    """一个会话里的多条产品需求 + 当前活跃那条（计划 §十四/§十五）。"""

    def __init__(self) -> None:
        self.requirements: List[Requirement] = []

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, product_domain: str) -> Optional[Requirement]:
        domain = str(product_domain or "").strip().upper()
        for item in self.requirements:
            if item.product_domain == domain:
                return item
        return None

    @property
    def active(self) -> Optional[Requirement]:
        for item in self.requirements:
            if item.status == STATUS_ACTIVE:
                return item
        return None

    # ── 登记 / 切换 ─────────────────────────────────────────────────────
    def activate(self, product_domain: str, slots: Optional[Dict[str, Any]] = None) -> Requirement:
        """把某个产品域设为当前活跃需求（不存在就新建；不删别的域的信息）。"""
        domain = str(product_domain or "").strip().upper() or UNKNOWN
        for item in self.requirements:
            item.status = STATUS_PENDING
        requirement = self.get(domain)
        if requirement is None:
            requirement = Requirement(product_domain=domain)
            self.requirements.append(requirement)
        requirement.status = STATUS_ACTIVE
        requirement.update(slots or {})
        return requirement

    def apply(self, understanding: "TurnUnderstanding") -> Dict[str, Any]:
        """按这一轮的理解更新需求簿，返回"What changed"（供留痕/测试）。"""
        domains = list(understanding.product_domains or ([understanding.product_domain] if understanding.product_domain else []))
        domains = [d for d in domains if d and d != UNKNOWN and d != MULTI] or [understanding.product_domain]
        active_before = self.active.product_domain if self.active else ""
        created: List[str] = []
        for domain in domains:
            if self.get(domain) is None and domain not in ("", UNKNOWN, MULTI):
                created.append(domain)
        # 先登记（pending），再把当前活跃/主域设为 active —— 多产品时保留其它域
        primary = domains[0] if domains else UNKNOWN
        for domain in domains:
            if domain in ("", UNKNOWN, MULTI):
                continue
            requirement = self.get(domain)
            if requirement is None:
                requirement = Requirement(product_domain=domain)
                self.requirements.append(requirement)
            requirement.update(understanding.requirement_updates)
        # 不登记 UNKNOWN / MULTI 这类"还没定品类"的占位需求（它们不是产品需求）
        active = (
            self.activate(primary, understanding.requirement_updates)
            if primary and primary not in (UNKNOWN, MULTI)
            else self.active
        )
        return {
            "active_before": active_before,
            "active_after": active.product_domain if active else "",
            "switched": bool(active_before and active and active_before != active.product_domain),
            "created": created,
            "multi_product": bool(understanding.multi_product),
        }

    def snapshot(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.requirements]

    @classmethod
    def from_payload(cls, payload: Any) -> "RequirementBook":
        """从 memory 里的 JSON 还原（跨轮共享同一条需求簿）。"""
        book = cls()
        data = payload if isinstance(payload, dict) else {}
        for item in data.get("requirements") or []:
            if not isinstance(item, dict):
                continue
            book.requirements.append(
                Requirement(
                    product_domain=str(item.get("product_domain") or UNKNOWN),
                    status=str(item.get("status") or STATUS_PENDING),
                    slots=dict(item.get("slots") or {}),
                )
            )
        return book


@dataclass
class TurnUnderstanding:
    """统一回合理解（计划 §四）。"""

    conversation_type: str = PURE_CONVERSATION
    intent: str = ""
    product_domain: str = UNKNOWN
    product_domains: List[str] = field(default_factory=list)
    business_signal: Dict[str, Any] = field(default_factory=dict)
    requirement_updates: Dict[str, Any] = field(default_factory=dict)
    active_requirement: str = ""
    switch_product: bool = False
    multi_product: bool = False
    comparison: bool = False
    # 这一回合客户是不是以打招呼开场（"hi, i need a display"）——
    # 计划 v2.9.2 §五：Greeting 是"自然业务入口"，它的接话必须能进最终回复
    greeting_present: bool = False

    @property
    def has_business_signal(self) -> bool:
        return bool(self.business_signal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_type": self.conversation_type,
            "intent": self.intent,
            "product_domain": self.product_domain,
            "product_domains": list(self.product_domains),
            "business_signal": dict(self.business_signal),
            "requirement_updates": dict(self.requirement_updates),
            "active_requirement": self.active_requirement,
            "switch_product": self.switch_product,
            "multi_product": self.multi_product,
            "comparison": self.comparison,
            "greeting_present": self.greeting_present,
        }


# 「想换产品」的说法（计划 §十四：LCD → LED 要切换 active，但不删信息）
_SWITCH_RE = None


def _switch_requested(message: str) -> bool:
    global _SWITCH_RE
    if _SWITCH_RE is None:
        import re

        _SWITCH_RE = re.compile(
            r"\b(?:actually|instead|switch to|change to|rather)\b|"
            r"其实|改成|换成|不要这个|另外|换一个",
            re.IGNORECASE,
        )
    return bool(_SWITCH_RE.search(str(message or "")))


def _comparison_requested(message: str) -> bool:
    import re

    return bool(
        re.search(
            r"\b(?:which is better|difference between|vs\.?|versus|compare)\b|"
            r"哪个好|区别|对比|比较",
            str(message or ""),
            re.IGNORECASE,
        )
    )


def understand_turn(
    message: str,
    *,
    intent: str = "",
    profile: Any = None,
    history: Optional[List[Any]] = None,
    book: Optional[RequirementBook] = None,
) -> TurnUnderstanding:
    """把一句话（或聚合后的一个回合）理解成统一结构。"""
    text = str(message or "")
    signals = business_signals(text)
    conversation_type = classify_turn_kind(text, profile)
    domain = detect_product_domain(text, profile)
    domains: List[str] = []
    if domain == MULTI:
        # "LED for the stadium and LCD for the meeting room" → 两条需求
        lower = text.lower()
        if "led" in lower:
            domains.append("LED")
        if "lcd" in lower or "video wall" in lower:
            domains.append("LCD")
        if "ifp" in lower or "interactive" in lower:
            domains.append("IFP")
    elif domain in ALL_PRODUCT_DOMAINS and domain != UNKNOWN:
        domains = [domain]
    comparison = _comparison_requested(text) or (domain == MULTI and not domains)
    switch_product = _switch_requested(text) and bool(domains)
    active_before = book.active.product_domain if book is not None and book.active else ""
    return TurnUnderstanding(
        conversation_type=conversation_type,
        intent=str(intent or ""),
        product_domain=domain,
        product_domains=domains,
        business_signal=signals,
        requirement_updates=dict(signals),
        active_requirement=domains[0] if domains else active_before,
        switch_product=switch_product,
        multi_product=len(domains) > 1,
        comparison=comparison,
        greeting_present=has_greeting_opener(text),
    )


__all__ = [
    "Requirement",
    "RequirementBook",
    "STATUS_ACTIVE",
    "STATUS_PENDING",
    "TurnUnderstanding",
    "has_greeting_opener",
    "understand_turn",
]
