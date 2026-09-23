"""产品类型路由器（计划 v2.9.3 §四/§五/§十/§十一）。

第一层只回答一件事：**客户该进 LED 还是 LCD**。

    LED
    LCD
    ├── 普通 LCD
    └── IFP（LCD 的子类型，**不**出现在第一层）

判断优先级（计划 §五，从强到弱）：

    1. 客户明确说 LED / LCD            → CONFIRMED / CUSTOMER
    2. 客户说了 IFP 特征（触控/手写/白板/交互/电子白板…）→ LCD + subtype=IFP（INFERRED）
    3. 图片（Vision）判断              → LED / LCD（INFERRED / VISION）
    4. 只说用途（场景 + 环境 + 尺寸 + 视距 + 安装 + 触控…）→ 辅助推断（INFERENCE）
    5. 客户不知道 LED/LCD 是什么        → 先解释，再问（needs_explanation）
    6. 信息不足                        → 自然询问（ask_customer）
    7. 所有办法都拿不到有效信息          → 最终 fallback：LED（DEFAULT）

以及状态管理（计划 §十二/§十三）：

    display_type / status / source / confidence / reason / locked
    客户确认过 → locked=True；之后普通信息不能改类型；
    只有客户明确改（"Actually, I want LCD."）才允许重新路由。

原则（计划 §七/§九/§十一）：场景只是证据，不能覆盖客户明确选择；
推断必须能被客户确认；**不许编造客户没提供的信息**。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 第一层产品类型（IFP 不出现在这里）
LED = "LED"
LCD = "LCD"
UNKNOWN = "UNKNOWN"

# 状态 / 来源（计划 §十二）
STATUS_UNKNOWN = "UNKNOWN"
STATUS_INFERRED = "INFERRED"
STATUS_CONFIRMED = "CONFIRMED"

SOURCE_CUSTOMER = "CUSTOMER"
SOURCE_VISION = "VISION"
SOURCE_INFERENCE = "INFERENCE"
SOURCE_DEFAULT = "DEFAULT"

# LCD 子类型（§十四：IFP 在 LCD 链路内部判断）
SUBTYPE_IFP = "IFP"
SUBTYPE_PLAIN_LCD = "LCD"

_LED_RE = re.compile(r"\bled\b|led屏|点间距|箱体|模组", re.IGNORECASE)
_LCD_RE = re.compile(r"\blcd\b|液晶|video ?wall|拼接屏|拼接", re.IGNORECASE)
_IFP_RE = re.compile(
    r"interactive (?:display|panel|flat panel|whiteboard)|touch ?screen|touch display|"
    r"smart ?board|whiteboard|ifp|"
    r"\b(?:write|writing|handwriting|annotate|annotation)\b|"
    r"电子白板|交互平板|触控|触摸|手写|书写|会议平板|一体机",
    re.IGNORECASE,
)
_OUTDOOR_RE = re.compile(r"\boutdoor\b|\boutside\b|户外|室外|露天", re.IGNORECASE)
_AD_RE = re.compile(r"advertis|billboard|signage|广告|招牌|幕墙", re.IGNORECASE)
_BIG_SCENE_RE = re.compile(
    r"stadium|arena|stage|concert|facade|exhibition hall|机场|体育|场馆|舞台|演唱会|幕墙|大楼",
    re.IGNORECASE,
)
_PURPOSE_RE = re.compile(
    r"meeting room|conference|classroom|training|showroom|lobby|church|retail|store|"
    r"会议室|教室|展厅|大厅|教堂|门店",
    re.IGNORECASE,
)
# "超大 / 远距离"这类信号：会议室也可能更合适 LED（计划 §七 的会议室 + 超大尺寸 + 远距离）
_LARGE_RE = re.compile(r"\b(?:huge|giant|very large|large|big|massive)\b|超大|很大|大型", re.IGNORECASE)
_DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", re.IGNORECASE)
_WHAT_IS_RE = re.compile(
    r"what(?:'s| is) (?:the difference between )?(?:an? )?(?:led|lcd)|"
    r"(?:led|lcd).{0,12}(?:是什么|什么意思)|区别|difference",
    re.IGNORECASE,
)
_DONT_KNOW_RE = re.compile(
    r"i (?:don'?t|do not) know|not sure|you (?:decide|choose|recommend)|"
    r"不知道|不确定|你(?:来)?决定|你帮我选|随便",
    re.IGNORECASE,
)
_AFFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|correct|right|that works|"
    r"sounds good|go ahead|please do|fine|好的|可以|行|对|是的|没错|没问题)\b",
    re.IGNORECASE,
)
_SWITCH_RE = re.compile(
    r"\b(?:actually|instead|switch to|change to|rather)\b|其实|改成|换成|不要这个|另外换",
    re.IGNORECASE,
)


@dataclass
class DisplayTypeDecision:
    """第一层产品类型判断结果（计划 §十二）。"""

    display_type: str = UNKNOWN
    subtype: str = ""
    status: str = STATUS_UNKNOWN
    source: str = ""
    confidence: float = 0.0
    reason: str = ""
    locked: bool = False
    needs_explanation: bool = False
    ask_customer: bool = False
    evidence: List[str] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.status == STATUS_CONFIRMED

    @property
    def is_ifp(self) -> bool:
        return self.display_type == LCD and self.subtype == SUBTYPE_IFP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_type": self.display_type,
            "subtype": self.subtype,
            "status": self.status,
            "source": self.source,
            "confidence": round(float(self.confidence), 2),
            "reason": self.reason,
            "locked": self.locked,
            "needs_explanation": self.needs_explanation,
            "ask_customer": self.ask_customer,
            "evidence": list(self.evidence),
        }


def _explicit_type(text: str) -> str:
    """客户**明确**说了 LED / LCD 吗（两个都说 → 以先出现的为准，由上层处理比较）。"""
    led = _LED_RE.search(text)
    lcd = _LCD_RE.search(text)
    if led and lcd:
        return LED if led.start() < lcd.start() else LCD
    if led:
        return LED
    if lcd:
        return LCD
    return ""


def _far_viewing(text: str, threshold_m: float = 6.0) -> bool:
    for match in _DISTANCE_RE.finditer(text):
        try:
            if float(match.group(1)) >= threshold_m:
                return True
        except ValueError:  # pragma: no cover - 防御式
            continue
    return False


def _infer_from_purpose(text: str) -> tuple[str, str, float, str]:
    """只说用途时，按"场景 + 环境 + 触控/手写"给倾向（计划 §六/§七）。"""
    if _IFP_RE.search(text):
        return LCD, SUBTYPE_IFP, 0.8, "interactive/touch features → LCD (IFP)"
    if _OUTDOOR_RE.search(text) or _AD_RE.search(text) or _BIG_SCENE_RE.search(text):
        return LED, "", 0.75, "outdoor / advertising / large venue → LED"
    if _LARGE_RE.search(text) or _far_viewing(text):
        # 会议室 + 超大尺寸 / 远距离观看 → LED 通常更合适（但仍要客户确认）
        return LED, "", 0.65, "very large screen or far viewing distance → LED is usually the better fit"
    if _PURPOSE_RE.search(text):
        # 会议室/教室这类室内场景：LED 与 LCD 都可能 → 给倾向但要客户确认
        return LCD, "", 0.55, "indoor meeting-type scene → LCD is usually the simpler fit"
    return "", "", 0.0, ""


def route_display_type(
    message: str,
    *,
    profile: Any = None,
    vision_display_type: str = "",
    vision_reason: str = "",
    current: Optional[DisplayTypeDecision] = None,
    customer_confirmed: Optional[str] = None,
) -> DisplayTypeDecision:
    """第一层产品类型判断（纯函数，方便单测）。"""
    text = str(message or "")
    current = current or DisplayTypeDecision()

    # 客户这一轮明确确认/修改了类型（也是锁定与重新路由的唯一入口）
    if customer_confirmed:
        confirmed = str(customer_confirmed).upper()
        if confirmed in (LED, LCD):
            return DisplayTypeDecision(
                display_type=confirmed,
                subtype=SUBTYPE_IFP if (confirmed == LCD and _IFP_RE.search(text)) else "",
                status=STATUS_CONFIRMED,
                source=SOURCE_CUSTOMER,
                confidence=1.0,
                reason="customer confirmed the product type",
                locked=True,
                evidence=[text[:120]],
            )

    # 已锁定：普通信息不能改类型；只有客户明确修改才允许重新路由
    if current.locked and current.display_type in (LED, LCD):
        if _SWITCH_RE.search(text):
            explicit = _explicit_type(text)
            if explicit and explicit != current.display_type:
                return DisplayTypeDecision(
                    display_type=explicit,
                    subtype=SUBTYPE_IFP if (explicit == LCD and _IFP_RE.search(text)) else "",
                    status=STATUS_CONFIRMED,
                    source=SOURCE_CUSTOMER,
                    confidence=1.0,
                    reason="customer explicitly changed the product type",
                    locked=True,
                    evidence=[text[:120]],
                )
        # 保持不变（信息照常进需求抽取，但类型不动）
        kept = DisplayTypeDecision(**{**current.to_dict(), "evidence": list(current.evidence)})
        kept.reason = kept.reason or "locked product type kept"
        return kept

    # ① 客户明确说 LED / LCD
    # 注意：先排除"LED 是什么 / LED 和 LCD 有什么区别"这种**提问**，
    # 否则会把解释类问题误判成"客户明确选择了 LED"（计划 §九）。
    if _WHAT_IS_RE.search(text):
        return DisplayTypeDecision(
            status=STATUS_UNKNOWN,
            source=SOURCE_INFERENCE,
            reason="customer asked what LED/LCD means",
            needs_explanation=True,
            ask_customer=True,
            evidence=[text[:120]],
        )

    explicit = _explicit_type(text)
    if explicit:
        subtype = SUBTYPE_IFP if (explicit == LCD and _IFP_RE.search(text)) else ""
        return DisplayTypeDecision(
            display_type=explicit,
            subtype=subtype,
            status=STATUS_CONFIRMED,
            source=SOURCE_CUSTOMER,
            confidence=1.0,
            reason="customer explicitly named the product type",
            locked=True,
            evidence=[text[:120]],
        )

    # ①.5 客户对我们"要不要按 X 继续"的确认（计划 §六：推断 → 客户确认 → 锁定）
    if (
        _AFFIRM_RE.search(text)
        and current.display_type in (LED, LCD)
        and current.status == STATUS_INFERRED
    ):
        return DisplayTypeDecision(
            display_type=current.display_type,
            subtype=current.subtype,
            status=STATUS_CONFIRMED,
            source=SOURCE_CUSTOMER,
            confidence=1.0,
            reason="customer confirmed our product-type suggestion",
            locked=True,
            evidence=[text[:120]],
        )

    # ② 客户描述了 IFP 特征（触控/手写/白板/交互）→ LCD + IFP 子类型
    if _IFP_RE.search(text):
        return DisplayTypeDecision(
            display_type=LCD,
            subtype=SUBTYPE_IFP,
            status=STATUS_INFERRED,
            source=SOURCE_INFERENCE,
            confidence=0.8,
            reason="interactive / touch / whiteboard features → LCD (IFP)",
            ask_customer=True,
            evidence=[text[:120]],
        )

    # ③ 图片（Vision）判断 → 需要客户确认
    vision = str(vision_display_type or "").upper()
    if vision in (LED, LCD):
        return DisplayTypeDecision(
            display_type=vision,
            subtype="",
            status=STATUS_INFERRED,
            source=SOURCE_VISION,
            confidence=0.7,
            reason=vision_reason or "inferred from the customer's image",
            ask_customer=True,
            evidence=[vision_reason] if vision_reason else [],
        )

    # ⑤ 只知道用途 → 辅助推断（要给客户确认）
    inferred, subtype, confidence, reason = _infer_from_purpose(text)
    if inferred:
        return DisplayTypeDecision(
            display_type=inferred,
            subtype=subtype,
            status=STATUS_INFERRED,
            source=SOURCE_INFERENCE,
            confidence=confidence,
            reason=reason,
            ask_customer=True,
            evidence=[text[:120]],
        )

    # ⑥ 客户明确表示"不知道 / 你帮我选" → 接受 AI 判断（若有），否则按用途兜底
    if _DONT_KNOW_RE.search(text):
        if current.display_type in (LED, LCD):
            adopted = DisplayTypeDecision(**{**current.to_dict(), "evidence": list(current.evidence)})
            adopted.status = STATUS_CONFIRMED
            adopted.source = SOURCE_INFERENCE
            adopted.locked = True
            adopted.ask_customer = False
            adopted.reason = "customer let us decide; using our own judgement"
            return adopted
        profile_type = str(getattr(profile, "display_type", "") or "").upper()
        if profile_type in (LED, LCD):
            return DisplayTypeDecision(
                display_type=profile_type,
                status=STATUS_CONFIRMED,
                source=SOURCE_INFERENCE,
                confidence=0.6,
                reason="customer let us decide; using the requirement profile",
                locked=True,
            )

    # ⑦ 信息不足 → 自然询问（不硬猜）
    return DisplayTypeDecision(
        status=STATUS_UNKNOWN,
        source=SOURCE_INFERENCE,
        reason="not enough information to tell LED from LCD",
        ask_customer=True,
    )


def final_fallback() -> DisplayTypeDecision:
    """所有办法都拿不到有效信息 → 最终 fallback：LED（计划 §五 最后一条）。"""
    return DisplayTypeDecision(
        display_type=LED,
        status=STATUS_INFERRED,
        source=SOURCE_DEFAULT,
        confidence=0.3,
        reason="final fallback: no usable signal, continue with LED",
        ask_customer=False,
    )


# 计划 §九：客户不知道 LED/LCD 是什么时，只给**简单业务解释**（不给技术文档）
EXPLANATION_LINES = {
    "en": (
        "LED displays can be built very large and have no visible seams between panels, "
        "and they work both indoors and outdoors.",
        "LCD displays are like commercial TV panels: they can be tiled into a video wall, "
        "but the seams between panels remain visible, and they are normally used indoors.",
    ),
    "zh": (
        "LED 屏可以做得很大，屏体之间基本没有可见拼缝，室内室外都能用。",
        "LCD 屏类似商用液晶屏，可以拼接成视频墙，但屏与屏之间会有可见拼缝，一般用于室内。",
    ),
}


def explanation_lines(language: str = "en") -> List[str]:
    """LED / LCD 的简单业务解释（计划 §九）。"""
    key = "zh" if str(language or "").lower().startswith("zh") else "en"
    return list(EXPLANATION_LINES[key])


def product_type_question(
    decision: DisplayTypeDecision,
    *,
    language: str = "en",
    seed: int = 0,
) -> str:
    """类型没确认时，这一轮该问的那句话（计划 §五/§六/§七/§九）。

    这是"问什么"的确定性部分（Python 决定），措辞由 ResponseGenerator 的 LLM 重写。
    """
    zh = str(language or "").lower().startswith("zh")
    if decision.needs_explanation:
        led_line, lcd_line = explanation_lines("zh" if zh else "en")
        tail = (
            "您打算把它用在哪里？我可以帮您判断哪种更合适。"
            if zh
            else "If you tell me what you plan to use it for, I can help you choose."
        )
        return f"{led_line} {lcd_line} {tail}"
    if decision.status == STATUS_INFERRED and decision.display_type in (LED, LCD):
        name = "LCD (interactive flat panel)" if decision.is_ifp else decision.display_type
        if zh:
            return f"按您说的情况，我先建议 {name}，要不要就按这个方向继续？"
        return (
            f"From what you have described, {name} looks like the better fit here, "
            "shall I continue in that direction?"
        )
    # 注意：这里只给"意思"，措辞由 ResponseGenerator 的 LLM 重新组织
    # （不要在这里堆多条写死的问法 —— 客户口径：话术太僵硬）
    if zh:
        return "您是想要 LED 显示屏，还是 LCD（拼接屏 / 交互平板）？"
    return (
        "Are you looking for an LED display, or an LCD such as a video wall "
        "or interactive flat panel?"
    )


def load_decision(payload: Any) -> DisplayTypeDecision:
    data = payload if isinstance(payload, dict) else {}
    return DisplayTypeDecision(
        display_type=str(data.get("display_type") or UNKNOWN),
        subtype=str(data.get("subtype") or ""),
        status=str(data.get("status") or STATUS_UNKNOWN),
        source=str(data.get("source") or ""),
        confidence=float(data.get("confidence") or 0.0),
        reason=str(data.get("reason") or ""),
        locked=bool(data.get("locked")),
        needs_explanation=bool(data.get("needs_explanation")),
        ask_customer=bool(data.get("ask_customer")),
        evidence=list(data.get("evidence") or []),
    )


__all__ = [
    "LCD",
    "LED",
    "SOURCE_CUSTOMER",
    "SOURCE_DEFAULT",
    "SOURCE_INFERENCE",
    "SOURCE_VISION",
    "STATUS_CONFIRMED",
    "STATUS_INFERRED",
    "STATUS_UNKNOWN",
    "SUBTYPE_IFP",
    "SUBTYPE_PLAIN_LCD",
    "UNKNOWN",
    "DisplayTypeDecision",
    "EXPLANATION_LINES",
    "explanation_lines",
    "product_type_question",
    "final_fallback",
    "load_decision",
    "route_display_type",
]
