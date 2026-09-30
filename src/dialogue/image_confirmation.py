"""图片确认层（Image Confirmation Layer）——《LCD_IFP_需求链路工程化整改计划》Phase 4。

职责（只做这三件事，不推荐、不选型）：

    1. 把 ImageRecognitionResult 作为**候选事实**并进档案（看见 ≠ 客户需要）；
    2. 与已有文字上下文对比 → 发现冲突（图说室内 / 客户说室外）→ 请客户确认；
    3. 客户确认 / 纠正之后才锁定：客户明确修改 > 客户明确确认 > 图片识别结果。

设计取舍：LED 的图片确认语义已经在 `src/vision/integration.py`
（`resolve_vision_confirmation`）里实现过一遍，这里**复用同一套来源规则**
（confirmed / vision_accepted / vision_corrections），只补 LCD 字段
（拼接 / 尺寸 / 摄像头 / 触控 / 手写 / 场景）与"图文冲突"的确认。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.vision.recognition import ImageRecognitionResult

logger = logging.getLogger(__name__)

# 图片候选字段 → 档案字段（LCD 专用命名，避免和 LED 的宽高/点间距混用）
LCD_FIELD_TO_PROFILE: Dict[str, str] = {
    "is_splicing": "lcd_is_splicing",
    "screen_size_inch": "lcd_size_inch",
    "splicing_layout": "lcd_splicing_layout",
    "screen_count": "lcd_screen_count",
    "bezel_mm": "lcd_bezel_mm",
    "camera_observed": "lcd_camera_observed",
    "touch_observed": "lcd_touch_observed",
    "handwriting_observed": "lcd_handwriting_observed",
    "application": "purpose",
}

# 客户"需要"的字段：图片看到**不代表**客户要（必须客户确认后才写）
REQUIRED_ONLY_FIELDS: Dict[str, str] = {
    "camera_observed": "lcd_camera_required",
    "touch_observed": "lcd_touch_required",
    "handwriting_observed": "lcd_handwriting_required",
}

# 冲突对比用的字段（图片 vs 文字上下文）
CONTEXT_COMPARE_FIELDS: tuple[str, ...] = (
    "display_type",
    "environment",
    "installation",
    "is_splicing",
    "screen_size_inch",
)

_AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|correct|right|right\b|exactly|confirm(ed)?|sure|that'?s right|"
    r"it is|it's)\b|对|是的|没错|正确|确认|可以|是的，|嗯",
    re.IGNORECASE,
)
_DENY_RE = re.compile(
    r"\b(no|nope|not|isn'?t|aren'?t|wrong|incorrect|actually|instead)\b|不是|不对|错误|其实|改成|改为",
    re.IGNORECASE,
)


@dataclass
class ContextCompare:
    """图片与文字上下文的对比结论（计划 §二十二/§二十三）。"""

    already_known: Dict[str, Any] = field(default_factory=dict)   # 文字里已经确认过的
    matched: Dict[str, Any] = field(default_factory=dict)         # 图片与文字一致（不用再问）
    observed: Dict[str, Any] = field(default_factory=dict)        # 图片看到的
    conflicts: List[str] = field(default_factory=list)            # 字段名（图片与文字不一致）
    to_confirm: List[str] = field(default_factory=list)           # 需要请客户确认的字段

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "already_known": dict(self.already_known),
            "matched": dict(self.matched),
            "observed": dict(self.observed),
            "conflicts": list(self.conflicts),
            "to_confirm": list(self.to_confirm),
            "context_conflict": bool(self.conflicts),
        }


# ── 1) 对比：图片 vs 文字上下文 ─────────────────────────────────────────────
def compare_with_context(
    profile: Any,
    recognition: Optional[ImageRecognitionResult],
) -> ContextCompare:
    """把"图片看到的"和"客户已经说过的"对齐，找出冲突与待确认项。"""
    result = ContextCompare()
    if profile is None or recognition is None:
        return result
    sources = dict(getattr(profile, "sources", None) or {})
    observed = recognition.candidates()
    result.observed = dict(observed)

    for name, value in observed.items():
        profile_field = LCD_FIELD_TO_PROFILE.get(name, name)
        current = getattr(profile, profile_field, None)
        source = str(sources.get(profile_field) or "")
        if name == "screen_size_inch" and current is not None:
            # LCD 尺寸落在 lcd_size_inch；客户明确说过就不再用图片值
            current = getattr(profile, "lcd_size_inch", None)
        if current in (None, "", [], {}):
            result.to_confirm.append(name)
            continue
        if _same_value(current, value):
            if _is_customer_source(source):
                # 图片与客户说的一致 → 不用再确认这一项（计划 §二十二）
                result.matched[name] = current
            continue
        if _is_customer_source(source):
            # 客户已经用**自己的话**说过 → 图片与文字冲突，必须请客户确认
            result.already_known[name] = current
            result.conflicts.append(name)
        else:
            result.to_confirm.append(name)
    return result


def _is_customer_source(source: str) -> bool:
    text = str(source or "").lower()
    return text in ("explicit", "confirmed", "customer_explicit", "customer_confirmed")


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    try:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) < 1e-6
    except (TypeError, ValueError):  # pragma: no cover - 防御式
        pass
    return str(left).strip().lower() == str(right).strip().lower()


# ── 2) 把图片结果作为候选并进档案（不锁定）────────────────────────────────
def apply_recognition_to_profile(
    profile: Any,
    recognition: Optional[ImageRecognitionResult],
) -> Dict[str, Any]:
    """图片候选事实入档：只写"档案里还没有"的字段，来源标成图片。"""
    stats: Dict[str, Any] = {"merged": [], "conflicts": [], "pending": []}
    if profile is None or recognition is None or recognition.is_empty():
        return stats
    sources = dict(getattr(profile, "sources", None) or {})
    assertions = dict(getattr(profile, "vision_assertions", None) or {})
    pending = list(getattr(profile, "vision_confirmation_pending", None) or [])
    conflicts = list(getattr(profile, "conflicts", None) or [])
    conflict_slots = list(getattr(profile, "conflict_slots", None) or [])

    compare = compare_with_context(profile, recognition)
    for name, value in recognition.candidates().items():
        profile_field = LCD_FIELD_TO_PROFILE.get(name, name)
        if name in CONTEXT_COMPARE_FIELDS and getattr(profile, profile_field, None) not in (
            None, "", [], {},
        ):
            # 客户已经说过 → 不覆盖；冲突在下面登记
            pass
        elif getattr(profile, profile_field, None) in (None, "", [], {}):
            setattr(profile, profile_field, value)
            sources[profile_field] = "vision_explicit"
            stats["merged"].append(profile_field)
        assertions[profile_field] = value
        if name not in pending and name in compare.to_confirm:
            pending.append(name)
            stats["pending"].append(name)

    if recognition.display_type and not getattr(profile, "display_type", None):
        profile.display_type = recognition.display_type
        sources["display_type"] = "vision_explicit"
        stats["merged"].append("display_type")

    for name in compare.conflicts:
        slot = name
        note = f"image said {recognition.candidates().get(name)}, context said {compare.already_known.get(name)} ({name})"
        if note not in conflicts:
            conflicts.append(note)
        if slot not in conflict_slots:
            conflict_slots.append(slot)
        # 冲突也要请客户确认（计划 §二十三）：把这一项挂进待确认清单
        if name not in pending:
            pending.append(name)
            stats["pending"].append(name)
        stats["conflicts"].append(name)

    profile.sources = sources
    profile.vision_assertions = assertions
    profile.vision_confirmation_pending = pending
    profile.conflicts = conflicts
    profile.conflict_slots = conflict_slots
    return stats


# ── 3) 客户确认 / 纠正之后才锁定 ───────────────────────────────────────────
def apply_customer_reply(
    profile: Any,
    message: str,
    *,
    recognitions: Sequence[ImageRecognitionResult] = (),
) -> Dict[str, Any]:
    """客户对图片识别的回应：确认 → 客户确认；纠正 → 客户值优先并记纠正。

    规则（计划第三章优先级）：

        客户明确修改 > 客户明确确认 > 图片识别结果
    """
    stats: Dict[str, Any] = {"confirmed": [], "accepted": [], "corrected": [], "conflicts_cleared": []}
    if profile is None:
        return stats
    # 共用字段（display_type / environment / installation …）复用 Vision 模块里
    # 已有的确认语义，避免同一套规则写两份（计划 §二十八：保留现有提取工具）。
    try:
        from src.vision.integration import resolve_vision_confirmation

        shared = resolve_vision_confirmation(profile, str(message or ""))
        stats["shared"] = shared
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("[ImageConfirm] shared confirmation failed: %s", exc)
    text = str(message or "")
    affirmed = bool(_AFFIRM_RE.search(text))
    denied = bool(_DENY_RE.search(text))
    sources = dict(getattr(profile, "sources", None) or {})
    assertions = dict(getattr(profile, "vision_assertions", None) or {})
    corrections = list(getattr(profile, "vision_corrections", None) or [])
    pending = list(getattr(profile, "vision_confirmation_pending", None) or [])

    # 客户这句话里带的新值（"不是，是室外" / "其实要 4K" 等）由上层已写进档案；
    # 这里只负责"升级来源 / 记录纠正 / 清 pending"。
    for name in list(pending):
        profile_field = LCD_FIELD_TO_PROFILE.get(name, name)
        value = getattr(profile, profile_field, None)
        asserted = assertions.get(profile_field) or assertions.get(name)
        source = str(sources.get(profile_field) or "")

        if value not in (None, "", [], {}) and asserted not in (None, "") and not _same_value(value, asserted):
            note = f"image said {asserted}, customer said {value} ({name})"
            if note not in corrections:
                corrections.append(note)
            stats["corrected"].append(name)
            sources[profile_field] = "confirmed"
            pending.remove(name)
            continue
        if affirmed and not denied:
            sources[profile_field] = "confirmed"
            stats["confirmed"].append(name)
            pending.remove(name)
            continue
        if source.startswith("vision"):
            # 上一轮已经把识别结果摆给客户核对过，客户没反对 → 视为已核对，不再重复问
            sources[profile_field] = "vision_accepted"
            stats["accepted"].append(name)
            pending.remove(name)

    # 客户明确纠正过的字段：图片与文字冲突随之消解
    conflict_slots = list(getattr(profile, "conflict_slots", None) or [])
    if denied and conflict_slots:
        stats["conflicts_cleared"].extend(conflict_slots)
        conflict_slots = []

    profile.sources = sources
    profile.vision_corrections = corrections
    profile.vision_confirmation_pending = [
        name for name in pending if name in _all_confirmable()
    ]
    profile.conflict_slots = conflict_slots
    return stats


def _all_confirmable() -> set:
    from src.vision.recognition import CONFIRMABLE_FIELDS

    return set(CONFIRMABLE_FIELDS)


# ── 4) 结合上下文的确认话术（计划 §二十二）────────────────────────────────
_FIELD_TEXT = {
    "environment": {"indoor": "indoor", "outdoor": "outdoor"},
    "installation": {"fixed": "fixed installation", "rental": "rental"},
    "is_splicing": {True: "a video wall (spliced)", False: "single displays (not spliced)"},
    "camera_observed": {True: "a camera above the screens", False: "no camera"},
    "screen_size_inch": {},
}


def confirmation_prompt(
    profile: Any,
    recognitions: Sequence[ImageRecognitionResult],
    *,
    language: str = "en",
) -> str:
    """把"文字里已经说过的 + 图片看到的"合成一句核对（最多一个问题）。"""
    recognitions = [r for r in (recognitions or []) if r is not None]
    if not recognitions or profile is None:
        return ""
    compare = compare_with_context(profile, recognitions[0])
    observed = compare.observed
    if not observed:
        return ""
    zh = str(language or "").lower().startswith("zh")
    display_type = str(recognitions[0].display_type or "").upper() or "LCD"
    known_bits = []
    for name in ("environment", "installation", "is_splicing"):
        if name in compare.already_known or name in compare.matched or _customer_said(profile, name):
            value = getattr(
                profile, LCD_FIELD_TO_PROFILE.get(name, name), compare.already_known.get(name)
            )
            text = _FIELD_TEXT.get(name, {}).get(value if value in (True, False) else str(value))
            if text:
                known_bits.append(text)

    if compare.conflicts:
        field = compare.conflicts[0]
        observed_text = _FIELD_TEXT.get(field, {}).get(
            observed.get(field) if observed.get(field) in (True, False) else str(observed.get(field))
        ) or str(observed.get(field))
        context_text = _FIELD_TEXT.get(field, {}).get(
            compare.already_known.get(field)
            if compare.already_known.get(field) in (True, False)
            else str(compare.already_known.get(field))
        ) or str(compare.already_known.get(field))
        if zh:
            return (
                f"你之前说的是 {context_text}，而图片看起来是 {observed_text}。"
                f"以哪个为准？"
            )
        return (
            f"You mentioned {context_text}, while the image looks like {observed_text}. "
            f"Which one should I use?"
        )

    matched = ", ".join(known_bits)
    extras = []
    if observed.get("camera_observed") is True:
        extras.append("a camera above the screens" if not zh else "屏幕上方有一个摄像头")
    if (
        observed.get("is_splicing") is True
        and "is_splicing" not in compare.already_known
        and "is_splicing" not in compare.matched
    ):
        extras.append("a spliced video wall" if not zh else "拼接屏")
    extra_text = ""
    if extras:
        extra_text = ("I can also see " if not zh else "我还看到 ") + " and ".join(extras)

    if zh:
        head = f"我把图片和你说的对了一下：看起来是 {display_type}"
        if matched:
            head += f"（{matched}）"
        ask = "这些和你说的一致吗？"
        return f"{head}。{extra_text}，{ask}".replace("，。", "。")

    head = f"I've matched the image with what you described: it appears to be an {display_type}"
    if matched:
        head += f" ({matched})"
    ask = "Does that match what you have in mind?"
    if extra_text:
        return f"{head}. {extra_text}. {ask}"
    return f"{head}. {ask}"


def _customer_said(profile: Any, name: str) -> bool:
    profile_field = LCD_FIELD_TO_PROFILE.get(name, name)
    sources = dict(getattr(profile, "sources", None) or {})
    return _is_customer_source(sources.get(profile_field, ""))


def recognition_from_profile(profile: Any) -> Optional[ImageRecognitionResult]:
    """从档案里还原"图片当时看到了什么"（`vision_assertions` 是留痕的事实源）。"""
    if profile is None:
        return None
    assertions = dict(getattr(profile, "vision_assertions", None) or {})
    if not assertions:
        return None
    reverse = {value: key for key, value in LCD_FIELD_TO_PROFILE.items()}
    payload: Dict[str, Any] = {}
    for field_name, value in assertions.items():
        name = reverse.get(field_name, field_name)
        payload[name] = value
    display_type = str(getattr(profile, "display_type", None) or "LCD").upper()
    payload.setdefault("display_type", display_type)
    return ImageRecognitionResult.from_payload(payload)


# 客户说"不对"时要问的"哪里不对"——把在核对的字段翻译成人话
_DENIED_LABEL = {
    "display_type": ("the screen type (LED or LCD)", "屏的类型（LED 还是 LCD）"),
    "is_splicing": ("whether it is a spliced video wall", "是不是拼接墙"),
    "lcd_is_splicing": ("whether it is a spliced video wall", "是不是拼接墙"),
    "camera_observed": ("whether there is a camera", "有没有摄像头"),
    "lcd_camera_observed": ("whether there is a camera", "有没有摄像头"),
    "environment": ("indoor or outdoor", "室内还是室外"),
    "installation": ("fixed installation or rental", "固装还是租赁"),
}


def denied_prompt_from_profile(profile: Any, *, language: str = "en") -> str:
    """客户说"图片识别得不对" → 请他指出哪里不对（客户口径 2026-09-30）。

    客户只说了"不对"、没说正确的值，**不能**当成"没反对"直接采用识别结果，
    也不能自己猜他指的是哪一项 —— 直接把在核对的那几项列出来请他指。
    """
    denied = list(getattr(profile, "vision_confirmation_denied", None) or [])
    if not denied:
        return ""
    zh = str(language or "").lower().startswith("zh")
    labels: List[str] = []
    for field in denied:
        text = _DENIED_LABEL.get(field, (str(field), str(field)))[1 if zh else 0]
        if text and text not in labels:
            labels.append(text)
    if not labels:
        return ""
    joined = "、".join(labels) if zh else _join_labels(labels)
    if zh:
        return f"抱歉我看错了。你说的不一样的是哪一项 —— {joined}？"
    return f"Sorry, I got that wrong. Which part doesn't match — {joined}?"


def _join_labels(labels: List[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " or " + labels[-1]


def prompt_from_profile(profile: Any, *, language: str = "en") -> str:
    """给"图片那一轮"用的核对话术（LCD/IFP；没有图片留痕时返回空串）。

    计划 §二十二：图片确认必须结合上下文 —— 客户已经说过的部分不再重复问。
    """
    denied = denied_prompt_from_profile(profile, language=language)
    if denied:
        # 客户已经说过"不对"了 → 这一轮问的是"哪里不对"，不再重复同一句核对
        return denied
    recognition = recognition_from_profile(profile)
    if recognition is None:
        return ""
    return confirmation_prompt(profile, [recognition], language=language)


__all__ = [
    "CONTEXT_COMPARE_FIELDS",
    "ContextCompare",
    "LCD_FIELD_TO_PROFILE",
    "denied_prompt_from_profile",
    "REQUIRED_ONLY_FIELDS",
    "apply_customer_reply",
    "apply_recognition_to_profile",
    "compare_with_context",
    "confirmation_prompt",
    "prompt_from_profile",
    "recognition_from_profile",
]
