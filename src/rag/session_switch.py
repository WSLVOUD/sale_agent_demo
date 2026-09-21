"""
会话内"需求重置"检测（客户换产品 / 换项目 / 改需求）。

背景（客户实测反馈）：
    同一个会话里，客户在拿到推荐之后又说"我想换个产品 / 重新推荐一下 /
    其实是户外的"，旧实现会把前面采集到的需求继续带下去，于是：
      1) 没有重新采集需求，AI 直接拿旧需求再次推荐；
      2) 旧需求（室内 / 教堂 / 5 米）继续参与打分，推荐结果自然不对。

设计原则（与 v2.0 Ready Gate 一致）：
    **纯规则、无 LLM、可单测**。只在"确有依据"时才重置：
      - 客户明确要求重来 / 换产品 / 换项目        → 全量重置（explicit_request）
      - 客户把屏幕类型换成另一种（LED↔LCD↔IFP）   → 全量重置（display_type_switch）
      - 已有推荐后，客户给出与已确认需求冲突的环境或场景 → 全量重置（requirement_conflict）
      - 已有推荐后，客户说要做另一个项目 / 另一块屏     → 全量重置（new_inquiry）

以下情况**不**视为重置，避免误伤（客户只是想多看几个 / 只是改一个槽位）：
    - "还有别的型号吗"、"再推荐几款"（想看更多选项，不是换需求）
    - "尺寸改成 8 米"、"换成 P3"（只更新对应槽位，不需要清空重来）
    - 产品参数提问、报价询问等

重置后**不是**丢给 LLM 自由发挥：Sales Agent 会清空需求档案，
回到"一次只问一个高价值问题"的需求采集流程（见 readiness.py 的 Gate）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 客户表达的"重新来过 / 换产品"─────────────────────────────────────────────
# 强信号：出现即要求重来（不受"想看更多"的护栏影响）
_STRONG_RESET_PATTERNS = (
    # 中文
    r"重新(?:帮我|给我|帮忙)?(?:来|选|推荐|做|问|开始|采集|收集|梳理|聊|了解|来过|整理|匹配|选型)",
    r"重来(?:一遍|一次|吧|下)?",
    r"从头(?:来|开始)",
    r"换(?:成|做|一|个|款|种|套|别|其他|其它|另)",
    r"换(?:产品|型号|方案|屏|品牌)",
    r"(?:不要|不想要|不喜欢|不选|不考虑|不满意)(?:这|那)?(?:个|款|种|块|台|套|方案)",
    r"另(?:外|选|换|做|来)(?:一)?(?:款|个|种)?",
    r"(?:想|要|得|能|可以)(?:再)?(?:换|改)(?:一|个|款|种|成)",
    # 英文
    r"\bstart (?:over|again|from scratch|fresh)\b",
    r"\b(?:let'?s|can we|i (?:want|need|would like|'?d like) to)\s+(?:re)?start\b",
    r"\breset\b",
    r"\brestart\b",
    r"\bchange[ds]? my mind\b",
    r"\bswitch(?:ing)? to\b",
    r"\bscratch that\b",
    r"\bforget (?:that|it)\b",
)

# 弱信号：需要配合"已有推荐"或"已有需求"才判定为重置
_WEAK_RESET_PATTERNS = (
    r"\bi (?:want|need|would like|'?d like|prefer)\s+(?:a\s+|the\s+)?(?:different|another|other|something else)\b",
    r"\b(?:rather have|let'?s try|try|show me)\s+(?:a\s+)?(?:different|another|other)\b",
    r"\b(?:different|another)\s+(?:one|model|product|option|unit|screen|display|series|solution)\b",
    r"\bchange (?:the )?(?:product|model|requirement|requirements|spec|specs|configuration)\b",
    r"\bnew (?:project|order|site|requirement|screen|display)\b",
    r"\banother (?:project|site|order|screen|display|room|venue)\b",
    r"\bdifferent (?:project|site|order|room|venue)\b",
    r"别的(?:项目|场地|工程)",
    r"另外(?:一)?(?:个项目|块屏|块屏幕)",
    r"还有(?:一)?个(?:(?:新)?项目|场地)",
    r"(?:推荐|来|出|给|要)(?:个|一|几)?(?:别的|其他|其它|另外)",
    r"(?:别的|其他|其它|另外)(?:的)?(?:产品|型号|方案|款|屏|品牌)",
)

# 护栏：命中这些说明客户是"想看更多选项"，不是要清空重来
_MORE_OPTIONS_GUARDS = (
    r"(?:还|又|再|多)(?:有|要|看|来|推荐)(?:别的|其他|其它|更多|几|一款|一个|几款|几个)",
    r"有没有(?:别的|其他|其它|更多)",
    r"(?:有没有|还有|有)(?:别的|其他|其它|另外)(?:的)?(?:推荐|选择|方案|产品|型号|款|屏)",
    r"(?:看看|看下|瞧瞧)(?:别的|其他|其它)",
    r"(?:再|多)推荐",
    r"多看(?:几)?(?:款|个)",
    # 实测 bug：客户说"你还能给我推荐其他的吗？"被当成"换产品" → 清空需求重问。
    # 下面这几条覆盖"推荐 + 其他/别的/更多"的常见口语顺序（中间可以有 能/给我/再）。
    r"(?:还|再|又|多)?(?:能|可以|能不能|可不可以)?(?:帮我|给我|帮忙)?(?:再)?推荐"
    r"[^。！？?]{0,8}?(?:其他|其它|别的|别款|更多|几款|几个|另一款)",
    r"(?:还有|有没有|有)[^。！？?]{0,4}(?:其他|其它|别的|更多)"
    r"[^。！？?]{0,8}?(?:推荐|型号|方案|选择|款|屏|吗|么)",
    r"其他(?:的)?(?:推荐|型号|方案|选择|款式)",
    r"\b(?:recommend|suggest|show)\s+(?:me\s+)?(?:others?|more|some\s+more|additional)\b",
    r"\b(?:any|some|more|other)\s+(?:other\s+)?(?:options?|models?|choices?|suggestions?|alternatives?)\b",
    r"\bothers?\s+(?:please|too)\b",
    r"\b(?:any|anything|something)\s+else\b",
    r"\bmore\s+(?:options|models|products|choices|recommendations)\b",
    r"\bother\s+(?:options|models|products|choices|recommendations)\b",
)

# 客户明确让销售"再来一款 / 另外推荐一款"（要更多型号）→ 这是"想看更多选项"，
# 不是换产品，绝不能清空需求重新采集（实测 bug：客户回"另外帮我推荐一款"，
# 系统回了句"我们重新来一遍"又开始问室内外）。
# 注意：这个判断要**优先于**强重置信号，因为 "另(?:外|…)" 也会命中强信号。
_OTHER_MODEL_REQUEST_RE = re.compile(
    r"(?:另外|再|又|还)(?:帮我|给我|帮忙)?(?:再)?推荐(?:一|几|两)?(?:款|个|种|台)"
    # 客户口语："你还能给我推荐其他的吗 / 还能再推荐几款吗 / 推荐点别的"
    r"|(?:还|再|又|多)?(?:能|可以|能不能|可不可以)?(?:帮我|给我|帮忙)?(?:再)?推荐"
    r"[^。！？?]{0,8}?(?:其他|其它|别的|别款|更多|几款|几个|另一款)"
    r"|(?:还有|有没有|有)[^。！？?]{0,4}(?:其他|其它|别的|更多)"
    r"[^。！？?]{0,8}?(?:推荐|型号|方案|选择|款)"
    r"|\b(?:recommend|suggest)\s+(?:me\s+)?(?:another|one more|a different)\b"
    # 英文："can you recommend others / any other options / more models"
    r"|\b(?:recommend|suggest|show)\s+(?:me\s+)?(?:others?|more|some\s+more|additional)\b"
    r"|\b(?:any|some|more|other)\s+(?:other\s+)?(?:options?|models?|choices?|suggestions?|alternatives?)\b",
    re.IGNORECASE,
)


# 重置后的口语确认（同一件事多种说法，按轮次轮换）
RESET_ACKS: Dict[str, tuple[str, ...]] = {
    "en": (
        "No problem — let's set the previous details aside and start over.",
        "Sure, let's collect the requirements again from the start.",
        "Got it, I'll treat this as a fresh enquiry.",
        "Understood — let me re-check your requirements for this one.",
    ),
    "zh": (
        "好的，我们把之前的需求放一边，重新来一遍。",
        "明白，那我重新帮您梳理一次需求。",
        "没问题，我们按新的需求重新收集。",
    ),
}


@dataclass
class ResetDecision:
    """会话内需求重置判定结果。"""

    should_reset: bool = False
    reason: str = ""            # explicit_request / display_type_switch / requirement_conflict / new_inquiry
    evidence: str = ""          # 命中的客户原话片段（日志用）
    conflicts: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_reset": self.should_reset,
            "reason": self.reason,
            "evidence": self.evidence,
            "conflicts": list(self.conflicts),
            "detail": self.detail,
        }


def _match_any(patterns: Tuple[str, ...], text: str) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


def _match_explicit_reset(text: str) -> Optional[str]:
    """客户是否明确要求"重新来 / 换一个"。强信号优先于"想看更多"的护栏。"""
    # "另外/再 帮我推荐一款" = 想看更多型号（给排行第二的备选），不是换产品。
    # 这一步必须放在强信号之前：强信号里的 "另(?:外|…)" 会把"另外"也当成重置。
    if _OTHER_MODEL_REQUEST_RE.search(text):
        return None
    strong = _match_any(_STRONG_RESET_PATTERNS, text)
    if strong:
        return strong
    if _match_any(_MORE_OPTIONS_GUARDS, text):
        return None
    return _match_weak_reset(text)


def _match_weak_reset(text: str) -> Optional[str]:
    """弱信号判定（同样要先过"想看更多选项"的护栏）。"""
    if _match_any(_MORE_OPTIONS_GUARDS, text):
        return None
    return _match_any(_WEAK_RESET_PATTERNS, text)


def _coerce_profile(profile: Any, requirements: Optional[Dict[str, Any]]) -> Any:
    """把 dict / 旧结构需求统一成 RequirementProfile（失败则返回 None）。"""
    if profile is None and requirements:
        profile = requirements
    if profile is None:
        return None
    if not isinstance(profile, dict):
        return profile
    try:
        from src.models.requirement import RequirementProfile

        data = dict(profile)
        if data.get("sources") is not None or data.get("purpose") is not None:
            return RequirementProfile.model_validate(data)
        return RequirementProfile.from_legacy(data)
    except Exception:  # pragma: no cover - 防御式
        return None


def _explicit_slots(message: str) -> Dict[str, Any]:
    """本轮消息里**客户明确说出**的槽位（排除系统推断出来的值）。"""
    try:
        from src.rag.query_understanding import extract_slots

        slots = dict(extract_slots(message) or {})
    except Exception:  # pragma: no cover - 防御式
        return {}
    inferred = {str(item) for item in (slots.pop("_inferred_slots", None) or [])}
    return {k: v for k, v in slots.items() if k not in inferred}


def find_requirement_conflicts(message: str, profile: Any) -> List[str]:
    """找出"本轮新说法"与"已确认需求"冲突的槽位。

    只检查会影响选型方向的硬约束（使用环境 / 使用场景）：
      - 客户之前确认"室内"，现在说"室外"        → 旧需求整体作废
      - 客户之前是"会议室"，现在说"教堂 / 商场"  → 旧需求整体作废

    尺寸 / 点间距 / 亮度这类可单独修改的参数**不算冲突**，
    只更新对应槽位即可，不需要清空重来。
    """
    if profile is None:
        return []

    explicit = _explicit_slots(message)
    conflicts: List[str] = []

    status = {}
    try:
        status = profile.status or {}
    except Exception:  # pragma: no cover - 防御式
        status = {}

    new_environment = explicit.get("environment")
    old_environment = getattr(profile, "environment", None)
    if (
        new_environment
        and old_environment
        and new_environment != old_environment
        and status.get("environment") == "confirmed"
    ):
        conflicts.append("environment")

    new_purpose = explicit.get("purpose")
    old_purpose = getattr(profile, "purpose", None)
    if new_purpose and old_purpose and new_purpose != old_purpose:
        conflicts.append("purpose")

    return conflicts


def detect_requirement_reset(
    message: str,
    *,
    requirements: Optional[Dict[str, Any]] = None,
    profile: Any = None,
    recommended: bool = False,
    display_type_change: Optional[Tuple[str, str]] = None,
) -> ResetDecision:
    """判断本轮是否需要清空已有需求、重新采集。

    Args:
        message: 客户本轮原话
        requirements: 会话里累计的需求（dict，可空）
        profile: 结构化需求档案（RequirementProfile 或 dict，可空）
        recommended: 本会话是否已经给过产品推荐
        display_type_change: 屏幕类型切换 (old, new)，由 runner 检测后传入
    """
    text = str(message or "").strip()
    if not text:
        return ResetDecision(detail="空消息，无需重置")

    if display_type_change:
        old_type, new_type = display_type_change
        return ResetDecision(
            should_reset=True,
            reason="display_type_switch",
            evidence=f"{old_type}→{new_type}",
            detail=f"客户把屏幕类型换成 {new_type}，之前的需求不再适用",
        )

    resolved_profile = _coerce_profile(profile, requirements)
    has_context = bool(requirements) or resolved_profile is not None

    matched = _match_explicit_reset(text)
    if matched:
        if not has_context and not recommended:
            return ResetDecision(detail="会话里还没有可清空的需求")
        return ResetDecision(
            should_reset=True,
            reason="explicit_request",
            evidence=text[:60],
            detail="客户明确要求重新来 / 换一个",
        )

    conflicts = find_requirement_conflicts(text, resolved_profile)
    if conflicts and recommended:
        return ResetDecision(
            should_reset=True,
            reason="requirement_conflict",
            evidence=text[:60],
            conflicts=conflicts,
            detail="客户给出的新需求与已推荐时的需求冲突："
                   + ", ".join(conflicts),
        )

    if recommended and _match_weak_reset(text):
        return ResetDecision(
            should_reset=True,
            reason="new_inquiry",
            evidence=text[:60],
            detail="客户提出另一个项目 / 另一块屏",
        )

    return ResetDecision(detail="未检测到需求变更")


def reset_acknowledgement(language: str = "en", seed: int = 0) -> str:
    """重置后的口语确认（同一语义多种说法，按 seed 轮换）。"""
    variants = RESET_ACKS.get(str(language or "en").lower()) or RESET_ACKS["en"]
    return variants[seed % len(variants)]


__all__ = [
    "RESET_ACKS",
    "ResetDecision",
    "detect_requirement_reset",
    "find_requirement_conflicts",
    "reset_acknowledgement",
]
