"""LCD / IFP 需求决策层 —— **唯一决策中心**（整改计划 §五/§九/§十三）。

    事实提取（lcd/parsing.py）→ RequirementProfile → **本模块** → 唯一 Next Action

本模块只负责：

    1. 当前 LCD 状态判断      2. 当前分支判断
    3. 判断下一步             4. 更新最终 LCD 状态
    5. 产生 Next Action

不做（整改计划 §九）：大量正则、参数解析、文本清洗、resolution/layout/seam/camera
parser、产品数据查询、RAG 检索、Response 文案。这些已经在：

    src/dialogue/lcd/parsing.py      事实解析（客户说了什么）
    src/dialogue/lcd/resolution.py   分辨率规则
    src/dialogue/lcd/splicing.py*(排布/拼缝解析在 parsing 内)
    src/dialogue/lcd/ifp.py          IFP 判定（§八：IFP 判断必须统一）
    src/dialogue/lcd/defaults.py     可选字段默认值

**这些 helper 一律不产生 next_action** —— "下一步做什么"只由本模块决定。
为兼容既有调用，helper 里的公共符号在本模块继续 re-export。

《LCD_IFP_需求链路工程化整改计划》Phase 5 / 6 / 7：

    当前消息 + 历史上下文 + 已确认档案 + 图片确认结果
        ↓
    唯一 Next Action（只问一个问题）

设计要点（对应计划章节）：

  · §六/§七 不靠"关键词 → 固定问题"：关键词只用来产出**候选事实**，
    判断"下一步问什么"由本模块结合上下文与档案统一决定；
  · §九~§十六 四个场景分支：Monitoring/Splicing、Advertising、Normal、
    Conference/Education →（手写/互动）→ IFP；
  · §十三/§十四 分辨率：客户明确优先；否则 <65"→2K、>65"→4K、
    65" 按产品库实际规格（2K/4K 都有 → 默认 4K，不再反问客户）；
  · §二十 一轮只问一个问题；§二十一 已锁定（客户来源）的字段永不重复问；
  · §十八/§十九 客户答非所问、多消息合并：只认"档案里新增了什么"，
    不看客户有没有答上一问。

本模块**不**做产品推荐（推荐仍走 RAG / RecommendationEngine）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ── 纯 helper（整改计划 §十）：事实解析 / 分辨率 / IFP 判定 / 默认值 ─────────
# 它们**只回答"客户说了什么"**；"下一步做什么"只由本模块决定。
from .lcd.defaults import (  # noqa: F401
    _DEGRADABLE_SLOTS,
    _DEGRADED_DEFAULTS,
    _DEGRADED_FIELD,
    _FILLABLE_DEFAULTS,
    _FILLABLE_FIELD,
    _SLOT_ASK_LIMIT,
    _apply_degraded_default,
    fill_defaults_for_recommendation,
)
from .lcd.ifp import _is_ifp_branch, effective_display_type, is_ifp_requirement  # noqa: F401
from .lcd.parsing import (  # noqa: F401
    _BEZEL_MM_RE,
    _BEZEL_RE,
    _CATEGORY_HINTS,
    _LAYOUT_RE,
    _LAYOUT_WORDS_RE,
    _LOCKED_CATEGORY_SOURCES,
    _PEOPLE_RE,
    _REAL_CATEGORIES,
    _ROOM_TYPES,
    _SIZE_INCH_RE,
    ADVERTISING,
    CONFERENCE_EDUCATION,
    LcdFacts,
    MONITORING,
    NORMAL,
    UNKNOWN,
    extract_lcd_facts,
)
from .lcd.resolution import catalog_resolutions_for_size, resolution_for_size  # noqa: F401


logger = logging.getLogger(__name__)


_UNKNOWN_ORDER: Tuple[str, ...] = (
    "purpose", "lcd_size", "environment", "lcd_touch",
)
_BRANCH_ORDER: Dict[str, Tuple[str, ...]] = {
    UNKNOWN: _UNKNOWN_ORDER,
    MONITORING: ("environment", "lcd_splicing", "lcd_layout", "lcd_bezel", "lcd_size"),
    ADVERTISING: ("environment", "lcd_size", "lcd_touch"),
    NORMAL: ("environment", "lcd_size"),
    CONFERENCE_EDUCATION: (
        "environment", "lcd_size", "lcd_handwriting",
        "lcd_tender", "lcd_ops", "lcd_camera",
    ),
}

# 这些问题在"不是 IFP / 不是会议场景"时不许问（计划 §十一：广告机不主动问 OPS）
IFP_ONLY_SLOTS: frozenset[str] = frozenset({"lcd_tender", "lcd_ops", "lcd_camera"})


_AFFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|correct|right|that'?s right|sounds good|"
    r"works for me|fine|对|是的|没错|可以|行|好的|嗯|是的，)",
    re.IGNORECASE,
)
_DENY_RE = re.compile(
    r"^\s*(?:no|nope|not|don'?t|does ?n'?t|不需要|不用|不是|不对|没有)",
    re.IGNORECASE,
)

# 客户**没指定**时可以按业务默认补上的字段（计划 §十/§十一/§十五/§十六）：


# 客户明确要推荐 / 报价 → 立刻用可补默认值收口，不再追问可选字段
_EXPLICIT_RECO_RE = re.compile(
    r"\b(?:recommend|suggest|quote|quotation|proposal|pro forma)\b|推荐|报价|方案|选型",
    re.IGNORECASE,
)

# LCD 槽位（用于"客户在回答哪一项"的判断）
_LCD_SLOTS: frozenset[str] = frozenset(
    {
        "purpose", "lcd_category", "lcd_size", "lcd_resolution", "lcd_splicing",
        "lcd_layout", "lcd_screen_count", "lcd_bezel", "lcd_touch",
        "lcd_handwriting", "lcd_tender", "lcd_ops", "lcd_camera", "lcd_room_type",
        "environment",
    }
)

_BEZEL_IN_TEXT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm", re.IGNORECASE)

# 客户"明确不要"的说法（用于 camera / touch / handwriting 这类"看见了但不需要"）


@dataclass
class LcdNextAction:
    """唯一 Next Action（计划 §二十六 的结构）。"""

    product_type: str = "lcd"
    lcd_category: str = UNKNOWN
    confirmed: bool = False
    updated_fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    next_action: str = ""
    question: str = ""
    question_slot: str = ""
    response_context: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    locked_facts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_type": self.product_type,
            "lcd_category": self.lcd_category,
            "confirmed": self.confirmed,
            "updated_fields": dict(self.updated_fields),
            "missing_fields": list(self.missing_fields),
            "next_action": self.next_action,
            "question": self.question,
            "question_slot": self.question_slot,
            "response_context": dict(self.response_context),
            "conflicts": list(self.conflicts),
            "locked_facts": dict(self.locked_facts),
        }


# ── 3) 决策主入口（Phase 5）───────────────────────────────────────────────
def decide_lcd_next_action(
    profile: Any,
    message: str = "",
    *,
    facts: Optional[LcdFacts] = None,
    recognition: Any = None,
    explicit_recommend: bool = False,
) -> LcdNextAction:
    """唯一的 LCD 下一步决策入口。

    Args:
        profile: RequirementProfile（**已经**被事实抽取更新过；本函数只读 + 记录）
        message: 客户这一轮的原话（用于承接语与候选事实）
        facts: 预先抽好的候选事实（没给就现抽）
        recognition: ImageRecognitionResult（有图片时用来算冲突）
    """
    facts = facts or extract_lcd_facts(message, profile=profile)
    action = LcdNextAction()
    if profile is None:
        return action

    sources = dict(getattr(profile, "sources", None) or {})
    category = _resolve_category(profile, facts)
    action.lcd_category = category
    action.updated_fields = facts.to_dict()
    action.locked_facts = _locked_facts(profile, sources)
    action.conflicts = list(getattr(profile, "conflicts", None) or [])

    if recognition is not None:
        # 图片与文字冲突 → 先请客户确认（计划 §二十三）
        try:
            from .image_confirmation import compare_with_context

            compare = compare_with_context(profile, recognition)
            if compare.has_conflict:
                action.next_action = "confirm_image_conflict"
                action.question_slot = compare.conflicts[0]
                action.question = _conflict_question(compare)
                action.response_context = _response_context(facts, ask_one=True)
                return action
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[LCD] image compare failed: %s", exc)

    # 客户刚说"这是招投标项目" → 先引导发招标文件（计划 §十六），一次即止。
    if facts.tender is True:
        action.next_action = "request_tender_documents"
        action.question_slot = "lcd_tender"
        action.question = (
            "Could you send over the tender documents so I can match the specification?"
        )
        action.missing_fields = []
        action.confirmed = False
        action.response_context = _response_context(facts, ask_one=True)
        return action

    missing = _missing_slots(profile, category, sources, facts=facts)
    # 客户明确要推荐 / 报价 → 可选字段（拼缝、触控、手写、OPS、摄像头…）按业务
    # 默认值补齐（计划 §十：客户没指定拼缝 → 推荐 3.5mm），不再卡着追问。
    if explicit_recommend and missing:
        applied = fill_defaults_for_recommendation(profile, missing)
        if applied:
            action.response_context["defaults_applied"] = applied
            sources = dict(getattr(profile, "sources", None) or {})
            missing = _missing_slots(profile, category, sources, facts=facts)
    action.missing_fields = list(missing)
    slot = missing[0] if missing else ""
    action.question_slot = slot
    action.next_action = f"ask_{slot}" if slot else "ready"
    action.confirmed = not missing
    action.question = _question_for(slot) if slot else ""
    action.response_context = _response_context(facts, ask_one=bool(slot))
    return action


def _resolve_category(profile: Any, facts: LcdFacts) -> str:
    """场景分支：**语境理解**定品类；关键词只做兜底，不能推翻已定的品类。

    客户口径（2026-09-30）："不允许有关键词来触发 AI 的某个链路，需要让 AI 结合
    语境上下文去选择品类。" 所以优先级是：

        语境理解（本轮改口）> 档案里已锁定的品类 > 语境理解（本轮首次定下）
        > 关键词兜底 > 拼接墙兜底 > unknown
    """
    current = str(getattr(profile, "lcd_category", None) or "").strip().lower()
    if current not in _REAL_CATEGORIES:
        current = ""
    incoming = facts.category if facts.category in _REAL_CATEGORIES else ""
    source = str(getattr(facts, "category_source", "") or "")

    if incoming and incoming != current:
        if current and source != "understanding":
            # 关键词只是候选事实，不能把已经定下的品类改掉
            return current
        return incoming
    if current:
        return current
    # 没有任何场景线索，但已经说是拼接墙 → 监控/拼接分支
    if getattr(profile, "lcd_is_splicing", None) is True:
        return MONITORING
    return UNKNOWN


def category_is_locked(profile: Any) -> bool:
    """品类是否已经"锁定"（客户说过 / 语境已理解出来）——锁定了就不再改口。"""
    if profile is None:
        return False
    current = str(getattr(profile, "lcd_category", None) or "").strip().lower()
    if current not in _REAL_CATEGORIES:
        return False
    source = str((getattr(profile, "sources", None) or {}).get("lcd_category") or "")
    return source in _LOCKED_CATEGORY_SOURCES


def _locked_facts(profile: Any, sources: Dict[str, str]) -> Dict[str, Any]:
    """已经定下来的事实（计划 §二十一）——给表达层，让它不要再问。

    来源口径（计划：品类由语境决定）：
      · explicit / confirmed …… 客户说过
      · understanding ………… 语境理解已经判出来（客户口径 2026-09-30 的主路径）
      · keyword ………………… 只有品类字段认：关键词兜底选出的场景**已经决定**
        了接下来问哪些问题，表达层必须当它已经定了，否则又会回头问用途。
    """
    locked: Dict[str, Any] = {}
    for field_name in (
        "display_type", "environment", "installation", "purpose",
        "lcd_category", "lcd_size_inch", "lcd_resolution", "lcd_is_splicing",
        "lcd_splicing_layout", "lcd_screen_count", "lcd_bezel_mm",
        "lcd_touch_required", "lcd_handwriting_required",
        "lcd_tender_project", "lcd_ops_required", "lcd_camera_required",
    ):
        value = getattr(profile, field_name, None)
        if value in (None, "", [], {}):
            continue
        source = str(sources.get(field_name) or "")
        if source in _LOCKED_CATEGORY_SOURCES:
            locked[field_name] = value
        elif field_name == "lcd_category" and source == "keyword":
            locked[field_name] = value
    return locked


def _missing_slots(
    profile: Any, category: str, sources: Dict[str, str], *, facts: Optional[LcdFacts] = None
) -> List[str]:
    """分支内"仍然缺失且值得问"的槽位（已锁定的一律不问）。"""
    order = _BRANCH_ORDER.get(category)
    if order is None:
        order = _UNKNOWN_ORDER
    is_ifp_branch = _is_ifp_branch(profile)
    missing: List[Tuple[int, str]] = []
    for index, slot in enumerate(order):
        if slot in IFP_ONLY_SLOTS and not is_ifp_branch:
            continue
        # 摄像头（计划 §十六）：会议场景可以问；学校/课堂只在上下文提到时才问
        if slot == "lcd_camera" and not _camera_question_allowed(profile, facts):
            continue
        # 单体屏不问排布：没有拼接就不存在"几列几行"（实测 2026-09-30：
        # 客户已经说 single displays，需求链还在问排布）
        if slot == "lcd_layout" and getattr(profile, "lcd_is_splicing", None) is False:
            continue
        asked_count = _ask_count(profile, slot)
        # 可选字段（拼缝默认 3.5mm、触控/手写/OPS/摄像头默认不需要）已经问过一次
        # 就按默认走，不再重复问 —— 否则会出现"拼缝 → 尺寸 → 排布 → 拼缝"的死循环
        # （实测 2026-09-30）。
        if slot in _FILLABLE_DEFAULTS:
            if asked_count >= 1 and getattr(
                profile,
                _FILLABLE_FIELD.get(slot, slot),
                None,
            ) in (None, "", [], {}):
                fill_defaults_for_recommendation(profile, [slot])
                sources = dict(getattr(profile, "sources", None) or {})
                continue
        # 问满上限还是没答 → 收口（绝不再问第三次，保证链路能走到推荐）
        if slot not in _FILLABLE_DEFAULTS and asked_count >= _SLOT_ASK_LIMIT:
            if _apply_degraded_default(profile, slot):
                sources = dict(getattr(profile, "sources", None) or {})
                continue
            if slot in _DEGRADABLE_SLOTS:
                logger.info(
                    "[LCD] slot=%s 问了 %d 次仍未答 → 不阻塞推荐（不编造值）",
                    slot, asked_count,
                )
                continue
        if _slot_answered(profile, slot, sources):
            continue
        missing.append((index, slot))
    # 问过的排到后面（客户答非所问时改问下一项，而不是把同一项再问一遍）
    missing.sort(key=lambda item: (_ask_count(profile, item[1]), item[0]))
    return [slot for _index, slot in missing]


def _ask_count(profile: Any, slot: str) -> int:
    """这个槽位已经问过几次（拿不到就按 0 算）。"""
    try:
        return int(profile.ask_count(slot))
    except Exception:  # pragma: no cover - 防御式
        return 0


def _camera_question_allowed(profile: Any, facts: Optional[LcdFacts]) -> bool:
    """要不要问摄像头：客户提过（yes/no 都算）或就是会议室；课堂不强制问。"""
    if facts is not None and facts.camera is not None:
        return True
    room = str(getattr(profile, "lcd_room_type", "") or "").strip().lower()
    return room == "meeting room"


def apply_answer_to_previous(
    profile: Any,
    message: str,
    *,
    last_question: str = "",
) -> Dict[str, Any]:
    """客户用 "yes / 对 / 可以"（或否认）回答上一轮 LCD 问题时，把提案落到档案。

    实测（2026-09-30）：AI 问 "would something around 3.5mm work for you?"，
    客户答 "yes" —— 旧实现只认"消息里有没有 3.5"，于是拼缝永远没记下来，
    需求链在 拼缝 → 尺寸 → 排布 → 拼缝 之间来回问，永远不推荐。

    规则：
      · 上一轮问的是哪一项（profile.last_asked_slot / 问题文本）→ 客户 yes 就采用
        **上一轮提出的那个值**（问题里的 3.5mm / 65" / 3x3 等）；
      · 布尔槽位（拼接 / 触控 / 手写 / OPS / 摄像头 / 招标）→ yes=True，no=False；
      · 值来源写 "explicit"（客户确认过就是客户的事实）。
    """
    stats: Dict[str, Any] = {}
    if profile is None:
        return stats
    text = str(message or "").strip()
    if not text:
        return stats
    slot = str(getattr(profile, "last_asked_slot", "") or "").strip()
    if slot not in _LCD_SLOTS:
        return stats
    affirmed = bool(_AFFIRM_RE.match(text))
    denied = bool(_DENY_RE.match(text))
    if not affirmed and not denied:
        return stats
    question = str(last_question or "")
    sources = dict(getattr(profile, "sources", None) or {})

    def write(field_name: str, value: Any, source: str = "explicit") -> None:
        setattr(profile, field_name, value)
        sources[field_name] = source
        stats[field_name] = value

    if slot == "lcd_bezel":
        proposed = _BEZEL_IN_TEXT_RE.search(question)
        value = float(proposed.group(1).replace(",", ".")) if proposed else 3.5
        write("lcd_bezel_mm", value)
    elif slot == "lcd_splicing":
        write("lcd_is_splicing", not denied)
    elif slot == "lcd_touch":
        write("lcd_touch_required", not denied)
    elif slot == "lcd_handwriting":
        write("lcd_handwriting_required", not denied)
    elif slot == "lcd_tender":
        write("lcd_tender_project", not denied)
    elif slot == "lcd_ops":
        write("lcd_ops_required", not denied)
    elif slot == "lcd_camera":
        write("lcd_camera_required", not denied)
    elif slot == "lcd_size":
        match = _SIZE_INCH_RE.search(question)
        if match:
            write("lcd_size_inch", float(match.group(1).replace(",", ".")))
            write("lcd_size_source", "customer_confirmed", source="explicit")
    elif slot == "lcd_layout":
        match = _LAYOUT_RE.search(question)
        if match:
            write("lcd_splicing_layout", f"{match.group(1)}x{match.group(2)}")
            write("lcd_screen_count", int(match.group(1)) * int(match.group(2)))
    elif slot == "lcd_resolution":
        # "Do you need 4K, or is 2K enough?" → "yes" 有歧义，不猜（计划 §十四）
        pass
    elif slot == "environment":
        # 客户确认的是环境：只有问题里提出了具体环境才写
        if re.search(r"indoor", question, re.IGNORECASE):
            write("environment", "indoor")
        elif re.search(r"outdoor", question, re.IGNORECASE):
            write("environment", "outdoor")
    profile.sources = sources
    return stats


def _slot_answered(profile: Any, slot: str, sources: Dict[str, str]) -> bool:
    from src.models.requirement import SLOT_TO_FIELD

    field_name = SLOT_TO_FIELD.get(slot, slot)
    value = getattr(profile, field_name, None)
    if value in (None, "", [], {}):
        return False
    # 图片**刚看到、还没跟客户核过**的还不算"客户已经回答"（计划 §三：看见 ≠ 需要）。
    # 但 vision_accepted 不一样 —— 那是"识别结果已经摆给客户核对过、客户没有纠正"
    # （客户口径 2026-09-30：客户没正面回答就按识别出来的走）→ 已经定下来了，
    # 不能再问一遍。实测 bug：图片说 indoor，客户纠正类型后系统又问"室内还是室外"。
    source = str(sources.get(field_name) or "")
    return source not in ("vision_explicit", "vision_inferred")


def _question_for(slot: str) -> str:
    return {
        "purpose": "What will the screens be used for (for example a control room, a meeting room, or advertising)?",
        "environment": "Will they be used indoors or outdoors?",
        "lcd_splicing": "Do you need a video wall (spliced screens) or single displays?",
        "lcd_layout": "How should the video wall be arranged (for example 6x2)?",
        "lcd_bezel": "How narrow does the bezel need to be (for example 3.5mm)?",
        "lcd_size": "What screen size do you have in mind (in inches)?",
        "lcd_resolution": "Do you need 4K, or is 2K enough?",
        "lcd_touch": "Do you need touch functionality?",
        "lcd_handwriting": "Do you need handwriting / whiteboard capability?",
        "lcd_tender": "Is this a tender project?",
        "lcd_ops": "Do you need an OPS slot (a built-in PC module)?",
        "lcd_camera": "Do you need a camera for video meetings?",
    }.get(slot, "")


def _conflict_question(compare: Any) -> str:
    return (
        "You mentioned one thing in text while the image shows another "
        f"({', '.join(compare.conflicts)}). Which one should I use?"
    )


def _response_context(facts: LcdFacts, *, ask_one: bool) -> Dict[str, Any]:
    """回复生成层需要知道的"这一轮允许表达什么"（计划 §二十六）。"""
    return {
        "acknowledge": True,
        "mention_locked_facts": True,
        "ask_one_question": bool(ask_one),
        "new_information": bool(facts.to_dict()),
    }


# ── 4) 把这一轮的事实写进档案（§十七/§十八/§二十一）──────────────────────
def _write_category(
    profile: Any,
    category: str,
    sources: Dict[str, str],
    stats: Dict[str, Any],
    *,
    source: str,
) -> None:
    """品类入档：来源锁定后不许被关键词推翻，只有"语境理解"能改口径。"""
    if category not in _REAL_CATEGORIES:
        return
    current = str(getattr(profile, "lcd_category", None) or "").strip().lower()
    if current not in _REAL_CATEGORIES:
        current = ""
    current_source = str(sources.get("lcd_category") or "")
    if (
        current
        and current != category
        and current_source in _LOCKED_CATEGORY_SOURCES
        and source != "understanding"
    ):
        stats["kept"].append("lcd_category")
        return
    if current == category and current_source in _LOCKED_CATEGORY_SOURCES and source == "keyword":
        return
    setattr(profile, "lcd_category", category)
    sources["lcd_category"] = source
    stats["updated"].append("lcd_category")


def apply_lcd_facts(profile: Any, facts: LcdFacts) -> Dict[str, Any]:
    """客户这一轮说的 LCD 事实入档：**客户明说 > 已有图片来源**。

    客户答非所问也没关系 —— 只要说了就记下来（计划 §十八），
    下一轮就不会再问同一项（§二十一）。
    """
    stats: Dict[str, Any] = {"updated": [], "kept": []}
    if profile is None or facts is None:
        return stats
    sources = dict(getattr(profile, "sources", None) or {})

    def write(field_name: str, value: Any, *, source: str = "explicit") -> None:
        if value in (None, "", []):
            return
        current = getattr(profile, field_name, None)
        current_source = str(sources.get(field_name) or "")
        if current not in (None, "", [], {}) and current_source in (
            "explicit", "confirmed", "customer_explicit", "customer_confirmed",
        ) and current != value:
            # 客户之前明确说过的值，本轮不是明确纠正就不覆盖
            stats["kept"].append(field_name)
            return
        setattr(profile, field_name, value)
        sources[field_name] = source
        stats["updated"].append(field_name)

    # 品类（计划：语境理解定品类）：写的时候带上来源，来源为 understanding / explicit
    # 的品类会被"锁定"——之后的分析里，关键词再也改不动它（除非客户改口）。
    if facts.category in _REAL_CATEGORIES:
        _write_category(
            profile,
            facts.category,
            sources,
            stats,
            source=str(getattr(facts, "category_source", "") or "") or "explicit",
        )
    if facts.room_type:
        write("lcd_room_type", facts.room_type)
        if not getattr(profile, "purpose", None):
            write("purpose", facts.room_type)
    if facts.environment:
        write("environment", facts.environment)
    if facts.installation:
        write("installation", facts.installation)
    if facts.people_count is not None:
        write("audience_count", facts.people_count)
    if facts.size_inch is not None:
        write("lcd_size_inch", facts.size_inch)
        write("lcd_size_source", "customer_explicit")
    if facts.is_splicing is not None:
        write("lcd_is_splicing", facts.is_splicing)
    if facts.layout:
        write("lcd_splicing_layout", facts.layout)
    if facts.screen_count is not None:
        write("lcd_screen_count", facts.screen_count)
    if facts.bezel_mm is not None:
        write("lcd_bezel_mm", facts.bezel_mm)
    if facts.touch is not None:
        write("lcd_touch_required", facts.touch)
    if facts.handwriting is not None:
        write("lcd_handwriting_required", facts.handwriting)
    if facts.tender is not None:
        write("lcd_tender_project", facts.tender)
    if facts.ops is not None:
        write("lcd_ops_required", facts.ops)
    if facts.camera is not None:
        write("lcd_camera_required", facts.camera)

    # 分辨率（Phase 7 / §十三/§十四）：客户明确优先，否则按尺寸规则
    if facts.resolution:
        write("lcd_resolution", facts.resolution)
        write("lcd_resolution_source", "customer_explicit")
    else:
        size = getattr(profile, "lcd_size_inch", None)
        if size is not None and not getattr(profile, "lcd_resolution", None):
            resolution, rule = resolution_for_size(
                float(size),
                catalog_resolutions=catalog_resolutions_for_size(float(size)),
            )
            if resolution:
                write("lcd_resolution", resolution)
                write("lcd_resolution_source", rule)

    profile.sources = sources
    return stats


def lcd_turn(
    profile: Any,
    message: str = "",
    *,
    recognition: Any = None,
    last_question: str = "",
    category_signal: Optional[Dict[str, Any]] = None,
    fact_signal: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, LcdNextAction]:
    """Sales / Solution 调用的一站式入口：抽事实 → 入档 → 决定下一步。

    ``last_question``：上一轮 AI 问的原话（客户用 "yes" 回答时，要能接住其中的提案，
    例如 "would something around 3.5mm work?" → yes → 拼缝 3.5mm）。

    ``category_signal``：语境理解的品类结论（``lcd_category_understanding`` 的返回）。
    给了就用它定品类；没给（模型不可用 / 单测直调）时退回关键词兜底。

    ``fact_signal``：语境理解这一轮给出的事实（同一个返回里的 ``facts``）。
    给了就以它为准，关键词只在它没覆盖到的字段上兜底（客户口径 2026-09-30：
    "只允许结合上下文语境整理需求，关键词只能是兜底"）。
    """
    # ① 先把"客户在回答上一轮 LCD 问题"的情况落地（yes / no + 提案值）
    answered = apply_answer_to_previous(profile, message, last_question=last_question)
    # ② 再抽这一轮的事实（客户自己说的值优先）
    # asked_slot：客户这一句是**在回答我们上一轮问的那一项** —— 排布 / 拼缝 / 尺寸
    # 的解析都要靠它，"3x3" 才算排布、"0.88mm" 才算拼缝（见 extract_lcd_facts 注释）。
    facts = extract_lcd_facts(
        message,
        profile=profile,
        category_signal=category_signal,
        asked_slot=str(getattr(profile, "last_asked_slot", "") or ""),
        fact_signal=fact_signal if fact_signal is not None else category_signal,
    )
    stats = apply_lcd_facts(profile, facts)
    if answered:
        stats["answered_previous"] = answered
    cleared = _clear_resolved_conflicts(profile, facts)
    if cleared:
        stats["conflicts_cleared"] = cleared
    explicit_recommend = bool(_EXPLICIT_RECO_RE.search(str(message or "")))
    action = decide_lcd_next_action(
        profile,
        message,
        facts=facts,
        recognition=recognition,
        explicit_recommend=explicit_recommend,
    )
    action.response_context["applied"] = stats
    action.response_context["explicit_recommend"] = explicit_recommend
    action.response_context["category_locked"] = category_is_locked(profile)
    if isinstance(category_signal, dict) and category_signal:
        action.response_context["category_signal"] = dict(category_signal)
    return profile, action


def _clear_resolved_conflicts(profile: Any, facts: LcdFacts) -> List[str]:
    """客户已经明确给了值 → 之前登记的"图文冲突"自动消解（计划 §二十三）。"""
    if profile is None:
        return []
    conflict_slots = list(getattr(profile, "conflict_slots", None) or [])
    if not conflict_slots:
        return []
    explicit_now = {
        "environment": facts.environment,
        "installation": facts.installation,
        "lcd_splicing": facts.is_splicing,
        "lcd_size": facts.size_inch,
        "display_type": str(getattr(profile, "display_type", None) or ""),
    }
    cleared: List[str] = []
    remaining: List[str] = []
    for slot in conflict_slots:
        value = explicit_now.get(slot)
        if value not in (None, "", []):
            cleared.append(slot)
        else:
            remaining.append(slot)
    if cleared:
        profile.conflict_slots = remaining
        profile.conflicts = [
            note for note in (getattr(profile, "conflicts", None) or [])
            if not any(str(slot) in str(note) for slot in cleared)
        ]
    return cleared

__all__ = [
    "ADVERTISING",
    "CONFERENCE_EDUCATION",
    "IFP_ONLY_SLOTS",
    "LcdFacts",
    "LcdNextAction",
    "MONITORING",
    "NORMAL",
    "UNKNOWN",
    "apply_lcd_facts",
    "catalog_resolutions_for_size",
    "category_is_locked",
    "decide_lcd_next_action",
    "effective_display_type",
    "extract_lcd_facts",
    "is_ifp_requirement",
    "lcd_turn",
    "resolution_for_size",
]
