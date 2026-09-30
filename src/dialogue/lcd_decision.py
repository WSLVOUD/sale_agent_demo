"""LCD / IFP 需求决策层（LCD Requirement Decision）。

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
    65" 按产品库实际规格（查不到就不猜，交给客户/业务规则）；
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

logger = logging.getLogger(__name__)

# ── 场景（LCD 分支）─────────────────────────────────────────────────────────
MONITORING = "monitoring"
ADVERTISING = "advertising"
NORMAL = "normal"
CONFERENCE_EDUCATION = "conference_education"
UNKNOWN = "unknown"

# 真正能定分支的品类（UNKNOWN 不算）
_REAL_CATEGORIES: Tuple[str, ...] = (
    MONITORING,
    ADVERTISING,
    NORMAL,
    CONFERENCE_EDUCATION,
)

# 品类来源里"已经定下来、不许被关键词推翻"的来源（客户口径 2026-09-30：锁定屏幕）
_LOCKED_CATEGORY_SOURCES: frozenset[str] = frozenset(
    {"understanding", "explicit", "confirmed", "customer_explicit", "customer_confirmed"}
)

# 关键词只产出**候选**场景；真正定分支还要看上下文（计划 §九/§二十七）
_CATEGORY_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (MONITORING, (
        "control room", "monitoring", "surveillance", "security", "command center",
        "video wall", "splicing", "spliced", "nvr", "cctv", "监控", "指挥中心", "拼接",
    )),
    (ADVERTISING, (
        "advertising", "advertisement", "digital signage", "signage", "billboard",
        "menu board", "kiosk", "广告", "广告机", "标牌", "导视",
    )),
    (CONFERENCE_EDUCATION, (
        "meeting", "meeting room", "conference", "classroom", "school", "university", "lecture",
        "training", "whiteboard", "education", "会议室", "教室", "学校", "培训", "白板",
    )),
)

# 场景 → 分支内的字段顺序（只问缺失且值得问的；OPS 只对 IFP 问，计划 §十一/§十六）
# UNKNOWN：还没场景线索时先问用途；用途问过仍没答就往前推进（§十八/§二十一：
# 客户答非所问也要往前，不反复问同一项）。
# 客户口径（2026-09-30）：**分辨率不许问客户** —— 客户自己提了（"要 4K"）就照他的，
# 没提就按尺寸规则定（<65" → 2K，≥65" → 4K，见 resolution_for_size）。
# 所以 lcd_resolution 从所有分支的提问顺序里拿掉。
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

_SIZE_INCH_RE = re.compile(
    # 客户写法很随意：65" / 65' / 65’ / 65 in / 65-inch / 65英寸 / 65寸
    # （实测：客户发 "65'" 时旧写法解析不出来 → 又被问了一遍尺寸）
    r"(?<![a-z0-9])(\d{2,3}(?:[.,]\d+)?)\s*"
    r"(?:\"|''|'|’|′|''|inch(?:es)?|in\b|-?inch|-?in\b|”|″|英寸|吋|寸)",
    re.IGNORECASE,
)
_LAYOUT_RE = re.compile(r"(?<![a-z0-9])(\d{1,2})\s*[x×*]\s*(\d{1,2})(?![a-z0-9])", re.IGNORECASE)
# "3 rows and 3 columns" / "3row 3columns" / "3列3行" / "3行3列"
_LAYOUT_WORDS_RE = re.compile(
    r"(?<![a-z0-9])(\d{1,2})\s*(?:rows?|行)\s*(?:and|,|，|和|\+)?\s*(\d{1,2})\s*(?:cols?|columns?|列)"
    r"|(?<![a-z0-9])(\d{1,2})\s*(?:cols?|columns?|列)\s*(?:and|,|，|和|\+)?\s*(\d{1,2})\s*(?:rows?|行)",
    re.IGNORECASE,
)
_BEZEL_RE = re.compile(r"(?:bezel|拼缝|边框)\D{0,8}(\d+(?:[.,]\d+)?)\s*mm", re.IGNORECASE)
# 只写数字 + mm（没有 bezel 字样）：客户答上一轮"拼缝要多窄"时就是这种写法
_BEZEL_MM_RE = re.compile(r"(?<![\d.])(\d+(?:[.,]\d+)?)\s*mm\b", re.IGNORECASE)
_PEOPLE_RE = re.compile(r"(\d{1,4})\s*(?:people|persons|students|seats|人)", re.IGNORECASE)
_ROOM_TYPES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("control room", ("control room", "监控室", "指挥中心", "control centre", "control center")),
    ("meeting room", ("meeting room", "conference room", "boardroom", "会议室")),
    ("classroom", ("classroom", "class room", "教室", "课堂")),
    ("lobby", ("lobby", "reception", "大堂", "前台")),
    ("retail store", ("retail", "store", "shop", "商场", "门店", "店铺")),
)

# 客户"回答上一轮 LCD 问题"的确认/否认说法
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
#   · 拼缝：客户没有指定 → 推荐 3.5mm（§十 明文）
#   · 手写/触控/OPS/摄像头：客户没提 → 不作为必需项（走对应的 NO 分支）
# 补的值来源标 "recommended"，不会伪装成客户要求（计划 §八）。
_FILLABLE_DEFAULTS: Dict[str, Any] = {
    "lcd_bezel": 3.5,
    "lcd_touch": False,
    "lcd_handwriting": False,
    "lcd_ops": False,
    "lcd_camera": False,
    "lcd_tender": False,
}
_FILLABLE_FIELD: Dict[str, str] = {
    "lcd_bezel": "lcd_bezel_mm",
    "lcd_touch": "lcd_touch_required",
    "lcd_handwriting": "lcd_handwriting_required",
    "lcd_ops": "lcd_ops_required",
    "lcd_camera": "lcd_camera_required",
    "lcd_tender": "lcd_tender_project",
}

# ── 防死循环（客户口径 2026-09-30："还是没有触发推荐"）──────────────────────
# 实测：视频墙对话里客户答的排布是 "3x3"，旧解析没记下来 → 需求链在"问排布"
# 上无限循环，永远不推荐。除了修解析，这里再加一道**收口**规则：
#
#   同一个槽位问满 _SLOT_ASK_LIMIT 次还拿不到答案 → 绝不再问第三次。
#   · 有安全业务默认值的（LCD 几乎全是室内屏）→ 按默认补齐（来源标 recommended）
#   · 没有安全默认值的（排布）→ 不阻塞推荐，也不编造：直接带着"待确认排布"推荐
#
# LED 链路不经过这个函数（计划 §二十九 LED Chain = Frozen），所以 LED 口径不受影响。
_SLOT_ASK_LIMIT = 2
_DEGRADED_DEFAULTS: Dict[str, Any] = {
    "environment": "indoor",
}
_DEGRADED_FIELD: Dict[str, str] = {
    "environment": "environment",
}
_DEGRADABLE_SLOTS: frozenset[str] = frozenset({"lcd_layout"})
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
_NEGATED_NEED_RE = re.compile(
    r"\b(no|not|don'?t|doesn'?t|isn'?t|without|only in the (photo|picture|image)|"
    r"just in the (photo|picture|image))\b[^.!?]{0,40}"
    r"\b(need|needed|required|require|necessary|want|use|for)\b"
    r"|\b(no|not|don'?t)\s+need\b"
    r"|不需要|不用|没必要|只是图片里|图片里有但",
    re.IGNORECASE,
)


def _negated_need(text: str) -> bool:
    """这句是不是在"明确不要"（看见 ≠ 需要 → 客户否认时记 false）。"""
    return bool(_NEGATED_NEED_RE.search(str(text or "")))


@dataclass
class LcdFacts:
    """本轮从客户消息里抽出的**候选**事实（关键词/语义只在这一层起作用）。"""

    category: str = UNKNOWN
    # 品类的来源：understanding（语境理解，客户口径要求的主路径）
    #            / keyword（关键词兜底，只在语境理解不可用时用）
    category_source: str = ""
    room_type: str = ""
    size_inch: Optional[float] = None
    resolution: str = ""
    is_splicing: Optional[bool] = None
    layout: str = ""
    screen_count: Optional[int] = None
    bezel_mm: Optional[float] = None
    touch: Optional[bool] = None
    handwriting: Optional[bool] = None
    tender: Optional[bool] = None
    ops: Optional[bool] = None
    camera: Optional[bool] = None
    people_count: Optional[int] = None
    environment: str = ""
    installation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "", UNKNOWN)}


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


# ── 1) 分辨率确定性规则（Phase 7 / §十三 / §十四）──────────────────────────
def resolution_for_size(
    size_inch: Optional[float],
    *,
    customer_resolution: str = "",
    catalog_resolutions: Optional[Iterable[str]] = None,
) -> Tuple[str, str]:
    """返回 ``(resolution, source)``。

    优先级：客户明确 > 尺寸规则（<65"→2K，>65"→4K）> 产品库实际规格（65" 边界）。

    客户口径（2026-09-30）：**分辨率不许反问客户**。"客户不说，就按大于 65 选 4K、
    小于 65 选 2K / 1080p"。恰好 65" 时先看产品库：只有一种规格就用它；
    两种都有（现在就是 2K/4K 都有）就按"大屏走 4K"的口径默认 4K，不再问客户。
    """
    explicit = _normalize_resolution(customer_resolution)
    if explicit:
        return explicit, "customer_explicit"
    if size_inch is None:
        return "", "unknown"
    size = float(size_inch)
    if size < 65:
        return "2K", "size_rule_lt_65"
    if size > 65:
        return "4K", "size_rule_gt_65"
    catalog = {_normalize_resolution(item) for item in (catalog_resolutions or [])}
    catalog.discard("")
    if len(catalog) == 1:
        return catalog.pop(), "product_catalog"
    return "4K", "size_default_4k_65"


def _normalize_resolution(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return ""
    if text in ("2K", "1080P", "FHD", "1920X1080", "1920*1080", "FULLHD"):
        return "2K"
    if text in ("4K", "2160P", "UHD", "3840X2160", "3840*2160", "4KUHD"):
        return "4K"
    return ""


def catalog_resolutions_for_size(size_inch: float, data_dir: str = "") -> List[str]:
    """产品库里该尺寸档已有的分辨率（65" 边界用；查不到就返回空）。"""
    try:
        import json
        import os

        from src.config import config

        path = os.path.join(data_dir or config.DATA_DIR, "lcd_products.json")
        with open(path, encoding="utf-8") as handle:
            products = json.load(handle)["products"]
    except Exception:  # pragma: no cover - 防御式
        return []
    found: List[str] = []
    for item in products:
        try:
            if abs(float(str(item.get("display_size_inch", "")).strip('"')) - float(size_inch)) > 0.01:
                continue
        except (TypeError, ValueError):
            continue
        for token in str(item.get("resolution") or "").replace("/", " ").split():
            normalized = _normalize_resolution(token)
            if normalized and normalized not in found:
                found.append(normalized)
    return found


# ── 2) 事实抽取（候选）─────────────────────────────────────────────────────
def extract_lcd_facts(
    message: str,
    *,
    profile: Any = None,
    category_signal: Optional[Dict[str, Any]] = None,
    asked_slot: Optional[str] = None,
    fact_signal: Optional[Dict[str, Any]] = None,
) -> LcdFacts:
    """从这一句里抽 LCD 候选事实。

    品类（category）**优先取语境理解的结论**（客户口径 2026-09-30：
    "不允许有关键词来触发 AI 的某个链路，需要让 AI 结合语境上下文去选择品类"）。
    关键词解析退居兜底 —— 只有在语境理解不可用（模型失败 / 离线 / 单测直调）时才用，
    并且会被标成 ``category_source="keyword"``，**不能推翻**已经由语境定下的品类。

    ``asked_slot``：我们上一轮问的是哪一项。**这一项是很多解析的前提**：

        实测（2026-09-30）视频墙对话：

            我们问 "How many columns and rows…something like 6x2?" → 客户答 "3x3"
            → 旧实现只在"这一句里有 wall / spliced 字样"时才把 3x3 当排布，
              于是排布永远没记下来，需求链在"问排布"上死循环、永远不推荐。

        规则：客户答的就是我们问的那一项 —— 问排布时，"3x3" 就是排布；
        问拼缝时，"0.88mm" 就是拼缝（不需要客户再说一遍 "bezel"）。

    ``fact_signal``：语境理解给出的事实（``understand_lcd_turn`` 的 ``facts``）。
    客户口径（2026-09-30）：**只允许结合上下文语境整理需求，关键词只能是兜底** ——
    所以这里的顺序是：先跑正则拿到兜底值，再用语境结论覆盖；语境没给的字段才用正则。
    模型不可用时 ``fact_signal`` 为空 → 全部退回正则（降级路径）。
    """
    text = str(message or "")
    lowered = text.lower()
    facts = LcdFacts()
    if asked_slot is None:
        asked_slot = str(getattr(profile, "last_asked_slot", "") or "")
    asked_slot = str(asked_slot or "")

    understood = ""
    if isinstance(category_signal, dict):
        candidate = str(category_signal.get("category") or "").strip().lower()
        if candidate in _REAL_CATEGORIES or candidate == UNKNOWN:
            understood = candidate
            facts.category_source = "understanding"

    if understood:
        facts.category = understood if understood in _REAL_CATEGORIES else UNKNOWN
    else:
        for pattern, value in _CATEGORY_HINTS:
            if any(hint in lowered for hint in value):
                facts.category = pattern
                facts.category_source = "keyword"
                break
    for name, hints in _ROOM_TYPES:
        if any(hint in lowered for hint in hints):
            facts.room_type = name
            break
    if facts.category == UNKNOWN:
        # 场景词只作为"候选"：room_type 明显属于会议/教室 → conference_education
        if facts.room_type in ("meeting room", "classroom"):
            facts.category = CONFERENCE_EDUCATION
            facts.category_source = facts.category_source or "keyword"
        elif facts.room_type == "control room":
            facts.category = MONITORING
            facts.category_source = facts.category_source or "keyword"
        elif facts.room_type in ("retail store", "lobby"):
            facts.category = ADVERTISING
            facts.category_source = facts.category_source or "keyword"

    size = _SIZE_INCH_RE.search(text)
    if size is None and asked_slot == "lcd_size":
        # 问的就是屏幕尺寸 → 裸数字（"65"）也按英寸理解
        bare_size = re.fullmatch(
            r"\s*(?:about\s+|around\s+|roughly\s+)?(\d{2,3}(?:[.,]\d+)?)\s*",
            lowered,
        )
        if bare_size:
            size = bare_size
    if size:
        try:
            facts.size_inch = float(size.group(1).replace(",", "."))
        except ValueError:  # pragma: no cover - 防御式
            facts.size_inch = None

    # 排布（6x2 / 3 rows and 3 columns）只有在"拼接语境"下才算拼接墙；
    # 否则可能是尺寸（3x5 米）。拼接语境的三个来源：
    #   ① 这一句里有 wall / spliced / 拼接；② 我们上一轮问的就是排布；
    #   ③ 档案里已经确认是拼接墙（客户答的是纯数字，本来就该按排布理解）。
    layout_ctx = (
        "wall" in lowered
        or "splic" in lowered
        or "拼接" in text
        or asked_slot == "lcd_layout"
        or getattr(profile, "lcd_is_splicing", None) is True
    )
    def _is_a_size_not_a_layout(match: "re.Match[str]") -> bool:
        """匹配到的 NxM 后面紧跟长度单位（"3x5 m" / "3x5米"）→ 那是尺寸，不是排布。

        注意只看**这个匹配后面**：整句里出现 "3.5mm 拼缝" 之类不该影响排布判断
        （实测：客户说 "6x2 video wall, 3.5mm bezel" 时整句有 mm，旧写法把排布漏掉了）。
        """
        tail = text[match.end() : match.end() + 4].lower()
        return bool(re.match(r"\s*(?:m\b|米|inch|inches|英寸|吋|寸)", tail))

    layout = _LAYOUT_RE.search(text)
    columns = rows = ""
    layout_match = layout
    if layout:
        columns, rows = layout.group(1), layout.group(2)
    else:
        words = _LAYOUT_WORDS_RE.search(text)
        if words:
            layout_match = words
            if words.group(1) and words.group(2):
                # "3 rows and 3 columns" → 列 x 行
                rows, columns = words.group(1), words.group(2)
            else:
                columns, rows = words.group(3), words.group(4)
    if asked_slot == "lcd_layout":
        # 问的就是排布 → 单位也不影响（客户可能写 "3m x 3m" 指墙体，但仍要拿到排布数字）
        _looks_like_size = False
    else:
        _looks_like_size = bool(layout_match) and _is_a_size_not_a_layout(layout_match)
    if columns and rows and layout_ctx and not _looks_like_size:
        facts.layout = f"{columns}x{rows}"
        facts.is_splicing = True
        facts.screen_count = int(columns) * int(rows)
    if any(word in lowered for word in ("video wall", "spliced", "splicing", "拼接屏", "拼接墙")):
        facts.is_splicing = True
    if re.search(r"\b(single display|not spliced|no video wall|不拼接|单体屏)\b", lowered):
        facts.is_splicing = False

    # 拼缝：客户答 "0.88mm"（我们问的就是拼缝）时，不该要求他再说一遍 "bezel"。
    bezel = _BEZEL_RE.search(text)
    if bezel is None and (asked_slot == "lcd_bezel" or "bezel" in lowered or "拼缝" in text):
        bezel = _BEZEL_MM_RE.search(text)
    if bezel is None and asked_slot == "lcd_bezel":
        # 连单位都没写的裸数字（"0.88" / "3.5"）——问的是拼缝，按 mm 理解
        bare = re.fullmatch(r"\s*(?:about\s+|around\s+|roughly\s+)?(\d+(?:[.,]\d+)?)\s*", lowered)
        if bare and float(bare.group(1).replace(",", ".")) <= 20:
            bezel = bare
    if bezel:
        try:
            facts.bezel_mm = float(bezel.group(1).replace(",", "."))
        except ValueError:  # pragma: no cover - 防御式
            facts.bezel_mm = None

    if re.search(r"\b(4k|uhd|3840\s*[x*]\s*2160)\b", lowered):
        facts.resolution = "4K"
    elif re.search(r"\b(2k|fhd|1080p|1920\s*[x*]\s*1080)\b", lowered):
        facts.resolution = "2K"

    if re.search(r"\b(touch|touchscreen|touch screen|interactive)\b|触摸|触控", lowered):
        facts.touch = False if _negated_need(lowered) else True
    if re.search(r"\b(whiteboard|handwriting|write on|annotation)\b|白板|手写", lowered):
        facts.handwriting = False if _negated_need(lowered) else True
        if facts.handwriting:
            facts.touch = True if facts.touch is None else facts.touch
    if re.search(r"\b(tender|bid|bidding)\b|招标|投标", lowered):
        facts.tender = True
    if re.search(r"\bops\b|插拔式电脑|电脑模块", lowered):
        facts.ops = bool(re.search(r"\b(need|with|require|yes)\b|需要|要|带", lowered)) or None
    # "cam" 也算：实测客户写 "i need a lcd witch cam"（拼写还有错），
    # 旧写法只认 camera/webcam → 摄像头这一项没被记下来。
    if re.search(r"\b(cam|camera|cameras|webcam|video conference)\b|摄像头|摄像机", lowered):
        # "只是图片里有 / 不需要" → 客户明确说不要（计划 §十一 Case 11）
        facts.camera = False if _negated_need(lowered) else True

    people = _PEOPLE_RE.search(text)
    if people:
        try:
            facts.people_count = int(people.group(1))
        except ValueError:  # pragma: no cover - 防御式
            facts.people_count = None

    if re.search(r"\b(indoor|inside)\b|室内", lowered):
        facts.environment = "indoor"
    elif re.search(r"\b(outdoor|outside)\b|室外|户外", lowered):
        facts.environment = "outdoor"
    if re.search(r"\b(rental|rent)\b", lowered):
        facts.installation = "rental"
    elif re.search(r"\b(fixed|permanent|wall[- ]mount)\b", lowered):
        facts.installation = "fixed"

    # ── 语境理解为主、关键词为兜底（客户口径 2026-09-30）──────────────────
    # 上面那些正则只是兜底值：模型读了"我们问过什么 + 客户答过什么 + 这一句"
    # 之后的结论，覆盖它们。模型没提到的字段才继续用正则的结果。
    _apply_understood_facts(facts, fact_signal)
    return facts


def _apply_understood_facts(facts: LcdFacts, signal: Optional[Dict[str, Any]]) -> None:
    """把语境理解给出的事实覆盖到候选事实上（只认认得的字段与合法取值）。"""
    if not isinstance(signal, dict):
        return
    understood = signal.get("facts")
    if not isinstance(understood, dict) or not understood:
        return

    environment = str(understood.get("environment") or "").strip().lower()
    if environment in ("indoor", "outdoor"):
        facts.environment = environment

    installation = str(understood.get("installation") or "").strip().lower()
    if installation in ("fixed", "rental"):
        facts.installation = installation

    splicing = understood.get("is_splicing")
    if isinstance(splicing, bool):
        facts.is_splicing = splicing

    layout = str(understood.get("splicing_layout") or "").strip()
    if layout:
        match = re.fullmatch(r"(\d{1,2})\s*[x×*]\s*(\d{1,2})", layout)
        if match:
            facts.layout = f"{match.group(1)}x{match.group(2)}"
            facts.screen_count = int(match.group(1)) * int(match.group(2))
            if facts.is_splicing is None:
                facts.is_splicing = True

    size = understood.get("screen_size_inch")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        if 10 <= float(size) <= 500:
            facts.size_inch = float(size)

    bezel = understood.get("bezel_mm")
    if isinstance(bezel, (int, float)) and not isinstance(bezel, bool):
        if 0 <= float(bezel) <= 100:
            facts.bezel_mm = float(bezel)

    resolution = _normalize_resolution(understood.get("resolution"))
    if resolution:
        facts.resolution = resolution

    for key, attribute in (
        ("touch", "touch"),
        ("handwriting", "handwriting"),
        ("camera", "camera"),
        ("ops", "ops"),
        ("tender", "tender"),
    ):
        value = understood.get(key)
        if isinstance(value, bool):
            setattr(facts, attribute, value)

    # 要手写就一定要触控（和正则那套口径一致）
    if facts.handwriting is True and facts.touch is None:
        facts.touch = True


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


def _apply_degraded_default(profile: Any, slot: str) -> bool:
    """问满上限仍无答案 → 按业务默认补齐（来源标 recommended，不伪装成客户要求）。"""
    if slot not in _DEGRADED_DEFAULTS:
        return False
    field_name = _DEGRADED_FIELD.get(slot, slot)
    if getattr(profile, field_name, None) not in (None, "", [], {}):
        return False
    value = _DEGRADED_DEFAULTS[slot]
    setattr(profile, field_name, value)
    sources = dict(getattr(profile, "sources", None) or {})
    sources[field_name] = "recommended"
    profile.sources = sources
    logger.info(
        "[LCD] slot=%s 问了 %d 次仍未答 → 按业务默认收口：%s=%r",
        slot, _SLOT_ASK_LIMIT, field_name, value,
    )
    return True


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


def fill_defaults_for_recommendation(profile: Any, missing: Iterable[str]) -> Dict[str, Any]:
    """客户要推荐 / 可选字段已经问过 → 用业务默认值补齐（来源标 recommended）。"""
    applied: Dict[str, Any] = {}
    if profile is None:
        return applied
    sources = dict(getattr(profile, "sources", None) or {})
    field_map = {
        "lcd_bezel": "lcd_bezel_mm",
        "lcd_touch": "lcd_touch_required",
        "lcd_handwriting": "lcd_handwriting_required",
        "lcd_ops": "lcd_ops_required",
        "lcd_camera": "lcd_camera_required",
        "lcd_tender": "lcd_tender_project",
    }
    for slot in list(missing or []):
        if slot not in _FILLABLE_DEFAULTS:
            continue
        field_name = field_map[slot]
        if getattr(profile, field_name, None) not in (None, "", [], {}):
            continue
        setattr(profile, field_name, _FILLABLE_DEFAULTS[slot])
        sources[field_name] = "recommended"
        applied[field_name] = _FILLABLE_DEFAULTS[slot]
    profile.sources = sources
    return applied


def is_ifp_requirement(profile: Any) -> bool:
    """客户要的是不是**交互平板（IFP）**（计划 §十四/§十五）。

    客户口径（2026-09-30）："客户要求很明显是可手写的会议室使用的 IFP，
    为什么会推荐普通的可触摸的 LCD？" —— 所以判定口径是：

        · 产品类型已经是 IFP；
        · 或者客户要**手写 / 白板**（这就是交互平板，不是普通 LCD）；
        · 或者会议 / 教育场景下客户要**触控**。

    判定为真以后，检索与选型都要按 IFP 来（否则会从 LCD 语料里挑出一台
    普通的商用显示器 —— 实测就是这样把 P65 推给了要手写白板的会议室）。
    """
    display_type = str(getattr(profile, "display_type", None) or "").upper()
    if display_type == "IFP":
        return True
    if getattr(profile, "lcd_handwriting_required", None) is True:
        return True
    if (
        getattr(profile, "lcd_touch_required", None) is True
        and str(getattr(profile, "lcd_category", "") or "") == CONFERENCE_EDUCATION
    ):
        return True
    return False


def effective_display_type(profile: Any) -> str:
    """检索 / 选型实际该按哪个产品类型走（LED / LCD / IFP）。

    客户明说 LCD、但要手写白板时，档案里的 display_type 仍是 LCD（那是客户原话），
    可**选型口径**必须是 IFP —— 这里给出"实际该查哪一类产品"。
    """
    display_type = str(getattr(profile, "display_type", None) or "").upper()
    if display_type == "LED":
        return "LED"
    if is_ifp_requirement(profile):
        return "IFP"
    return display_type


def _is_ifp_branch(profile: Any) -> bool:
    """会议 / 教育 + 需要手写或互动 → IFP 分支（计划 §十五/§十六）。"""
    return is_ifp_requirement(profile)


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
