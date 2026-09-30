"""LCD 事实解析（纯 helper）：从客户这句话里读出**候选事实**。

搬迁自 lcd_decision.py（整改计划 §十一"第一批应该移出的内容"）：
分辨率 / 拼接 / 排布 / 拼缝 / 尺寸 / 摄像头 / 手写 / TouCH / OPS / 招标 等解析。

**不做决策**：只产出 LcdFacts（"客户说了什么"）；
"下一步问什么"仍然只由 lcd_decision 决定。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .resolution import _normalize_resolution

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


__all__ = [
    "ADVERTISING",
    "CONFERENCE_EDUCATION",
    "LcdFacts",
    "MONITORING",
    "NORMAL",
    "UNKNOWN",
    "extract_lcd_facts",
]
