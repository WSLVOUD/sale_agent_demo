"""Rule-based extraction of product and requirement slots."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .measurement_parser import (
    _extract_axis_measurements,
    _extract_bare_measurement,
    _extract_room_dimensions,
    _extract_size_axis,
    _extract_space_facts,
    _extract_target_size,
    _extract_viewing_distance,
    _resolve_axis_measurements,
)

logger = logging.getLogger(__name__)



# ── 关键词表 ────────────────────────────────────────────────────────────────
_INDOOR_KEYWORDS = (
    "室内", "户内", "屋内", "indoors", "indoor", "innenbereich", "innen",
    "interior", "interno", "interno", "intérieur", "interieur", "в помещении", "屋内",
)

_OUTDOOR_KEYWORDS = (
    "户外", "室外", "露天", "屋外", "外墙", "幕墙", "outside", "outdoor", "outdoors",
    "open-air", "open air",
    "exterior", "extérieur", "exterieur", "außenbereich", "aussenbereich", "externo",
    "улиц", "наружн", "屋外",
)

_SEMI_OUTDOOR_KEYWORDS = ("半户外", "半室外", "遮阳", "semi-outdoor", "half outdoor")


_RENTAL_KEYWORDS = (
    "租赁", "租用", "短租", "临时", "快闪", "巡演", "演出", "演出用",
    "演唱会", "音乐会", "舞台", "活动", "巡展", "车展", "赛事",
    # "reantal" / "renatl"：客户实测笔误（"fixed for indoor and reantal for outdoor"）
    "rental", "reantal", "renatl", "rent", "hire", "mieten", "louer", "location",
    "alquiler", "locação",
    "аренда", "レンタル", "租赁屏",
)

_FIXED_KEYWORDS = (
    "固定安装", "固装", "永久", "挂墙", "壁挂", "安装固定",
    "fixed", "fixed installation", "permanent", "wall mount",
)


_DISPLAY_TYPE_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "IFP": (
        "ifp", "交互平板", "交互式平板", "会议一体机", "触摸一体机", "电子白板",
        "interactive flat panel", "interactive display", "whiteboard",
    ),
    "LCD": ("lcd", "拼接屏", "拼接墙", "拼接", "液晶", "video wall", "splicing"),
    # 计划 v2.9.3 §五：**去掉"泛指屏就是 LED"的默认**。
    # "显示屏 / 屏幕 / 屏 / display / screen" 这些词只说明客户要一块屏，
    # 不代表是 LED —— 它们交给 Product Type Router 判断（推断→确认 / 询问）。
    "LED": ("led", "led屏", "点间距屏", "箱体屏"),
}


# 场景 → 规范 purpose（同时给出英文检索词）
# 顺序即优先级：越具体的场景越靠前；hall / exhibition / rental 这类泛化场景放最后。
_PURPOSE_KEYWORDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("concert", ("演唱会", "音乐会", "concert", "konzert", "концерт", "コンサート"), "concert live event"),
    ("stage", ("舞台", "演出", "表演", "剧场", "stage", "performance", "bühne", "сцена"), "stage performance"),
    ("wedding", ("婚礼", "婚宴", "婚庆", "结婚", "wedding", "marriage", "hochzeit", "свадьба", "結婚式"), "wedding event marriage"),
    ("control_room", ("指挥中心", "监控中心", "控制室", "中控室", "command center", "control room"), "control room command center"),
    ("classroom", ("教室", "培训室", "培训", "教学", "学校", "课堂", "classroom", "class room", "smart classroom", "training room", "school", "university", "lecture hall", "language lab"), "classroom education training"),
    ("conference", ("会议室", "会议", "meeting room", "conference", "boardroom", "会議室"), "conference room meeting room"),
    ("church", ("教堂", "礼拜", "宗教", "礼拜堂", "教会", "church", "worship", "place of worship"), "church place of worship"),
    ("museum", ("博物馆", "museum", "gallery", "美术馆"), "museum exhibition display"),
    ("showroom", ("展厅", "展馆", "展览", "showroom", "ausstellung"), "showroom exhibition hall"),
    # 户外广告用具体关键词，避免把"商场店铺做广告展示"误判为广告场景
    ("advertising", ("幕墙", "外墙", "楼体", "广告牌", "广告屏", "标牌", "橱窗", "传媒", "advertising", "billboard", "signage", "facade", "building wall"), "advertising billboard digital signage"),
    ("retail", ("商场", "商店", "零售", "店铺", "超市", "retail", "store", "shop", "mall"), "retail store shop"),
    ("stadium", ("体育场", "体育馆", "球场", "赛场", "stadium", "arena", "sports venue"), "stadium sports venue"),
    ("airport", ("机场", "车站", "码头", "地铁", "airport", "terminal", "station"), "airport terminal transport hub"),
    ("bank", ("银行", "bank"), "bank branch"),
    ("hotel", ("酒店", "大堂", "hotel", "lobby"), "hotel lobby"),
    ("restaurant", ("餐厅", "酒吧", "咖啡厅", "restaurant", "bar", "cafe"), "restaurant bar"),
    ("office", ("办公室", "办公", "office", "workspace"), "office workspace"),
    ("hospital", ("医院", "医疗", "诊所", "hospital", "clinic", "medical"), "hospital medical"),
    ("exhibition", ("展会", "展览会", "展销", "trade show", "expo", "exhibition"), "exhibition trade show"),
    ("hall", ("大厅", "hall", "concourse"), "large indoor hall"),
    # 注：**不要**把 租赁/rental/event 当作 purpose —— 租赁是安装方式
    # （installation=rental 已单独记录）。实测（2026-09-28 第四次）：客户说
    # "fixed for indoor and rental for outdoor"，purpose 被写成 "rental"，
    # 客户看到 "Screen 1 (indoor / rental)" 这种标签，以为需求收错了；
    # 更严重的是逐屏检索句里也带上了这词，把这块屏的 installation 又翻成租赁。
)


_WATERPROOF_KEYWORDS = ("防水", "waterproof", "ip65", "ip66", "防雨")

_COB_KEYWORDS = ("cob",)

_HDR_KEYWORDS = ("hdr",)

_GOB_KEYWORDS = ("gob",)

_FLEXIBLE_KEYWORDS = ("柔性", "可弯曲", "异形", "弧形", "弯曲", "flexible", "curved", "bendable")


_BUDGET_LOW_KEYWORDS = (
    "预算有限", "预算低", "预算不高", "预算少", "便宜", "低价", "经济", "低成本",
    "cheap", "budget", "economical", "low price", "低价位", "affordable",
)

_BUDGET_HIGH_KEYWORDS = ("高端", "顶配", "旗舰", "premium", "high-end", "high end", "flagship")

_BUDGET_MID_KEYWORDS = ("中档", "中等", "mid-price", "mid range", "middle")


# 默认固装的场景（租赁必须由客户显式说明）
_INDOOR_FIXED_PURPOSES = frozenset({
    "conference", "classroom", "retail", "showroom", "museum", "hall",
    "control_room", "office", "hospital", "bank", "hotel", "restaurant", "airport",
    "exhibition",
    # 教堂（church）几乎都是室内，客户反馈"说了 church 还问室内室外"很傻
    "church",
})


# 仅在无显式室内/室外关键词时用于环境推断的场景
# 注意：演唱会 / 舞台 / 演出既可能室内也可能室外，不能仅凭场景断定
_OUTDOOR_PURPOSES = frozenset({"advertising", "stadium"})


# 默认固装的场景（含户外广告 / 场馆；租赁必须显式说明）
_FIXED_PURPOSES = _INDOOR_FIXED_PURPOSES | _OUTDOOR_PURPOSES


# 室内外都可能、必须问客户的场景：
#   stage / concert（舞台、演唱会，室内体育馆也可能）、rental（租赁流动场景）、wedding（室内外都可能）
_AMBIGUOUS_ENVIRONMENT_PURPOSES = frozenset({"stage", "concert", "rental", "wedding"})



def environment_from_purpose(purpose: Optional[str]) -> Optional[str]:
    """场景 → 使用环境（只在"一眼就能定"时给值，否则返回 None 让系统去问）。

    室内：会议室 / 教室 / 门店 / 展厅 / 博物馆 / 大厅 / 指挥中心 / 办公室 /
          医院 / 银行 / 酒店 / 餐厅 / 机场 / 展会 / 教堂
    室外：户外广告 / 体育场馆
    其余（舞台、演唱会、租赁等）室内外都可能 → 返回 None，仍然要问客户。
    """
    name = str(purpose or "").strip().lower()
    if not name or name in _AMBIGUOUS_ENVIRONMENT_PURPOSES:
        return None
    if name in _OUTDOOR_PURPOSES:
        return "outdoor"
    if name in _INDOOR_FIXED_PURPOSES:
        return "indoor"
    return None



_ASCII_WORD_RE_CACHE: Dict[str, "re.Pattern[str]"] = {}



def _keyword_pattern(token: str) -> Optional["re.Pattern[str]"]:
    """纯英文字母关键词 → 整词正则（允许复数 s），其余返回 None（按子串匹配）。

    子串匹配会带来一类"看起来像、其实无关"的误判，例如：
      - ``different`` 里的 ``rent``      → 被判成"租赁需求"
      - 客户名字 ``Akbar`` 里的 ``bar``  → 被判成"餐厅场景"
    因此英文关键词一律按整词（含可选复数）匹配；中文 / 含数字的关键词（如 ip65）
    仍按子串匹配。
    """
    token = token.lower().strip()
    if not token:
        return None
    if token.isascii() and all(ch.isalpha() or ch.isspace() for ch in token):
        pattern = _ASCII_WORD_RE_CACHE.get(token)
        if pattern is None:
            words = r"\s+".join(re.escape(part) for part in token.split())
            # 注意：不能用 \b —— Python 的 \w 把中文也算作"单词字符"，
            # 于是 "会议室用LCD" 里的 lcd 前面就不是边界，关键词会匹配不到。
            # 这里用"前后不是英文字母/数字"来界定英文关键词。
            # 词形变化也认：permanent→permanently / bank→banking / mount→mounted / store→stores
            pattern = re.compile(
                rf"(?<![a-z0-9]){words}(?:s|es|d|ed|ing|ly)?(?![a-z0-9])",
                re.IGNORECASE,
            )
            _ASCII_WORD_RE_CACHE[token] = pattern
        return pattern
    return None



def _any(text: str, keywords: Sequence[str]) -> bool:
    """关键词匹配：英文整词（可复数），中文/含数字按子串。"""
    for keyword in keywords:
        token = str(keyword).lower().strip()
        pattern = _keyword_pattern(token)
        if pattern is not None:
            if pattern.search(text):
                return True
        elif token and token in text:
            return True
    return False



# 兼容旧调用名（语义与 _any 完全一致）
_any_word = _any



def _detect_purpose(lowered: str) -> Optional[str]:
    for purpose, keywords, _ in _PURPOSE_KEYWORDS:
        if _any(lowered, [k.lower() for k in keywords]):
            return purpose
    return None



def purpose_english(purpose: Optional[str]) -> str:
    """规范 purpose → 英文检索词。"""
    if not purpose:
        return ""
    for name, _, english in _PURPOSE_KEYWORDS:
        if name == purpose:
            return english
    return str(purpose)



# ── 疑问句判定（Sales / Solution 共用一份，避免两处规则漂移）──────────────────
_QUESTION_RE = re.compile(
    r"[?？]|"
    r"吗|呢|是否|有没有|有没|能不能|可不可以|怎么|如何|为什么|为何|哪家|哪个|哪些|多少钱|多久|什么时候|"
    r"\b(?:do you|can you|could you|would you|is there|are there|how much|how long|how do|how can|"
    r"what about|anything|any )\b",
    re.IGNORECASE,
)



def looks_like_question(text: str) -> bool:
    """客户这句话是不是在"提问"（而不是在陈述需求）。

    用途：避免把"你们在肯尼亚有代理商吗？"、"这个屏多久能发货？"这类问题
    当成"要推荐产品"（句子里出现"屏/LED"并不代表客户想要推荐）。
    """
    return bool(_QUESTION_RE.search(str(text or "")))



# 内容类型（视频 / 图片 / 两者都有）：只记录，不参与选型
_CONTENT_MIXED_KEYWORDS = (
    "两种都有", "两个都有", "两者都有", "都要", "都放", "都会用", "都用到", "混合", "both", "a mix", "mixed",
)

_CONTENT_VIDEO_KEYWORDS = ("视频", "影片", "动态画面", "视频素材", "video", "videos", "motion")

_CONTENT_IMAGE_KEYWORDS = (
    "图片", "照片", "静态画面", "静图", "图文", "幻灯片", "image", "images", "photo", "photos",
    "picture", "pictures", "static",
)


# 价格 / 质量取向（推荐前那一问）
_PREFERENCE_BOTH_KEYWORDS = (
    # 强信号：本身就是"权衡"的说法
    "都看重", "都看中", "都重要", "都要好", "both matter", "either is fine", "both are important",
    # 客户口径（2026-09-18）：问"最看重价格还是质量"时，客户只回
    # "both are fine" / "either works" 这一类，也必须算"两者都行"（走默认档）。
    # 注意：单独的 "both" / "都行" 有歧义（也可能是在回答"视频还是图片"），
    # 由 RequirementExtractor 结合"上一轮问的是哪一项"落地，不放在这里。
    "both are fine", "both is fine", "both are good", "both work", "both works",
    "either works", "either one is fine", "either way is fine",
    "两者都行", "两个都行", "两种都行", "两个都可以",
)

_PREFERENCE_BOTH_WEAK_KEYWORDS = (
    # 弱信号：需要上下文里出现价格/质量才算
    "两个都", "两者都", "都要", "both",
)

_PREFERENCE_QUALITY_KEYWORDS = (
    "质量优先", "看重质量", "看中质量", "质量为主", "不在乎价格", "不看价格", "价格无所谓", "品质", "要最好的",
    "效果更好", "高端", "quality", "best quality", "premium", "top quality", "not about price",
)

_PREFERENCE_PRICE_KEYWORDS = (
    "价格优先", "看重价格", "看中价格", "价格为主", "要便宜", "便宜的", "省钱", "预算有限", "性价比",
    "价格", "price", "cheaper", "budget", "cost", "affordable", "economical",
)



def _extract_content_type(text: str) -> Optional[str]:
    """客户回答"放视频还是放图片" → video / image / mixed（只记录）。"""
    lowered = str(text or "").lower()
    if any(keyword in lowered for keyword in _CONTENT_MIXED_KEYWORDS):
        return "mixed"
    has_video = any(keyword in lowered for keyword in _CONTENT_VIDEO_KEYWORDS)
    has_image = any(keyword in lowered for keyword in _CONTENT_IMAGE_KEYWORDS)
    if has_video and has_image:
        return "mixed"
    if has_video:
        return "video"
    if has_image:
        return "image"
    return None



def _extract_price_preference(text: str) -> Optional[str]:
    """客户回答"最看重价格还是质量" → price / both / quality。"""
    lowered = str(text or "").lower()
    has_quality = any(keyword in lowered for keyword in _PREFERENCE_QUALITY_KEYWORDS)
    has_price = any(keyword in lowered for keyword in _PREFERENCE_PRICE_KEYWORDS)
    if any(keyword in lowered for keyword in _PREFERENCE_BOTH_KEYWORDS):
        return "both"
    # 弱信号："都要 / both" 这类要确认是在说价格与质量（避免"视频和图片都要"误判）
    if any(keyword in lowered for keyword in _PREFERENCE_BOTH_WEAK_KEYWORDS) and (
        has_price
        or has_quality
        or any(word in lowered for word in ("价格", "价钱", "质量", "品质", "price", "quality"))
    ):
        return "both"
    if has_quality and has_price:
        return "both"
    if has_quality:
        return "quality"
    if has_price:
        return "price"
    return None



def _extract_pixel_pitch(text: str) -> Optional[float]:
    match = re.search(r"(?<![A-Za-z0-9])[Pp](\d+(?:\.\d+)?)(?![A-Za-z0-9])", text)
    if match:
        return float(match.group(1))
    match = re.search(
        r"(?:点间距|间距|pixel\s*pitch|pitch)\s*(?:是|为|约|:)?\s*"
        r"(?:around|about|approx(?:imately)?|roughly|左右的?)?\s*"
        r"(\d+(?:\.\d+)?)\s*mm",
        text, re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    # 反向语序："1.9mm 点间距"
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*mm\s*(?:的)?\s*(?:点间距|间距|pixel\s*pitch|pitch)",
        text, re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return None



def _extract_brightness_min(text: str) -> Optional[int]:
    match = re.search(r"(\d{3,5})\s*(?:nit|nits|cd/m2|cd/m²|cd|尼特)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"亮度\s*(?:要|是|为|约|至少|不低于|>=?)?\s*(\d{3,5})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:brightness|亮度)\s*(?:>=|at least|above|over|最低)?\s*(\d{3,5})", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None



def _extract_model_mention(text: str) -> tuple[Optional[str], Optional[str]]:
    """识别客户直接点名的型号 / 系列。"""
    normalized = text
    for hyphen in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
        normalized = normalized.replace(hyphen, "-")
    model_match = re.search(
        r"\bTW\s*(\d{2})\s*-\s*(IRHD|HOD|COB|3216|IR|OD)\s*-\s*[Pp]?(\d+(?:\.\d+)?)\s*(H|E)?\s*(?:\(\s*GOB\s*\))?",
        normalized, re.IGNORECASE,
    )
    if model_match:
        tw, kind, pitch, suffix = model_match.groups()
        gob = "GOB" if "gob" in model_match.group(0).lower() else None
        model = f"TW{tw}-{kind.upper()}-P{pitch}"
        if suffix:
            model += suffix.upper()
        if gob:
            model += "(GOB)"
        return f"TW{tw}-{kind.upper()}", model
    series_match = re.search(
        r"\bTW\s*(\d{2})\s*-\s*(IRHD|HOD|COB|3216|IR|OD)\b", normalized, re.IGNORECASE
    )
    if series_match:
        return f"TW{series_match.group(1)}-{series_match.group(2).upper()}", None
    return None, None



def _extract_budget_level(lowered: str) -> Optional[str]:
    if _any(lowered, _BUDGET_LOW_KEYWORDS):
        return "low"
    if _any(lowered, _BUDGET_HIGH_KEYWORDS):
        return "high"
    if _any(lowered, _BUDGET_MID_KEYWORDS):
        return "mid"
    return None



# ── 主入口 ──────────────────────────────────────────────────────────────────
def extract_slots(message: str) -> Dict[str, Any]:
    """把自然语言需求转换成结构化槽位（纯规则、无 LLM）。"""
    text = str(message or "").strip()
    if not text:
        return {}
    lowered = text.lower()
    slots: Dict[str, Any] = {}

    # 1) 产品类型（IFP > LCD > LED 优先）
    for display_type in ("IFP", "LCD", "LED"):
        if _any(lowered, [k.lower() for k in _DISPLAY_TYPE_KEYWORDS[display_type]]):
            slots["display_type"] = display_type
            break
    # 计划 v2.9.3 §五：**不做"没点名就默认 LED"**。
    # 客户没说 LED / LCD / IFP 时 display_type 就是未知，交给 Product Type Router
    # 去推断（要客户确认）或者直接问客户；LED 只在"所有办法都拿不到信息"时兜底。

    # 2) 室内 / 室外 / 半户外（显式关键词优先，其次场景推断）
    if _any(lowered, _SEMI_OUTDOOR_KEYWORDS):
        slots["environment"] = "semi_outdoor"
    else:
        # 室内/室外都出现时（冲突需求）以先出现的那个为准
        outdoor_pos = [lowered.find(k) for k in _OUTDOOR_KEYWORDS if k in lowered]
        indoor_pos = [lowered.find(k) for k in _INDOOR_KEYWORDS if k in lowered]
        first_outdoor = min(outdoor_pos) if outdoor_pos else None
        first_indoor = min(indoor_pos) if indoor_pos else None
        if first_outdoor is not None and (first_indoor is None or first_outdoor < first_indoor):
            slots["environment"] = "outdoor"
        elif first_indoor is not None:
            slots["environment"] = "indoor"

    # 3) 场景 purpose（也用于环境推断）
    purpose = _detect_purpose(lowered)
    if purpose:
        slots["purpose"] = purpose
    
    # 3b) 识别"play video"、"play music"这类应用描述为舞台/演出场景
    if not purpose:
        if any(kw in lowered for kw in ("play video", "play music", "display video", "show video")):
            slots["purpose"] = "stage"
            purpose = "stage"

    if "environment" not in slots:
        # 会议室 / 教室 / 教堂 / 展厅 / 机场 → 室内；户外广告 / 体育场 → 室外。
        # 这类场景"一眼就能定"，客户反馈不该再追问室内外。
        # 【M2】来源记为 scenario_derived（客户原话场景直接判定），
        # 既不是"算法估算"，也不是"系统默认"，可以用于放行环境判定。
        # 舞台 / 演唱会 / 租赁等室内外都可能 → 返回 None，继续问。
        derived_environment = environment_from_purpose(purpose)
        if derived_environment:
            slots["environment"] = derived_environment
            slots.setdefault("_scenario_derived", []).append("environment")

    # 4) 固装 / 租赁
    # 注意：这里的英文关键词必须整词匹配 —— "different" 里含 "rent"，
    # 用子串匹配会把"想换一个型号"误判成租赁需求（并污染后续 Gate 判定）。
    if _any_word(lowered, _RENTAL_KEYWORDS):
        slots["installation"] = "rental"
    elif _any_word(lowered, _FIXED_KEYWORDS):
        slots["installation"] = "fixed"
    elif purpose in _FIXED_PURPOSES:
        # 会议室 / 教室 / 展厅 … 这类场景默认固装（租赁必须由客户显式说明）
        # 【M2】来源记为 default（系统业务默认值）：可以参与打分，但不能单独放行 Gate。
        slots["installation"] = "fixed"
        slots.setdefault("_default_slots", []).append("installation")
    elif slots.get("environment") in ("indoor", "outdoor"):
        # 已经确定使用环境但没有租赁信号 → 固装（租赁必须显式说明）
        slots["installation"] = "fixed"
        slots.setdefault("_default_slots", []).append("installation")

    # 5) 观看距离 / 目标尺寸
    #    先摘出"场地尺寸"（房间/大厅多大）：那是场地，不是屏体 —— 摘掉之后
    #    屏体尺寸仍然按原规则解析（客户没说屏多大就继续问客户）。
    room_facts, text = _extract_room_dimensions(text)
    if room_facts:
        slots.update(room_facts)
        logger.info("场地尺寸：%s（不计入屏体尺寸）", room_facts)
    distance = _extract_viewing_distance(text)
    if distance is not None:
        slots["viewing_distance_m"] = distance
    width_mm, height_mm = _extract_target_size(text)
    if width_mm:
        slots["target_width_mm"] = width_mm
    if height_mm:
        slots["target_height_mm"] = height_mm
    # 5a) "45cm is the width" / "129,2cm 是长度" 这类"数字 + 方向词"的写法
    #     （旧解析只认 "45cm wide" 这种紧贴语序，第二个尺寸会整条丢掉）
    axis_measurements = _extract_axis_measurements(text)
    if axis_measurements:
        width_mm, height_mm = _resolve_axis_measurements(axis_measurements)
        if width_mm:
            slots["target_width_mm"] = width_mm
        if height_mm:
            slots["target_height_mm"] = height_mm
    # 5b) 客户只报了一个长度（如 "129,2cm" / "1292mm"）且没说宽高：
    #     记成"尺寸线索"，由系统追问这是宽、高还是对角线，绝不替客户猜。
    if not width_mm and not height_mm and distance is None:
        hint_mm = _extract_bare_measurement(text)
        if hint_mm:
            slots["screen_size_hint_mm"] = hint_mm
    # 5c) 客户指认了方向但没给数字（"it's the width" / "宽度"）：
    #     配合上一轮的尺寸线索使用。
    axis = _extract_size_axis(text)
    if axis:
        slots["size_axis"] = axis

    # 6) 点间距 / 亮度 / 特殊功能
    pitch = _extract_pixel_pitch(text)
    if pitch is not None:
        slots["pixel_pitch_mm"] = pitch
    brightness_min = _extract_brightness_min(text)
    if brightness_min is not None:
        slots["brightness_min"] = brightness_min
    if _any(lowered, _WATERPROOF_KEYWORDS):
        slots["waterproof"] = True
    if _any(lowered, _COB_KEYWORDS):
        slots["cob"] = True
    if _any(lowered, _HDR_KEYWORDS):
        slots["hdr"] = True
    if _any(lowered, _GOB_KEYWORDS):
        slots["gob"] = True
    if _any(lowered, _FLEXIBLE_KEYWORDS):
        slots["flexible"] = True

    # 7) 预算档位
    budget = _extract_budget_level(lowered)
    if budget:
        slots["budget_level"] = budget

    # 7c) 内容类型（视频 / 图片 / 两者都有）—— 只记录，不参与选型
    content_type = _extract_content_type(text)
    if content_type:
        slots["content_type"] = content_type

    # 7d) 价格 / 质量取向（推荐前那一问；客户已说预算时不问）
    preference = _extract_price_preference(text)
    if preference:
        slots["price_preference"] = preference

    # 7b) 交互需求（IFP 场景的关键卖点）
    if _any(lowered, ("手写", "书写", "触控", "触摸", "白板", "touch", "whiteboard", "annotation", "interactive")):
        slots["interaction"] = True

    # 7e) 场地几何（人数 / 面积 / 进深）—— 观看距离的另外几种来源
    slots.update(_extract_space_facts(text))

    # 8) 客户点名型号 / 系列
    series_id, model = _extract_model_mention(text)
    if series_id:
        slots["series_id"] = series_id
    if model:
        slots["model"] = model

    return slots

__all__ = ['_AMBIGUOUS_ENVIRONMENT_PURPOSES', '_ASCII_WORD_RE_CACHE', '_BUDGET_HIGH_KEYWORDS', '_BUDGET_LOW_KEYWORDS', '_BUDGET_MID_KEYWORDS', '_COB_KEYWORDS', '_CONTENT_IMAGE_KEYWORDS', '_CONTENT_MIXED_KEYWORDS', '_CONTENT_VIDEO_KEYWORDS', '_DISPLAY_TYPE_KEYWORDS', '_FIXED_KEYWORDS', '_FIXED_PURPOSES', '_FLEXIBLE_KEYWORDS', '_GOB_KEYWORDS', '_HDR_KEYWORDS', '_INDOOR_FIXED_PURPOSES', '_INDOOR_KEYWORDS', '_OUTDOOR_KEYWORDS', '_OUTDOOR_PURPOSES', '_PREFERENCE_BOTH_KEYWORDS', '_PREFERENCE_BOTH_WEAK_KEYWORDS', '_PREFERENCE_PRICE_KEYWORDS', '_PREFERENCE_QUALITY_KEYWORDS', '_PURPOSE_KEYWORDS', '_QUESTION_RE', '_RENTAL_KEYWORDS', '_SEMI_OUTDOOR_KEYWORDS', '_WATERPROOF_KEYWORDS', '_any', '_any_word', '_detect_purpose', '_extract_brightness_min', '_extract_budget_level', '_extract_content_type', '_extract_model_mention', '_extract_pixel_pitch', '_extract_price_preference', '_keyword_pattern', 'environment_from_purpose', 'extract_slots', 'looks_like_question', 'purpose_english']
