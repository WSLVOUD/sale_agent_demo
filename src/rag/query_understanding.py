"""
Phase 4：Query Understanding + Query Rewrite。

目标（来自计划文档）：
    不要直接使用客户原始自然语言进行 RAG。

    "I need a screen for a conference room around 5m viewing distance."
        ↓ 结构化
    {"display_type": "LED", "environment": "indoor", "purpose": "conference",
     "installation": "fixed", "viewing_distance_m": 5}
        ↓ 标准化检索式
    "indoor fixed installation LED display conference room viewing distance 5m"

设计要点：
  - 纯规则实现（确定性 + 零延迟 + 可测试），LLM 只作为可选补充（``use_llm=True``）
  - 支持中英文与常见多语言关键词（西/法/德/葡/俄/日），为 Phase 13 打底
  - 输出的 slots 直接对应 Phase 6 的 Requirement Profile 与 Phase 8 的推荐评分
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── 语言检测 ────────────────────────────────────────────────────────────────
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")

_LANGUAGE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 注意：提示词必须是该语言**特有**的，不能混入英文单词
    # （曾经把英文 "display" 放进德语提示 → 任何含 display 的英文都被判成德语）
    ("de", (
        "ich brauche", "wir brauchen", "brauchen", "bildschirm", "leinwand",
        "innenbereich", "außenbereich", "abstand", "mieten", "metern",
    )),
    ("fr", (
        "je cherche", "nous avons besoin", "besoin d", "écran", "ecran",
        "extérieur", "exterieur", "salle de",
    )),
    ("es", ("necesito", "pantalla", "centro comercial", "alquiler", "distancia de")),
    ("pt", ("preciso", "painel", "distância", "aluguel")),
    ("it", ("cerco", "schermo", "distanza", "esterno")),
    ("ru", ("нам нужен", "нужен", "экран", "расстояние")),
)


def detect_language(text: str) -> str:
    """粗粒度语言识别（用于选择关键词表与回复语种）。"""
    if not text:
        return "en"
    if _KANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _ARABIC_RE.search(text):
        return "ar"
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _CJK_RE.search(text):
        return "zh"
    lowered = text.lower()
    for lang, hints in _LANGUAGE_HINTS:
        if any(hint in lowered for hint in hints):
            return lang
    return "en"


# 语言代码 → 英文语言名（用于"跟随客户语言回复"的指令）
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
}


def language_name(code: Optional[str]) -> str:
    """语言代码 → 语言名（未知代码原样返回）。"""
    if not code:
        return "English"
    return LANGUAGE_NAMES.get(str(code).lower(), str(code))


def response_language_rule(language: Optional[str]) -> str:
    """v2.0 Phase 14：根据策略生成"用哪种语言回复"的指令。

    - ``RESPONSE_LANGUAGE_POLICY=en``（默认）→ 始终英语
    - ``RESPONSE_LANGUAGE_POLICY=auto``      → 跟随客户语言
    """
    try:
        from src.config import config

        policy = str(getattr(config, "RESPONSE_LANGUAGE_POLICY", "en")).lower()
    except Exception:  # pragma: no cover - 防御式
        policy = "en"

    if policy == "auto" and language and str(language).lower() != "en":
        return f"Reply in {language_name(language)} (the customer's language)."
    return "ALWAYS use English, regardless of the customer's language."


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
    "rental", "rent", "hire", "mieten", "louer", "location", "alquiler", "locação",
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
    "LED": ("led", "显示屏", "屏幕", "屏", "display", "screen", "pantalla", "écran", "экран"),
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
    ("rental", ("租赁", "rental", "event"), "rental event"),
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

_DISTANCE_VALUE = r"(\d+(?:[.,]\d+)?)"

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
_DISTANCE_UNITS = (
    r"米|m\b|meters?|metres?|meter|metros?|mètres?|metern|метр(?:ов|а)?|メートル|メーター"
    # 英制单位：客户常用 "100 feet" / "30ft away" / "40 inches"
    r"|feet|foot|ft|inches?|inch|yard|yards|yd|英尺|英寸|码"
)
_DISTANCE_PREFIXES = (
    "视距", "可视距离", "观看距离", "距离", "viewing distance", "view distance",
    "distance", "abstand", "distancia", "distanza", "distance de visionnage",
    "расстояние просмотра", "расстояние", "視距離", "視距离",
)

# 距离单位 → 米（默认米；英制单位按换算）
# 注意：用 (?<![a-z]) 而不是 \b —— "100ft" 里数字与字母之间没有 \b 边界。
_DISTANCE_UNIT_TO_METER: tuple[tuple[str, float], ...] = (
    (r"(?<![a-z])(?:ft|feet|foot)(?![a-z])|英尺", 0.3048),
    # "in" 只在紧贴数字时当作英寸，避免把 "distance in metres" 里的 in 误判
    (r"(?<![a-z])(?:inch|inches)(?![a-z])|\d\s*in(?![a-z])|英寸", 0.0254),
    (r"(?<![a-z])(?:yd|yards?)(?![a-z])|码", 0.9144),
)


def _distance_to_meters(value: float, matched_text: str) -> float:
    """把"数值 + 单位"换算成米（英制单位要换算，否则 100 feet 会被当成 100 米）。"""
    for unit_pattern, factor in _DISTANCE_UNIT_TO_METER:
        if re.search(unit_pattern, matched_text, re.IGNORECASE):
            return value * factor
    return value


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


# ── 观看距离的"模糊回答"（降门槛提问后，客户常说 close / near / far）──────────
# 计划（Phase 7）明确要求允许客户回答 approximately / roughly / near / far /
# more than 10m；这里把这类回答映射为一个**近似**距离，让流程能继续往下走，
# 而不是卡在观看距离上反复问。
_ROUGH_DISTANCE_PATTERNS: "tuple[tuple[Any, float], ...]" = (
    (re.compile(r"\b(?:very close|really close|right in front|extremely close)\b|非常近|很近|特别近", re.IGNORECASE), 2.0),
    (re.compile(
        r"\b(?:close|closer|closest|near|nearby)\b(?!\s+(?:the|a|an|my|your|our|it|them|that|this)\b)"
        r"|近距离|比较近|离得近|挺近",
        re.IGNORECASE,
    ), 3.0),
    (re.compile(r"\b(?:medium|middle|moderate|average|halfway)\b|中等|不远不近", re.IGNORECASE), 7.0),
    (re.compile(r"\b(?:far|farther|further|far away)\b|远处|比较远|很远|挺远", re.IGNORECASE), 15.0),
)

_DISTANCE_CONTEXT_RE = re.compile(
    r"\b(?:away|distance|viewers?|audience|screen|sitting|seated)\b|离|距离|观众|屏幕",
    re.IGNORECASE,
)

_RANGE_UNITS = r"(?:meters?|metres?|m|feet|foot|ft|米|英尺)"


def _extract_rough_viewing_distance(text: str) -> Optional[float]:
    """"近 / 远 / 中等"这类模糊回答 → 近似观看距离（米）。

    只在"像在回答观看距离"的短句里生效，避免把 "close the deal"、
    "near the airport" 这类说法误判成距离。
    """
    lowered = str(text or "").lower().strip()
    if not lowered:
        return None
    words = re.findall(r"[a-z\u4e00-\u9fff]+", lowered)
    if len(words) > 6 and not _DISTANCE_CONTEXT_RE.search(lowered):
        return None
    for pattern, value in _ROUGH_DISTANCE_PATTERNS:
        if pattern.search(lowered):
            return value
    return None


def _extract_ranged_viewing_distance(text: str) -> Optional[float]:
    """区间 / 上下界 → 近似观看距离："5-10 metres" → 7.5、"more than 10m" → 15。"""
    lowered = str(text or "").lower()

    ranged = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:-|–|—|~|～|to|到|至)\s*(\d+(?:[.,]\d+)?)\s*" + _RANGE_UNITS,
        lowered,
    )
    if ranged:
        try:
            low = float(ranged.group(1).replace(",", "."))
            high = float(ranged.group(2).replace(",", "."))
        except ValueError:
            return None
        if 0 < low <= high:
            return round(_distance_to_meters((low + high) / 2, ranged.group(0)), 3)

    upper = re.search(
        r"(?:more than|over|above|greater than|>|超过|以上|至少)\s*"
        r"(\d+(?:[.,]\d+)?)\s*" + _RANGE_UNITS,
        lowered,
    ) or re.search(r"(\d+(?:[.,]\d+)?)\s*" + _RANGE_UNITS + r"\s*(?:\+|以上|多)", lowered)
    if upper:
        try:
            value = float(upper.group(1).replace(",", "."))
        except ValueError:
            return None
        if value > 0:
            return round(_distance_to_meters(value * 1.5, upper.group(0)), 3)

    lower = re.search(
        r"(?:less than|under|below|within|no more than|以内|不到|少于)\s*"
        r"(\d+(?:[.,]\d+)?)\s*" + _RANGE_UNITS,
        lowered,
    )
    if lower:
        try:
            value = float(lower.group(1).replace(",", "."))
        except ValueError:
            return None
        if value > 0:
            return round(_distance_to_meters(value * 0.6, lower.group(0)), 3)
    return None


def _extract_viewing_distance(text: str) -> Optional[float]:
    lowered = text.lower()
    # 允许"距离"与数值之间有少量修饰词（如 "distancia de visión 15 metros"）
    # 注意：中间修饰词必须是纯字母（[^\W\d_]），否则会吃掉数值的高位数字
    pattern = re.compile(
        r"(?:" + "|".join(_DISTANCE_PREFIXES) + r")"
        r"(?:\s*(?:是|为|约|大概|大约|around|about|approx\.?|:|[^\W\d_]{1,12})){0,3}\s*"
        + _DISTANCE_VALUE + r"\s*(?:" + _DISTANCE_UNITS + r")",
        re.IGNORECASE,
    )
    match = pattern.search(lowered)
    if not match:
        # 反向语序："3米视距"、"5m viewing distance"
        pattern_rev = re.compile(
            _DISTANCE_VALUE + r"\s*(?:" + _DISTANCE_UNITS + r")\s*(?:的)?\s*(?:视距|可视距离|观看距离|viewing distance|視距離)",
            re.IGNORECASE,
        )
        match = pattern_rev.search(lowered)
    if not match:
        # 区间 / 上下界："5-10 metres" → 7.5、"more than 10m" → 15、"less than 5m" → 3
        ranged = _extract_ranged_viewing_distance(lowered)
        if ranged is not None:
            return ranged
    if not match:
        # 模糊回答："close / near / far / 近 / 远"（第二轮降门槛提问时客户的常见回答）
        rough = _extract_rough_viewing_distance(lowered)
        if rough is not None:
            return rough
    if not match:
        # 兜底：客户直接回答裸数值 + 单位（"5m"、"about 5 meters"、"大约5米"）。
        # 排除面积（平米）与尺寸（5m x 3m）表达，避免误判。
        _has_area = any(token in lowered for token in ("平米", "平方米", "平方"))
        # 尺寸表达（含单位、by/乘 等写法）以 _extract_target_size 的判定为准
        _size_width, _size_height = _extract_target_size(lowered)
        _has_size = (
            _size_width is not None
            or _size_height is not None
            # "宽度 1.29m" 这类带方向词的尺寸同样不算观看距离
            or bool(_extract_axis_measurements(lowered))
        )
        if not _has_area and not _has_size:
            bare = re.compile(
                # 注意排除前面的 "." / ","：否则 "长1.29米" 会被从 "29米" 开始匹配成 29 米
                r"(?<![\w.,])(?:about|around|approx\.?|approximately|roughly|约|大概|大约|差不多)?\s*"
                + _DISTANCE_VALUE
                + r"\s*(?:meters?|metres?|m|米|feet|foot|ft|英寸|英尺)"
                # 单位后面可以是空白/结尾/标点，也可以是中文（"6 米远"）
                + r"(?=\s|$|[，。,.?？!]|[\u4e00-\u9fff])",
                re.IGNORECASE,
            )
            match = bare.search(lowered)
    if not match:
        return None
    raw = match.group(1).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    # 英制单位先换算成米，再按"合理视距"范围校验（100 feet ≈ 30.5 m 才是对的）
    value = _distance_to_meters(value, match.group(0))
    return round(value, 3) if 0 < value <= 200 else None


# 长度单位 → mm 换算系数
_SIZE_UNIT_TO_MM: Dict[str, float] = {
    "mm": 1.0, "毫米": 1.0,
    "cm": 10.0, "厘米": 10.0, "公分": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
    "metre": 1000.0, "metres": 1000.0, "米": 1000.0,
    "ft": 304.8, "feet": 304.8, "foot": 304.8, "英尺": 304.8,
    "in": 25.4, "inch": 25.4, "inches": 25.4, "英寸": 25.4, "寸": 25.4,
}

_SIZE_UNIT_PATTERN = (
    r"(mm|cm|m|meters?|metres?|feet|foot|ft|inches?|inch|毫米|厘米|公分|米|英尺|英寸|寸)"
)

# 客户只报一个"屏幕长度"时最常见的单位（m / ft 更可能是观看距离，不在此列）
_BARE_MEASUREMENT_RE = re.compile(
    r"(?<![a-z0-9])(\d+(?:[.,]\d+)?)\s*(mm|cm|毫米|厘米|公分|inch|inches|英寸)(?![a-z0-9])",
    re.IGNORECASE,
)

# 客户指认方向（不带数字时才有意义："it's the width" / "宽度"）
# 注意：客户口中的"长/长边(length)"= 水平方向 = 我们档案里的"宽"；
# 客户口中的"宽/width"在同时出现"长"时 = 竖直方向 = 我们档案里的"高"。
_SIZE_AXIS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("width", r"(?<![a-z])(?:width|wide|length|long side)(?![a-z])|宽度|宽|长度|长边|长"),
    ("height", r"(?<![a-z])(?:height|tall)(?![a-z])|高度|高"),
    ("diagonal", r"(?<![a-z])(?:diagonal|diag)(?![a-z])|对角线|对角"),
)

# "数字 + 单位 + 方向词"的单条尺寸（支持 "45cm is the width" 这种语序）
_AXIS_WORD = (
    r"(?:length|long|width|wide|height|tall|diagonal|diag"
    r"|长度|长边|长|宽度|宽|高度|高|对角线|对角)"
)

_AXIS_WORD_TO_SLOT: Dict[str, str] = {
    "length": "length", "long": "length", "长度": "length", "长边": "length", "长": "length",
    "width": "width", "wide": "width", "宽度": "width", "宽": "width",
    "height": "height", "tall": "height", "高度": "height", "高": "height",
    "diagonal": "diagonal", "diag": "diagonal", "对角线": "diagonal", "对角": "diagonal",
}

# 一次扫描里的两种 token：方向词 / "数值 + 可选单位"
_SIZE_UNIT_BODY = (
    r"mm|cm|m|meters?|metres?|feet|foot|ft|inches?|inch|毫米|厘米|公分|米|英尺|英寸|寸"
)
_SIZE_TOKEN_RE = re.compile(
    r"(?P<axis>(?<![A-Za-z])" + _AXIS_WORD + r"(?![A-Za-z]))"
    r"|(?P<num>\d+(?:[.,]\d+)?)(?:\s*(?P<unit>" + _SIZE_UNIT_BODY + r"))?",
    re.IGNORECASE,
)

# 数值与方向词之间的"填充词"只能是这样（不能夹着另一个数字或句号）
_AXIS_GAP_RE = re.compile(
    r"^[\s，,、:：=＝\-–—_()（）]*"
    r"(?:(?:is|are|as|us|the|a|an|of|it|its|and|about|roughly|approx\.?)\s+)*"
    r"[\s，,、:：=＝\-–—_()（）]*"
    r"(?:是|为|的|约|大概|大约|有|宽|高|长)?"
    r"[\s，,、:：=＝\-–—_()（）]*$",
    re.IGNORECASE,
)


def _fix_unit_typos(text: str) -> str:
    """修正常的单位笔误："45xm" → "45cm"、"1290 x m" → "1290cm"。

    只处理"数字 + x + m"这一种写法（客户把 cm 打成 xm），
    其它内容一律不动，避免误伤 "5m x 3m" 这类真正的乘号表达。
    """
    return re.sub(r"(?<=\d)\s*[xX]\s*m(?![a-z])", "cm", str(text or ""))


def _extract_axis_measurements(text: str) -> List[tuple[str, float]]:
    """抽出"带方向词的尺寸"，**从左到右扫描**，避免一个方向词被两次归给不同数字。

    支持（客户口径：怎么写都要认）：
      方向词在前："长是5，高是3" / "长1.29米，宽0.45米" / "width: 3m, height 5m"
      数字在前：  "45cm is the width" / "3m wide 5m long" / "5米宽"
      不带单位：  "长是5，高是3" → 5m / 3m（量级推断见 ``_size_to_mm``）
    """
    fixed = _fix_unit_typos(str(text or ""))
    results: List[tuple[str, float]] = []
    pending_axis: Optional[str] = None          # 方向词在前，等后面的数值
    pending_axis_end = -1
    last_number: Optional[tuple[float, Optional[str]]] = None   # 数值在前，等后面的方向词
    last_number_end = -1

    for match in _SIZE_TOKEN_RE.finditer(fixed):
        if match.group("axis"):
            axis = _AXIS_WORD_TO_SLOT.get(match.group("axis").lower().strip())
            if not axis:
                continue
            # 前面刚出现过一个数值，而且中间只有填充词 → 这个方向词是在修饰那个数值
            if last_number is not None:
                gap = fixed[last_number_end:match.start()]
                if len(gap) <= 16 and _AXIS_GAP_RE.match(gap):
                    value, unit = last_number
                    mm = _size_to_mm(value, unit)
                    if mm:
                        results.append((axis, mm))
                    last_number = None
                    continue
            pending_axis = axis
            pending_axis_end = match.end()
            continue

        # 数值 token
        try:
            value = float(str(match.group("num")).replace(",", "."))
        except (TypeError, ValueError):  # pragma: no cover - 防御式
            continue
        unit = (match.group("unit") or "").lower() or None
        if pending_axis is not None:
            gap = fixed[pending_axis_end:match.start()]
            if len(gap) <= 16 and _AXIS_GAP_RE.match(gap):
                mm = _size_to_mm(value, unit)
                if mm:
                    results.append((pending_axis, mm))
                pending_axis = None
                last_number = None
                continue
            pending_axis = None
        last_number = (value, unit)
        last_number_end = match.end()
    return results


def _resolve_axis_measurements(
    measurements: Sequence[tuple[str, float]],
) -> tuple[Optional[float], Optional[float]]:
    """把"客户说的方向"翻译成档案里的宽 / 高。

    客户描述一块屏的矩形时用词常常是"长 × 宽"，而 LED 行业口径是"宽 × 高"：
      - 只给"长(length)"        → 水平方向 = 我们的宽
      - 给了"长 + 宽"           → 长边 = 我们的宽，另一条边 = 我们的高
      - 只给"宽(width)"         → 我们的宽
      - "对角线(diagonal)"      → 换算不了箱体，交给系统继续问宽高
    """
    by_axis: Dict[str, float] = {}
    for axis, mm in measurements:
        by_axis.setdefault(axis, mm)

    width_mm = by_axis.get("width")
    height_mm = by_axis.get("height")
    length_mm = by_axis.get("length")

    if length_mm:
        if width_mm and not height_mm:
            # 客户用"长 + 宽"描述这块屏：长边和短边，分别落到宽和高
            width_mm, height_mm = length_mm, width_mm
        elif not width_mm:
            width_mm = length_mm
        elif height_mm:
            # 三个方向都给了：长边归宽度，另一条边归高度
            width_mm, height_mm = length_mm, min(width_mm, height_mm)

    if abs(float(width_mm or 0) - float(height_mm or 0)) < 1e-6:
        height_mm = None
    return width_mm, height_mm


def _extract_bare_measurement(text: str) -> Optional[float]:
    """裸尺寸（"129,2cm" / "1292 mm" / "51 inch"）→ 毫米。

    客户报裸数字时**不替他判断**这是宽、高还是对角线，
    只记成线索，由系统追问（见 readiness 的 size_axis）。
    """
    match = _BARE_MEASUREMENT_RE.search(str(text or ""))
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except ValueError:  # pragma: no cover - 防御式
        return None
    return _size_to_mm(value, match.group(2).lower())


def _extract_size_axis(text: str) -> Optional[str]:
    """客户是否指认了尺寸方向（width / height / diagonal）。"""
    lowered = str(text or "").lower()
    if re.search(r"\d", lowered):
        # 带数字的表达（"129.2cm wide"）由 _extract_target_size 处理
        return None
    for axis, pattern in _SIZE_AXIS_PATTERNS:
        if re.search(pattern, lowered):
            return axis
    return None


def _size_to_mm(value: float, unit: Optional[str]) -> Optional[float]:
    """把"数值 + 单位"换算成毫米；没写单位时按量级推断（>50 视为 mm，否则视为 m）。"""
    if value <= 0:
        return None
    if unit:
        return round(value * _SIZE_UNIT_TO_MM.get(unit.lower(), 1000.0), 3)
    inferred_unit = "mm" if value > 50 else "m"
    return round(value * _SIZE_UNIT_TO_MM[inferred_unit], 3)


def _extract_target_size(text: str) -> tuple[Optional[float], Optional[float]]:
    """解析目标尺寸，支持多单位并统一换算为毫米。

    支持：``5m x 3m`` / ``5米*3米`` / ``5000x3000mm`` / ``500cm x 300cm`` /
    ``16ft x 9ft`` / ``120 x 90``（无单位时按量级推断）/ ``5.5 by 3 meters``。
    也支持只给一个方向：``5米宽`` / ``3米高``（另一个方向返回 None，交给 Gate 追问）。
    """
    lowered = str(text).lower().strip()
    # 统一分隔符为 " x "
    # （客户写法很随意：3*5 / 3x5 / 3×5 / 3✕5 / 3＊5(全角) / 3 by 5 / 3乘5 都要认）
    normalized = re.sub(r"\s*(?:x|×|✕|╳|＊|\*|by|乘)\s*", " x ", lowered, flags=re.IGNORECASE)

    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*" + _SIZE_UNIT_PATTERN + r"?\s*x\s*"
        r"(\d+(?:[.,]\d+)?)\s*" + _SIZE_UNIT_PATTERN + r"?",
        normalized,
    )
    if match:
        width_raw, width_unit, height_raw, height_unit = match.groups()
        try:
            width = float(width_raw.replace(",", "."))
            height = float(height_raw.replace(",", "."))
        except ValueError:
            return None, None

        # 只写了一侧单位 → 另一侧继承（"5m x 3" / "5000 x 3000mm"）
        if not width_unit and height_unit:
            width_unit = height_unit
        if not height_unit and width_unit:
            height_unit = width_unit
        # 两侧都没写单位 → 各自按量级推断
        return _size_to_mm(width, width_unit), _size_to_mm(height, height_unit)

    # 客户不带分隔符："3米5米" / "3 m 5 m"（两个相邻的尺寸，中间没有 x）
    if not match:
        pair = re.search(
            r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*" + _SIZE_UNIT_PATTERN
            + r"\s*(?:,|，|、|和|and)?\s*"
            r"(\d+(?:[.,]\d+)?)\s*" + _SIZE_UNIT_PATTERN,
            normalized,
        )
        if pair:
            # 别把"观看距离 5 米 + 尺寸 3 米"这种句子当成宽高
            before = normalized[max(0, pair.start() - 14):pair.start()]
            if not re.search(
                r"视距|观看距离|可视距离|距离|distance|away|far", before, re.IGNORECASE
            ):
                width = _size_to_mm(
                    float(pair.group(1).replace(",", ".")), (pair.group(2) or "").lower() or None
                )
                height = _size_to_mm(
                    float(pair.group(3).replace(",", ".")), (pair.group(4) or "").lower() or None
                )
                if width and height:
                    return width, height

    # 只给一个方向：宽 / 高
    single = re.search(
        r"(\d+(?:[.,]\d+)?)\s*" + _SIZE_UNIT_PATTERN + r"?\s*(宽|高|width|height|tall|wide)",
        normalized,
    )
    if single:
        value_raw, unit, axis = single.groups()
        try:
            value = float(value_raw.replace(",", "."))
        except ValueError:
            return None, None
        mm = _size_to_mm(value, unit)
        if axis in ("宽", "width", "wide"):
            return mm, None
        return None, mm

    return None, None


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


# ── 场地几何事实（v2.2）：观众人数 / 场地面积 / 场地纵深 ─────────────────────
# 这三个量本身**不是推荐规则**，只是"观看距离"的不同来源。系统用同一组物理公式
# 把它们折算成观看距离区间（见 parameter_inference.estimate_viewing_distance），
# 所以客户换一种说法（多少人 / 多少平米 / 进深几米）不需要新增推荐分支。
_AUDIENCE_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:people|persons?|viewers?|seats?|guests?|attendees?|audience\s*members?|"
    r"观众|人员|座位|个人|人)",
    re.IGNORECASE,
)
_ROOM_AREA_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(?:sq\.?\s*m(?:etres?|eters?)?\.?|square\s*(?:metres?|meters?)|m2|m²|㎡|"
    r"平米|平方米|个平方|平方)",
    re.IGNORECASE,
)
_ROOM_DEPTH_RES: Tuple["re.Pattern[str]", ...] = (
    re.compile(
        r"(?:depth|deep|纵深|进深|长度)[^\d\n]{0,12}?(\d+(?:\.\d+)?)", re.IGNORECASE
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?|米)\s*(?:deep|深|纵深|进深)",
        re.IGNORECASE,
    ),
)


def _positive_number(raw: str) -> Optional[float]:
    try:
        value = float(str(raw).replace(",", "").replace("，", ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _extract_space_facts(text: str) -> Dict[str, Any]:
    """从客户原话里读出"人数 / 面积 / 进深"（纯规则、可单测）。

    这些值一律按"客户明说"记录（来源 explicit），但它们**不会**直接决定型号：
    只有"观看距离"才是选型输入，这里只是给观看距离提供另一种来源。
    """
    facts: Dict[str, Any] = {}
    text = str(text or "")
    if not text:
        return facts

    match = _AUDIENCE_RE.search(text)
    if match:
        people = _positive_number(match.group(1))
        if people and 1 <= people <= 100000:
            facts["audience_count"] = int(people)

    match = _ROOM_AREA_RE.search(text)
    if match:
        area = _positive_number(match.group(1))
        if area and 1 <= area <= 100000:
            facts["room_area_sqm"] = round(float(area), 2)

    for pattern in _ROOM_DEPTH_RES:
        match = pattern.search(text)
        if not match:
            continue
        depth = _positive_number(match.group(1))
        if depth and 1 <= depth <= 200:
            facts["room_depth_m"] = round(float(depth), 2)
            break
    return facts


# ── 结果结构 ────────────────────────────────────────────────────────────────
@dataclass
class QueryUnderstanding:
    """Query 理解结果：结构化槽位 + 标准化检索式。"""
    raw_query: str
    language: str = "en"
    slots: Dict[str, Any] = field(default_factory=dict)
    retrieval_query: str = ""
    method: str = "rule"
    # Phase 6：结构化需求档案（历史 + 本轮合并）
    profile: Any = None

    # 便捷访问
    @property
    def display_type(self) -> Optional[str]:
        return self.slots.get("display_type")

    @property
    def environment(self) -> Optional[str]:
        return self.slots.get("environment")

    @property
    def installation(self) -> Optional[str]:
        return self.slots.get("installation")

    @property
    def purpose(self) -> Optional[str]:
        return self.slots.get("purpose")

    @property
    def viewing_distance_m(self) -> Optional[float]:
        return self.slots.get("viewing_distance_m")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "language": self.language,
            "slots": dict(self.slots),
            "retrieval_query": self.retrieval_query,
            "method": self.method,
            "profile": self.profile.model_dump() if self.profile is not None else None,
        }


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
    if "display_type" not in slots:
        # 没有显式产品类型词时，只要有"型号 / 点间距 / 场景"这类销售上下文，
        # 且未指明 LCD/IFP，就按目录主力品类 LED 处理（后续 Phase 8 的硬约束仍会校验）
        sales_context = (
            _extract_pixel_pitch(text) is not None
            or _extract_model_mention(text)[0] is not None
            or _detect_purpose(lowered) is not None
            or _any(lowered, _INDOOR_KEYWORDS)
            or _any(lowered, _OUTDOOR_KEYWORDS)
        )
        if sales_context:
            slots["display_type"] = "LED"

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


def merge_slots(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """合并两轮槽位，并正确维护来源标记（M2 四态）。

    来源标记：
      - ``_inferred_slots``    ：算法估算（不能放行 Gate）
      - ``_scenario_derived``  ：客户原话场景直接判定（可以放行对应字段）
      - ``_default_slots``     ：系统业务默认值（参与打分，不单独放行）

    关键：当**后面某一轮客户明确说出**某个字段时，该字段必须从历史标记中移除
    —— 否则前一轮"场景默认固装"这类值会一直污染后续判定，出现
    "客户已经明说固装，系统仍当成没说过"的错误。
    """
    markers = ("_inferred_slots", "_scenario_derived", "_default_slots")
    merged = dict(base or {})
    incoming_marker: Dict[str, str] = {}
    for marker in markers:
        for key in incoming.get(marker) or []:
            incoming_marker[str(key)] = marker

    for key, value in incoming.items():
        if key in markers:
            continue
        merged[key] = value
        source_marker = incoming_marker.get(key)
        for marker in markers:
            current = merged.get(marker)
            if marker == source_marker:
                if current is None:
                    merged[marker] = [key]
                elif key not in current:
                    current.append(key)
            elif current and key in current:
                # incoming 里这个字段是客户明说的 → 撤销历史标记
                current.remove(key)

    for marker in markers:
        if not merged.get(marker):
            merged.pop(marker, None)
    return merged


def build_retrieval_query(slots: Dict[str, Any], fallback: str = "") -> str:
    """把结构化槽位重写成标准化英文检索式。"""
    if not slots:
        return fallback or "LED display product"

    parts: List[str] = []
    environment = slots.get("environment")
    if environment == "indoor":
        parts.append("indoor")
    elif environment == "outdoor":
        parts.append("outdoor")
    elif environment == "semi_outdoor":
        parts.append("semi outdoor")

    installation = slots.get("installation")
    if installation == "rental":
        parts.append("rental")
    elif installation == "fixed":
        parts.append("fixed installation")

    display_type = slots.get("display_type") or "LED"
    parts.append(f"{display_type} display")

    english_purpose = purpose_english(slots.get("purpose"))
    if english_purpose:
        parts.append(english_purpose)

    if slots.get("viewing_distance_m") is not None:
        parts.append(f"viewing distance {slots['viewing_distance_m']:g}m")
    if slots.get("pixel_pitch_mm") is not None:
        parts.append(f"pixel pitch {slots['pixel_pitch_mm']:g}mm")
    if slots.get("brightness_min") is not None:
        parts.append(f"brightness above {slots['brightness_min']}nit")
    if slots.get("waterproof"):
        parts.append("waterproof IP65")
    if slots.get("cob"):
        parts.append("COB")
    if slots.get("hdr"):
        parts.append("HDR")
    if slots.get("interaction"):
        parts.append("touch whiteboard interactive")
    if slots.get("model"):
        parts.append(str(slots["model"]))
    elif slots.get("series_id"):
        parts.append(str(slots["series_id"]))
    if slots.get("budget_level"):
        parts.append({"low": "low price", "mid": "mid price", "high": "premium"}[slots["budget_level"]])

    return " ".join(part for part in parts if part)


def understand_query(
    message: str,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    llm_slots: Optional[Dict[str, Any]] = None,
    profile: Any = None,
) -> QueryUnderstanding:
    """Query 理解主入口。

    Args:
        message: 本轮用户消息
        history: 对话历史（用于补全多轮槽位）
        llm_slots: 可选，由 LLM 提取的槽位（LLM 只做"提取事实"，不做技术参数推断）
    """
    from src.models.requirement import RequirementProfile, merge_profiles

    text = str(message or "").strip()
    slots: Dict[str, Any] = {}
    history_slots: Dict[str, Any] = {}
    for turn in history or []:
        role = turn.get("role") or turn.get("type")
        if role in ("user", "human"):
            history_slots = merge_slots(history_slots, extract_slots(str(turn.get("content", ""))))
    slots = merge_slots({}, history_slots)

    current_slots = extract_slots(text)
    slots = merge_slots(slots, current_slots)
    if llm_slots:
        # LLM 只补充它独有的信息，不覆盖规则已确定的事实
        for key, value in llm_slots.items():
            if value in (None, "", [], {}):
                continue
            slots.setdefault(key, value)

    # Phase 6：历史事实标记为 inferred，本轮明确说出的标记为 explicit
    base = profile if isinstance(profile, RequirementProfile) else None
    if base is None and history_slots:
        base = RequirementProfile.from_slots(history_slots)
    merged_profile = merge_profiles(base, current_slots, explicit_keys=set(current_slots))
    if llm_slots:
        merged_profile = merge_profiles(merged_profile, llm_slots)

    return QueryUnderstanding(
        raw_query=text,
        language=detect_language(text) if text else "en",
        slots=slots,
        retrieval_query=build_retrieval_query(slots, fallback=text),
        method="llm+rule" if llm_slots else "rule",
        profile=merged_profile,
    )
