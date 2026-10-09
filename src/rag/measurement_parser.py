"""Deterministic parsing of physical measurements and venue geometry."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_DISTANCE_VALUE = r"(\d+(?:[.,]\d+)?)"

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
    r"平米|平方米|个平方|平方|平)",
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
        # "the screen is 6m deep" 说的是屏，不是场地 → 不算场地纵深
        if _screen_spoken_near(text, match.start()):
            continue
        depth = _positive_number(match.group(1))
        if depth and 1 <= depth <= 200:
            facts["room_depth_m"] = round(float(depth), 2)
            break
    return facts



# ── 场地尺寸（和"屏体尺寸"是两回事）─────────────────────────────────────────
# 实测风险：客户说 "the room is 8m x 5m"（回答"场地多大"）时，旧解析会把它当成
# **8×5m 的屏体**，算出完全错误的箱体。这里把"场地尺寸"单独识别出来，并从文本里
# 挖掉，屏体尺寸继续按原来的规则解析（该问客户还是问客户）。
_ROOM_CONTEXT_RE = re.compile(
    r"\b(?:room|hall|venue|auditorium|lobby|space|floor)\b|房间|场地|屋里|大厅|空间|室内尺寸",
    re.IGNORECASE,
)

_SCREEN_CONTEXT_RE = re.compile(
    r"\b(?:screen|display|panel|video\s*wall|led\s*wall)\b|屏幕|屏体|显示屏|大屏",
    re.IGNORECASE,
)

_DIM_PAIR_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:m|metres?|meters?|米)?\s*(?:x|×|✕|\*|by|乘)\s*"
    r"(\d+(?:[.,]\d+)?)\s*(?:m|metres?|meters?|米)?",
    re.IGNORECASE,
)

_ROOM_WIDE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:m|metres?|meters?|米)?\s*(?:wide|宽|宽度)", re.IGNORECASE
)

_ROOM_DEEP_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:m|metres?|meters?|米)?\s*(?:deep|深|纵深|进深)",
    re.IGNORECASE,
)



def _screen_spoken_near(text: str, position: int) -> bool:
    """这个测量的上文是不是在说"屏幕"？"""
    window = str(text or "")[max(0, position - 30): position]
    return bool(_SCREEN_CONTEXT_RE.search(window))



def _looks_like_room(text: str, position: int, implied: bool = False) -> bool:
    """这个测量的上下文是在说"场地"吗（并且没说"屏幕"）？

    ``implied=True``：整段话里出现了"深 / 纵深 / 进深"这类**只有场地才有**的说法
    （屏幕不说"深"），于是同一段话里的"5 米宽"也按场地宽度理解。

    判断用"最近的上下文词"：例如 "indoor fixed LED screen, the room is 8m x 5m"
    里既出现 screen 又出现 room，但紧挨着数字的是 room → 按场地理解。
    """
    window = str(text or "")[max(0, position - 40): position]
    room_hits = list(_ROOM_CONTEXT_RE.finditer(window))
    screen_hits = list(_SCREEN_CONTEXT_RE.finditer(window))
    last_screen = screen_hits[-1].start() if screen_hits else None
    if room_hits and (last_screen is None or room_hits[-1].start() > last_screen):
        return True
    return implied and last_screen is None



def _extract_room_dimensions(text: str) -> Tuple[Dict[str, Any], str]:
    """抽出场地的宽 / 深，并把这些数字从文本里挖掉。

    支持两种说法：

        the room is 8m x 5m               → 面积 40㎡，纵深 8m
        the hall is 10m wide and 6m deep  → 面积 60㎡，纵深 6m（"deep" 为准）
        5米宽8米深                        → 同上

    返回 ``(facts, masked_text)``：masked_text 里场地数字被空格替换，
    后续"屏体尺寸"解析就不会把它误当成屏幕。
    """
    source = str(text or "")
    facts: Dict[str, Any] = {}
    masked = source
    # "深 / 纵深 / 进深"本身就是场地词（屏幕不会说"深"）→ 同一段话按场地理解
    implied_room = bool(_ROOM_DEEP_RE.search(source))
    if not source or not (_ROOM_CONTEXT_RE.search(source) or implied_room):
        return facts, masked

    def _mask(start: int, end: int) -> None:
        nonlocal masked
        masked = masked[:start] + " " * (end - start) + masked[end:]

    # ① "8m x 5m" 这种成对写法
    for match in _DIM_PAIR_RE.finditer(source):
        if not _looks_like_room(source, match.start(), implied=implied_room):
            continue
        first = _positive_number(match.group(1))
        second = _positive_number(match.group(2))
        if not first or not second:
            continue
        if not (1 <= first <= 100 and 1 <= second <= 100):
            continue
        wide, deep = max(first, second), min(first, second)
        facts["room_area_sqm"] = round(float(wide * deep), 2)
        facts["room_depth_m"] = round(float(wide), 2)
        _mask(match.start(), match.end())
        return facts, masked

    # ② "10m wide … 6m deep" / "5米宽8米深"
    width = depth = None
    for match in _ROOM_WIDE_RE.finditer(source):
        if _looks_like_room(source, match.start(), implied=implied_room):
            width = _positive_number(match.group(1))
            _mask(match.start(), match.end())
            break
    for match in _ROOM_DEEP_RE.finditer(source):
        if _looks_like_room(source, match.start(), implied=implied_room):
            depth = _positive_number(match.group(1))
            _mask(match.start(), match.end())
            break
    if depth:
        facts["room_depth_m"] = round(float(depth), 2)
    if width and depth:
        facts["room_area_sqm"] = round(float(width * depth), 2)
    return facts, masked

__all__ = ['_AUDIENCE_RE', '_AXIS_GAP_RE', '_AXIS_WORD', '_AXIS_WORD_TO_SLOT', '_BARE_MEASUREMENT_RE', '_DIM_PAIR_RE', '_DISTANCE_CONTEXT_RE', '_DISTANCE_PREFIXES', '_DISTANCE_UNITS', '_DISTANCE_UNIT_TO_METER', '_DISTANCE_VALUE', '_RANGE_UNITS', '_ROOM_AREA_RE', '_ROOM_CONTEXT_RE', '_ROOM_DEEP_RE', '_ROOM_DEPTH_RES', '_ROOM_WIDE_RE', '_ROUGH_DISTANCE_PATTERNS', '_SCREEN_CONTEXT_RE', '_SIZE_AXIS_PATTERNS', '_SIZE_TOKEN_RE', '_SIZE_UNIT_BODY', '_SIZE_UNIT_PATTERN', '_SIZE_UNIT_TO_MM', '_distance_to_meters', '_extract_axis_measurements', '_extract_bare_measurement', '_extract_ranged_viewing_distance', '_extract_room_dimensions', '_extract_rough_viewing_distance', '_extract_size_axis', '_extract_space_facts', '_extract_target_size', '_extract_viewing_distance', '_fix_unit_typos', '_looks_like_room', '_positive_number', '_resolve_axis_measurements', '_screen_spoken_near', '_size_to_mm']
