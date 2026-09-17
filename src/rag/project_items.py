"""一个项目下的多条屏体需求（客户口径 2026-09-18）。

客户可能会说："教堂里一块室内屏，门口再来一块室外屏" —— 这是一个项目、两块屏，
每块屏有各自的场景 / 室内外 / 尺寸 / 视距，最终要给出**两份**推荐 + **两份**
箱体/模组计算。也可能混着要 LED + LCD/IFP。

模块职责：
  - ``detect_new_item``    判断客户这句话是不是在说"另一块屏"
  - ``screen_label``       给第二块（及以后）的推荐加"这是哪一块"的标签
  - ``combined_summary``   两块（或多块）屏都推荐完之后，给一份汇总

注：客户口径（2026-09-18 二次确认）**不再**在推荐后主动追问
"By the way — is this the only screen in the project…"，多屏只在客户自己
提到第二块屏时才启用。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── "还有一块屏"的显式信号 ──────────────────────────────────────────────
_NEW_ITEM_PATTERNS = (
    # 英文
    r"\banother\s+(?:one|screen|display|led|lcd|panel|unit|set)\b",
    r"\bsecond\s+(?:screen|display|led|lcd|panel|unit|one)\b",
    r"\bone\s+more\s+(?:screen|display|led|lcd|panel|unit)\b",
    r"\btwo\s+(?:screens|displays|leds|lcds|panels|units)\b",
    r"\b(?:2|two)\s+x\s+(?:screen|display|led)",
    r"\balso\s+(?:need|want|looking)\b",
    r"\bwe\s+also\s+need\b",
    # 中文
    r"另外(?:一?块|一个|还要|再要|需要)",
    r"再(?:来|要|加|做|装)一?[块个台]",
    r"第二块|第二个位置|另一个位置|其他位置|别的?位置",
    r"还要(?:一?块|一?个|加)",
    r"两块|两个位置|两台",
    r"门口|入口处?|大门口|正门|店招|门头",
)

_NEW_ITEM_RE = re.compile("|".join(_NEW_ITEM_PATTERNS), re.IGNORECASE)


# 「另一个位置」的方位词：只有出现这些词，环境/屏类型冲突才当成"第二块屏"
_LOCATION_HINT_RE = re.compile(
    r"门口|入口|大门|正门|外面|外侧|另(?:一)?(?:处|个位置|边)|别的?位置|"
    r"entrance|outside|outdoor area|front (?:door|of)|another (?:spot|location|area)|"
    r"second (?:spot|location|area)",
    re.IGNORECASE,
)

_MULTI_ITEM_HEADER = {
    "en": "Summary for your project — {count} screens:",
    "zh": "这个项目的整体方案汇总 —— 共 {count} 块屏：",
}

_ITEM_LABEL = {
    "en": "Screen {index}",
    "zh": "第 {index} 块屏",
}


def _norm_environment(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("indoor", "室内", "户内"):
        return "indoor"
    if text in ("outdoor", "semi_outdoor", "室外", "户外", "半户外", "露天"):
        return "outdoor"
    return text


def _norm_display_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text in ("LED", "LCD", "IFP") else ""


def _screen_where(profile: Any, lang: str) -> str:
    """这一块屏的"环境 / 场景"短语（用于标签与汇总）。"""
    data = profile if isinstance(profile, dict) else getattr(profile, "model_dump", lambda: {})()
    data = data or {}
    scene = str(data.get("purpose") or data.get("usage") or "").strip()
    env = _norm_environment(data.get("environment"))
    env_text = {
        "indoor": "室内" if lang == "zh" else "indoor",
        "outdoor": "室外" if lang == "zh" else "outdoor",
    }.get(env, env)
    return " / ".join([part for part in (env_text, scene) if part])


def detect_new_item(
    message: str,
    profile: Any = None,
    *,
    already_recommended: bool = False,
) -> Tuple[bool, str]:
    """客户这句话是不是在说"另一块屏"。

    只在**信号明确**时才开新条目，避免把"客户改需求"误判成"第二块屏"：
      1. 显式说法（"另外一块 / 第二块 / 门口 / another screen / also need …"）；
      2. 或者：第一块屏已经推荐完了，而这句同时给出**方位词**和冲突的
         环境 / 屏类型（例如当前是室内、客户说"门口那块是室外的"）。

    Returns:
        (是否开新条目, 原因)
    """
    text = str(message or "")
    if not text.strip():
        return False, ""

    if _NEW_ITEM_RE.search(text):
        return True, "explicit_multi_item"

    if not already_recommended or profile is None:
        return False, ""

    try:
        from src.rag.query_understanding import extract_slots

        slots = extract_slots(text)
    except Exception:  # pragma: no cover - 防御式
        return False, ""

    has_location_hint = bool(_LOCATION_HINT_RE.search(text))
    new_env = _norm_environment(slots.get("environment"))
    current_env = _norm_environment(getattr(profile, "environment", None))
    if has_location_hint and new_env and current_env and new_env != current_env:
        return True, f"environment_switch:{current_env}->{new_env}"

    new_type = _norm_display_type(slots.get("display_type"))
    current_type = _norm_display_type(getattr(profile, "display_type", None))
    if new_type and current_type and new_type != current_type:
        return True, f"display_type_switch:{current_type}->{new_type}"

    return False, ""


def screen_label(index: int, profile: Any, language: str = "en") -> str:
    """多屏会话里给每块屏加个前缀，避免"把第二块说成第一块"。

    实测：客户开了第二块屏（门口室外广告屏）之后，推荐话术里仍会出现
    "For your church indoor screen …"，客户分不清在说哪一块。
    """
    lang = "zh" if str(language or "").lower().startswith("zh") else "en"
    where = _screen_where(profile, lang)
    number = index + 1
    if lang == "zh":
        return f"第 {number} 块屏（{where}）：" if where else f"第 {number} 块屏："
    return f"Screen {number} ({where}): " if where else f"Screen {number}: "


def item_summary_line(item: Dict[str, Any], index: int, language: str = "en") -> str:
    """汇总里的一行：第 N 块屏 + 场景/环境 + 型号（含箱体计算摘要）。"""
    lang = "zh" if str(language or "").lower().startswith("zh") else "en"
    label = _ITEM_LABEL[lang].format(index=index)
    where = _screen_where(item.get("profile") or {}, lang)
    model = str(item.get("model") or "").strip()
    calc = str(item.get("calculation") or "").strip()

    bits: List[str] = [label]
    if where:
        bits.append(f"（{where}）" if lang == "zh" else f"({where})")
    if model:
        bits.append(f"→ {model}")
    line = " ".join(bits)
    if calc:
        line = f"{line} —— {calc}" if lang == "zh" else f"{line} — {calc}"
    return line


def combined_summary(items: Sequence[Dict[str, Any]], language: str = "en") -> Optional[str]:
    """多块屏都推荐完之后，给一份"两份推荐 + 两份计算"的汇总。"""
    usable = [item for item in items or [] if str(item.get("model") or "").strip()]
    if len(usable) < 2:
        return None
    lang = "zh" if str(language or "").lower().startswith("zh") else "en"
    header = _MULTI_ITEM_HEADER[lang].format(count=len(usable))
    lines = [item_summary_line(item, index, lang) for index, item in enumerate(usable, start=1)]
    return header + "\n" + "\n".join(lines)


def _clauses(text: str) -> List[str]:
    """把一句话按逗号 / and / 顿号 等切成小分句（用于拆分多块屏的规格）。"""
    parts = re.split(
        r"[,，;；]|\band\b|\bplus\b|\balso\b|、|另外|还有|\+|以及", str(text or "")
    )
    return [part.strip() for part in parts if part.strip()]


_SIZE_PAIR_RE = re.compile(
    r"(?P<w>\d+(?:[.,]\d+)?)\s*(?P<wu>mm|cm|m|meters?|metres?|米|厘米|毫米)?\s*"
    r"(?:wide|width|宽|长度)?\s*"
    r"(?:x|×|\*|by)\s*"
    r"(?P<h>\d+(?:[.,]\d+)?)\s*(?P<hu>mm|cm|m|meters?|metres?|米|厘米|毫米|high|高|高度)?",
    re.IGNORECASE,
)

_ENV_TOKENS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"indoor|室内|户内", re.IGNORECASE), "indoor"),
    (re.compile(r"outdoor|室外|户外|半户外|露天", re.IGNORECASE), "outdoor"),
)

# "……for indoor" / "……室外" 这样的分句：每块屏的说法通常落在自己那句里
_ENV_CHUNK_RE = re.compile(
    r"([^,，;；]*?)((?:indoor|室内|户内|outdoor|室外|户外|半户外|露天))", re.IGNORECASE
)

_INSTALL_TOKENS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"permanent|fixed|固定|固装|长期|墙上", re.IGNORECASE), "fixed"),
    (re.compile(r"rental|rent|租赁|快装|快拆|便携|带走|可搬|临时|活动", re.IGNORECASE), "rental"),
)

# 点间距：P3 / 3mm / 点间距3 / 3 毫米
_PITCH_RE = re.compile(
    r"(?<![A-Za-z0-9])[Pp](\d+(?:\.\d+)?)(?![0-9])"
    r"|(\d+(?:\.\d+)?)\s*(?:mm|毫米)\s*(?:pitch|点间距)?"
    r"|点间距\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _attrs_from_chunk(chunk: str) -> Dict[str, Any]:
    """从一个分句里取出"这块屏"的属性（环境 / 尺寸 / 安装方式 / 点间距）。"""
    env = next((name for pattern, name in _ENV_TOKENS if pattern.search(chunk)), None)
    installation = next(
        (name for pattern, name in _INSTALL_TOKENS if pattern.search(chunk)), None
    )
    pitch = None
    match = _PITCH_RE.search(chunk)
    if match:
        raw = next((group for group in match.groups() if group), "")
        try:
            pitch = float(raw)
        except (TypeError, ValueError):
            pitch = None
    width = height = None
    size_match = _SIZE_PAIR_RE.search(chunk)
    if size_match:
        width = _to_meters(size_match.group("w"), size_match.group("wu"))
        height = _to_meters(size_match.group("h"), size_match.group("hu") or size_match.group("wu"))
    return {
        "environment": env,
        "installation": installation,
        "pixel_pitch_mm": pitch,
        "width_m": width,
        "height_m": height,
        "clause": chunk,
    }


def _to_meters(value: str, unit: str) -> Optional[float]:
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    unit = str(unit or "").lower()
    if unit in ("mm", "毫米"):
        return number / 1000.0
    if unit in ("cm", "厘米"):
        return number / 100.0
    return number


def split_multi_screen_specs(message: str) -> List[Dict[str, Any]]:
    """一句话里出现**多块屏**的规格 → 拆成每块屏一份。

    例："4m wide x 2.5 high for indoor and 3m x 2m for the outdoor"
        → [{'environment': 'indoor',  'width_m': 4.0, 'height_m': 2.5},
           {'environment': 'outdoor', 'width_m': 3.0, 'height_m': 2.0}]

    客户口径：分不清哪组参数属于哪块屏时，两块**都记成一样的**。
    """
    text = str(message or "")

    by_refs = _split_by_references(text)
    if by_refs:
        return by_refs

    # ① 句子里出现了两个及以上"室内/室外" → 按"**离哪个环境词最近就归哪块屏**"
    #    来分配每项参数。这样以下说法都能拆对：
    #      "4m wide x2.5 high for indoor and 3m x2m for the outdoor"
    #      "permanent for indoor , rental for outdoor"
    #      "p3 for indoor p5 for outdoor"
    env_marks: List[Tuple[int, int, str]] = []
    for match in _ENV_CHUNK_RE.finditer(text):
        name = next((n for pattern, n in _ENV_TOKENS if pattern.search(match.group(2))), None)
        if name:
            env_marks.append((match.start(2), match.end(2), name))

    if len({mark[2] for mark in env_marks}) >= 2:
        ordered = list(dict.fromkeys(mark[2] for mark in env_marks))
        screens = {name: {"environment": name} for name in ordered}

        def _owner(start: int, end: int) -> str:
            """这个参数属于哪块屏。

            规则：先看它**后面**紧跟的是不是 "for (the) indoor/outdoor"
            （"p3 for indoor" / "4m x 2.5m for the outdoor"）；
            不是的话，就归给它**前面**最近的那个环境词
            （"indoor 4m x 2.5m … outdoor 3m x 2m"）。
            """
            suffix = text[end:end + 24]
            follow = re.match(
                r"\s*(?:(?:for|used\s+for|是|用于|用来|使用)\s*)?(?:the|a|an)?\s*"
                r"(indoor|室内|户内|outdoor|室外|户外|半户外|露天)",
                suffix,
                re.IGNORECASE,
            )
            if follow:
                if re.match(r"\s*(?:for|used\s+for|是|用于|用来|使用)\b", suffix, re.IGNORECASE):
                    return next(n for p, n in _ENV_TOKENS if p.search(follow.group(1)))
            before = [mark for mark in env_marks if mark[1] <= start]
            if before:
                return before[-1][2]
            centre = (start + end) / 2
            return min(env_marks, key=lambda m: abs((m[0] + m[1]) / 2 - centre))[2]

        for match in _SIZE_PAIR_RE.finditer(text):
            width = _to_meters(match.group("w"), match.group("wu"))
            height = _to_meters(match.group("h"), match.group("hu") or match.group("wu"))
            if width and height:
                screen = screens[_owner(match.start(), match.end())]
                screen.setdefault("width_m", width)
                screen.setdefault("height_m", height)
        for pattern, name in _INSTALL_TOKENS:
            for match in pattern.finditer(text):
                screens[_owner(match.start(), match.end())].setdefault("installation", name)
        for match in _PITCH_RE.finditer(text):
            raw = next((group for group in match.groups() if group), "")
            try:
                pitch = float(raw)
            except (TypeError, ValueError):
                continue
            screens[_owner(match.start(), match.end())].setdefault("pixel_pitch_mm", pitch)

        result = [screens[name] for name in ordered]
        return _share_single_values(result)

    chunks = _clauses(text)

    entries: List[Dict[str, Any]] = []
    for clause in chunks:
        entry = _attrs_from_chunk(clause)
        if any(
            entry.get(key) not in (None, "", [], {})
            for key in ("environment", "installation", "pixel_pitch_mm", "width_m", "height_m")
        ):
            entries.append(entry)
    return _merge_screen_entries(entries)


# ── 泛化的"分屏指代"：不止室内/室外 ─────────────────────────────────────
# (种类, 正则, 规范值)；序数只用来定顺序，不写进需求字段
_REFERENCE_TOKENS: Tuple[Tuple[str, str, str], ...] = (
    ("environment", r"indoor|室内|户内", "indoor"),
    ("environment", r"outdoor|室外|户外|半户外|露天", "outdoor"),
    ("display_type", r"\bLED\b", "LED"),
    ("display_type", r"\bLCD\b", "LCD"),
    ("display_type", r"\bIFP\b", "IFP"),
    ("purpose", r"church|教堂", "church"),
    ("purpose", r"advertis\w*|广告", "advertising"),
    ("purpose", r"conference|meeting room|会议室", "conference"),
    ("purpose", r"classroom|教室", "classroom"),
    ("purpose", r"stage|舞台|演出", "stage"),
    ("position", r"entrance|门口|入口|大门口|正门|门头", "entrance"),
    ("position", r"lobby|大堂|大厅", "lobby"),
    ("position", r"reception|前台", "reception"),
)
_ORDINAL_TOKENS: Tuple[Tuple[str, int], ...] = (
    (r"第一(?:块|个|台)|first (?:one|screen)|the first", 0),
    (r"第二(?:块|个|台)|second (?:one|screen)|the second", 1),
    (r"第三(?:块|个|台)|third (?:one|screen)", 2),
)


def _split_by_references(text: str) -> List[Dict[str, Any]]:
    """按句中出现的**任意指代**分屏："室内/室外"、"LED/LCD"、"教堂/广告"、
    "门口/大堂"、"第一块/第二块" 都算。参数按"就近归属"落到对应那块屏。"""
    refs: List[Tuple[int, int, str, Any]] = []
    for kind, pattern, value in _REFERENCE_TOKENS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            refs.append((match.start(), match.end(), kind, value))
    ordinals: List[Tuple[int, int, int]] = []
    for pattern, index in _ORDINAL_TOKENS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            refs.append((match.start(), match.end(), "ordinal", index))
            ordinals.append((match.start(), match.end(), index))
    if not refs:
        return []

    # 去重：同一种类的同一个值只留一次
    seen = set()
    unique = []
    for ref in sorted(refs):
        key = (ref[2], ref[3])
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)

    # 必须是**同一类指代出现两次**才算多屏（"LED 教堂屏"只是"一块屏的两个属性"）
    per_kind: Dict[str, set] = {}
    for _, _, kind, value in unique:
        if kind == "ordinal":
            continue
        per_kind.setdefault(kind, set()).add(value)
    if not ordinals and not any(len(values) >= 2 for values in per_kind.values()):
        return []
    # 只有"出现两次的那种指代"才是分屏依据；只出现一次的（例如句首的 "LED"）
    # 只是这块/那些屏的共同属性，不能自己算一块屏。
    driver_kinds = {kind for kind, values in per_kind.items() if len(values) >= 2}
    if driver_kinds and not ordinals:
        unique = [ref for ref in unique if ref[2] in driver_kinds or ref[2] == "ordinal"]
        refs_sorted = sorted(unique, key=lambda item: item[0])

    refs_sorted = sorted(unique, key=lambda item: item[0])

    def owner(start: int, end: int) -> Tuple[str, Any]:
        """参数归属：先看后面是不是 'for (the) <指代>'，否则归前面最近的那个。"""
        suffix = text[end:end + 24]
        for kind, pattern, value in _REFERENCE_TOKENS:
            if re.match(
                rf"\s*(?:(?:for|used for|是|用于|用来|使用)\s*)?(?:the|a|an)?\s*(?:{pattern})",
                suffix,
                re.IGNORECASE,
            ):
                return kind, value
        before = [ref for ref in refs_sorted if ref[1] <= start]
        if before:
            return before[-1][2], before[-1][3]
        centre = (start + end) / 2
        best = min(refs_sorted, key=lambda ref: abs((ref[0] + ref[1]) / 2 - centre))
        return best[2], best[3]

    # 每块屏：先按序数定序号，否则按出现顺序
    screens: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for start, end, kind, value in refs_sorted:
        if kind == "ordinal":
            key = f"#{value}"
        else:
            key = (kind, value)
        if key not in screens:
            screens[key] = {}
            order.append(key)
        if kind != "ordinal":
            screens[key].setdefault(kind, value)
    if ordinals:
        order = sorted(order, key=lambda key: int(str(key)[1:]) if str(key).startswith("#") else 99)

    def _screen_of(kind: str, value: Any) -> Optional[Dict[str, Any]]:
        key = (kind, value)
        if key in screens:
            return screens[key]
        return screens.get(next(iter(screens))) if screens else None

    for match in _SIZE_PAIR_RE.finditer(text):
        width = _to_meters(match.group("w"), match.group("wu"))
        height = _to_meters(match.group("h"), match.group("hu") or match.group("wu"))
        if width and height:
            screen = screens.get(owner(match.start(), match.end()))
            if screen is None:
                continue
            screen.setdefault("width_m", width)
            screen.setdefault("height_m", height)
    for pattern, name in _INSTALL_TOKENS:
        for match in pattern.finditer(text):
            screen = screens.get(owner(match.start(), match.end()))
            if screen is not None:
                screen.setdefault("installation", name)
    for match in _PITCH_RE.finditer(text):
        raw = next((group for group in match.groups() if group), "")
        try:
            pitch = float(raw)
        except (TypeError, ValueError):
            continue
        screen = screens.get(owner(match.start(), match.end()))
        if screen is not None:
            screen.setdefault("pixel_pitch_mm", pitch)

    result = [screens[key] for key in order if key in screens]
    result = [screen for screen in result if screen]
    if len(result) < 2:
        return []
    return _share_single_values(result)


def _share_single_values(screens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """客户口径：没说清"哪块要什么"时两块记成一样的 ——

    某个属性只有一块写了（另一块没提），就补到另一块上。
    """
    for key in ("installation", "pixel_pitch_mm", "width_m", "height_m"):
        values = [
            screen.get(key)
            for screen in screens
            if screen.get(key) not in (None, "", [], {})
        ]
        if len(values) == 1:
            for screen in screens:
                if screen.get(key) in (None, "", [], {}):
                    screen[key] = values[0]
    return screens


def _merge_screen_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把分句属性合并成"每块屏一份"（分不清归属时两块记成一样的）。"""
    if len(entries) < 2:
        return []

    envs = [entry["environment"] for entry in entries if entry["environment"]]
    if len(envs) >= 2:
        # 每块屏一份：按环境归位，同环境的属性合并
        screens: List[Dict[str, Any]] = []
        for env in dict.fromkeys(envs):
            merged: Dict[str, Any] = {"environment": env}
            for entry in entries:
                if entry["environment"] not in (None, env):
                    continue
                for key in ("installation", "pixel_pitch_mm", "width_m", "height_m"):
                    if merged.get(key) in (None, "", [], {}) and entry.get(key) not in (None, "", [], {}):
                        merged[key] = entry[key]
            screens.append(merged)
        return _share_single_values(screens)

    sized = [entry for entry in entries if entry["width_m"] and entry["height_m"]]
    if len(sized) >= 2:
        return sized
    return []


def _legacy_split(message: str) -> List[Dict[str, Any]]:
    """旧实现保留（供对照/回退）；当前入口是上面的 split_multi_screen_specs。"""
    entries: List[Dict[str, Any]] = []
    for clause in _clauses(message):
        env = None
        for pattern, name in _ENV_TOKENS:
            if pattern.search(clause):
                env = name
                break
        width = height = None
        match = _SIZE_PAIR_RE.search(clause)
        if match:
            width = _to_meters(match.group("w"), match.group("wu"))
            # 高度没写单位时，沿用宽度的单位（"4m wide x 2.5 high"）
            height = _to_meters(match.group("h"), match.group("hu") or match.group("wu"))
        if env or (width and height):
            entries.append(
                {
                    "environment": env,
                    "width_m": width,
                    "height_m": height,
                    "clause": clause,
                }
            )

    sized = [entry for entry in entries if entry["width_m"] and entry["height_m"]]
    envs = [entry["environment"] for entry in entries if entry["environment"]]

    if len(sized) >= 2:
        screens: List[Dict[str, Any]] = []
        for entry in sized:
            env = entry["environment"]
            if env is None and len(envs) >= len(sized):
                # 环境和尺寸不在同一个分句里 → 按出现顺序一一对应
                env = envs[len(screens)]
            screens.append({**entry, "environment": env})
        return screens

    if len(sized) == 1 and len(envs) >= 2:
        # 两块屏、只有一组尺寸 → 两块用同一组尺寸（没说清就都记一样的）
        same = sized[0]
        return [
            {
                "environment": env,
                "width_m": same["width_m"],
                "height_m": same["height_m"],
                "clause": same["clause"],
                "shared_spec": True,
            }
            for env in envs
        ]

    return []


_ORDINAL_TARGETS: Tuple[Tuple["re.Pattern[str]", int], ...] = (
    (re.compile(r"第一块|第一个|第一台|the first", re.IGNORECASE), 0),
    (re.compile(r"第二块|第二个|第二台|the second", re.IGNORECASE), 1),
    (re.compile(r"第三块|第三个|第三台|the third", re.IGNORECASE), 2),
    (re.compile(r"第四块|第四个|the fourth", re.IGNORECASE), 3),
)

_CHANGE_VERB_RE = re.compile(
    r"改成|改为|换成|变成|调整|修改|change|update|make it|set it to|turn it into",
    re.IGNORECASE,
)


def detect_screen_target(
    message: str,
    items: Sequence[Dict[str, Any]],
    *,
    require_change: bool = True,
) -> Optional[int]:
    """客户这句话在说**哪一块屏**（多屏项目里）。

    支持两种指法：
      - 序数："第一块 / 第二块 / the second one"
      - 环境："室内那块 / 室外那块 / the outdoor one"（按已记录的环境匹配）
    只有同时出现"改动意图或新数值"时才返回下标，避免客户只是提到某一块就被切走。
    """
    if len(items or []) < 2:
        return None

    text = str(message or "")
    if not text:
        return None

    if require_change:
        from src.rag.query_understanding import extract_slots

        try:
            slots = extract_slots(text) or {}
        except Exception:  # pragma: no cover - 防御式
            slots = {}
        has_value = any(
            slots.get(key) not in (None, "", [], {})
            for key in (
                "target_width_m", "target_height_m", "size", "pixel_pitch",
                "viewing_distance_m", "brightness_min", "content_type", "installation",
            )
        )
        if not (_CHANGE_VERB_RE.search(text) or has_value):
            return None

    for pattern, index in _ORDINAL_TARGETS:
        if pattern.search(text) and index < len(items):
            return index

    for pattern, name in _ENV_TOKENS:
        if not pattern.search(text):
            continue
        for index, item in enumerate(items):
            profile = item.get("profile") or {}
            if _norm_environment(profile.get("environment")) == name:
                return index
    return None


def product_model(product: Any) -> str:
    """从推荐结果里取型号名。

    实际返回的产品既可能是 ``{"model": ...}``，也可能是 RAG 的 Document 结构
    （``{"metadata": {"model": ...}}`` 或带 ``.metadata`` 的对象）—— 实测踩过：
    只认 ``["model"]`` 时多屏流程会静默跳过。
    """
    if product is None:
        return ""

    def _from_metadata(meta: Any) -> str:
        if isinstance(meta, dict):
            return str(meta.get("model") or meta.get("product_id") or "").strip()
        return ""

    if isinstance(product, dict):
        model = str(product.get("model") or product.get("product_id") or "").strip()
        model = model or _from_metadata(product.get("metadata"))
        if model:
            return model
        raw = str(product.get("id") or "")
    else:
        model = _from_metadata(getattr(product, "metadata", None))
        if model:
            return model
        raw = str(getattr(product, "id", "") or "")

    # 兜底：Document 的 id 形如 "model-TW11-3216-P3.0"
    if raw.startswith("model-"):
        return raw[len("model-"):].strip()
    return ""


__all__ = [
    "combined_summary",
    "detect_new_item",
    "detect_screen_target",
    "item_summary_line",
    "product_model",
    "screen_label",
    "split_multi_screen_specs",
]
