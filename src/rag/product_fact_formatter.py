"""Format customer-provided product facts and vision confirmations."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .language_utils import _lang, reply_language



# ── 复述客户刚给出的需求信息 ────────────────────────────────────────────────
_PURPOSE_LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "conference": "a conference room",
        "classroom": "a classroom",
        "retail": "a retail store",
        "showroom": "a showroom",
        "museum": "a museum",
        "hall": "a large hall",
        "control_room": "a control room",
        "office": "an office",
        "hospital": "a hospital",
        "bank": "a bank",
        "hotel": "a hotel",
        "restaurant": "a restaurant",
        "airport": "a transport hub",
        "advertising": "outdoor advertising",
        "stadium": "a stadium",
        "concert": "a concert",
        "stage": "a stage performance",
        "church": "a church",
        "exhibition": "an exhibition",
        "rental": "a rental event",
    },
    "zh": {
        "conference": "会议室",
        "classroom": "教室",
        "retail": "门店零售",
        "showroom": "展厅",
        "museum": "博物馆",
        "hall": "大厅",
        "control_room": "监控指挥中心",
        "office": "办公室",
        "hospital": "医院",
        "bank": "银行",
        "hotel": "酒店",
        "restaurant": "餐厅",
        "airport": "交通枢纽",
        "advertising": "户外广告",
        "stadium": "体育场馆",
        "concert": "演唱会",
        "stage": "舞台演出",
        "church": "教堂",
        "exhibition": "展会",
        "rental": "租赁活动",
    },
}



def purpose_phrase(value: Any, language: Optional[str] = None) -> str:
    """客户说的使用场景 → **客户能看到的说法**（唯一词表：``_PURPOSE_LABELS``）。

    用途：给"面向客户的提示词"提供说人话的场景。提示词里**不能**出现内部
    品类 token（``advertising`` / ``monitoring`` / ``normal`` /
    ``conference_education``）—— 模型会把它们当成客户的说法照抄进文案。

    实测（2026-10）：客户全程说 "exhibition"（展会），内部品类判成
    ``advertising``，推荐提示词里写了 ``Customer's category: advertising``，
    模型就写成 "For your **advertising** video wall…" —— 客户从没说过广告。
    改成传 ``purpose_phrase("exhibition")`` = "an exhibition" 之后，模型才有
    正确的"客户要拿它干什么"可依据。

    认不出 token 时原样返回，不编造场景。
    """
    if value in (None, "", [], {}):
        return ""
    lang = _lang(language)
    labels = _PURPOSE_LABELS.get(lang) or {}
    token = str(value).strip()
    return labels.get(token) or labels.get(token.lower()) or token


_ENVIRONMENT_LABELS = {
    "en": {"indoor": "indoor", "outdoor": "outdoor", "semi_outdoor": "semi-outdoor"},
    "zh": {"indoor": "室内", "outdoor": "室外", "semi_outdoor": "半户外"},
}


# 复述用的开场（同一件事多种说法，按轮次轮换）
_ACK_ECHO_LEADS = {
    "en": (
        "Got it — {echo}.",
        "Thanks, {echo} — noted.",
        "Understood, {echo}.",
        "Noted: {echo}.",
        "Alright, {echo}.",
        "Thanks for that — {echo}.",
        # 客户口径：不要每轮都是 "Got it / Understood"，下面这些换着用
        "That makes sense — {echo}.",
        "Good to know — {echo}.",
        "Right, {echo} — that helps.",
        "Appreciate you sharing that — {echo}.",
        "Okay, {echo}, got it noted.",
    ),
    "zh": (
        "好的，{echo}。",
        "明白，{echo}。",
        "收到，{echo}。",
        "了解，{echo}。",
        "记下了，{echo}。",
        "这个信息有用，{echo}。",
        "好，{echo}，我记一下。",
        "清楚，{echo}。",
    ),
}



# 各槽位的"事实特征"：回应里出现这些，就说明它已经确认了这一项
_SLOT_FACT_PATTERNS: Dict[str, str] = {
    "environment": r"(?<![a-z])(?:indoor|outdoor|indoors|outdoors)(?![a-z])|室内|室外|半户外",
    "installation": r"(?<![a-z])(?:fixed|rental|permanent)(?![a-z])|固装|固定安装|租赁",
    "viewing_distance": (
        r"\d[\d.,]*\s*(?<![a-z])(?:m|metres?|meters?|ft|feet|foot)(?![a-z])"
        r"|\d[\d.,]*\s*(?:米|英尺|英寸|寸)"
    ),
    "width": r"\d[\d.,]*\s*(?<![a-z])(?:m|mm|cm|ft|feet|inch|inches)(?![a-z])|\d[\d.,]*\s*(?:米|毫米|厘米|英尺|英寸)",
    "height": r"\d[\d.,]*\s*(?<![a-z])(?:m|mm|cm|ft|feet|inch|inches)(?![a-z])|\d[\d.,]*\s*(?:米|毫米|厘米|英尺|英寸)",
    "size": (
        r"\d[\d.,]*\s*(?:x|×|by)\s*\d"
        r"|\d[\d.,]*\s*(?<![a-z])(?:m|mm|cm|ft|feet|inch|inches)(?![a-z])"
        r"|\d[\d.,]*\s*(?:米|毫米|厘米|英尺|英寸)"
        r"|(?<![a-z])(?:wide|high|tall)(?![a-z])|米宽|米高"
    ),
    "display_type": r"(?<![A-Za-z])(?:LED|LCD|IFP)(?![A-Za-z])",
    # 追问"这个数字是宽/高/对角线吗"时，回应里若已经替客户定性了方向，就是矛盾
    "size_axis": (
        r"(?<![a-z])(?:width|height|length|wide|tall|diagonal)(?![a-z])"
        r"|宽度|高度|长度|长边|对角线"
    ),
}



def _purpose_tokens(lang: str) -> Sequence[str]:
    """场景标签的可识别片段（"a conference room" → "conference room"）。"""
    tokens = []
    for label in _PURPOSE_LABELS.get(lang, {}).values():
        cleaned = re.sub(r"^(?:a|an)\s+", "", str(label), flags=re.IGNORECASE).strip()
        if cleaned:
            tokens.append(cleaned)
    return tuple(tokens)



def ack_conflicts_with_slot(ack: str, slot: str, language: str = "en") -> bool:
    """回应里是否已经确认了"系统接下来还要追问"的那一项。

    实测 bug：先确认"about 100 feet viewing distance"，紧接着又追问观看距离，
    客户看到的就是自相矛盾。（根因是英制单位没被解析，已修；
    这里作为兜底护栏，保证"确认过的不再追问"。）
    """
    ack = str(ack or "")
    slot = str(slot or "")
    if not ack or not slot:
        return False
    lang = _lang(language)
    if slot == "purpose":
        return any(token.lower() in ack.lower() for token in _purpose_tokens(lang))
    pattern = _SLOT_FACT_PATTERNS.get(slot)
    if not pattern:
        return False
    return bool(re.search(pattern, ack, re.IGNORECASE))



def _slot_tokens(message: str) -> Dict[str, Any]:
    """本轮消息里客户**明确说出**的槽位（排除系统推断值）。"""
    try:
        from src.rag.query_understanding import extract_slots

        slots = dict(extract_slots(message) or {})
    except Exception:  # pragma: no cover - 防御式
        return {}
    inferred = {str(item) for item in (slots.pop("_inferred_slots", None) or [])}
    return {k: v for k, v in slots.items() if k not in inferred}



def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"



def requirement_echo(
    message: str,
    requirement: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    exclude_slot: str = "",
) -> Optional[str]:
    """把客户刚说的需求复述成一小段话（没有可复述的信息则返回 None）。

    ``exclude_slot``：接下来还要追问的那一项 —— 复述时跳过它，
    否则会出现"先确认、再追问同一件事"的自相矛盾。
    """
    slots = _slot_tokens(message)
    # 兜底：解析器没给出场景时，用累计需求里的 usage 也不合适（那不是本轮说的），
    # 因此这里只看本轮消息。
    purpose = slots.get("purpose")
    environment = None if purpose else slots.get("environment")
    pitch = slots.get("pixel_pitch_mm") or slots.get("pixel_pitch")
    distance = slots.get("viewing_distance_m") or slots.get("distance")
    width = slots.get("target_width_mm") or slots.get("target_width")
    height = slots.get("target_height_mm") or slots.get("target_height")

    has_facts = any([purpose, environment, pitch, distance, width, height])
    if not has_facts:
        return None

    # 用"回复语言"而不是"客户语言"：策略为 en 时，即使用户写中文也回英文
    lang = _lang(language or reply_language(message))
    parts: List[str] = []

    if purpose and exclude_slot != "purpose":
        parts.append(_PURPOSE_LABELS[lang].get(str(purpose), str(purpose)))
    if environment and exclude_slot != "environment":
        parts.append(_ENVIRONMENT_LABELS[lang].get(str(environment), str(environment)))
    if distance and exclude_slot != "viewing_distance":
        # 客户用英制说，就用英制复述回去（100 feet 不要换算成 30.48 m 再念一遍）
        imperial = re.search(r"(?<![a-z])(?:ft|feet|foot)(?![a-z])", str(message), re.IGNORECASE)
        if lang == "en":
            if imperial:
                shown = _format_number(round(float(distance) / 0.3048, 1))
                parts.append(f"about {shown} ft viewing distance")
            else:
                parts.append(f"about {_format_number(distance)} m viewing distance")
        else:
            parts.append(f"观看距离约 {_format_number(distance)} 米")
    if width and height and exclude_slot not in ("size", "width", "height"):
        # 小尺寸（如 129.2 x 45 cm）用厘米复述更贴近客户的说法
        if min(float(width), float(height)) < 1000:
            shown = (
                f"{_format_number(width / 10)} x {_format_number(height / 10)} cm"
                if lang == "en"
                else f"{_format_number(width / 10)} x {_format_number(height / 10)} 厘米"
            )
        else:
            shown = (
                f"{_format_number(width / 1000)} x {_format_number(height / 1000)} m"
                if lang == "en"
                else f"{_format_number(width / 1000)} x {_format_number(height / 1000)} 米"
            )
        parts.append(f"a {shown} screen" if lang == "en" else f"{shown}的屏")
    if pitch:
        parts.append(f"P{_format_number(pitch)}")
    for slot_name, label_en, label_zh in (
        ("cob", "COB", "COB"),
        ("hdr", "HDR", "HDR"),
        ("waterproof", "waterproof", "防水"),
        ("flexible", "flexible", "柔性"),
    ):
        if slots.get(slot_name):
            parts.append(label_en if lang == "en" else label_zh)

    separator = ", " if lang == "en" else "、"
    return separator.join(parts)



# ── 图片识别结果 → 跟客户确认（客户口径：识别完先核对，不对就按客户说的记）──
_VISION_FIELD_PHRASES: Dict[str, Dict[str, Dict[str, str]]] = {
    "display_type": {
        "en": {"LED": "an LED screen", "LCD": "an LCD display", "IFP": "an interactive flat panel"},
        "zh": {"LED": "LED 屏", "LCD": "LCD 屏", "IFP": "交互平板"},
    },
    "environment": {
        "en": {
            "indoor": "indoor use",
            "outdoor": "outdoor use",
            "semi_outdoor": "semi-outdoor use",
        },
        "zh": {"indoor": "室内使用", "outdoor": "室外使用", "semi_outdoor": "半户外使用"},
    },
    "installation": {
        "en": {"fixed": "a fixed installation", "rental": "a rental setup"},
        "zh": {"fixed": "固定安装", "rental": "租赁使用"},
    },
}


_VISION_PURPOSE_PHRASES: Dict[str, Dict[str, str]] = {
    "en": {
        "conference": "a conference room", "classroom": "a classroom", "retail": "a retail space",
        "advertising": "advertising", "stadium": "a stadium", "church": "a church",
        "stage": "a stage", "concert": "a concert venue", "museum": "a museum",
        "showroom": "a showroom", "airport": "an airport", "bank": "a bank",
        "hotel": "a hotel", "restaurant": "a restaurant", "office": "an office",
        "hospital": "a hospital", "exhibition": "an exhibition hall", "hall": "a hall",
        "control_room": "a control room", "rental": "rental and events",
    },
    "zh": {
        "conference": "会议室", "classroom": "教室", "retail": "零售门店",
        "advertising": "广告传媒", "stadium": "体育场馆", "church": "教堂",
        "stage": "舞台", "concert": "演唱会", "museum": "博物馆",
        "showroom": "展厅", "airport": "机场", "bank": "银行",
        "hotel": "酒店", "restaurant": "餐厅", "office": "办公室",
        "hospital": "医院", "exhibition": "展馆", "hall": "大厅",
        "control_room": "监控中心", "rental": "租赁活动",
    },
}


_VISION_CONFIRM_TEMPLATES = {
    "en": (
        # 一句话说完：看到什么 + 请客户纠正（避免"三句话各说各的"）
        "Thanks for the photo — it looks like {items}, so correct me if I've misread it.",
        "From your picture I'd say {items}, and let me know if that's not right.",
        "The photo reads to me as {items} (tell me if I'm off).",
        "Looking at your photo, I'd take it as {items} — feel free to correct me.",
    ),
    "zh": (
        "照片收到了，看着像是{items}，我理解不对的话你纠正我。",
        "从照片看应该是{items}，跟实际不一样的话跟我说一声。",
        "我看照片判断是{items}（说得不对你直接纠正我）。",
        "按照片来看是{items}，如果不对提醒我一下。",
    ),
}



def vision_confirmation_items(profile, language: Optional[str] = None) -> List[str]:
    """把"图片识别出、还没确认"的字段转成人话（用于跟客户核对）。"""
    lang = _lang(language)
    fields = list(getattr(profile, "vision_confirmation_pending", None) or [])
    items: List[str] = []
    for field in fields:
        value = getattr(profile, field, None)
        if value in (None, "", [], {}):
            continue
        if field == "purpose":
            phrase = (_VISION_PURPOSE_PHRASES.get(lang) or {}).get(str(value))
        else:
            phrase = ((_VISION_FIELD_PHRASES.get(field) or {}).get(lang) or {}).get(str(value))
        if phrase and phrase not in items:
            items.append(phrase)
    return items



def vision_confirmation_sentence(profile, language: Optional[str] = None, seed: int = 0) -> str:
    """生成"图片里看到的是 XXX，对吗？"这一句（多种说法轮换，不死板）。

    客户纠正后按客户说的记录（见 RequirementProfile.merge：客户明说 > 图片识别）。
    """
    if profile is None:
        return ""
    items = vision_confirmation_items(profile, language)
    if not items:
        return ""
    lang = _lang(language)
    if lang == "zh":
        item_text = "、".join(items)
    elif len(items) == 1:
        item_text = items[0]
    else:
        item_text = ", ".join(items[:-1]) + " and " + items[-1]
    templates = _VISION_CONFIRM_TEMPLATES.get(lang) or _VISION_CONFIRM_TEMPLATES["en"]
    return templates[seed % len(templates)].format(items=item_text).strip()

__all__ = ['_ACK_ECHO_LEADS', '_ENVIRONMENT_LABELS', '_PURPOSE_LABELS', '_SLOT_FACT_PATTERNS', '_VISION_CONFIRM_TEMPLATES', '_VISION_FIELD_PHRASES', '_VISION_PURPOSE_PHRASES', '_format_number', '_purpose_tokens', '_slot_tokens', 'ack_conflicts_with_slot', 'purpose_phrase', 'requirement_echo', 'vision_confirmation_items', 'vision_confirmation_sentence']
