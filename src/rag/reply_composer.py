"""
"先回应客户这句话 + 再追问一个需求" 的组合回复。

背景（客户实测反馈）：
    客户问 "Do u have P 1.2 COB Led"，系统只回了一句
    "…will it be indoors or outdoors?"；客户问 "can I get ur representative in
    Indonesia"，系统又答了一堆通用话术、完全没提需求。销售不能只会问问题。

设计原则：
  1. **先回应** —— 客户这句话本身要被接住：
       - 问"有没有某规格 / 某型号" → 拿产品数据**核实后**正面回答（不编造）
       - 客户给出需求信息（场景 / 室内外 / 视距 / 尺寸…） → 把它复述一遍
       - 别的问题 → 由 RAG 答案负责回答（这里不再硬加一句空话）
  2. **再追问** —— 追问的问题文本由 Ready Gate 决定，**原样保留**，
     这里只负责给不同轮次换不同的引导语，避免每次都同一套固定话术。
  3. 纯规则 / 复用既有解析器，不额外增加 LLM 调用。
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── 语言策略 ───────────────────────────────────────────────────────────────
def reply_language(message: str = "") -> str:
    """回复语言：`.env` 里是 auto 时跟随客户语言，否则统一英语。"""
    try:
        from src.config import config

        if getattr(config, "RESPONSE_LANGUAGE_POLICY", "en") != "auto":
            return "en"
        from src.rag.query_understanding import detect_language

        return detect_language(message)
    except Exception:  # pragma: no cover - 防御式
        return "en"


def _lang(language: str) -> str:
    return "zh" if str(language or "").lower().startswith("zh") else "en"


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
    ),
    "zh": (
        "好的，{echo}。",
        "明白，{echo}。",
        "收到，{echo}。",
        "了解，{echo}。",
        "记下了，{echo}。",
    ),
}

# 已有答案为前提，往追问过渡的引导语
_CONNECTORS = {
    "en": (
        "So I can match the right model,",
        "In the meantime,",
        "While we're at it,",
        "And so I can point you to the right one,",
        "To narrow it down,",
        "That said,",
    ),
    "zh": (
        "另外，",
        "顺便问一下，",
        "为了给您匹配更合适的型号，",
        "同时，",
        "再确认一下，",
        "这样我好帮您缩小范围，",
    ),
}

# 实在没有可回应内容时的中性兜底（避免出现"只丢一个问题"的回复）
_ACK_GENERIC = {
    "en": (
        "Understood.",
        "Got it.",
        "Noted, thanks.",
        "Alright.",
        "Thanks for that.",
    ),
    "zh": (
        "好的。",
        "明白了。",
        "收到。",
        "了解。",
    ),
}


_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _clean_llm_ack(text: str, language: str = "en") -> str:
    """清洗 LLM 生成的回应：去换行 / 去引号 / 丢弃含问句的内容 / 语言纠偏。"""
    ack = " ".join(str(text or "").split()).strip()
    if not ack:
        return ""
    ack = ack.strip('"\'“”')
    if any(mark in ack for mark in ("?", "？")):
        # 追问由系统另外拼接，避免一句话里出现两个问题
        return ""
    if _lang(language) == "en" and _CJK_RE.search(ack):
        # 策略要求全英文时，模型偶尔仍会用中文回一句
        # （实测 "Sello 你好，很高兴认识你。" + 英文追问），这里直接丢弃
        return ""
    return ack[:240].rstrip()


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


# ── 客户问"有没有某规格 / 某型号" → 用产品数据核实后正面回答 ────────────────
_AVAILABILITY_PATTERNS = (
    r"\bdo (?:you|u|ya) (?:have|carry|stock|sell|make|offer|produce)\b",
    r"\bhave (?:you|u) got\b",
    r"\b(?:can|could) (?:i|we) (?:get|have|order|buy)\b",
    r"\bis there (?:a|an|any)\b",
    r"\bare there any\b",
    r"有没有",
    r"你们有",
    r"有(?:没有)?(?:这种|那种)",
    r"能(?:做|提供|生产)(?:吗|么)",
    r"可以(?:做|提供)",
)

_AVAILABILITY_TEMPLATES = {
    "en": {
        "yes": "Yes — we do carry {spec}.",
        "yes_named": "Yes — we do have {spec}.",
        "unsure": "Let me double-check that spec for you.",
    },
    "zh": {
        "yes": "有的，我们有 {spec} 的 LED 屏。",
        "yes_named": "有的，我们有 {spec}。",
        "unsure": "这个规格我帮您确认一下。",
    },
}

# 客户写 "P 1.2" / "p1.2" 时的兜底点间距识别（只用于"正面回应"，不参与选型 Gate）
_LOOSE_PITCH_RE = re.compile(r"(?<![a-z0-9])p\s*(\d{1,2}(?:\.\d{1,2})?)", re.IGNORECASE)


@lru_cache(maxsize=4)
def _catalog_models(data_dir: str) -> Tuple[Any, ...]:
    """加载（并缓存）Model 级产品数据，用于核实"有没有这个规格"。"""
    try:
        from src.rag.json_loader import load_canonical_models

        return tuple(load_canonical_models(data_dir))
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Reply composer catalog load failed: %s", exc)
        return ()


def _default_data_dir() -> str:
    try:
        from src.config import config

        return str(getattr(config, "DATA_DIR", "data"))
    except Exception:  # pragma: no cover - 防御式
        return "data"


def _pitch_matches(model_pitch: float, asked: float) -> bool:
    """点间距比对：既接受数值接近（1.2 ≈ 1.25），也接受型号命名口径一致。"""
    if model_pitch is None or asked is None:
        return False
    if abs(float(model_pitch) - float(asked)) <= 0.15:
        return True
    return abs(round(float(model_pitch), 1) - round(float(asked), 1)) < 1e-6


def _asked_pitch(slots: Dict[str, Any], message: str = "") -> Optional[float]:
    """客户问的点间距：优先解析器结果，退而求其次用 "P 1.2" 这种宽松写法。"""
    pitch = slots.get("pixel_pitch_mm") or slots.get("pixel_pitch")
    if pitch:
        return float(pitch)
    match = _LOOSE_PITCH_RE.search(str(message or ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:  # pragma: no cover - 防御式
            return None
    return None


def _spec_text(slots: Dict[str, Any], lang: str, message: str = "") -> Optional[str]:
    """把客户问的规格整理成一句话里的名词短语。"""
    pitch = _asked_pitch(slots, message)
    pieces: List[str] = []
    if pitch:
        pieces.append(f"{_format_number(pitch)} mm" if lang == "en" else f"{_format_number(pitch)}mm")
    if slots.get("cob"):
        pieces.append("COB")
    if slots.get("hdr"):
        pieces.append("HDR")
    if slots.get("waterproof"):
        pieces.append("waterproof" if lang == "en" else "防水")
    display_type = str(slots.get("display_type") or "").strip()
    if not pieces and not display_type and not (slots.get("series_id") or slots.get("model")):
        return None
    if lang == "en":
        head = " ".join(pieces)
        # LED / LCD / IFP 走 "an LED"，其余走 "a 1.2 mm COB LED"
        if head and display_type:
            text = f"{head} {display_type}"
        elif head:
            text = head
        else:
            text = display_type or "that"
        article = "an" if text.split()[0][:1].lower() in ("l", "i", "a", "e", "o") else "a"
        return f"{article} {text}"
    head = " ".join(pieces)
    return (head + display_type).strip() or display_type or "该规格"


def availability_answer(
    message: str,
    *,
    language: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[str]:
    """客户问"你们有没有某规格"时，用产品数据核实后正面回答。"""
    text = str(message or "")
    if not text or not any(re.search(p, text, re.IGNORECASE) for p in _AVAILABILITY_PATTERNS):
        return None

    slots = _slot_tokens(text)
    lang = _lang(language or reply_language(text))
    models = _catalog_models(str(data_dir or _default_data_dir()))
    templates = _AVAILABILITY_TEMPLATES[lang]

    # 1) 客户点名型号 / 系列 → 直接查目录
    named = str(slots.get("model") or slots.get("series_id") or "").strip()
    if named:
        for model in models:
            if named.lower() in {str(getattr(model, "model", "")).lower(),
                                 str(getattr(model, "series_id", "")).lower(),
                                 str(getattr(model, "series", "")).lower()}:
                return templates["yes_named"].format(spec=named)
        return templates["unsure"]

    # 2) 客户问的是规格（点间距 / COB / HDR / 防水）
    spec = _spec_text(slots, lang, text)
    if not spec:
        return None
    pitch = _asked_pitch(slots, text)
    matched = False
    for model in models:
        if pitch and not _pitch_matches(getattr(model, "pixel_pitch_mm", None), pitch):
            continue
        if slots.get("cob") and not getattr(model, "cob", False):
            continue
        if slots.get("hdr") and not getattr(model, "hdr", False):
            continue
        if slots.get("waterproof") and not getattr(model, "waterproof", False):
            continue
        matched = True
        break

    if matched:
        return templates["yes"].format(spec=spec)
    # 没查到就不硬说"有"，也不要编造：如实说需要确认
    if pitch or slots.get("cob") or slots.get("hdr") or slots.get("waterproof"):
        return templates["unsure"]
    return None


def acknowledge(
    message: str,
    *,
    requirement: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    seed: int = 0,
    data_dir: Optional[str] = None,
    llm_ack: str = "",
    slot: str = "",
) -> str:
    """生成"接住客户这句话"的一小段回应（没有可接的内容则返回空串）。

    优先级：
      1. 用产品数据**核实过**的可用性回答（"有没有 P1.2 COB" → 不能靠猜）
      2. LLM 依据客户原话生成的口语回应（自我介绍 / 提问 / 要报价等）
      3. 规则复述客户刚说的需求（场景 / 视距 / 尺寸…）
      4. 中性兜底（"Got it."）—— 保证销售永远不会只丢一个问题过去

    ``slot`` 是接下来要继续追问的那一项：如果回应里已经确认了这一项，
    就退回到中性兜底，避免"刚确认完又问同一件事"。
    """
    lang = _lang(language or reply_language(message))
    candidates: List[str] = []
    availability = availability_answer(message, language=lang, data_dir=data_dir)
    if availability:
        candidates.append(availability)
    llm_cleaned = _clean_llm_ack(llm_ack, lang)
    if llm_cleaned:
        candidates.append(llm_cleaned)
    echo = requirement_echo(message, requirement, language=lang, exclude_slot=slot)
    if echo:
        candidates.append(_ACK_ECHO_LEADS[lang][seed % len(_ACK_ECHO_LEADS[lang])].format(echo=echo))

    for candidate in candidates:
        if not ack_conflicts_with_slot(candidate, slot, lang):
            return candidate
    generic = _ACK_GENERIC[lang]
    return generic[seed % len(generic)]


def _already_asks(answer: str, slot: str, language: str) -> bool:
    """答案里是否已经在问同一个槽位（避免同一轮把同一个问题问两遍）。"""
    if not answer or not slot:
        return False
    try:
        from src.rag.readiness import QUESTION_VARIANTS

        for variant in (QUESTION_VARIANTS.get(slot) or {}).get(_lang(language)) or ():
            if variant.strip() and variant.strip() in answer:
                return True
    except Exception:  # pragma: no cover - 防御式
        return False
    return False


def _tail_question(question: str) -> str:
    """取问句本体：去掉 "Just so I match the right models —" 这类前置铺垫。"""
    for dash in ("—", "–"):
        if dash in question:
            tail = question.rsplit(dash, 1)[-1].strip()
            if tail:
                return tail
    return question


def _lower_first_word(text: str) -> str:
    """引导语（以逗号结尾）后面的问句首字母小写，读起来才自然。

    例："…how many cabinets would fit best. In the meantime, Will it be an
    indoor or outdoor setup?" → 末尾问句改成小写 w。
    """
    text = str(text or "")
    if not text:
        return text
    head = text.split(" ", 1)[0]
    if head in ("I", "I'd", "I'll", "I'm") or head.isupper():
        return text
    return text[0].lower() + text[1:]


def compose_requirement_reply(
    *,
    answer: str = "",
    question: str = "",
    slot: str = "",
    message: str = "",
    language: Optional[str] = None,
    seed: int = 0,
    requirement: Optional[Dict[str, Any]] = None,
    include_ack: bool = True,
    data_dir: Optional[str] = None,
    llm_ack: str = "",
) -> str:
    """把"回应"与"追问"合成一句自然的销售回复。

    - ``answer`` 为空（Sales 自己收需求）：``回应 + 追问``
    - ``answer`` 已经是同一个追问（Solution 的 clarify）：只补一句回应
    - ``answer`` 是别的答复（RAG 答客户问题）：``答复 + 引导语 + 追问``
    """
    answer = str(answer or "").strip()
    question = str(question or "").strip()
    if not question:
        return answer

    lang = _lang(language or reply_language(message))
    ack = (
        acknowledge(
            message,
            requirement=requirement,
            language=lang,
            seed=seed,
            data_dir=data_dir,
            llm_ack=llm_ack,
            slot=slot,
        )
        if include_ack
        else ""
    )

    if answer and _already_asks(answer, slot, lang):
        return f"{ack} {answer}".strip() if ack else answer
    if answer:
        connectors = _CONNECTORS[lang]
        tail = _tail_question(question)
        if lang == "en":
            tail = _lower_first_word(tail)
        return f"{answer} {connectors[seed % len(connectors)]} {tail}".strip()
    return f"{ack} {question}".strip() if ack else question


__all__ = [
    "ack_conflicts_with_slot",
    "acknowledge",
    "availability_answer",
    "compose_requirement_reply",
    "reply_language",
    "requirement_echo",
]
