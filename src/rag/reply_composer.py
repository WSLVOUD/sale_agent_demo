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


# ── 回复语言兜底（客户口径：策略=en 时绝不能出现中文）────────────────────────
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")

_ENFORCE_EN_PROMPT = """Rewrite the customer reply below in English ONLY.

Rules:
1. Keep every fact, number, model code and question exactly as it is.
2. Output English only — absolutely no Chinese characters.
3. Keep the same conversational sales tone and roughly the same length.
4. Do not add, remove or answer anything; just say the same thing in English.
5. Output the rewritten reply only — no quotes, no explanation, no markdown.

Reply:
{text}"""


def contains_cjk(text: str) -> bool:
    """文本里是否含中日韩字符（用于"必须是英文"的语言护栏）。"""
    return bool(_CJK_RE.search(str(text or "")))


def enforce_english(text: str, *, message: str = "") -> str:
    """语言护栏：策略为 en 时，保证发给客户的文本里没有中文。

    - 没有中文 → 原样返回（不增加任何 LLM 调用）；
    - 有中文 → 用**一次** LLM 调用重写成英文；
    - 重写失败或结果仍是中文 → 返回空串，由调用方退回英文兜底话术，
      绝不让中文发给客户。

    策略为 ``auto`` 时不做任何处理（跟随客户语言是预期行为）。
    """
    text = str(text or "")
    if not text or not contains_cjk(text):
        return text
    if reply_language(message) != "en":
        return text
    logger.warning("CJK reply detected under policy=en — rewriting to English: %r", text[:80])
    try:
        from src.core.llm import get_llm

        response = get_llm(temperature=0).invoke(_ENFORCE_EN_PROMPT.format(text=text[:1500]))
        rewritten = str(getattr(response, "content", response) or "").strip()
        if rewritten and not contains_cjk(rewritten):
            return rewritten
        logger.warning("English rewrite still contains CJK — dropping it")
    except Exception as exc:  # pragma: no cover - 网络/额度问题
        logger.warning("English rewrite failed: %s", exc)
    return ""


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
# 刚"回答完客户的问题"（公司/价格/规格核实）之后再抛需求问题时的过渡语，
# 直接硬接问句会很生硬（实测反馈："…from there. Is this a permanent installation…?"）
_BRIDGES = {
    "en": (
        "By the way —",
        "On that note —",
        "While we're at it —",
        "So I can point you to the right model —",
        "That said —",
        "Now —",
    ),
    "zh": (
        "另外，",
        "顺便问一下，",
        "说到这个，",
        "同时，",
    ),
}


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

# 只有一句空泛客套、完全没有客户内容的"接话"（客户口径：这种不要发）
_CLICHE_ACK_RE = re.compile(
    r"^(?:(?:got it|ok|okay|understood|sure|noted|thanks|thank you|thanks for that|alright|fine|"
    r"noted thanks|好的|收到|了解|明白|知道了|嗯|行|可以)[.。！!,，、\s]*)+$",
    re.IGNORECASE,
)


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
    if _CLICHE_ACK_RE.match(ack):
        # 只有 "Got it / 好的" 这种没有任何客户内容的空话 → 丢弃，
        # 交给带客户内容的 echo 兜底（客户口径：接话要顺着客户说）
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

    # 公司 / 办事处 / 地址类问题不是"有没有某规格"，交给 company_info 照实回答
    # （实测 bug：客户问"western region 有没有你们的人"，这里回了一句
    #  "Yes — we do carry an LED."）
    from src.rag.company_info import is_company_question

    if is_company_question(text):
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
    #    必须"有具体规格"才算可用性提问：只问"有没有 LED 屏"这种泛问，
    #    回一句 "Yes — we do carry an LED." 没有信息量（实测 bug）
    pitch = _asked_pitch(slots, text)
    has_specific_spec = bool(
        pitch
        or slots.get("cob")
        or slots.get("hdr")
        or slots.get("waterproof")
        or slots.get("brightness_min")
    )
    if not has_specific_spec:
        return None

    spec = _spec_text(slots, lang, text)
    if not spec:
        return None
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


# ── 价格 / 报价类提问：需求没问清之前，先说"要确认产品和配置才能报价" ────────
_PRICE_QUESTION_RE = re.compile(
    r"(?<![a-z])(?:price|prices|pricing|cost|costs|quote|quotation|"
    r"how much|discount|cheaper|expensive|budget)\b|"
    r"价格|报价|多少钱|价钱|优惠|折扣|便宜|贵",
    re.IGNORECASE,
)

_PRICE_POLICY_ANSWERS = {
    "en": (
        "Pricing depends on the exact model and cabinet configuration, so I need to confirm the product first — then I'll put together a quotation for you.",
        "I can quote you properly once the model and cabinet layout are confirmed — let me pin those down first.",
        "Prices are per configuration, so I'll confirm the right model first and follow up with the quotation.",
        "The quotation depends on which model and configuration we land on — I'll confirm those details first, then price it up.",
    ),
    "zh": (
        "价格要按具体型号和箱体配置来算，确认好产品之后我马上给您报价。",
        "不同型号和配置价格差别比较大，我先把型号确认下来，随后给您准确报价。",
        "报价需要先确定型号和箱体排布，我们先把需求确认清楚。",
    ),
}


def is_price_question(message: str) -> bool:
    """客户这句话是不是在问价格 / 报价。"""
    return bool(_PRICE_QUESTION_RE.search(str(message or "")))


def price_policy_answer(language: str = "en", seed: int = 0) -> str:
    """"要先确认产品才能报价"的标准应答（多种说法轮换）。"""
    variants = _PRICE_POLICY_ANSWERS.get(_lang(language)) or _PRICE_POLICY_ANSWERS["en"]
    return variants[seed % len(variants)]


# ── "没有匹配的产品" → 改成"能不能放宽某个参数"（客户口径）─────────────────
# 禁止对客户说"找不到 / 没有匹配的产品"，一律改成邀请客户放宽条件。
_NO_PRODUCT_RE = re.compile(
    r"no matching products?|no suitable (?:outdoor |indoor )?model|"
    r"(?:i )?(?:could ?n[o']t|cannot|can't|am unable to) find (?:a|any|the) "
    r"(?:model|product|match)|couldn't find a model|"
    r"no (?:models?|products?) (?:found|available|match)|"
    r"没有匹配的?(?:产品|型号)|没有合适的?(?:产品|型号)|"
    r"找不到(?:合适的?|匹配的?|适合的?)?(?:产品|型号|屏|大屏|方案)|"
    r"(?:没有|未)找到(?:合适的?|匹配的?)?(?:产品|型号)|"
    r"无匹配(?:产品|型号)|无法推荐|推荐不出来",
    re.IGNORECASE,
)

_RELAXATION_ANSWERS = {
    "en": (
        "Let's take a slightly different angle — if one of the requirements can be relaxed "
        "(for example the pixel pitch, the screen size, or the viewing distance), I can match "
        "a model for you right away.",
        "Happy to get you the closest fit — would you be open to adjusting one requirement, "
        "say the pixel pitch or the screen size? Then I can put the right options in front of you.",
        "One quick option: if any of the requirements is flexible — pitch, size, or installation — "
        "I can match a suitable model immediately.",
        "If you can give a little on one of the conditions (pitch, brightness or screen size), "
        "I'll find you the best matching model straight away.",
    ),
    "zh": (
        "我们换个角度：如果某个条件可以放宽一点（比如点间距、屏体尺寸或观看距离），"
        "我马上就能帮您匹配到合适的型号。",
        "方便的话，看看哪个条件能松一点（例如点间距、亮度或尺寸），我好帮您找到最合适的型号。",
        "只要有一个条件可以灵活一点（点间距 / 尺寸 / 安装方式都行），我就能立刻帮您匹配合适的型号。",
    ),
}


def has_no_product_phrase(text: str) -> bool:
    """回复里是否出现了"找不到 / 没有匹配产品"这类话术。"""
    return bool(_NO_PRODUCT_RE.search(str(text or "")))


def relaxation_answer(language: Optional[str] = None, seed: int = 0) -> str:
    """"能不能放宽某个参数"的应答（多种说法轮换）。"""
    variants = _RELAXATION_ANSWERS.get(_lang(language)) or _RELAXATION_ANSWERS["en"]
    return variants[seed % len(variants)]


# ── Phase 15：Best-effort 推荐时"缺了什么 + 会影响什么"的自然说法 ────────────
# 客户不知道某项需求 → 仍然按已有信息推荐，但必须**自然**说明缺的是什么、
# 可能影响什么；绝不能说"信息不足无法推荐"。
_UNKNOWN_IMPACT: Dict[str, Dict[str, Any]] = {
    "viewing_distance": {
        "en": ("the exact viewing distance",
               "the final pixel pitch may need a small adjustment once you know it"),
        "zh": ("具体的观看距离", "拿到后最终点间距可能还需要微调"),
    },
    "installation": {
        "en": ("the installation type",
               "I've assumed a fixed installation for now and can switch you to a rental "
               "series if that changes"),
        "zh": ("安装方式", "目前按固定安装来选，如果改成租赁我可以换租用系列"),
    },
    "size": {
        "en": ("the exact screen size",
               "the cabinet count and final screen dimensions can be worked out as soon as "
               "we have the width and height"),
        "zh": ("具体的屏体尺寸", "拿到宽高后就能算出箱体数量和最终屏体尺寸"),
    },
    "width": {
        "en": ("the target screen width",
               "the cabinet layout can be finalised once we have the width"),
        "zh": ("目标屏幕宽度", "有宽度后就能确定箱体排布"),
    },
    "height": {
        "en": ("the target screen height",
               "the cabinet layout can be finalised once we have the height"),
        "zh": ("目标屏幕高度", "有高度后就能确定箱体排布"),
    },
    "environment": {
        "en": ("the indoor/outdoor setup",
               "an outdoor install would need a brighter, weatherproofed model"),
        "zh": ("室内还是室外", "如果改成室外需要更亮、防护等级更高的型号"),
    },
    "purpose": {
        "en": ("the exact application",
               "the feature set can be tuned once we know how the screen will be used"),
        "zh": ("具体使用场景", "明确场景后功能配置还能再优化"),
    },
    "brightness": {
        "en": ("the required brightness level",
               "a different brightness option can be quoted if the site needs more"),
        "zh": ("亮度要求", "如果现场需要更高亮度可以再换型号"),
    },
    "pixel_pitch": {
        "en": ("the preferred pixel pitch",
               "we can move to a finer or coarser pitch whenever you decide"),
        "zh": ("偏好的点间距", "确定后可以在更细或更粗的点间距之间切换"),
    },
}

_UNKNOWN_ITEM_FALLBACK = {
    "en": ("one of the details", "I can fine-tune the recommendation once we have it"),
    "zh": ("其中一项细节", "拿到后可以再把推荐调得更准"),
}

_DEGRADED_TEMPLATES = {
    "en": (
        "One thing to flag: {items} {verb} not confirmed yet, so this is the best match "
        "for what you've told me — {impacts}.",
        "Just so it's clear: I've based this on your confirmed requirements, as {items} "
        "{verb} still open. {impacts_cap}.",
    ),
    "zh": (
        "有一点先说明：{items}还没确认，所以这是基于您已提供信息的最佳匹配 —— {impacts}。",
        "补充一句：目前是按您已确认的信息来选的，{items}还在待定，{impacts}。",
    ),
}


def missing_impact(slot: str, language: Optional[str] = None) -> Tuple[str, str]:
    """某个未确认槽位 → （"缺的是什么", "可能影响什么"）。"""
    lang = _lang(language)
    table = _UNKNOWN_IMPACT.get(slot)
    if not table:
        table = {"en": _UNKNOWN_ITEM_FALLBACK["en"], "zh": _UNKNOWN_ITEM_FALLBACK["zh"]}
    return tuple(table.get(lang) or table["en"])  # type: ignore[return-value]


def degraded_note(
    slots: Sequence[str],
    language: Optional[str] = None,
    seed: int = 0,
) -> str:
    """Phase 15：存在 unknown 字段时，给推荐话术补一句"缺什么 + 影响什么"。"""
    wanted = [str(slot) for slot in (slots or []) if str(slot).strip()]
    if not wanted:
        return ""
    lang = _lang(language)
    items = [missing_impact(slot, lang)[0] for slot in wanted]
    impacts = [missing_impact(slot, lang)[1] for slot in wanted]
    if lang == "zh":
        item_text = "、".join(items)
        impact_text = "；".join(impacts)
        verb = "还没确认"
    else:
        item_text = " and ".join(items)
        impact_text = " Also, ".join(impacts)
        verb = "is" if len(items) == 1 else "are"
    templates = _DEGRADED_TEMPLATES.get(lang) or _DEGRADED_TEMPLATES["en"]
    template = templates[seed % len(templates)]
    return template.format(
        items=item_text,
        verb=verb,
        impacts=impact_text,
        impacts_cap=impact_text[:1].upper() + impact_text[1:] if impact_text else "",
    ).strip()


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

# 图片确认句 → 追问之间的过渡词（让两句接得上，而不是硬拼）
_VISION_BRIDGES = {
    "en": ("So, ", "Then, ", "Now, ", "Also, ", "So I can match the right model, "),
    "zh": ("那么，", "这样的话，", "那个，", "顺便问一下，"),
}

# 纯客套、没有实质信息的"回应"：已经有图片确认句时就不再叠一遍
_GENERIC_ACK_STARTS = {
    "en": (
        "got it", "thanks", "thank you", "sure", "ok", "okay", "understood",
        "alright", "sounds good", "happy to help", "no problem", "noted",
    ),
    "zh": ("好的", "收到", "明白", "了解", "没问题", "谢谢", "嗯", "可以"),
}


def _is_generic_ack(text: str, language: str) -> bool:
    """这句"回应"是不是纯客套（是的话，图片确认句在场时就不重复了）。"""
    cleaned = str(text or "").strip().lower()
    if not cleaned:
        return False
    if language == "en" and len(cleaned.split()) > 14:
        return False
    return any(cleaned.startswith(prefix) for prefix in _GENERIC_ACK_STARTS.get(language, ()))


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
    # 公司 / 办事处 / 地址类问题：按 data/company_profile.txt 照实回答
    from src.rag.company_info import company_answer

    company = company_answer(message, language=lang, seed=seed)
    if company:
        candidates.append(company)
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


# 问句模板里的"固定铺垫"（纯过渡话术，没有实际信息）——接了过渡语之后就该去掉；
# 而 "Fixed installation or rental —" 这种是**问句内容**，必须保留。
_CANNED_PREAMBLES = (
    "that's okay", "that is okay", "that's fine", "that is fine", "no problem",
    "just so i quote", "just so i use it correctly", "just so i match", "just so i plan",
    "just so i can point you", "quick check", "quick one", "one quick question",
    "while we're at it", "if you're not sure", "if you are not sure",
    "so i plan this properly", "in the meantime",
    "没关系", "不确定也没关系", "顺便", "另外",
)


def _tail_question(question: str) -> str:
    """取问句本体：只去掉"固定铺垫"，保留问句本身的内容。

    例：
      "Just so I match the right models — is this a fixed install or a rental?"
        → "is this a fixed install or a rental?"          （铺垫是固定话术 → 去掉）
      "Fixed installation or rental — which one is it for you?"
        → 原样保留（"Fixed installation or rental" 是问句内容，不能砍）
    """
    text = str(question or "").strip()
    for dash in ("—", "–"):
        if dash not in text:
            continue
        head, tail = text.rsplit(dash, 1)
        head_clean = head.strip().lower()
        tail = tail.strip()
        if not tail:
            continue
        if head_clean.startswith(_CANNED_PREAMBLES) or len(head_clean.split()) <= 3:
            return tail
    return text


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


def _ack_is_customer_answer(message: str, data_dir: Optional[str] = None) -> bool:
    """这一轮的"回应"是不是在**回答客户提出的问题**（公司 / 价格 / 规格核实）。

    这类回应之后再抛需求问题需要过渡语；而"复述客户刚说的需求"（Got it — a
    church.）直接接问句是自然的。
    """
    text = str(message or "")
    if not text:
        return False
    if is_price_question(text):
        return True
    try:
        from src.rag.company_info import is_company_question

        if is_company_question(text):
            return True
    except Exception:  # pragma: no cover - 防御式
        pass
    return bool(availability_answer(text, data_dir=data_dir))


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
    vision_confirmation: str = "",
) -> str:
    """把"回应"与"追问"合成一句自然的销售回复。

    - ``answer`` 为空（Sales 自己收需求）：``回应 + 追问``
    - ``answer`` 已经是同一个追问（Solution 的 clarify）：只补一句回应
    - ``answer`` 是别的答复（RAG 答客户问题）：``答复 + 引导语 + 追问``
    """
    answer = str(answer or "").strip()
    question = str(question or "").strip()
    vision_confirmation = str(vision_confirmation or "").strip()

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
    # 图片确认句本身就包含了"接住客户这句话"（照片收到 + 我看到什么），
    # 所以纯客套的 ack（"Got it — happy to help you find a display like that."）
    # 不再叠上去 —— 否则就是三句话各说各的，读起来很生硬。
    if vision_confirmation and _is_generic_ack(ack, lang):
        ack = ""
    # 【实测 bug】图片确认句里刚说了"看起来是固定安装"，紧接着又问"是固装还是租赁" → 自相矛盾。
    # 图片确认本身就是"跟客户核对这一项"，所以这一项本轮不再追问。
    if question and vision_confirmation and slot and ack_conflicts_with_slot(
        vision_confirmation, slot, lang
    ):
        logger.info(
            "Dropping question for slot=%s — already covered by the vision confirmation",
            slot,
        )
        question = ""
    # 图片识别结果先跟客户确认：放在"回应"之后、追问之前
    lead = " ".join(part for part in (ack, vision_confirmation) if part).strip()

    if not question:
        # 没有待问项时保持原行为（不要把 ack 硬拼上来）
        if not vision_confirmation:
            return answer
        return " ".join(part for part in (answer, vision_confirmation) if part).strip()

    if answer and _already_asks(answer, slot, lang):
        return f"{lead} {answer}".strip() if lead else answer
    if answer:
        connectors = _CONNECTORS[lang]
        tail = _tail_question(question)
        if lang == "en":
            tail = _lower_first_word(tail)
        body = f"{answer} {connectors[seed % len(connectors)]} {tail}".strip()
        return f"{lead} {body}".strip() if lead else body
    if lead:
        # 有图片确认句时：用过渡词把"确认"和"追问"接起来，并用完整问句
        # （"…looks like an indoor LED screen for a conference room, correct me if I've misread it.
        #   So, is this a permanent install, or is it for rental/events?"）
        if vision_confirmation:
            bridges = _VISION_BRIDGES[lang]
            full_question = question
            if lang == "en":
                full_question = _lower_first_word(full_question)
            # 中文不加空格，英文加空格
            separator = "" if lang == "zh" else " "
            return f"{lead}{separator}{bridges[seed % len(bridges)]}{full_question}".strip()
        # 【客户口径】只要既有"接话"又有"追问"，就必须有一个过渡把它们连起来，
        # 不能两句硬拼（"…enjoy a good walk. Where's it going to be used?"）。
        # 过渡语按轮次轮换，英文还会把问句首字母小写，读起来是一段话。
        bridges = _BRIDGES[lang]
        tail = _tail_question(question)
        if lang == "en":
            tail = _lower_first_word(tail)
        separator = "" if lang == "zh" else " "
        return f"{lead}{separator}{bridges[seed % len(bridges)]} {tail}".strip()
    return question


__all__ = [
    "ack_conflicts_with_slot",
    "acknowledge",
    "availability_answer",
    "compose_requirement_reply",
    "contains_cjk",
    "degraded_note",
    "enforce_english",
    "has_no_product_phrase",
    "is_price_question",
    "missing_impact",
    "price_policy_answer",
    "relaxation_answer",
    "reply_language",
    "requirement_echo",
    "vision_confirmation_items",
    "vision_confirmation_sentence",
]
