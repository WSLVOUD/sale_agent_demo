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
    "item_summary_line",
    "product_model",
    "screen_label",
]
