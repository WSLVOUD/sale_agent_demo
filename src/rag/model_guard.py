"""自由问答 / 异议回答的"不越界"护栏（客户口径 2026-09-21）。

实测 bug：客户是「室内 / 教堂 / 固装」，AI 在回答"能不能 10 天到货"时突然写出

    "... that's really tight for an outdoor fixed cabinet setup like this.
     The TW21-OD-P10 needs production plus waterproofing and testing ..."

原因：那条路径（异议回答 / 自由问答）**既没有客户的已确认需求，也没有硬约束**，
检索到什么型号就往答复里写。这个模块提供三件事：

    requirement_lines(...)   客户已确认需求的简短清单（喂给 LLM 当"不可违背"的事实）
    retrieval_filters(...)   按环境 / 安装方式构造检索过滤条件（矛盾型号直接不进候选）
    strip_model_mentions(...) 自由问答默认**不给型号**（型号只能由推荐链路给出）
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 型号编码（TW21-OD-P10 / TW31-COB-P0.9H / T65Omni-N4 …）
MODEL_CODE_RE = re.compile(
    r"\b(?:TW|TC|TS|T)\s?\d{1,3}[A-Za-z0-9]*(?:-[A-Za-z0-9.]+)+\b",
    re.IGNORECASE,
)

# 户外型号的型号名特征（OD / OO / OUT…）
_OUTDOOR_MODEL_RE = re.compile(r"-(?:OD|OO|OR|OH)\b|-?(?:OD|OO|OR|OH)-", re.IGNORECASE)


def _value(source: Any, *names: str) -> Any:
    for name in names:
        if isinstance(source, dict) and source.get(name) not in (None, "", [], {}):
            return source[name]
        if hasattr(source, name):
            value = getattr(source, name)
            if value not in (None, "", [], {}):
                return value
    return None


def requirement_summary(source: Any) -> Dict[str, Any]:
    """客户已确认的需求（environment / installation / size / distance / purpose）。

    兼容两种来源：

      · RequirementProfile（Sales 的权威档案）—— environment / installation /
        target_width_mm / viewing_distance_m …；
      · legacy 投影字典（``profile_to_solution_requirement`` / Solution 的
        ``state["requirement"]``）—— indoor / outdoor / is_rental / size / distance …。

    实测 bug（2026-09-22）：自由问答把 legacy 字典传进来，而这里只认 profile
    风格的键 → "已确认需求"清单里**环境、安装方式、屏体尺寸全部丢失** →
    模型看不到客户已经答过的 permanent / 3×5，就又问了一遍。
    """
    if source is None:
        return {}
    out: Dict[str, Any] = {}
    environment = _value(source, "environment")
    if not environment:
        # legacy：indoor / outdoor 两个布尔
        if _value(source, "indoor") is True:
            environment = "indoor"
        elif _value(source, "outdoor") is True:
            environment = "outdoor"
    if environment:
        out["environment"] = str(environment)
    installation = _value(source, "installation")
    if not installation:
        # legacy：is_rental（True=租赁 / False=固装）
        rental = _value(source, "is_rental")
        if rental is True:
            installation = "rental"
        elif rental is False:
            installation = "fixed"
    if installation:
        out["installation"] = str(installation)
    purpose = _value(source, "purpose")
    if purpose:
        out["purpose"] = str(purpose)
    width = _value(source, "target_width_mm", "target_width_m")
    height = _value(source, "target_height_mm", "target_height_m")
    if width or height:
        out["size"] = f"{width or '?'}x{height or '?'} mm"
    elif _value(source, "size"):
        # legacy：已经是 "3米x5米" 这样的字符串
        out["size"] = str(_value(source, "size"))
    distance = _value(source, "viewing_distance_m", "distance")
    if distance:
        out["viewing_distance"] = str(distance)
    display_type = _value(source, "display_type")
    if display_type:
        out["display_type"] = str(display_type)
    return out


def requirement_lines(source: Any) -> List[str]:
    """给 LLM 看的"已确认需求"清单（没有就不返回）。"""
    summary = requirement_summary(source)
    lines: List[str] = []
    labels = {
        "environment": "使用环境（客户已确认）",
        "installation": "安装方式（客户已确认）",
        "purpose": "使用场景（客户已确认）",
        "size": "屏体尺寸（客户已确认）",
        "viewing_distance": "观看距离（客户已确认）",
        "display_type": "产品品类（客户已确认）",
    }
    for key, label in labels.items():
        value = summary.get(key)
        if value:
            lines.append(f"- {label}：{value}")
    return lines


def retrieval_filters(source: Any) -> Dict[str, Any]:
    """按已确认需求构造检索过滤（Chroma where 条件）。"""
    summary = requirement_summary(source)
    filters: Dict[str, Any] = {}
    environment = str(summary.get("environment") or "").lower()
    if environment in ("indoor", "室内"):
        filters["indoor"] = True
    elif environment in ("outdoor", "semi_outdoor", "户外", "室外", "半户外"):
        filters["outdoor"] = True
    installation = str(summary.get("installation") or "").lower()
    if installation in ("fixed", "固装"):
        filters["is_rental"] = False
    elif installation in ("rental", "租赁"):
        filters["is_rental"] = True
    display_type = str(summary.get("display_type") or "").upper()
    if display_type in ("LED", "LCD", "IFP"):
        filters["display_type"] = display_type
    return filters


def chunk_conflicts(chunk: Any, source: Any) -> bool:
    """这条检索结果是否与客户已确认需求相矛盾（矛盾就不要进 prompt）。"""
    summary = requirement_summary(source)
    if not summary:
        return False
    meta: Dict[str, Any] = {}
    if isinstance(chunk, dict):
        meta = dict(chunk.get("metadata") or {})
        text = str(chunk.get("text") or chunk.get("page_content") or "")
    else:  # pragma: no cover - Document 对象
        meta = dict(getattr(chunk, "metadata", {}) or {})
        text = str(getattr(chunk, "page_content", "") or "")

    environment = str(summary.get("environment") or "").lower()
    if environment in ("indoor", "室内"):
        if meta.get("outdoor") is True or meta.get("indoor") is False:
            return True
        if _OUTDOOR_MODEL_RE.search(text):
            return True
    if environment in ("outdoor", "semi_outdoor", "户外", "室外", "半户外"):
        if meta.get("indoor") is True and meta.get("outdoor") is not True:
            return True

    installation = str(summary.get("installation") or "").lower()
    if installation in ("fixed", "固装") and meta.get("is_rental") is True:
        return True
    if installation in ("rental", "租赁") and meta.get("is_rental") is False:
        return True
    return False


def drop_conflicting_chunks(chunks: Iterable[Any], source: Any) -> Tuple[List[Any], int]:
    """过滤掉与需求矛盾的检索结果，返回 ``(保留的, 丢掉的数量)``。"""
    kept: List[Any] = []
    dropped = 0
    for chunk in chunks or []:
        if chunk_conflicts(chunk, source):
            dropped += 1
            continue
        kept.append(chunk)
    return kept, dropped


def strip_model_mentions(text: str, *, allow: Iterable[str] = ()) -> Tuple[str, List[str]]:
    """去掉回答里的型号（自由问答默认不给型号），返回 ``(新文本, 去掉的型号)``。

    型号通常整句都是在讲它（"The TW21-OD-P10 needs production plus waterproofing"），
    所以**整句删掉**，而不是只抠掉型号名（否则会剩半句病句）。
    """
    allowed = {str(item).lower() for item in (allow or []) if item}
    removed: List[str] = []
    kept_sentences: List[str] = []
    for sentence in re.split(r"(?<=[.!?。！？])\s*", str(text or "")):
        piece = sentence.strip()
        if not piece:
            continue
        models = [m for m in MODEL_CODE_RE.findall(piece) if m.lower() not in allowed]
        if models:
            removed.extend(models)
            continue
        kept_sentences.append(piece)
    cleaned = " ".join(kept_sentences).strip()
    return cleaned, removed


# "在说客户这套配置"的特征词（区分"室内屏和户外屏不一样"这种泛泛比较）
_SETUP_WORDS_RE = re.compile(
    r"\byour\b|\bcabinet|\bsetup|\bpanel|\binstallation|\binstalled\b|\bthis screen\b|"
    r"\bthe screen\b|\bfor a\b|\byou (?:need|want|are)\b|你的|您的|这块屏|这套",
    re.IGNORECASE,
)
_INDOOR_WORDS_RE = re.compile(r"\bindoor\b|室内", re.IGNORECASE)
_OUTDOOR_WORDS_RE = re.compile(r"\boutdoor\b|\boutside\b|户外|室外", re.IGNORECASE)


def strip_environment_contradictions(text: str, source: Any) -> Tuple[str, List[str]]:
    """删掉与客户已确认环境相矛盾的句子（室内客户却讲 outdoor 配置）。

    只处理"在描述客户这套配置"的句子（带 your / cabinet / setup / screen… 之类特征），
    泛泛的对比句（"室内屏和户外屏不一样"）不会被误删。
    """
    summary = requirement_summary(source)
    environment = str(summary.get("environment") or "").lower()
    if environment not in ("indoor", "outdoor", "semi_outdoor", "室内", "室外", "户外"):
        return str(text or ""), []
    opposite = (
        _OUTDOOR_WORDS_RE if environment in ("indoor", "室内") else _INDOOR_WORDS_RE
    )
    kept: List[str] = []
    removed: List[str] = []
    for sentence in re.split(r"(?<=[.!?。！？])\s*", str(text or "")):
        piece = sentence.strip()
        if not piece:
            continue
        if opposite.search(piece) and _SETUP_WORDS_RE.search(piece):
            removed.append(piece[:120])
            continue
        kept.append(piece)
    cleaned = " ".join(kept).strip()
    return cleaned, removed


__all__ = [
    "MODEL_CODE_RE",
    "chunk_conflicts",
    "drop_conflicting_chunks",
    "requirement_lines",
    "requirement_summary",
    "retrieval_filters",
    "strip_environment_contradictions",
    "strip_model_mentions",
]
