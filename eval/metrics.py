"""
Phase 0 评估指标与公共工具。

本模块被 ``retrieval_eval.py`` / ``recommendation_eval.py`` /
``calculator_eval.py`` 共用，提供：

1. 指标计算（Recall@K / MRR / Slot Accuracy / Hard Constraint Violation）
2. Golden Dataset 加载
3. 产品代号解析（Series / Model，兼容数据文件里的非 ASCII 连字符）
4. 检索过滤条件构造（与生产链路保持一致）

设计原则：纯函数，无副作用，不依赖 LLM，可在无网络环境下运行。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = PROJECT_ROOT / "eval" / "golden_dataset.json"
REPORT_DIR = PROJECT_ROOT / "eval" / "reports"

# ── 产品代号解析 ─────────────────────────────────────────────────────────────
# 数据文件里使用了 U+2011 (non-breaking hyphen) 等非 ASCII 连字符，统一归一化。
_HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212-"
_HYPHEN_CLASS = f"[{re.escape(_HYPHENS)}]"

_SERIES_CODES = ("IRHD", "IR", "HOD", "COB", "3216", "OD")

_SERIES_RE = re.compile(
    r"\bTW\s*(\d{2})\s*" + _HYPHEN_CLASS + r"\s*(IRHD|HOD|COB|3216|IR|OD)\b",
    re.IGNORECASE,
)

_MODEL_RE = re.compile(
    r"\bTW\s*(\d{2})\s*" + _HYPHEN_CLASS + r"\s*(IRHD|HOD|COB|3216|IR|OD)"
    r"\s*" + _HYPHEN_CLASS + r"\s*[Pp]?\s*(\d+(?:\.\d+)?)\s*(H|E)?\s*(?:\(\s*(GOB)\s*\))?",
    re.IGNORECASE,
)


def normalize_hyphens(text: str) -> str:
    """把所有非 ASCII 连字符替换为 '-'。"""
    if not text:
        return ""
    for ch in _HYPHENS:
        text = text.replace(ch, "-")
    return text


def canonical_series(tw: str, kind: str) -> str:
    """拼装规范化的 Series ID，例如 ``TW21-3216``。"""
    return f"TW{tw}{'-'}{kind.upper()}"


def extract_series(text: str) -> List[str]:
    """从文本中抽取 Series ID（去重、保持出现顺序）。"""
    normalized = normalize_hyphens(text or "")
    found: List[str] = []
    for tw, kind in _SERIES_RE.findall(normalized):
        series = canonical_series(tw, kind)
        if series not in found:
            found.append(series)
    return found


def extract_models(text: str) -> List[str]:
    """从文本中抽取完整 Model 代号（去重、保持出现顺序）。"""
    normalized = normalize_hyphens(text or "")
    found: List[str] = []
    for tw, kind, pitch, suffix, gob in _MODEL_RE.findall(normalized):
        model = f"{canonical_series(tw, kind)}-P{pitch}"
        if suffix:
            model += suffix.upper()
        if gob:
            model += "(GOB)"
        if model not in found:
            found.append(model)
    return found


def series_of_model(model: str) -> str:
    """从 Model 代号反推 Series ID，例如 ``TW21-3216-P2.5`` → ``TW21-3216``。"""
    series = extract_series(model)
    return series[0] if series else ""


def identifiers_from_items(items: Sequence[Any]) -> Tuple[List[str], List[str]]:
    """从检索/推荐结果中抽取 (series, models)。

    ``items`` 可以是 Document、dict 或纯文本；本函数优先使用 metadata 里的
    ``series_id`` / ``model``，其次回退到文本解析（兼容 Phase 2 之前的语料）。
    """
    series: List[str] = []
    models: List[str] = []
    for item in items or []:
        meta: Dict[str, Any] = {}
        text = ""
        if isinstance(item, dict):
            meta = item.get("metadata") or {}
            text = str(item.get("text") or item.get("page_content") or "")
        else:
            meta = getattr(item, "metadata", None) or {}
            text = str(getattr(item, "page_content", "") or "")

        for key in ("series_id", "series", "product_id"):
            value = meta.get(key)
            if not value:
                continue
            for found in extract_series(str(value)):
                if found not in series:
                    series.append(found)
        model_value = meta.get("model")
        if model_value and str(model_value) not in models:
            models.append(str(model_value))
        for found in extract_series(text):
            if found not in series:
                series.append(found)
        for found in extract_models(text):
            if found not in models:
                models.append(found)
    return series, models


# ── 指标 ────────────────────────────────────────────────────────────────────
def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Recall@K：前 K 个结果命中了多少比例的期望条目。"""
    expected_set = {e for e in expected if e}
    if not expected_set:
        return 0.0
    top = {r for r in list(retrieved)[:k] if r}
    return len(top & expected_set) / len(expected_set)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """MRR：第一个命中的排名倒数。"""
    expected_set = {e for e in expected if e}
    if not expected_set:
        return 0.0
    for index, item in enumerate(retrieved, start=1):
        if item in expected_set:
            return 1.0 / index
    return 0.0


def mean(values: Iterable[float]) -> float:
    """求均值，空集合返回 0.0。"""
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


def slot_match_rate(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    numeric_tolerance: float = 0.2,
) -> Tuple[float, Dict[str, bool]]:
    """逐字段比较结构化需求槽位（Requirement Slot Accuracy）。

    - 数值字段：允许 ``numeric_tolerance`` 的相对误差 + 0.1 绝对容差
    - 布尔字段：严格相等
    - 字符串：子串匹配（大小写不敏感）
    - 期望值为 None / "" / [] 的字段视为"未要求"，不计入分母
    """
    expected_clean = {
        k: v for k, v in (expected or {}).items()
        if v is not None and v != "" and v != [] and v != {}
    }
    if not expected_clean:
        return 1.0, {}

    details: Dict[str, bool] = {}
    for key, exp_val in expected_clean.items():
        act_val = (actual or {}).get(key)
        if act_val is None or act_val == "" or act_val == []:
            details[key] = False
            continue
        exp_val = normalize_slot_value(key, exp_val)
        act_val = normalize_slot_value(key, act_val)
        if isinstance(exp_val, bool):
            details[key] = bool(act_val) == exp_val
        elif isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
            tolerance = abs(float(exp_val)) * numeric_tolerance + 0.1
            details[key] = abs(float(act_val) - float(exp_val)) <= tolerance
        else:
            details[key] = str(exp_val).lower() in str(act_val).lower()

    return sum(details.values()) / len(details), details


# 场景词表：Golden Dataset 用中文描述场景，Query Understanding 输出规范英文 token。
# 比较前统一归一到规范 token，避免"口径不同"造成的假阴性。
_PURPOSE_ALIASES: Dict[str, str] = {
    # 会议室 / 办公
    "会议室": "conference", "会议": "conference", "会議室": "conference", "boardroom": "conference",
    "办公室": "office", "办公": "office",
    # 教学
    "教室": "classroom", "培训室": "classroom", "培训": "classroom", "教学": "classroom",
    # 展陈
    "展厅": "showroom", "展馆": "showroom", "展览": "showroom", "showroom": "showroom",
    "博物馆": "museum", "美术馆": "museum",
    "展会": "exhibition", "展览会": "exhibition", "expo": "exhibition",
    "大厅": "hall", "大堂": "hall",
    # 商业
    "商场": "retail", "商店": "retail", "店铺": "retail", "零售": "retail", "超市": "retail",
    "广告": "advertising", "楼体广告": "advertising", "幕墙广告": "advertising",
    "广告牌": "advertising", "幕墙": "advertising", "外墙": "advertising",
    # 演出 / 活动
    "演唱会": "concert", "音乐会": "concert", "演唱会租赁": "concert",
    "演出": "stage", "舞台": "stage", "表演": "stage",
    # 场馆 / 交通 / 其他
    "体育场": "stadium", "体育馆": "stadium", "球场": "stadium", "赛场": "stadium",
    "机场": "airport", "航站楼": "airport", "车站": "airport",
    "银行": "bank", "银行网点": "bank",
    "酒店": "hotel", "酒店大堂": "hotel",
    "餐厅": "restaurant", "医院": "hospital", "诊所": "hospital",
    "指挥中心": "control_room", "监控中心": "control_room", "控制室": "control_room",
    "室内大厅": "hall", "室内": "hall",
    "租赁": "rental", "临时": "rental", "活动": "rental",
    "商场外墙": "advertising", "幕墙广告屏": "advertising",
}


def normalize_slot_value(key: str, value: Any) -> Any:
    """槽位值归一化：场景词统一到规范 token，环境/安装方式统一大小写。"""
    if value is None or isinstance(value, bool):
        return value
    if key == "purpose":
        text = str(value).strip()
        if text in _PURPOSE_ALIASES:
            return _PURPOSE_ALIASES[text]
        lowered = text.lower()
        for alias, canonical in _PURPOSE_ALIASES.items():
            if alias.lower() in lowered:
                return canonical
        return lowered
    if key in ("environment", "installation", "display_type"):
        return str(value).strip().lower()
    return value


# ── 硬约束校验 ───────────────────────────────────────────────────────────────
def build_retrieval_filters(hard: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """把 Golden Dataset 的 ``hard`` 描述翻译成检索层 metadata filters。"""
    hard = hard or {}
    filters: Dict[str, Any] = {}
    if hard.get("display_type"):
        filters["display_type"] = hard["display_type"]
    environment = hard.get("environment")
    if environment == "indoor":
        filters["indoor"] = True
    elif environment == "outdoor":
        filters["outdoor"] = True
    installation = hard.get("installation")
    if installation == "fixed":
        filters["is_rental"] = False
    elif installation == "rental":
        filters["is_rental"] = True
    return filters


def hard_constraint_violations(
    items: Sequence[Any],
    hard: Optional[Dict[str, Any]],
    include_pitch: bool = True,
) -> List[str]:
    """检查结果是否违反硬约束，返回违规说明列表（空列表 = 全部合规）。

    对 metadata 缺失的条目采取"无法判定即不判违规"的保守策略，避免误报；
    唯一例外是 environment / installation，因为生产链路里它们一定存在。
    每条违规带类型前缀（``[environment]`` 等），便于汇总与定位。

    ``include_pitch=False`` 时忽略点间距：按计划文档的分工，观看距离推导出的点间距
    属于**软条件**（用于排序与最终校验），不该在"检索候选生成"阶段判违规。
    """
    hard = hard or {}
    if not hard:
        return []

    violations: List[str] = []
    for index, item in enumerate(items or [], start=1):
        meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
        text = str(
            (item.get("text") if isinstance(item, dict) else getattr(item, "page_content", "")) or ""
        )
        label = f"#{index}"

        display_type = hard.get("display_type")
        if display_type and meta.get("display_type") and meta["display_type"] != display_type:
            violations.append(
                f"[display_type] {label} display_type={meta['display_type']} != {display_type}"
            )

        environment = hard.get("environment")
        if environment == "indoor" and meta.get("indoor") is False:
            violations.append(f"[environment] {label} 环境不匹配: 期望 indoor")
        if environment == "outdoor" and meta.get("outdoor") is False:
            violations.append(f"[environment] {label} 环境不匹配: 期望 outdoor")
        if environment == "indoor" and meta.get("indoor") is None and "outdoor" in text.lower() and "indoor" not in text.lower():
            violations.append(f"[environment] {label} 疑似户外内容")

        installation = hard.get("installation")
        if installation and meta.get("is_rental") is not None:
            expected_rental = installation == "rental"
            if bool(meta["is_rental"]) != expected_rental:
                violations.append(f"[installation] {label} 安装方式不匹配: 期望 {installation}")

        brightness_min = hard.get("brightness_min")
        if brightness_min is not None and meta.get("brightness_max_cd") is not None:
            if meta["brightness_max_cd"] < brightness_min:
                violations.append(
                    f"[brightness] {label} 亮度不足: {meta['brightness_max_cd']} < {brightness_min}"
                )

        pitch_min = hard.get("pixel_pitch_min")
        pitch_max = hard.get("pixel_pitch_max")
        if include_pitch and pitch_min is not None and meta.get("pixel_pitch_max_mm") is not None:
            if meta["pixel_pitch_max_mm"] < pitch_min:
                violations.append(
                    f"[pitch] {label} 点间距上限 {meta['pixel_pitch_max_mm']} < {pitch_min}"
                )
        if include_pitch and pitch_max is not None and meta.get("pixel_pitch_min_mm") is not None:
            if meta["pixel_pitch_min_mm"] > pitch_max:
                violations.append(
                    f"[pitch] {label} 点间距下限 {meta['pixel_pitch_min_mm']} > {pitch_max}"
                )

    return violations


def pitch_fit_at_k(
    items: Sequence[Any],
    hard: Optional[Dict[str, Any]],
    k: int = 3,
) -> Optional[float]:
    """前 K 个结果里是否存在点间距落在期望区间的型号（软条件达成率）。

    期望区间为空时返回 ``None``（不参与汇总）。
    """
    hard = hard or {}
    low = hard.get("pixel_pitch_min")
    high = hard.get("pixel_pitch_max")
    if low is None and high is None:
        return None
    for item in list(items or [])[:k]:
        meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
        pitch = meta.get("pixel_pitch_mm")
        if pitch is None:
            pitch = meta.get("pixel_pitch_min_mm")
        if pitch is None:
            continue
        if (low is None or pitch >= low - 1e-6) and (high is None or pitch <= high + 1e-6):
            return 1.0
    return 0.0


def violation_type_counts(violations: Sequence[str]) -> Dict[str, int]:
    """按类型前缀统计违规数量。"""
    counts: Dict[str, int] = {}
    for item in violations or []:
        match = re.match(r"\[(\w+)\]", str(item))
        key = match.group(1) if match else "other"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def hit_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Coverage@K：前 K 个结果是否命中期望集合中的任意一个。"""
    expected_set = {e for e in expected if e}
    if not expected_set:
        return 0.0
    return 1.0 if set(list(retrieved)[:k]) & expected_set else 0.0


# ── Golden Dataset ──────────────────────────────────────────────────────────
def load_golden_cases(
    path: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """加载 Golden Dataset，支持按 id / tag 过滤与限量。"""
    dataset_path = Path(path) if path else GOLDEN_DATASET_PATH
    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases", [])

    if ids:
        wanted = {i.strip() for i in ids if i.strip()}
        cases = [c for c in cases if c.get("id") in wanted]
    if tags:
        wanted_tags = {t.strip() for t in tags if t.strip()}
        cases = [c for c in cases if wanted_tags & set(c.get("tags", []))]
    if limit is not None:
        cases = cases[: int(limit)]
    return cases


def write_report(report: Dict[str, Any], out_path: Optional[str] = None) -> Path:
    """把评估报告写入 ``eval/reports/``。"""
    if out_path:
        target = Path(out_path)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
    else:
        target = REPORT_DIR / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return target


def aggregate_by_tag(cases: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, float]:
    """按 tag 汇总某个逐条指标的平均值。"""
    buckets: Dict[str, List[float]] = {}
    for case in cases:
        value = case.get(metric_key)
        if value is None:
            continue
        for tag in case.get("tags", []) or ["untagged"]:
            buckets.setdefault(tag, []).append(float(value))
    return {tag: round(mean(values), 4) for tag, values in sorted(buckets.items())}
