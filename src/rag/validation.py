"""
Phase 11：确定性校验（替代原来的 LLM Reflection 重新决策）。

计划文档要求 Reflection 不再让 LLM 决定产品，而是逐项校验：

  1. Model 是否真实存在
  2. Product 是否来自 canonical data
  3. Environment 是否匹配
  4. Installation 是否匹配
  5. Pixel Pitch 是否符合需求
  6. Cabinet 数据是否真实
  7. Module 数据是否真实
  8. Cabinet Count 是否正确
  9. Module Count 是否正确
  10. 回复是否存在虚构参数

校验只读 canonical data，纯函数、可测试、无 LLM。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from src.models.product import CanonicalModel
from src.models.requirement import RequirementProfile

logger = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"\bTW\d{2}-(?:IRHD|HOD|COB|3216|IR|OD)-P\d+(?:\.\d+)?(?:[HE])?(?:\(GOB\))?", re.IGNORECASE)
_NIT_RE = re.compile(r"(\d{3,5})\s*(?:nit|nits|cd/m2|cd/m²)", re.IGNORECASE)
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_PITCH_RE = re.compile(r"(?<![A-Za-z0-9])[Pp](\d+(?:\.\d+)?)(?![A-Za-z0-9])")


def _product_model_names(products: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for item in products or []:
        meta = (item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)) or {}
        name = meta.get("model") or meta.get("product_id")
        if name and name not in names:
            names.append(name)
    return names


def validate_recommendation(
    recommendation: str,
    products: Sequence[Any],
    profile: Optional[RequirementProfile],
    selection: Optional[Dict[str, Any]] = None,
    calculation: Optional[Dict[str, Any]] = None,
    model_index: Optional[Dict[str, CanonicalModel]] = None,
) -> Dict[str, Any]:
    """执行 10 项校验，返回 ``{score, errors, checks, valid_products, summary}``。"""
    if model_index is None:
        from src.config import config
        from src.rag.json_loader import canonical_model_index

        model_index = canonical_model_index(config.DATA_DIR)

    errors: List[str] = []
    checks: Dict[str, bool] = {}
    recommendation = recommendation or ""

    # 关联的型号：优先用选型结果，其次用 product metadata
    selected_models: List[str] = []
    for rec in (selection or {}).get("recommendations") or []:
        if rec.get("model"):
            selected_models.append(rec["model"])
    for name in _product_model_names(products):
        if name not in selected_models:
            selected_models.append(name)

    # 1. Model 是否真实存在
    unknown = [m for m in selected_models if m not in model_index]
    checks["model_exists"] = not unknown
    if unknown:
        errors.append(f"型号不存在于产品库: {', '.join(unknown)}")

    # 2. Product 是否来自 canonical data
    checks["source_is_canonical"] = bool(selected_models) and not unknown
    if not selected_models:
        errors.append("推荐结果中没有可识别的型号")

    canonical: List[CanonicalModel] = [model_index[m] for m in selected_models if m in model_index]
    primary = canonical[0] if canonical else None

    # 3. Environment 匹配
    if profile is not None and profile.environment and canonical:
        mismatch = [
            m.model for m in canonical
            if (profile.environment == "indoor" and not m.indoor)
            or (profile.environment == "outdoor" and not m.outdoor)
        ]
        checks["environment_match"] = not mismatch
        if mismatch:
            errors.append(f"环境不匹配: {', '.join(mismatch)}")
    else:
        checks["environment_match"] = True

    # 4. Installation 匹配
    if profile is not None and profile.installation and canonical:
        mismatch = [m.model for m in canonical if m.installation != profile.installation]
        checks["installation_match"] = not mismatch
        if mismatch:
            errors.append(f"安装方式不匹配: {', '.join(mismatch)}")
    else:
        checks["installation_match"] = True

    # 5. Pixel Pitch 是否符合需求（客户明确值 / 推断区间）
    if primary is not None:
        if profile is not None and profile.pixel_pitch_mm is not None:
            tolerance = max(0.2, profile.pixel_pitch_mm * 0.15)
            ok = abs(primary.pixel_pitch_mm - profile.pixel_pitch_mm) <= tolerance
        else:
            technical = (selection or {}).get("technical_parameters") or {}
            low = technical.get("pixel_pitch_min_mm")
            high = technical.get("pixel_pitch_max_mm")
            ok = True
            if low is not None:
                ok = ok and primary.pixel_pitch_mm >= low - 1e-6
            if high is not None:
                ok = ok and primary.pixel_pitch_mm <= high + 1e-6
        checks["pixel_pitch_match"] = bool(ok)
        if not ok:
            errors.append(f"点间距不符合需求: {primary.model} = {primary.pixel_pitch_mm}mm")

    # 6/7. Cabinet / Module 数据是否真实
    checks["cabinet_data_real"] = bool(primary and primary.cabinet_width_mm and primary.cabinet_height_mm)
    checks["module_data_real"] = bool(primary and primary.module_width_mm and primary.module_height_mm)
    if primary is not None:
        if not checks["cabinet_data_real"]:
            errors.append(f"{primary.model} 缺少箱体数据")
        if not checks["module_data_real"]:
            errors.append(f"{primary.model} 缺少模组数据")

    # 5b. 室外点间距下限：室内外没点明点间距时，室外必须 P6 及以上
    if (
        primary is not None
        and profile is not None
        and profile.pixel_pitch_mm is None
        and profile.environment in ("outdoor", "semi_outdoor")
    ):
        from src.rag.parameter_inference import OUTDOOR_MIN_PITCH_MM

        ok = primary.pixel_pitch_mm >= OUTDOOR_MIN_PITCH_MM - 1e-6
        checks["outdoor_pitch_boundary"] = bool(ok)
        if not ok:
            errors.append(
                f"室外点间距低于 P{OUTDOOR_MIN_PITCH_MM:g}: "
                f"{primary.model} = {primary.pixel_pitch_mm}mm"
            )

    # 8/9. Cabinet / Module 数量是否正确（用 canonical 尺寸重算一遍）
    if calculation and primary is not None:
        expected_columns = -(-int(calculation["target_width_mm"]) // int(primary.cabinet_width_mm)) if primary.cabinet_width_mm else None
        import math

        expected_columns = math.ceil(calculation["target_width_mm"] / primary.cabinet_width_mm)
        expected_rows = math.ceil(calculation["target_height_mm"] / primary.cabinet_height_mm)
        checks["cabinet_count_correct"] = (
            calculation.get("columns") == expected_columns
            and calculation.get("rows") == expected_rows
            and calculation.get("cabinet_count") == expected_columns * expected_rows
        )
        if not checks["cabinet_count_correct"]:
            errors.append(
                f"箱体数量错误: 期望 {expected_columns}x{expected_rows}="
                f"{expected_columns * expected_rows} 个，实际 "
                f"{calculation.get('columns')}x{calculation.get('rows')}="
                f"{calculation.get('cabinet_count')} 个"
            )
        expected_modules = calculation.get("cabinet_count", 0) * primary.modules_per_cabinet
        checks["module_count_correct"] = calculation.get("total_modules") == expected_modules
        if not checks["module_count_correct"]:
            errors.append(
                f"模组数量错误: 期望 {expected_modules}，实际 {calculation.get('total_modules')}"
            )
    else:
        checks["cabinet_count_correct"] = True
        checks["module_count_correct"] = True

    # 10. 回复是否存在虚构参数
    fabricated: List[str] = []
    if canonical:
        allowed_models = {m.model.upper() for m in model_index.values()}
        allowed_pitches = {round(m.pixel_pitch_mm, 2) for m in canonical}
        allowed_brightness = {m.brightness_nit for m in canonical}

        for found in _MODEL_RE.findall(recommendation):
            if found.upper() not in allowed_models:
                fabricated.append(f"型号 {found}")
        for value in _PITCH_RE.findall(recommendation):
            if round(float(value), 2) not in allowed_pitches:
                fabricated.append(f"点间距 P{value}")
        for value in _NIT_RE.findall(recommendation):
            if int(value) not in allowed_brightness:
                fabricated.append(f"亮度 {value}nit")
        for value in _MM_RE.findall(recommendation):
            number = float(value)
            allowed_mm = {
                primary.module_width_mm, primary.module_height_mm,
                primary.cabinet_width_mm, primary.cabinet_height_mm,
            }
            if primary is not None and not any(abs(number - item) < 0.01 for item in allowed_mm):
                # 允许"实际尺寸/面积"里出现的其它 mm 数值，避免误报
                continue
    checks["no_fabricated_params"] = not fabricated
    if fabricated:
        errors.append("回复中出现产品数据之外的参数: " + ", ".join(sorted(set(fabricated))[:5]))

    # 评分：10 分制，每个错误扣 1.5 分
    score = max(0.0, 10.0 - 1.5 * len(errors))
    valid_products = [
        item for item in products or []
        if (_product_model_names([item]) or [""])[0] in selected_models
    ]
    summary = (
        "validation passed"
        if not errors
        else f"{len(errors)} issue(s): " + "; ".join(errors[:3])
    )
    logger.info("Validation: score=%.1f errors=%d", score, len(errors))
    return {
        "score": round(score, 2),
        "errors": errors,
        "checks": checks,
        "valid_products": valid_products,
        "summary": summary,
    }


__all__ = ["validate_recommendation"]
