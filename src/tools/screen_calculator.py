"""
Phase 9：Screen Calculator（箱体 / 模组工程计算）。

职责（计划文档「十四、Phase 9」）：
    输入  Model + 目标尺寸
    输出  箱体列数/行数/总数、实际尺寸、每箱模组数、模组总数、实际分辨率、面积

关键原则：
  - **LLM 不参与工程计算**，全部由本模块的确定性 Python 逻辑完成
  - 尺寸一律以物理尺寸（cabinet / module 的 mm）为准，
    不依赖厂商规格表里可能不自洽的 cabinet_resolution 字段

用法::

    from src.tools.screen_calculator import calculate_screen
    calc = calculate_screen("TW21-3216-P2.5", target_width_mm=5000, target_height_mm=3000)
    # columns=8 rows=7 cabinet_count=56 actual=5120x3360mm modules=336
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from src.models.product import CanonicalModel, parse_resolution

logger = logging.getLogger(__name__)

# 型号索引缓存（避免每次计算都重新读 JSON）
_model_cache: Optional[Dict[str, CanonicalModel]] = None
_model_cache_dir: Optional[str] = None


def _get_models(data_dir: Optional[str] = None) -> Dict[str, CanonicalModel]:
    """加载 Model 索引（带缓存）。"""
    global _model_cache, _model_cache_dir
    if data_dir is None:
        from src.config import config

        data_dir = config.DATA_DIR
    if _model_cache is None or _model_cache_dir != data_dir:
        from src.rag.json_loader import canonical_model_index

        _model_cache = canonical_model_index(data_dir)
        _model_cache_dir = data_dir
    return _model_cache


def reset_cache() -> None:
    """清空型号缓存（测试或数据更新后使用）。"""
    global _model_cache, _model_cache_dir
    _model_cache = None
    _model_cache_dir = None


def get_model(model: str, data_dir: Optional[str] = None) -> CanonicalModel:
    """按型号取产品档案；不存在时抛出 ``KeyError``（不猜测、不近似）。"""
    models = _get_models(data_dir)
    if model not in models:
        raise KeyError(
            f"未知型号 {model}；可用型号示例: {', '.join(sorted(models)[:5])} ..."
        )
    return models[model]


def calculate_screen(
    model: str,
    target_width_mm: float,
    target_height_mm: float,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """计算箱体 / 模组方案。

    Args:
        model: 实际销售型号，如 ``TW21-3216-P2.5``
        target_width_mm: 目标宽度（毫米）
        target_height_mm: 目标高度（毫米）

    Returns:
        dict，含列数、行数、箱体总数、实际尺寸、模组数量、分辨率、面积等。
    """
    product = get_model(model, data_dir)

    try:
        target_w = float(target_width_mm)
        target_h = float(target_height_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"目标尺寸必须是数字（毫米）: {target_width_mm} x {target_height_mm}") from exc
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"目标尺寸必须为正数，收到 {target_w} x {target_h}")

    cabinet_w = product.cabinet_width_mm
    cabinet_h = product.cabinet_height_mm
    if not cabinet_w or not cabinet_h:
        raise ValueError(f"{model} 缺少箱体尺寸，无法计算")

    # 箱体排列：向上取整，保证覆盖客户目标尺寸
    columns = math.ceil(round(target_w / cabinet_w, 6))
    rows = math.ceil(round(target_h / cabinet_h, 6))
    cabinet_count = columns * rows

    actual_w = columns * cabinet_w
    actual_h = rows * cabinet_h

    total_modules = cabinet_count * product.modules_per_cabinet
    area_sqm = round(actual_w * actual_h / 1_000_000, 3)

    # 实际分辨率：以物理尺寸 ÷ 点间距 推导（不依赖可能不自洽的 cabinet_resolution）
    pitch = product.pixel_pitch_mm
    horizontal_px = int(round(actual_w / pitch)) if pitch else None
    vertical_px = int(round(actual_h / pitch)) if pitch else None

    cabinet_px = parse_resolution(product.cabinet_resolution)
    module_px = parse_resolution(product.module_resolution)

    result: Dict[str, Any] = {
        "model": product.model,
        "series_id": product.series_id,
        "pixel_pitch_mm": pitch,
        "brightness_nit": product.brightness_nit,
        "environment": "outdoor" if product.outdoor else "indoor",
        "installation": product.installation,
        # 目标
        "target_width_mm": round(target_w, 3),
        "target_height_mm": round(target_h, 3),
        # 箱体
        "cabinet_width_mm": cabinet_w,
        "cabinet_height_mm": cabinet_h,
        "columns": columns,
        "rows": rows,
        "cabinet_count": cabinet_count,
        # 实际尺寸
        "actual_width_mm": round(actual_w, 3),
        "actual_height_mm": round(actual_h, 3),
        "actual_width_m": round(actual_w / 1000, 3),
        "actual_height_m": round(actual_h / 1000, 3),
        "area_sqm": area_sqm,
        # 模组
        "module_width_mm": product.module_width_mm,
        "module_height_mm": product.module_height_mm,
        "modules_per_cabinet": product.modules_per_cabinet,
        "total_modules": total_modules,
        # 分辨率
        "resolution_px": [horizontal_px, vertical_px] if horizontal_px else None,
        "cabinet_resolution": product.cabinet_resolution,
        "module_resolution": product.module_resolution,
        # 与目标的偏差
        "width_extra_mm": round(actual_w - target_w, 3),
        "height_extra_mm": round(actual_h - target_h, 3),
    }
    if cabinet_px:
        result["cabinet_resolution_px"] = list(cabinet_px)
    if module_px:
        result["module_resolution_px"] = list(module_px)

    logger.info(
        "calculate_screen(%s, %sx%s) → %dx%d cabinets, actual %sx%smm, %d modules",
        model, target_w, target_h, columns, rows, actual_w, actual_h, total_modules,
    )
    return result


def format_screen_spec(calc: Dict[str, Any]) -> str:
    """把计算结果格式化成可直接写进销售回复的英文短句。"""
    if not calc:
        return ""
    resolution = calc.get("resolution_px")
    resolution_text = (
        f"{resolution[0]}x{resolution[1]}px" if resolution and resolution[0] else "n/a"
    )
    return (
        f"{calc['model']}: {calc['columns']}x{calc['rows']} = {calc['cabinet_count']} cabinets, "
        f"actual size {calc['actual_width_m']}m x {calc['actual_height_m']}m "
        f"({calc['area_sqm']} sqm), {calc['total_modules']} modules, "
        f"resolution {resolution_text}"
    )


def estimate_screen(
    model: str,
    width_m: float,
    height_m: float,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """便捷入口：目标尺寸用米传入。"""
    return calculate_screen(model, width_m * 1000, height_m * 1000, data_dir=data_dir)


__all__ = [
    "calculate_screen",
    "estimate_screen",
    "format_screen_spec",
    "get_model",
    "reset_cache",
]
