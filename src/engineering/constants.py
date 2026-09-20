"""v2.3 §4：Engineering Rule 的**唯一**常量来源。

以前距离 / 点间距 / 亮度这几个工程规则的常量分散在 `parameter_inference.py` 里，
容易出现"同一输入、不同结果"。现在所有工程常量集中在本模块，其它模块只允许
从这里 import（`src/engineering/` 之外不再重复定义规则表）。
"""
from __future__ import annotations

from typing import Dict, Tuple

# ── 点间距规则 ──────────────────────────────────────────────────────────────
DEFAULT_PITCH_TOLERANCE = 0.5

# 观看距离 → 推荐点间距区间（通用距离表；计划文档示例：5m → 2.5~3.0mm）
VIEWING_DISTANCE_PITCH_TABLE: Tuple[Tuple[float, float, float], ...] = (
    (2.0, 0.6, 1.5),      # < 2m
    (4.0, 0.9, 2.0),      # 2–4m
    (8.0, 1.5, 3.0),      # 4–8m
    (15.0, 2.5, 5.0),     # 8–15m
    (30.0, 4.0, 8.0),     # 15–30m
    (float("inf"), 6.0, 10.0),  # ≥30m
)

# 环境 + 距离 → 点间距（业务规则，优先于通用距离表）
# 室内：≤3m → P2.5 及以下；>3m → P3 及以上；>30m → P10
INDOOR_PITCH_TABLE: Tuple[Tuple[float, float, float, float], ...] = (
    (3.0, 0.6, 2.5, 2.5),
    (30.0, 3.0, 10.0, 3.0),
    (float("inf"), 8.0, 10.0, 10.0),
)
# 室外：≤4m → P4；4–5m → P4/P5；6–20m → P5；20–25m → P6.67；25–30m → P8；>30m → P10
OUTDOOR_PITCH_TABLE: Tuple[Tuple[float, float, float, float], ...] = (
    (4.0, 3.9, 5.0, 4.0),
    (5.0, 3.9, 5.5, 4.5),
    (20.0, 4.5, 6.7, 5.0),
    (25.0, 5.0, 8.0, 6.7),
    (30.0, 8.0, 10.0, 8.0),
    (float("inf"), 8.0, 10.0, 10.0),
)

# 点间距 ↔ 观看距离（行业口径）：1mm 点间距最佳观看距离约 3m，
# 再远（>5m/mm）就吃力；最近观看距离不能小于 1m/mm（否则看到像素结构）。
PITCH_FAR_LIMIT_M_PER_MM = 5.0
PITCH_OPTIMAL_M_PER_MM = 3.0
PITCH_NEAR_LIMIT_M_PER_MM = 1.0

# 连观看距离都推不出来时的兜底档（按环境给保守区间，偏粗不偏细）
FALLBACK_PITCH_BAND: Dict[str, Tuple[float, float, float]] = {
    "indoor": (2.5, 4.0, 3.0),
    "semi_outdoor": (4.0, 6.0, 5.0),
    "outdoor": (5.0, 8.0, 5.0),
}
DEFAULT_FALLBACK_PITCH_BAND: Tuple[float, float, float] = (2.5, 5.0, 3.0)

# ── 场地方案 → 观看距离（物理公式，不是场景枚举）────────────────────────────
SEAT_WIDTH_M = 0.6            # 每个观众沿屏宽方向占的座位宽度（含扶手/间距）
SEAT_ROW_DEPTH_M = 0.9        # 每排座椅纵深
FRONT_OFFSET_M = 2.5          # 第一排到屏面的留距（走道 / 视线）
SCREEN_NEAR_FACTOR = 1.5      # 最近观众 ≈ 1.5 × 屏高
SCREEN_FAR_FACTOR = 3.0       # 最远观众 ≈ 3 × 屏高

# ── 亮度 ────────────────────────────────────────────────────────────────────
BRIGHTNESS_BY_ENVIRONMENT: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "outdoor": (4500, None),
    "semi_outdoor": (800, None),
    "indoor": (400, 800),
}

# ── 屏幕尺寸参考（LCD / IFP 场景）────────────────────────────────────────────
SCREEN_SIZE_BY_DISTANCE: Tuple[Tuple[float, str], ...] = (
    (3.0, "65-75英寸"),
    (4.0, "75-86英寸"),
    (6.0, "86-98英寸"),
    (float("inf"), "98英寸以上"),
)

# ── v2.5：分辨率"接近程度"阈值（集中在这里，后续用 Golden Dataset 调）───────
# 客户口径：不要写死 `if deviation < 5%: PASS`，也不要求实际分辨率与目标一模一样。
RESOLUTION_FIT_NEAR = 0.02             # ≤2% 视为 NEAR_MATCH
RESOLUTION_FIT_ACCEPTABLE = 0.08       # ≤8% 视为 ACCEPTABLE
RESOLUTION_FIT_NOT_ACCEPTABLE = 0.20   # ≤20% 视为 NOT_ACCEPTABLE，超过即 IMPOSSIBLE
ASPECT_DEVIATION_ACCEPTABLE = 0.03     # 比例偏差超过 3% 认为"比例冲突"需澄清

# 客户授权 AI 决定尺寸（DELEGATED）时的确定性参考尺寸：
# height = clamp(distance / 3, 1.0m, 12.0m)，width = height × 16/9
DELEGATED_SIZE_MIN_H_M = 1.0
DELEGATED_SIZE_MAX_H_M = 12.0
DELEGATED_SIZE_ASPECT = 16 / 9
DELEGATED_SIZE_DISTANCE_FACTOR = 3.0


def __getattr__(name: str):  # pragma: no cover - 兼容旧的私有名
    """兼容历史私有名（`_INDOOR_PITCH_TABLE` 等）—— 只做别名，不复制数据。"""
    aliases = {
        "_INDOOR_PITCH_TABLE": INDOOR_PITCH_TABLE,
        "_OUTDOOR_PITCH_TABLE": OUTDOOR_PITCH_TABLE,
    }
    if name in aliases:
        return aliases[name]
    raise AttributeError(name)


__all__ = [
    "BRIGHTNESS_BY_ENVIRONMENT",
    "DEFAULT_FALLBACK_PITCH_BAND",
    "DEFAULT_PITCH_TOLERANCE",
    "DELEGATED_SIZE_ASPECT",
    "DELEGATED_SIZE_DISTANCE_FACTOR",
    "DELEGATED_SIZE_MAX_H_M",
    "DELEGATED_SIZE_MIN_H_M",
    "FALLBACK_PITCH_BAND",
    "FRONT_OFFSET_M",
    "INDOOR_PITCH_TABLE",
    "OUTDOOR_PITCH_TABLE",
    "PITCH_FAR_LIMIT_M_PER_MM",
    "PITCH_NEAR_LIMIT_M_PER_MM",
    "PITCH_OPTIMAL_M_PER_MM",
    "SCREEN_FAR_FACTOR",
    "SCREEN_NEAR_FACTOR",
    "SCREEN_SIZE_BY_DISTANCE",
    "RESOLUTION_FIT_ACCEPTABLE",
    "RESOLUTION_FIT_NEAR",
    "RESOLUTION_FIT_NOT_ACCEPTABLE",
    "ASPECT_DEVIATION_ACCEPTABLE",
    "SEAT_ROW_DEPTH_M",
    "SEAT_WIDTH_M",
    "VIEWING_DISTANCE_PITCH_TABLE",
]
