"""v2.7：TurnBuilder 已更名为 :mod:`src.input.turn_manager`（Turn Manager）。

计划 §6/§47：Turn 聚合只能有一套实现 —— 这里只做兼容转发，
真正的实现在 ``turn_manager.TurnManager``。
"""
from __future__ import annotations

from .turn_manager import (  # noqa: F401
    DEFAULT_GRACE_MS,
    DEFAULT_MAX_WINDOW_MS,
    TurnBuffer,
    TurnBuilder,
    TurnManager,
    turn_grace_seconds,
    turn_max_window_seconds,
)

__all__ = [
    "DEFAULT_GRACE_MS",
    "DEFAULT_MAX_WINDOW_MS",
    "TurnBuffer",
    "TurnBuilder",
    "TurnManager",
    "turn_grace_seconds",
    "turn_max_window_seconds",
]
