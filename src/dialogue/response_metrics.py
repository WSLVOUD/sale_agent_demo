"""回复链路的可观测计数（计划 v2.9 §二十一 Phase 13）。

要在删除旧模板链之前先有数据，所以把这几类轮次记下来：

    total_turns              所有回复轮次
    llm_native_turns         LLM 原生生成（正常路径）
    repair_turns             校验失败 → LLM 重写一次后通过
    structured_fallback_turns 结构化兜底（LLM 不可用或重写仍不合格）
    legacy_reply_turns       旧模板链路（script_generator 的 fallback 标记）
    validator_failures       校验判不合格的次数（含重写前后）

线程安全（uvicorn 多线程下也会被调用）；`snapshot()` 给 /health 与日志用。
"""
from __future__ import annotations

import threading
from typing import Any, Dict

_COUNTERS = (
    "total_turns",
    "llm_native_turns",
    "repair_turns",
    "structured_fallback_turns",
    "legacy_reply_turns",
    "validator_failures",
)

_lock = threading.RLock()
_counts: Dict[str, int] = {name: 0 for name in _COUNTERS}


def record(name: str, *, amount: int = 1) -> None:
    """记一次（未知名字直接忽略，避免污染指标）。"""
    key = str(name or "").strip()
    if key not in _counts:
        return
    with _lock:
        _counts[key] = _counts.get(key, 0) + int(amount)


def snapshot() -> Dict[str, Any]:
    """当前计数快照 + 派生比率（便于判断旧链路是否接近 0）。"""
    with _lock:
        data = dict(_counts)
    total = data.get("total_turns", 0) or 0
    data["legacy_reply_ratio"] = (
        round(data.get("legacy_reply_turns", 0) / total, 4) if total else 0.0
    )
    data["structures"] = {
        "llm_native": "正常路径（计划 v2.9 §十九）",
        "repair": "校验失败后 LLM 重写一次",
        "structured_fallback": "LLM 不可用 / 重写仍不合格",
        "legacy_reply": "旧模板链路，目标长期接近 0",
    }
    return data


def reset() -> None:
    """清空计数（测试用）。"""
    with _lock:
        for name in _COUNTERS:
            _counts[name] = 0


__all__ = ["reset", "record", "snapshot"]
