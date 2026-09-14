"""
Phase 0/9：箱体 / 模组工程计算评估。

验证目标（Phase 9 交付物）::

    columns        = ceil(target_width / cabinet_width)
    rows           = ceil(target_height / cabinet_height)
    cabinet_count  = columns * rows
    actual_size    = (columns * cabinet_width, rows * cabinet_height)
    total_modules  = cabinet_count * modules_per_cabinet

覆盖率要求：整除 / 不整除 / 边界 / 超大尺寸 / 不同 Cabinet / 不同 Module 全部正确。
目标 Calculator Accuracy = 100%。

在 Phase 9 之前 ``src/tools/screen_calculator.py`` 不存在，本脚本会输出
``status: pending_phase_9`` 并以退出码 0 结束（不阻塞 Phase 0 基线）。

用法::

    python -m eval.calculator_eval
    python -m eval.calculator_eval --out eval/reports/calculator_report.json
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.metrics import write_report  # noqa: E402

TOLERANCE_MM = 1e-6

# ── Golden Calculator Cases ──────────────────────────────────────────────────
# 覆盖：整除 / 不整除 / 小尺寸 / 超大尺寸 / 不同 cabinet / 不同 module / 小数 cabinet
CASES: List[Dict[str, Any]] = [
    {
        "id": "c001", "model": "TW21-3216-P2.5", "target_width_mm": 5000, "target_height_mm": 3000,
        "expected": {"columns": 8, "rows": 7, "cabinet_count": 56, "actual_width_mm": 5120,
                     "actual_height_mm": 3360, "modules_per_cabinet": 6, "total_modules": 336},
        "note": "计划文档示例：5m x 3m",
    },
    {
        "id": "c002", "model": "TW21-3216-P2.5", "target_width_mm": 10000, "target_height_mm": 6000,
        "expected": {"columns": 16, "rows": 13, "cabinet_count": 208, "actual_width_mm": 10240,
                     "actual_height_mm": 6240, "modules_per_cabinet": 6, "total_modules": 1248},
        "note": "10m x 6m",
    },
    {
        "id": "c003", "model": "TW11-3216-P2.5", "target_width_mm": 3000, "target_height_mm": 2000,
        "expected": {"columns": 5, "rows": 5, "cabinet_count": 25, "actual_width_mm": 3200,
                     "actual_height_mm": 2400, "modules_per_cabinet": 6, "total_modules": 150},
        "note": "3m x 2m",
    },
    {
        "id": "c004", "model": "TW21-3216-P2.5", "target_width_mm": 6400, "target_height_mm": 4800,
        "expected": {"columns": 10, "rows": 10, "cabinet_count": 100, "actual_width_mm": 6400,
                     "actual_height_mm": 4800, "modules_per_cabinet": 6, "total_modules": 600},
        "note": "刚好整除",
    },
    {
        "id": "c005", "model": "TW11-3216-P1.2", "target_width_mm": 2000, "target_height_mm": 1500,
        "expected": {"columns": 4, "rows": 4, "cabinet_count": 16, "actual_width_mm": 2560,
                     "actual_height_mm": 1920, "modules_per_cabinet": 6, "total_modules": 96},
        "note": "小尺寸",
    },
    {
        "id": "c006", "model": "TW31-COB-P1.2H", "target_width_mm": 4000, "target_height_mm": 2000,
        "expected": {"columns": 7, "rows": 6, "cabinet_count": 42, "actual_width_mm": 4200,
                     "actual_height_mm": 2025, "modules_per_cabinet": 8, "total_modules": 336},
        "note": "小数 cabinet（337.5mm）",
    },
    {
        "id": "c007", "model": "TW31-COB-P0.7H", "target_width_mm": 1200, "target_height_mm": 900,
        "expected": {"columns": 2, "rows": 3, "cabinet_count": 6, "actual_width_mm": 1200,
                     "actual_height_mm": 1012.5, "modules_per_cabinet": 8, "total_modules": 48},
        "note": "横向整除 / 纵向不整除",
    },
    {
        "id": "c008", "model": "TW11-IR-P2.6", "target_width_mm": 3000, "target_height_mm": 3000,
        "expected": {"columns": 6, "rows": 6, "cabinet_count": 36, "actual_width_mm": 3000,
                     "actual_height_mm": 3000, "modules_per_cabinet": 4, "total_modules": 144},
        "note": "租赁 500x500 cabinet",
    },
    {
        "id": "c009", "model": "TW21-IRHD-P2.9H", "target_width_mm": 6000, "target_height_mm": 4000,
        "expected": {"columns": 12, "rows": 8, "cabinet_count": 96, "actual_width_mm": 6000,
                     "actual_height_mm": 4000, "modules_per_cabinet": 4, "total_modules": 384},
        "note": "租赁 6m x 4m",
    },
    {
        "id": "c010", "model": "TW11-OD-P6", "target_width_mm": 8000, "target_height_mm": 4000,
        "expected": {"columns": 9, "rows": 5, "cabinet_count": 45, "actual_width_mm": 8640,
                     "actual_height_mm": 4800, "modules_per_cabinet": 18, "total_modules": 810},
        "note": "户外 960x960 cabinet + 320x160 module",
    },
    {
        "id": "c011", "model": "TW31-HOD-P5.7E", "target_width_mm": 12000, "target_height_mm": 6000,
        "expected": {"columns": 13, "rows": 7, "cabinet_count": 91, "actual_width_mm": 12480,
                     "actual_height_mm": 6720, "modules_per_cabinet": 6, "total_modules": 546},
        "note": "户外高亮 480x320 module",
    },
    {
        "id": "c012", "model": "TW11-OD-P2.5", "target_width_mm": 2000, "target_height_mm": 1000,
        "expected": {"columns": 3, "rows": 2, "cabinet_count": 6, "actual_width_mm": 2880,
                     "actual_height_mm": 1920, "modules_per_cabinet": 18, "total_modules": 108},
        "note": "目标尺寸远小于单个箱体",
    },
    {
        "id": "c013", "model": "TW21-3216-P4.0", "target_width_mm": 7500, "target_height_mm": 4000,
        "expected": {"columns": 12, "rows": 9, "cabinet_count": 108, "actual_width_mm": 7680,
                     "actual_height_mm": 4320, "modules_per_cabinet": 6, "total_modules": 648},
        "note": "7.5m x 4m 不整除",
    },
    {
        "id": "c014", "model": "TW21-3216-P2.5", "target_width_mm": 20000, "target_height_mm": 12000,
        "expected": {"columns": 32, "rows": 25, "cabinet_count": 800, "actual_width_mm": 20480,
                     "actual_height_mm": 12000, "modules_per_cabinet": 6, "total_modules": 4800},
        "note": "超大尺寸",
    },
]

COMPARED_FIELDS = (
    "columns", "rows", "cabinet_count",
    "actual_width_mm", "actual_height_mm",
    "modules_per_cabinet", "total_modules",
)


def run_cases() -> Dict[str, Any]:
    """执行全部计算用例。"""
    import importlib.util

    if importlib.util.find_spec("src.tools.screen_calculator") is None:
        return {
            "status": "pending_phase_9",
            "reason": "src/tools/screen_calculator.py 尚未实现（计划 Phase 9 交付）",
            "cases": len(CASES),
            "passed": 0,
            "accuracy": None,
            "details": [],
        }

    try:
        from src.tools.screen_calculator import calculate_screen  # type: ignore
    except Exception as exc:
        return {
            "status": "import_error",
            "reason": f"screen_calculator 导入失败: {exc}",
            "cases": len(CASES),
            "passed": 0,
            "accuracy": None,
            "details": [],
        }

    details: List[Dict[str, Any]] = []
    passed = 0
    for case in CASES:
        try:
            actual = calculate_screen(
                model=case["model"],
                target_width_mm=case["target_width_mm"],
                target_height_mm=case["target_height_mm"],
            )
        except Exception as exc:
            details.append({"id": case["id"], "passed": False, "error": str(exc)})
            continue

        mismatches: Dict[str, Any] = {}
        for field in COMPARED_FIELDS:
            expected_value = case["expected"][field]
            actual_value = actual.get(field)
            if actual_value is None:
                mismatches[field] = {"expected": expected_value, "actual": None}
                continue
            if abs(float(actual_value) - float(expected_value)) > TOLERANCE_MM:
                mismatches[field] = {"expected": expected_value, "actual": actual_value}

        ok = not mismatches
        passed += 1 if ok else 0
        details.append({
            "id": case["id"],
            "model": case["model"],
            "target": [case["target_width_mm"], case["target_height_mm"]],
            "passed": ok,
            "mismatches": mismatches,
            "note": case.get("note", ""),
        })

    return {
        "status": "ok",
        "cases": len(CASES),
        "passed": passed,
        "accuracy": round(passed / len(CASES), 4) if CASES else 0.0,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0/9 箱体模组计算评估")
    parser.add_argument("--out", type=str, default="eval/reports/calculator_report.json")
    args = parser.parse_args()

    outcome = run_cases()

    report = {
        "report": "calculator_eval",
        "phase": "Phase 0 (占位) / Phase 9 (正式)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": outcome["status"],
        "summary": {k: v for k, v in outcome.items() if k != "details"},
        "cases": outcome.get("details", []),
    }
    path = write_report(report, args.out)

    if outcome["status"] != "ok":
        print(f"计算器评估: {outcome['status']} - {outcome.get('reason', '')}")
        print(f"已准备 {outcome['cases']} 条计算用例，Phase 9 落地后自动生效")
    else:
        print(f"Calculator Accuracy: {outcome['accuracy']} ({outcome['passed']}/{outcome['cases']})")
        for item in outcome["details"]:
            if not item["passed"]:
                print(f"  FAIL {item['id']}: {item.get('mismatches') or item.get('error')}")
    print(f"报告已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
