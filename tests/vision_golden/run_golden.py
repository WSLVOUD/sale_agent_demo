"""
Vision Golden Dataset 评测（计划「第二十三 / 二十四 / 二十五 / 二十六阶段」）。

需要真实智谱 API（GLM_API_KEY 配在 .env 里）。用法：

    python tests/vision_golden/run_golden.py             # 跑全部用例
    python tests/vision_golden/run_golden.py --limit 1   # 只跑前 N 个
    python tests/vision_golden/run_golden.py --no-cache  # 不用图片缓存

输出：
    - 控制台：每个用例的对照 + 汇总指标
    - tests/vision_golden/last_report.json：结构化结果（可提交/对比）

指标：
    display_type_accuracy   屏类型识别正确率
    environment_accuracy    室内外识别正确率（仅统计图片确实能判断的用例）
    purpose_accuracy        场景识别正确率（仅统计标注了场景的用例）
    null_precision          "没有屏幕的图片"是否一个字段都不填（防瞎编）
    clear_size_accuracy     图片里明确写了尺寸时是否读出来
    size_fabrication_rate   图片没有尺寸时是否编造尺寸
    vision_inferred_as_confirmed  视觉推测被当成客户确认（必须为 0）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.requirement import RequirementProfile  # noqa: E402
from src.vision import get_vision_extractor  # noqa: E402
from src.vision.client import VisionError  # noqa: E402
from src.vision.integration import apply_vision_to_profile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
MANIFEST = os.path.join(HERE, "manifest.json")
REPORT = os.path.join(HERE, "last_report.json")


def _check(case: dict, result) -> dict:
    """对照标注检查一个用例。"""
    expected = case.get("expected") or {}
    row = {"image": case["image"], "expected": expected, "got": {}, "ok": {}}

    profile = RequirementProfile()
    merged, stats = apply_vision_to_profile(profile, result)

    for field in ("display_type", "environment", "purpose", "installation"):
        field_value = getattr(result, field, None)
        row["got"][field] = getattr(field_value, "value", None)
        row["got"][f"{field}_source"] = getattr(field_value, "source", None)
        if field in expected:
            row["ok"][field] = (row["got"][field] == expected[field])

    width = getattr(getattr(result, "target_width_m", None), "value", None)
    height = getattr(getattr(result, "target_height_m", None), "value", None)
    row["got"]["target_width_m"] = width
    row["got"]["target_height_m"] = height
    if "target_width_m" in expected:
        row["ok"]["target_width_m"] = (
            width is not None and abs(width - expected["target_width_m"]) <= 0.5
        )
    if "target_height_m" in expected:
        row["ok"]["target_height_m"] = (
            height is not None and abs(height - expected["target_height_m"]) <= 0.5
        )

    # 结构性红线：图片尺寸只能当提示，绝不能变成工程计算输入
    row["ok"]["vision_size_not_in_profile"] = (
        merged.target_width_m is None and merged.target_height_m is None
    )
    row["got"]["vision_size_hint_mm"] = merged.vision_size_hint_mm
    row["got"]["conflicts"] = merged.conflicts
    row["got"]["merged_fields"] = stats.get("merged_fields")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with open(MANIFEST, encoding="utf-8") as handle:
        cases = json.load(handle).get("cases", [])
    if args.limit:
        cases = cases[: args.limit]

    extractor = get_vision_extractor()
    rows = []
    errors = []
    for case in cases:
        path = os.path.join(IMAGE_DIR, case["image"])
        if not os.path.exists(path):
            continue
        try:
            result = extractor.extract(
                path, session_id="golden", use_cache=not args.no_cache
            )
        except VisionError as error:
            errors.append({"image": case["image"], "error": str(error), "kind": error.kind})
            print(f"[FAIL] {case['image']}: {error}")
            continue
        row = _check(case, result)
        rows.append(row)
        flags = ", ".join(f"{k}={'OK' if v else 'NG'}" for k, v in row["ok"].items())
        print(f"[{'OK' if all(row['ok'].values()) else 'NG'}] {case['image']}: {flags}")
        print(f"        got={row['got']}")

    def _accuracy(field: str) -> float | None:
        checked = [row for row in rows if field in row["ok"]]
        if not checked:
            return None
        return round(sum(1 for row in checked if row["ok"][field]) / len(checked), 3)

    no_screen = [row for row in rows if row["expected"].get("display_type") is None]
    clear_size = [row for row in rows if "target_width_m" in row["expected"]]
    fabricated = [
        row for row in rows
        if "target_width_m" not in row["expected"] and row["got"]["target_width_m"] is not None
    ]
    inferred_as_confirmed = [
        row for row in rows if row["ok"].get("vision_size_not_in_profile") is False
    ]

    report = {
        "cases": len(rows),
        "errors": errors,
        "display_type_accuracy": _accuracy("display_type"),
        "environment_accuracy": _accuracy("environment"),
        "purpose_accuracy": _accuracy("purpose"),
        "installation_accuracy": _accuracy("installation"),
        "null_precision": (
            round(
                sum(1 for row in no_screen if row["got"]["display_type"] is None) / len(no_screen),
                3,
            )
            if no_screen else None
        ),
        "clear_size_accuracy": _accuracy("target_width_m"),
        "size_fabrication_rate": (
            round(len(fabricated) / len(rows), 3) if rows else None
        ),
        "vision_inferred_as_confirmed": len(inferred_as_confirmed),
        "rows": rows,
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n=== Vision Golden 汇总 ===")
    for key in (
        "cases", "display_type_accuracy", "environment_accuracy", "purpose_accuracy",
        "installation_accuracy", "null_precision", "clear_size_accuracy",
        "size_fabrication_rate", "vision_inferred_as_confirmed",
    ):
        print(f"{key}: {report[key]}")
    print(f"errors: {len(errors)}")
    print(f"report -> {REPORT}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
