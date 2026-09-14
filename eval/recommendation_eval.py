"""
Phase 0：需求理解与推荐基线评估。

分三层评估，互不依赖，可单独运行：

  1. slots  —— Requirement Slot Accuracy
              用当前生产的规则提取器（ParameterInference）解析 Query，
              与 Golden Dataset 的 ``slots`` 逐字段比较，并给出逐槽位命中率。

  2. route  —— 三层路由准确率
              用 ``classify_complexity`` 判断 fast / normal / agent 是否与期望一致。

  3. recommend —— Top-1 / Top-3 推荐准确率（可选）
              需要 Phase 8 的 ``src.rag.recommendation_engine``；
              若尚未实现则标记为 skipped，不计入总分。

可选 ``--agent``：调用完整 Solution Agent（需要 LLM 网络），端到端测 Top-1/Top-3。

用法::

    python -m eval.recommendation_eval
    python -m eval.recommendation_eval --limit 20
    python -m eval.recommendation_eval --out eval/reports/recommendation_baseline.json
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.metrics import (  # noqa: E402
    load_golden_cases,
    mean,
    slot_match_rate,
    write_report,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval.recommendation")

# Golden Dataset 里参与 Slot 打分的字段（其余字段仅作记录）
SCORED_SLOTS = (
    "display_type",
    "environment",
    "installation",
    "purpose",
    "viewing_distance_m",
    "pixel_pitch_mm",
    "brightness_min",
)


def _predicted_slots(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """把生产规则提取结果映射到 Golden Dataset 的 slot 词汇表。"""
    slots: Dict[str, Any] = {}
    if constraints.get("display_type"):
        slots["display_type"] = constraints["display_type"]
    if constraints.get("environment"):
        slots["environment"] = constraints["environment"]
    elif constraints.get("indoor") is True:
        slots["environment"] = "indoor"
    elif constraints.get("outdoor") is True:
        slots["environment"] = "outdoor"
    if constraints.get("installation"):
        slots["installation"] = constraints["installation"]
    elif constraints.get("is_rental") is not None:
        slots["installation"] = "rental" if constraints["is_rental"] else "fixed"
    if constraints.get("purpose"):
        slots["purpose"] = constraints["purpose"]
    pitch = constraints.get("pixel_pitch_mm")
    if pitch is None:
        pitch = constraints.get("pixel_pitch")
    if pitch is not None:
        slots["pixel_pitch_mm"] = pitch
    if constraints.get("brightness_min") is not None:
        slots["brightness_min"] = constraints["brightness_min"]
    if constraints.get("viewing_distance_m") is not None:
        slots["viewing_distance_m"] = constraints["viewing_distance_m"]
    return slots


def _expected_slots(case: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (case.get("slots") or {}).items() if k in SCORED_SLOTS}


def _history_constraints(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从对话历史中累积约束，用于多轮用例的路由判断。"""
    from src.rag.parameter_inference import ParameterInference

    extractor = ParameterInference()
    merged: Dict[str, Any] = {}
    for msg in history or []:
        if msg.get("role") not in ("user", "human"):
            continue
        merged.update(extractor.extract_constraints(str(msg.get("content", ""))))
    return merged


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """评估单条用例的 slots / route。"""
    from src.rag.query_understanding import understand_query
    from src.rag.router import classify_complexity

    query = case["query"]
    history = case.get("history") or []

    # Phase 4：Query Understanding 是槽位提取的正式入口
    understanding = understand_query(query, history=history)
    predicted = _predicted_slots(understanding.slots)
    constraints = dict(understanding.slots)
    expected = _expected_slots(case)
    slot_rate, slot_details = slot_match_rate(expected, predicted)

    decision = classify_complexity(query, existing_requirements=constraints)
    predicted_route = decision.route.value
    route_ok = predicted_route == case.get("route")

    # 硬约束抓取率：Query 里明确写出的硬条件是否被提取出来
    hard = case.get("hard") or {}
    hard_checks: Dict[str, bool] = {}
    if hard.get("environment"):
        hard_checks["environment"] = predicted.get("environment") == hard["environment"]
    if hard.get("installation"):
        hard_checks["installation"] = predicted.get("installation") == hard["installation"]
    if hard.get("display_type"):
        hard_checks["display_type"] = predicted.get("display_type") == hard["display_type"]
    if hard.get("brightness_min") is not None:
        value = predicted.get("brightness_min")
        hard_checks["brightness_min"] = value is not None and value >= hard["brightness_min"] * 0.8
    if hard.get("pixel_pitch_min") is not None or hard.get("pixel_pitch_max") is not None:
        value = predicted.get("pixel_pitch_mm")
        if value is None:
            hard_checks["pixel_pitch"] = False
        else:
            low = hard.get("pixel_pitch_min")
            high = hard.get("pixel_pitch_max")
            hard_checks["pixel_pitch"] = (
                (low is None or value >= low - 0.2) and (high is None or value <= high + 0.2)
            )

    return {
        "id": case["id"],
        "query": query,
        "tags": case.get("tags", []),
        "expected_slots": expected,
        "predicted_slots": predicted,
        "retrieval_query": understanding.retrieval_query,
        "language": understanding.language,
        "profile_completeness": (
            understanding.profile.completeness() if understanding.profile is not None else None
        ),
        "profile_missing": (
            understanding.profile.missing_slots() if understanding.profile is not None else None
        ),
        "profile_sufficient": (
            understanding.profile.is_sufficient() if understanding.profile is not None else None
        ),
        "slot_match_rate": round(slot_rate, 4),
        "slot_details": slot_details,
        "expected_route": case.get("route"),
        "predicted_route": predicted_route,
        "route_correct": route_ok,
        "route_reason": decision.reason,
        "hard_checks": hard_checks,
        "hard_capture_rate": (
            round(sum(hard_checks.values()) / len(hard_checks), 4) if hard_checks else None
        ),
    }


def evaluate_with_recommendation_engine(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """用统一 RecommendationService（Gate → Engine）测 Top-1 / Top-3。

    符合排查计划 v1.0：推荐必须先过 Recommendation Ready Gate。
    返回 NEED_CLARIFICATION 的用例不计入推荐准确率，单独统计为"需追问"。
    """
    try:
        from src.rag.recommendation_service import RecommendationService
    except Exception:
        return {
            "status": "skipped",
            "reason": "src.rag.recommendation_service 尚未实现",
            "cases": 0,
            "top1_accuracy": None,
            "top3_coverage": None,
            "hard_constraint_violation_rate": None,
        }

    service = RecommendationService()
    top1_hits: List[float] = []
    top3_hits: List[float] = []
    pitch_band_hits: List[float] = []
    violations: List[int] = []
    details: List[Dict[str, Any]] = []
    needs_clarification: List[str] = []

    for case in cases:
        expected_models = case.get("models") or []
        if not expected_models:
            continue
        try:
            from src.rag.query_understanding import understand_query

            profile = understand_query(
                case["query"], history=case.get("history") or []
            ).profile
            result = service.recommend(profile=profile, top_k=3)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.error("[%s] 推荐引擎失败: %s", case["id"], exc)
            continue
        if result.get("recommendation_status") == "NEED_CLARIFICATION":
            needs_clarification.append(case["id"])
            details.append({
                "id": case["id"],
                "expected_models": expected_models,
                "predicted_models": [],
                "status": "NEED_CLARIFICATION",
                "missing_fields": result.get("missing_fields", []),
            })
            continue
        models = [r.get("model") for r in (result.get("recommendations") or [])]
        top1_hits.append(1.0 if models[:1] and models[0] in expected_models else 0.0)
        top3_hits.append(1.0 if set(models[:3]) & set(expected_models) else 0.0)

        # 点间距区间命中：Top-3 里是否有型号落在期望区间（软条件的达成率）
        hard = case.get("hard") or {}
        low, high = hard.get("pixel_pitch_min"), hard.get("pixel_pitch_max")
        if low is not None or high is not None:
            pitches = [
                r.get("pixel_pitch_mm") for r in (result.get("recommendations") or [])[:3]
            ]
            pitch_band_hits.append(
                1.0 if any(
                    p is not None
                    and (low is None or p >= low - 1e-6)
                    and (high is None or p <= high + 1e-6)
                    for p in pitches
                ) else 0.0
            )
        violations.append(len(result.get("violations") or []))
        details.append({
            "id": case["id"],
            "expected_models": expected_models,
            "predicted_models": models[:3],
        })

    return {
        "status": "ok",
        "cases": len(top1_hits),
        "needs_clarification_count": len(needs_clarification),
        "needs_clarification": needs_clarification,
        "top1_accuracy": round(mean(top1_hits), 4),
        "top3_coverage": round(mean(top3_hits), 4),
        "pitch_band_coverage_at_3": (
            round(mean(pitch_band_hits), 4) if pitch_band_hits else None
        ),
        "hard_constraint_violation_rate": round(
            (sum(1 for v in violations if v > 0) / len(violations)) if violations else 0.0, 4
        ),
        "cases_detail": details,
    }


def evaluate_with_agent(cases: List[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
    """端到端：调用 Solution Agent（需要 LLM）。"""
    from src.agents.solution.runner import SolutionAgentRunner
    from src.config import config
    from src.core.embeddings import load_vectorstore
    from src.rag.corpus import build_retrieval_corpus

    vectorstore = load_vectorstore(persist_dir=config.VECTORSTORE_DIR)
    documents = build_retrieval_corpus(config.DATA_DIR)
    runner = SolutionAgentRunner(vectorstore, documents)

    top1_hits: List[float] = []
    top3_hits: List[float] = []
    details: List[Dict[str, Any]] = []
    selected = [c for c in cases if c.get("models")][: limit] if limit else [c for c in cases if c.get("models")]

    for case in selected:
        try:
            result = runner.run(message=case["query"], history=case.get("history") or [])
        except Exception as exc:
            logger.error("[%s] Agent 失败: %s", case["id"], exc)
            details.append({"id": case["id"], "error": str(exc)})
            continue
        from eval.metrics import identifiers_from_items

        _, models = identifiers_from_items(result.get("products") or [])
        expected = case["models"]
        top1_hits.append(1.0 if models[:1] and models[0] in expected else 0.0)
        top3_hits.append(1.0 if set(models[:3]) & set(expected) else 0.0)
        details.append({
            "id": case["id"],
            "expected_models": expected,
            "predicted_models": models[:5],
            "route": result.get("route"),
        })

    return {
        "status": "ok" if top1_hits else "no_data",
        "cases": len(top1_hits),
        "top1_accuracy": round(mean(top1_hits), 4) if top1_hits else None,
        "top3_coverage": round(mean(top3_hits), 4) if top3_hits else None,
        "cases_detail": details,
    }


def summarize_slots(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总 Slot 指标，并给出逐槽位命中率。"""
    scored = [c for c in case_results if c["expected_slots"]]
    per_slot: Dict[str, List[bool]] = {}
    for case in scored:
        for key, value in case["slot_details"].items():
            per_slot.setdefault(key, []).append(bool(value))

    hard_rates = [c["hard_capture_rate"] for c in case_results if c["hard_capture_rate"] is not None]
    return {
        "cases_total": len(case_results),
        "cases_scored": len(scored),
        "requirement_slot_accuracy": round(mean([c["slot_match_rate"] for c in scored]), 4),
        "profile_completeness": round(
            mean([c["profile_completeness"] for c in case_results if c.get("profile_completeness") is not None]), 4
        ),
        "profile_sufficient_rate": round(
            mean([1.0 if c.get("profile_sufficient") else 0.0 for c in case_results
                  if c.get("profile_sufficient") is not None]), 4
        ),
        "per_slot_accuracy": {
            key: round(sum(values) / len(values), 4) for key, values in sorted(per_slot.items())
        },
        "hard_constraint_capture_rate": round(mean(hard_rates), 4) if hard_rates else None,
        "route_accuracy": round(
            mean([1.0 if c["route_correct"] else 0.0 for c in case_results]), 4
        ),
        "route_confusion": _route_confusion(case_results),
    }


def _route_confusion(case_results: List[Dict[str, Any]]) -> Dict[str, int]:
    confusion: Dict[str, int] = {}
    for case in case_results:
        key = f"{case['expected_route']}->{case['predicted_route']}"
        confusion[key] = confusion.get(key, 0) + 1
    return dict(sorted(confusion.items(), key=lambda kv: -kv[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 需求理解 / 推荐基线评估")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--out", type=str, default="eval/reports/recommendation_report.json")
    parser.add_argument("--agent", action="store_true", help="额外跑端到端 Solution Agent（需要 LLM）")
    args = parser.parse_args()

    cases = load_golden_cases(
        ids=args.ids.split(",") if args.ids else None,
        tags=args.tags.split(",") if args.tags else None,
        limit=args.limit,
    )
    if not cases:
        print("没有匹配的用例")
        return 1

    case_results = [evaluate_case(case) for case in cases]
    for case in case_results:
        print(
            f"{case['id']} slots={case['slot_match_rate']:.2f} "
            f"route={case['predicted_route']}({'ok' if case['route_correct'] else 'x'}) "
            f"hard_capture={case['hard_capture_rate']}"
        )

    summary = summarize_slots(case_results)

    engine_result = evaluate_with_recommendation_engine(cases)
    agent_result = None
    if args.agent:
        print("\n运行端到端 Solution Agent（需要 LLM 网络）...")
        agent_result = evaluate_with_agent(cases)

    report = {
        "report": "recommendation_eval",
        "phase": "Phase 0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "eval/golden_dataset.json",
        "summary": summary,
        "recommendation_engine": engine_result,
        "agent_end_to_end": agent_result,
        "cases": case_results,
    }
    path = write_report(report, args.out)

    print("\n===== 需求理解 / 路由基线 =====")
    print(f"  Requirement Slot Accuracy: {summary['requirement_slot_accuracy']}")
    print(f"  Hard Constraint Capture  : {summary['hard_constraint_capture_rate']}")
    print(f"  Route Accuracy           : {summary['route_accuracy']}")
    print(f"  逐槽位命中率: {summary['per_slot_accuracy']}")
    print(f"  路由混淆: {summary['route_confusion']}")
    print(f"  推荐引擎: {engine_result['status']} ({engine_result.get('reason', '')})")
    print(f"\n报告已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
