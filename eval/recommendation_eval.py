"""
Phase 0：需求理解与推荐基线评估。

分层评估，互不依赖，可单独运行：

  1. slots  —— Requirement Slot Accuracy
              用当前生产的槽位提取器（query_understanding / extract_slots）解析 Query，
              与 Golden Dataset 的 ``slots`` 逐字段比较，并给出逐槽位命中率。

  2. recommend —— Top-1 / Top-3 推荐准确率（可选）
              需要 Phase 8 的 ``src.rag.recommendation_engine``；
              若尚未实现则标记为 skipped，不计入总分。

注（整改计划 §六/§十七）：原来还有一项 ``route``——用 ``classify_complexity`` 比
fast / normal / agent 三层路由。那套路由在生产路径上**零调用**（只被本脚本使用），
而且里面还有第二套"关键词判 LCD / IFP"，与"IFP 判断必须统一"冲突，已随旧链一并删除。
所以这里不再评测 route，其余（slots / 硬约束 / 字段混淆 / 产品类型 / 推荐引擎）照常评测。

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


# 计划 update_v2.9.10 §11【1】：场景词表归一（中文场景词 ↔ 英文枚举）。
# 首选**系统自己的词表**（`purpose_normalizer.PURPOSE_MAPPING`，单一来源）；
# 下面只补系统词表还没收录的少数标签，避免"会议室"与 "conference" 被判成两个值。
_PURPOSE_FALLBACK = {
    "培训室": "classroom", "培训": "conference", "报告厅": "conference",
    "酒店大堂": "hotel", "室内大厅": "hall",
    "幕墙广告": "advertising", "楼体广告": "advertising", "户外广告": "advertising",
    "舞台": "stage", "演唱会": "concert",
}


def _canon_purpose(value: Any) -> Any:
    if value in (None, ""):
        return value
    text = str(value).strip()
    try:
        from src.core.purpose_normalizer import PURPOSE_MAPPING

        for key in (text, text.lower()):
            hit = PURPOSE_MAPPING.get(key)
            if hit is not None:
                return str(getattr(hit, "value", hit))
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("purpose mapping unavailable: %s", exc)
    return _PURPOSE_FALLBACK.get(text, text.lower())


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
        slots["purpose"] = _canon_purpose(constraints["purpose"])
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
    expected = {k: v for k, v in (case.get("slots") or {}).items() if k in SCORED_SLOTS}
    if "purpose" in expected:
        expected["purpose"] = _canon_purpose(expected["purpose"])
    return expected


def _product_type_check(query: str, expected: Dict[str, Any]) -> Dict[str, Any]:
    """产品类型路由准确率（计划 update_v2.9.10 §14/§15）。

    只对"客户明确点了类型"的用例判分：这种用例的正确答案是唯一的。
    没点名的用例按口径应当是 UNKNOWN 或"推断后要客户确认"，不在这里计分。
    """
    want = expected.get("display_type")
    if not want:
        return {"expected": None, "predicted": None, "correct": None}
    try:
        from src.dialogue.product_type_router import route_display_type

        decision = route_display_type(query)
        got = decision.display_type
        # 口径（v2.9.x 已确认）：IFP 是 LCD 的子类型，首层只出 LED / LCD。
        # 所以标注为 IFP 的用例，路由器给 LCD + subtype=IFP 即为正确。
        if str(want).upper() == "IFP" and got == "LCD" and decision.subtype == "IFP":
            got = "IFP"
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("product type route failed: %s", exc)
        got = None
    return {"expected": want, "predicted": got, "correct": got == want}


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """评估单条用例的 slots / 硬约束 / 产品类型（route 已随旧路由删除）。"""
    from src.rag.query_understanding import understand_query

    query = case["query"]
    history = case.get("history") or []

    # Phase 4：Query Understanding 是槽位提取的正式入口
    understanding = understand_query(query, history=history)
    predicted = _predicted_slots(understanding.slots)
    constraints = dict(understanding.slots)
    expected = _expected_slots(case)
    slot_rate, slot_details = slot_match_rate(expected, predicted)

    # 硬约束抓取率（计划 update_v2.9.10 第一/七阶段）：
    #   hard    = 客户**明确说的**约束 → 必须抓到
    #   derived = 场景/规则推断 + 系统默认 → 只做信息统计，不作为硬指标
    # （以前所有标签都塞在 hard 里，模型行为正确也会被判错 —— 这是 0.5923 的主因）
    hard = case.get("hard") or {}
    derived = case.get("derived") or {}
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

    # 推断/默认字段的抓取情况（只统计，不判对错）
    derived_checks: Dict[str, bool] = {}
    for key, value in derived.items():
        slot = "pixel_pitch_mm" if key.startswith("pixel_pitch") else key
        if key == "pixel_pitch_min" or key == "pixel_pitch_max":
            derived_checks.setdefault(key, predicted.get("pixel_pitch_mm") is not None)
            continue
        derived_checks[key] = predicted.get(slot) == value

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
        "hard_checks": hard_checks,
        "hard_capture_rate": (
            round(sum(hard_checks.values()) / len(hard_checks), 4) if hard_checks else None
        ),
        "derived_checks": derived_checks,
        "derived_capture_rate": (
            round(sum(derived_checks.values()) / len(derived_checks), 4)
            if derived_checks else None
        ),
        # 计划 update_v2.9.10 §10/§15：每个字段来自哪一类事实（客户明确 / 场景推断 /
        # 系统推算 / 默认），供"显式错才扣分"的判定使用
        "predicted_facts": (
            understanding.profile.fact_sources()
            if getattr(understanding, "profile", None) is not None else {}
        ),
        "product_type": _product_type_check(query, expected),
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
    clarifications = 0
    details: List[Dict[str, Any]] = []
    selected = [c for c in cases if c.get("models")][: limit] if limit else [c for c in cases if c.get("models")]

    for case in selected:
        try:
            # 计划 update_v2.9.10【任务 2】：端到端评测必须走**生产链路** ——
            # query → RequirementExtractor → RequirementProfile → Solution。
            # （Solution 已经不再自己重建需求：不传档案就会得到 REQUIREMENT_NOT_READY）
            from src.rag.query_understanding import understand_query

            understanding = understand_query(case["query"], history=case.get("history") or [])
            result = runner.run(
                message=case["query"],
                history=case.get("history") or [],
                profile=understanding.profile,
            )
        except Exception as exc:
            logger.error("[%s] Agent 失败: %s", case["id"], exc)
            details.append({"id": case["id"], "error": str(exc)})
            continue
        from eval.metrics import identifiers_from_items

        _, models = identifiers_from_items(result.get("products") or [])
        expected = case["models"]
        if not models:
            # 计划 update_v2.9.10 §17：推荐必须先过 Recommendation Gate。
            # 硬性条件没齐时 Gate 返回 NEED_CLARIFICATION（正确的业务行为），
            # 这类用例不算"推荐错误"，单独统计。
            clarifications += 1
            details.append({
                "id": case["id"],
                "expected_models": expected,
                "predicted_models": [],
                "status": "need_clarification",
                "answer": str(result.get("answer") or "")[:120],
            })
            continue
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
        "cases_selected": len(selected),
        "need_clarification": clarifications,
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
        # 计划 update_v2.9.10 第四阶段：逐字段 TP / FP / FN / Accuracy / Recall
        "per_field_confusion": _per_field_confusion(scored),
        # 计划 update_v2.9.10 §14/§15：产品类型路由准确率（唯一决策入口）
        "product_type_accuracy": _product_type_accuracy(case_results),
        "hard_constraint_capture_rate": round(mean(hard_rates), 4) if hard_rates else None,
        "derived_capture_rate": round(
            mean([c["derived_capture_rate"] for c in case_results
                  if c.get("derived_capture_rate") is not None]), 4
        ),
    }


def _per_field_confusion(case_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """逐字段 TP / FP / FN / Accuracy / Recall（计划 update_v2.9.10 第四阶段）。

    只统计"标注里出现过该字段"的用例 + "预测里多给该字段"的用例：

        TP = 标注有、系统也给了且相等
        FP = 标注没有（或值不同）、系统却给了
        FN = 标注有、系统没给
    """
    stats: Dict[str, Dict[str, int]] = {}
    for case in case_results:
        expected = case.get("expected_slots") or {}
        predicted = case.get("predicted_slots") or {}
        facts = case.get("predicted_facts") or {}
        for key in sorted(set(expected) | set(predicted)):
            row = stats.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "fp_derived": 0})
            want = expected.get(key)
            got = predicted.get(key)
            if want is None:
                if got is not None:
                    # 计划 §10：系统**推断/默认**出来的额外字段不算"错"（信息更全），
                    # 只有"系统当成客户明确说的"才算硬错。
                    if facts.get(key) == "CUSTOMER_EXPLICIT":
                        row["fp"] += 1
                    else:
                        row["fp_derived"] += 1
                continue
            if got is None:
                row["fn"] += 1
            elif got == want:
                row["tp"] += 1
            else:
                row["fp"] += 1
                row["fn"] += 1
    result: Dict[str, Dict[str, Any]] = {}
    for key, row in sorted(stats.items()):
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        total = tp + fp + fn
        result[key] = {
            **row,
            "accuracy": round(tp / total, 4) if total else None,
            "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        }
    return result


def _product_type_accuracy(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """产品类型路由准确率（计划 §24 的 Product Type Accuracy）。"""
    checked = [
        c for c in case_results
        if (c.get("product_type") or {}).get("correct") is not None
    ]
    if not checked:
        return {"cases": 0, "accuracy": None}
    hits = [1.0 if c["product_type"]["correct"] else 0.0 for c in checked]
    misses = [
        {"id": c["id"], "query": c["query"], **c["product_type"]}
        for c in checked if not c["product_type"]["correct"]
    ]
    return {"cases": len(checked), "accuracy": round(mean(hits), 4), "misses": misses}


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

    print("\n===== 需求理解 / 推荐基线 =====")
    print(f"  Requirement Slot Accuracy: {summary['requirement_slot_accuracy']}")
    print(f"  Hard Constraint Capture  : {summary['hard_constraint_capture_rate']}  （仅客户明确说的约束）")
    print(f"  Derived Capture          : {summary['derived_capture_rate']}  （推断/默认字段，仅供参考）")
    print(f"  Product Type Accuracy    : {summary['product_type_accuracy']['accuracy']}  "
          f"（{summary['product_type_accuracy']['cases']} 条客户点名的用例，唯一决策入口=ProductTypeRouter）")
    print("  Per-field TP/FP/FN:")
    for key, row in (summary.get("per_field_confusion") or {}).items():
        print(
            f"    {key:18s} tp={row['tp']:3d} fp={row['fp']:3d} fn={row['fn']:3d} "
            f"fp_derived={row.get('fp_derived', 0):3d} acc={row['accuracy']} recall={row['recall']}"
        )
    print(f"  逐槽位命中率: {summary['per_slot_accuracy']}")
    print(f"  推荐引擎: {engine_result['status']} ({engine_result.get('reason', '')})")
    print(f"\n报告已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
