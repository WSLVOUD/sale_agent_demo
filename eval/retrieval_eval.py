"""
Phase 0：检索基线评估（Retrieval Baseline）。

对 Golden Dataset 逐条执行与生产链路一致的混合检索，输出：

  Recall@5 / Recall@10 / MRR（Series 级 + Model 级）
  Hard Constraint Violation Rate（硬约束违规率）
  延迟（平均 / P95）
  按 tag 拆分的分项指标

两种检索模式同时评估，便于定位问题归属：

  ranked  ：只应用环境/安装方式/产品类型 + 亮度硬过滤，衡量"排序质量"
  filtered：额外应用点间距过滤（与生产 retrieval_node 一致），衡量"过滤是否可用"

用法::

    python -m eval.retrieval_eval                  # 全量
    python -m eval.retrieval_eval --limit 10       # 只跑前 10 条
    python -m eval.retrieval_eval --ids g001,g025  # 只跑指定用例
    python -m eval.retrieval_eval --out eval/reports/retrieval_v0.json
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.metrics import (  # noqa: E402
    aggregate_by_tag,
    build_retrieval_filters,
    hard_constraint_violations,
    hit_at_k,
    identifiers_from_items,
    load_golden_cases,
    mean,
    pitch_fit_at_k,
    recall_at_k,
    reciprocal_rank,
    violation_type_counts,
    write_report,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval.retrieval")


def ensure_vectorstore(documents, force_rebuild: bool = False):
    """加载向量库；若与当前语料不一致则重建，保证评估针对最新数据。"""
    from src.config import config
    from src.core.embeddings import load_vectorstore, recreate_vectorstore

    if force_rebuild:
        return recreate_vectorstore(documents)

    vectorstore = load_vectorstore(persist_dir=config.VECTORSTORE_DIR)
    try:
        count = vectorstore._collection.count()
    except Exception:  # pragma: no cover - 防御式
        count = 0
    if count != len(documents):
        logger.warning(
            "向量库记录数 %d 与语料 %d 不一致，重建中（Phase 2 之后语料会变为 Model 级）",
            count, len(documents),
        )
        vectorstore = recreate_vectorstore(documents)
    return vectorstore


def build_pipeline(force_rebuild: bool = False):
    """构建与生产一致的混合检索管线。"""
    from src.config import config
    from src.rag.bm25 import BM25Search
    from src.rag.corpus import build_retrieval_corpus, corpus_as_dicts
    from src.rag.fusion import HybridSearch
    from src.rag.sparse import get_sparse_search

    documents = build_retrieval_corpus(config.DATA_DIR)
    docs = corpus_as_dicts(documents)
    vectorstore = ensure_vectorstore(documents, force_rebuild=force_rebuild)
    sparse = get_sparse_search(docs, config.VECTORSTORE_DIR)
    bm25 = BM25Search(docs)
    hybrid = HybridSearch(vectorstore, sparse, bm25)
    return hybrid, documents


def evaluate_case(hybrid, case: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """评估单条用例，返回两个模式下的指标。"""
    query = case["query"]
    hard = case.get("hard") or {}
    filters = build_retrieval_filters(hard)
    brightness_min = hard.get("brightness_min")

    pitch_kwargs = {}
    if hard.get("pixel_pitch_min") is not None:
        pitch_kwargs["pitch_min"] = hard["pixel_pitch_min"]
    if hard.get("pixel_pitch_max") is not None:
        pitch_kwargs["pitch_max"] = hard["pixel_pitch_max"]

    result: Dict[str, Any] = {
        "id": case["id"],
        "query": query,
        "tags": case.get("tags", []),
        "expected_series": case.get("series", []),
        "expected_models": case.get("models", []),
    }

    for mode in ("ranked", "filtered"):
        kwargs: Dict[str, Any] = {"top_k": top_k, "filters": filters or None}
        if brightness_min is not None:
            kwargs["brightness_min"] = brightness_min
        if mode == "filtered":
            kwargs.update(pitch_kwargs)

        start = time.time()
        try:
            items = hybrid.search(query=query, **kwargs)
        except Exception as exc:  # pragma: no cover - 防御式
            logger.error("[%s] 检索失败(%s): %s", case["id"], mode, exc)
            items = []
        elapsed_ms = (time.time() - start) * 1000

        series, models = identifiers_from_items(items)
        expected_series = case.get("series", [])
        expected_models = case.get("models", [])
        # 点间距按计划文档属于软条件（Phase 8 评分 / Phase 11 校验），
        # 这里只用环境/安装/产品类型/亮度这些真正的硬约束判违规。
        violations = hard_constraint_violations(items, hard, include_pitch=False)

        block: Dict[str, Any] = {
            "latency_ms": round(elapsed_ms, 1),
            "retrieved_count": len(items),
            "retrieved_series": series[:10],
            "retrieved_models": models[:10],
            "violations": violations,
            "violation_count": len(violations),
        }
        pitch_fit_3 = pitch_fit_at_k(items, hard, 3)
        pitch_fit_5 = pitch_fit_at_k(items, hard, 5)
        if pitch_fit_3 is not None:
            block["pitch_fit_at_3"] = pitch_fit_3
            block["pitch_fit_at_5"] = pitch_fit_5
        if expected_series:
            block["recall_at_5"] = round(recall_at_k(series, expected_series, 5), 4)
            block["recall_at_10"] = round(recall_at_k(series, expected_series, 10), 4)
            block["mrr"] = round(reciprocal_rank(series, expected_series), 4)
            block["series_hit_at_3"] = hit_at_k(series, expected_series, 3)
        if expected_models:
            block["model_recall_at_5"] = round(recall_at_k(models, expected_models, 5), 4)
            block["model_recall_at_10"] = round(recall_at_k(models, expected_models, 10), 4)
            block["model_mrr"] = round(reciprocal_rank(models, expected_models), 4)
            block["model_hit_at_1"] = hit_at_k(models, expected_models, 1)
            block["model_hit_at_3"] = hit_at_k(models, expected_models, 3)
            block["model_hit_at_5"] = hit_at_k(models, expected_models, 5)
        result[mode] = block

    return result


def summarize(cases: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """汇总某一模式下的整体指标。"""
    def pick(key: str) -> List[float]:
        return [c[mode][key] for c in cases if key in c.get(mode, {})]

    violations = [c[mode]["violation_count"] for c in cases if mode in c]
    latencies = pick("latency_ms")
    scored = [c for c in cases if mode in c and "recall_at_5" in c[mode]]
    model_scored = [c for c in cases if mode in c and "model_recall_at_5" in c[mode]]

    return {
        "cases_total": len(cases),
        "cases_scored_series": len(scored),
        "cases_scored_model": len(model_scored),
        "recall_at_5": round(mean(pick("recall_at_5")), 4),
        "recall_at_10": round(mean(pick("recall_at_10")), 4),
        "mrr": round(mean(pick("mrr")), 4),
        "series_hit_at_3": round(mean(pick("series_hit_at_3")), 4),
        "model_recall_at_5": round(mean(pick("model_recall_at_5")), 4),
        "model_recall_at_10": round(mean(pick("model_recall_at_10")), 4),
        "model_mrr": round(mean(pick("model_mrr")), 4),
        "model_hit_at_1": round(mean(pick("model_hit_at_1")), 4),
        "model_hit_at_3": round(mean(pick("model_hit_at_3")), 4),
        "model_hit_at_5": round(mean(pick("model_hit_at_5")), 4),
        "pitch_fit_at_3": round(mean(pick("pitch_fit_at_3")), 4),
        "pitch_fit_at_5": round(mean(pick("pitch_fit_at_5")), 4),
        "hard_constraint_violation_rate": round(
            (sum(1 for v in violations if v > 0) / len(violations)) if violations else 0.0, 4
        ),
        "total_violations": int(sum(violations)),
        "violation_types": violation_type_counts(
            [v for c in cases if mode in c for v in c[mode]["violations"]]
        ),
        "avg_latency_ms": round(mean(latencies), 1),
        "p95_latency_ms": round(
            statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0),
            1,
        ),
        "empty_result_rate": round(
            (sum(1 for c in cases if mode in c and c[mode]["retrieved_count"] == 0) / len(cases)) if cases else 0.0,
            4,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 检索基线评估")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条用例")
    parser.add_argument("--ids", type=str, default=None, help="逗号分隔的用例 id")
    parser.add_argument("--tags", type=str, default=None, help="逗号分隔的 tag 过滤")
    parser.add_argument("--top-k", type=int, default=10, help="每次检索返回数量（默认 10）")
    parser.add_argument("--out", type=str, default="eval/reports/retrieval_report.json")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    args = parser.parse_args()

    cases = load_golden_cases(
        ids=args.ids.split(",") if args.ids else None,
        tags=args.tags.split(",") if args.tags else None,
        limit=args.limit,
    )
    if not cases:
        print("没有匹配的用例")
        return 1

    print(f"加载 Golden Dataset: {len(cases)} 条用例")
    hybrid, documents = build_pipeline(force_rebuild=args.rebuild)
    print(f"语料规模: {len(documents)} chunks")

    case_results: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        result = evaluate_case(hybrid, case, top_k=args.top_k)
        case_results.append(result)
        ranked = result["ranked"]
        print(
            f"[{index}/{len(cases)}] {case['id']} "
            f"retrieved={ranked['retrieved_count']} "
            f"violations={ranked['violation_count']} "
            f"hit={','.join(ranked['retrieved_series'][:3]) or '-'}"
        )

    summary = {mode: summarize(case_results, mode) for mode in ("ranked", "filtered")}

    # 只为 Series 级指标做 tag 拆分（Model 级在 Phase 2 之后才有意义）
    tag_cases = [
        {"tags": c["tags"], "recall_at_5": c["ranked"].get("recall_at_5")}
        for c in case_results
    ]
    tag_summary = aggregate_by_tag(tag_cases, "recall_at_5")

    try:
        import subprocess
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        commit = ""

    report = {
        "report": "retrieval_eval",
        "phase": "Phase 0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset": "eval/golden_dataset.json",
        "corpus_chunks": len(documents),
        "top_k": args.top_k,
        "summary": summary,
        "recall_at_5_by_tag": tag_summary,
        "cases": case_results,
    }
    path = write_report(report, args.out)

    print("\n===== 检索基线（ranked：不含点间距过滤）=====")
    for key, value in summary["ranked"].items():
        print(f"  {key}: {value}")
    print("\n===== 检索基线（filtered：含点间距过滤，与生产一致）=====")
    for key, value in summary["filtered"].items():
        print(f"  {key}: {value}")
    print(f"\n报告已写入: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
