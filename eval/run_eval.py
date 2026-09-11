"""
评估脚本：运行 Golden Dataset 并输出评估报告。

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import logging
import time
import sys
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

# ── 项目路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.agents.solution.runner import SolutionAgentRunner
from src.agents.solution.state import SolutionState
from src.core.embeddings import load_vectorstore
from src.rag.loader import load_all_product_files
from src.agents.solution.nodes.requirement import detect_intent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── 结果数据结构 ────────────────────────────────────────────────────────────
@dataclass
class QueryUnderstandingResult:
    intent_correct: bool
    intent_predicted: str
    intent_expected: str
    constraint_match_rate: float  # 0.0 ~ 1.0
    constraint_details: dict[str, bool]  # key → matched?


@dataclass
class RetrievalResult:
    recall_at_5: float
    recall_at_10: float
    mrr: float
    retrieved_ids: list[str]
    expected_ids: list[str]


@dataclass
class GenerationResult:
    mentions_expected_points: bool
    has_hallucination: bool
    factual_accuracy: float  # 0.0 ~ 1.0
    response_length: int


@dataclass
class SystemMetrics:
    ttft_ms: float  # 首 token 延迟（毫秒）
    total_latency_ms: float  # 完整响应延迟
    llm_call_count: int
    input_tokens: int
    output_tokens: int


@dataclass
class CaseResult:
    case_id: str
    query: str
    # 子维度
    query_understanding: QueryUnderstandingResult | None = None
    retrieval: RetrievalResult | None = None
    generation: GenerationResult | None = None
    system: SystemMetrics | None = None
    # 业务指标
    product_accuracy: float = 0.0
    error_rate: float = 0.0
    error_type: str = ""


@dataclass
class EvalReport:
    total_cases: int
    passed_cases: int
    query_understanding_accuracy: float
    avg_constraint_match_rate: float
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_mrr: float
    avg_factual_accuracy: float
    avg_latency_ms: float
    total_llm_calls: int
    total_tokens: int
    cases: list[CaseResult] = field(default_factory=list)


# ── 数据加载 ────────────────────────────────────────────────────────────────
def load_dataset(path: str = "eval/dataset.json") -> list[dict[str, Any]]:
    with open(PROJECT_ROOT / path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


# ── Intent 评估 ─────────────────────────────────────────────────────────────
def evaluate_intent(predicted: str, expected: str) -> bool:
    """判断预测意图是否与期望意图匹配。"""
    expected = expected.lower().strip()
    predicted = predicted.lower().strip()

    # 意图别名映射
    intent_aliases: dict[str, list[str]] = {
        "recommendation": ["recommendation", "need_query", "recommend"],
        "product_question": ["product_question", "parameter_query", "product_query"],
        "conversation": ["conversation", "small_talk", "greeting"],
        "objection_handling": ["objection", "objection_handling"],
    }
    for canonical, aliases in intent_aliases.items():
        if predicted in aliases and expected == canonical:
            return True
        if predicted == canonical and expected == canonical:
            return True
    return predicted == expected


# ── 约束匹配 ────────────────────────────────────────────────────────────────
def evaluate_constraints(
    inferred: dict[str, Any],
    expected: dict[str, Any],
) -> tuple[float, dict[str, bool]]:
    """计算约束匹配率。返回 (match_rate, details)。"""
    # 过滤空值
    expected_clean = {k: v for k, v in expected.items() if v is not None and v != "" and v != []}
    if not expected_clean:
        return 1.0, {}

    details: dict[str, bool] = {}
    for key, exp_val in expected_clean.items():
        inf_val = inferred.get(key)
        if inf_val is None or inf_val == "" or inf_val == []:
            details[key] = False
            continue
        # 数值范围容差
        if isinstance(exp_val, (int, float)) and isinstance(inf_val, (int, float)):
            # 容差 ±20%
            tolerance = abs(float(exp_val)) * 0.2 + 0.1
            details[key] = abs(float(inf_val) - float(exp_val)) <= tolerance
        elif isinstance(exp_val, bool):
            details[key] = bool(inf_val) == exp_val
        else:
            details[key] = str(exp_val).lower() in str(inf_val).lower()

    match_rate = sum(details.values()) / len(details) if details else 1.0
    return match_rate, details


# ── 幻觉检测 ───────────────────────────────────────────────────────────────
_HALLUCINATION_PATTERNS = [
    "根据我的资料",
    "我手头",
    "产品资料里没有",
    "数据库中",
    "检索结果",
    "召回了",
    "资料库里暂无",
    "没有专门针对",
    "资料里没有说明",
    "资料中未提及",
    "根据知识库",
]


_MODEL_RE = re.compile(r"(?:TW(?:11|21|31)[\w\-./]+|T\d{2,3}Omni[\w\-]+|H(?:G)?\d{4}[\w\-]*|DS-TW|DLP-60|RGB-70)", re.I)


def extract_product_ids(
    products: list[Any],
    expected_products: list[str] | None = None,
    extra_text: str = "",
) -> list[str]:
    """Collect product/model IDs from dicts, Documents, and free text."""
    ids: list[str] = []
    blobs: list[str] = [extra_text or ""]

    def _add(value: Any) -> None:
        if not value:
            return
        text = str(value).strip()
        if text and text not in ids:
            ids.append(text)

    for p in products or []:
        if isinstance(p, dict):
            meta = p.get("metadata") or {}
            _add(p.get("product_id") or p.get("model") or p.get("id"))
            if isinstance(meta, dict):
                _add(meta.get("product_id") or meta.get("model") or meta.get("chunk_id"))
            blobs.append(str(p.get("text") or p.get("page_content") or p.get("content") or ""))
            continue
        meta = getattr(p, "metadata", None) or {}
        _add(getattr(p, "product_id", "") or getattr(p, "model", "") or getattr(p, "id", ""))
        if isinstance(meta, dict):
            _add(meta.get("product_id") or meta.get("model") or meta.get("chunk_id"))
        blobs.append(str(getattr(p, "page_content", "") or getattr(p, "content", "")))

    haystack = "\n".join(blobs)
    for match in _MODEL_RE.findall(haystack):
        _add(match)
    for expected in expected_products or []:
        if expected and expected in haystack:
            _add(expected)
    return ids


def has_hallucination(response: str) -> bool:
    for pattern in _HALLUCINATION_PATTERNS:
        if pattern in response:
            return True
    return False


# ── 初始化 Agent（全局复用）───────────────────────────────────────────────
_agent_runner: SolutionAgentRunner | None = None


def get_agent_runner() -> SolutionAgentRunner:
    global _agent_runner
    if _agent_runner is not None:
        return _agent_runner

    from src.config import config

    vectorstore = load_vectorstore(persist_dir=config.VECTORSTORE_DIR)
    documents = load_all_product_files(config.DATA_DIR)
    _agent_runner = SolutionAgentRunner(vectorstore, documents)
    return _agent_runner


# ── 主评估循环 ─────────────────────────────────────────────────────────────
def run_case(case: dict[str, Any]) -> CaseResult:
    """运行单个测试用例。"""
    case_id = case["id"]
    query = case["query"]
    expected_intent = case.get("intent", "")
    expected_constraints = case.get("constraints", {})
    expected_products = case.get("expected_products", [])
    expected_points = case.get("expected_points", [])

    logger.info(f"[{case_id}] Running: {query[:60]}")

    start = time.time()
    ttft = 0.0
    llm_calls = 0
    in_tokens = 0
    out_tokens = 0
    response_text = ""
    predicted_intent = ""
    inferred_constraints: dict[str, Any] = {}
    retrieved_ids: list[str] = []

    try:
        # ── Intent 检测 ────────────────────────────────────────────────
        predicted_intent = detect_intent(query)

        # ── 调用 Solution Agent ─────────────────────────────────────────
        agent = get_agent_runner()
        result = agent.run(message=query, history=[])

        response_text = result.get("answer", "") or ""
        inferred_constraints = result.get("requirement", {})

        retrieved_ids = extract_product_ids(
            result.get("products", []),
            expected_products,
            extra_text=response_text,
        )

    except Exception as exc:
        logger.warning(f"[{case_id}] Agent error: {exc}")
        response_text = ""

    total_latency = (time.time() - start) * 1000

    # ── Query Understanding 评估 ─────────────────────────────────────────
    intent_correct = evaluate_intent(predicted_intent, expected_intent)
    constraint_rate, constraint_details = evaluate_constraints(inferred_constraints, expected_constraints)

    qu_result = QueryUnderstandingResult(
        intent_correct=intent_correct,
        intent_predicted=predicted_intent,
        intent_expected=expected_intent,
        constraint_match_rate=constraint_rate,
        constraint_details=constraint_details,
    )

    # ── Retrieval 评估（基于预期产品 ID）──────────────────────────────────
    if expected_products:
        recall5 = len(set(retrieved_ids[:5]) & set(expected_products)) / max(len(expected_products), 1)
        recall10 = len(set(retrieved_ids[:10]) & set(expected_products)) / max(len(expected_products), 1)
        # MRR
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids[:10], 1):
            if rid in expected_products:
                mrr = 1.0 / i
                break
    else:
        recall5 = recall10 = mrr = 0.0

    ret_result = RetrievalResult(
        recall_at_5=recall5,
        recall_at_10=recall10,
        mrr=mrr,
        retrieved_ids=retrieved_ids,
        expected_ids=expected_products,
    )

    # ── Generation 评估 ──────────────────────────────────────────────────
    mentions_points = any(
        kw.lower() in response_text.lower()
        for kw in expected_points
    )
    hallucination = has_hallucination(response_text)
    factual = 0.0 if hallucination else 1.0  # 简化版

    gen_result = GenerationResult(
        mentions_expected_points=mentions_points,
        has_hallucination=hallucination,
        factual_accuracy=factual,
        response_length=len(response_text),
    )

    # ── System Metrics ───────────────────────────────────────────────────
    sys_result = SystemMetrics(
        ttft_ms=ttft,
        total_latency_ms=total_latency,
        llm_call_count=llm_calls,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
    )

    return CaseResult(
        case_id=case_id,
        query=query,
        query_understanding=qu_result,
        retrieval=ret_result,
        generation=gen_result,
        system=sys_result,
    )


def run_evaluation(dataset_path: str = "eval/dataset.json", limit: int = 0) -> EvalReport:
    """运行完整评估。"""
    cases_data = load_dataset(dataset_path)
    if limit and limit > 0:
        cases_data = cases_data[:limit]
    results: list[CaseResult] = []

    for case in cases_data:
        result = run_case(case)
        results.append(result)

    # 汇总统计
    n = len(results)
    qu_acc = sum(
        1 for r in results
        if r.query_understanding and r.query_understanding.intent_correct
    ) / n
    avg_constraint = sum(
        r.query_understanding.constraint_match_rate
        for r in results if r.query_understanding
    ) / n
    avg_recall5 = sum(r.retrieval.recall_at_5 for r in results if r.retrieval) / n
    avg_recall10 = sum(r.retrieval.recall_at_10 for r in results if r.retrieval) / n
    avg_mrr = sum(r.retrieval.mrr for r in results if r.retrieval) / n
    avg_factual = sum(r.generation.factual_accuracy for r in results if r.generation) / n
    avg_latency = sum(r.system.total_latency_ms for r in results if r.system) / n
    total_calls = sum(r.system.llm_call_count for r in results if r.system)
    total_tokens = sum(
        r.system.input_tokens + r.system.output_tokens
        for r in results if r.system
    )

    report = EvalReport(
        total_cases=n,
        passed_cases=qu_acc * n,
        query_understanding_accuracy=qu_acc,
        avg_constraint_match_rate=avg_constraint,
        avg_recall_at_5=avg_recall5,
        avg_recall_at_10=avg_recall10,
        avg_mrr=avg_mrr,
        avg_factual_accuracy=avg_factual,
        avg_latency_ms=avg_latency,
        total_llm_calls=total_calls,
        total_tokens=total_tokens,
        cases=results,
    )
    return report


def print_report(report: EvalReport) -> None:
    """格式化输出评估报告。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LED RAG System Evaluation Report")
    print(f"  Total Cases: {report.total_cases}")
    print(f"  Passed (Intent Correct): {int(report.passed_cases)} / {report.total_cases}")
    print(f"{sep}")

    print(f"\n[ Query Understanding ]")
    print(f"  Intent Accuracy:       {report.query_understanding_accuracy:.1%}")
    print(f"  Avg Constraint Match:  {report.avg_constraint_match_rate:.1%}")

    print(f"\n[ Retrieval ]")
    print(f"  Recall@5:              {report.avg_recall_at_5:.1%}")
    print(f"  Recall@10:             {report.avg_recall_at_10:.1%}")
    print(f"  MRR:                  {report.avg_mrr:.3f}")

    print(f"\n[ Generation ]")
    print(f"  Avg Factual Accuracy:  {report.avg_factual_accuracy:.1%}")

    print(f"\n[ System Performance ]")
    print(f"  Avg Latency:           {report.avg_latency_ms:.0f} ms")
    print(f"  Total LLM Calls:       {report.total_llm_calls}")
    print(f"  Total Tokens:          {report.total_tokens:,}")

    print(f"\n[ Per-Case Details ]")
    for case in report.cases:
        qu = case.query_understanding
        ret = case.retrieval
        gen = case.generation
        flag = "Y" if qu and qu.intent_correct else "N"
        halluc_flag = "Y" if gen and gen.has_hallucination else "N"
        constraint = f"{qu.constraint_match_rate:.0%}" if qu else "n/a"
        recall5 = f"{ret.recall_at_5:.0%}" if ret else "n/a"
        latency = f"{case.system.total_latency_ms:.0f}ms" if case.system else "n/a"
        print(
            f"  [{case.case_id}] "
            f"intent={flag} "
            f"constraint={constraint} "
            f"recall5={recall5} "
            f"halluc={halluc_flag} "
            f"lat={latency}"
        )

    print(f"\n{sep}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LED RAG System Evaluator")
    parser.add_argument("--dataset", default="eval/dataset.json", help="Dataset path")
    parser.add_argument("--output", default="", help="Output JSON path (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases (0 = all)")
    args = parser.parse_args()

    report = run_evaluation(args.dataset, limit=args.limit)
    print_report(report)

    if args.output:
        out = {
            "report_version": "1.0",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            **{k: v for k, v in asdict(report).items() if k != "cases"},
            "cases": [
                {k: v for k, v in asdict(c).items() if k != "system"}
                for c in report.cases
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {args.output}")
