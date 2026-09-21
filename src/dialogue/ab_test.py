"""v2.5++（僵硬话术优化 · 第十二阶段）：A/B 对比（计划 §15）。

两套策略跑同一批客户输入，比较自然度指标：

    Strategy A  Template → LLM Polish            （旧：模板腔仍留在草稿里）
    Strategy B  ResponseContext → LLM Native     （新：默认）

对比项（计划 §15）：ACK Rate / Echo Rate / Connector Rate / Question Repeat Rate /
Question Answer Rate / Unsupported Fact Rate / 平均回复长度。
确认 B 稳定后再全量切换（默认已经是 B，A 只用于对比与回退）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .response_context import ResponseContext
from .response_generator import (
    STRATEGY_NATIVE,
    STRATEGY_TEMPLATE_POLISH,
    generate_response,
)
from .response_validator import compute_metrics, validate_response


def run_strategy(
    cases: Iterable[Dict[str, Any]],
    strategy: str,
    *,
    llm_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, float]:
    """用指定策略跑一批用例，返回自然度指标。"""
    samples: List[Dict[str, Any]] = []
    for case in cases:
        context: ResponseContext = case["context"]
        llm = llm_factory(strategy) if llm_factory is not None else None
        text = generate_response(context, llm=llm, strategy=strategy)
        validation = validate_response(
            text,
            allow_ack=context.allow_ack,
            allow_connector=context.allow_connector,
            one_question=context.one_question,
            customer_question=bool(context.answer),
            answer=context.answer,
            customer_message=context.customer_message,
            supported_parameters=list(case.get("supported_parameters") or []) or None,
            required_question=context.question or context.required_question,
        )
        samples.append({
            "is_question": bool(case.get("is_question", context.answer != "")),
            "validation": validation,
        })
    return compute_metrics(samples)


def compare_strategies(
    cases: Iterable[Dict[str, Any]],
    *,
    llm_factory: Optional[Callable[[str], Any]] = None,
    strategies: Sequence[str] = (STRATEGY_TEMPLATE_POLISH, STRATEGY_NATIVE),
) -> Dict[str, Dict[str, float]]:
    """两套策略跑同一批用例 → 指标对比表。"""
    items = list(cases)
    return {
        strategy: run_strategy(items, strategy, llm_factory=llm_factory)
        for strategy in strategies
    }


__all__ = ["compare_strategies", "run_strategy"]
