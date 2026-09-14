"""Reflection node - evaluates retrieved chunks and recommendation quality."""
from typing import Dict, Any, List
import json
import logging
import re

from ..state import SolutionState
from ....core.llm import get_llm

logger = logging.getLogger(__name__)

# Minimum chunk quality score (0-10) to keep
CHUNK_MIN_SCORE = 4.0

# ── Reflection 预算控制 ───────────────────────────────────────────────────────
# Phase 6 优化：Reflection → Quality Gate
# 最多 1 次修正轮，不追求多轮迭代质量
DEFAULT_MAX_REFLECTION_ROUNDS = 1       # Phase 6: 最多 1 轮（原为 3）
DEFAULT_MAX_REFLECTION_TOKENS = 2000     # 单轮最大 token 消耗估算
DEFAULT_MAX_REFLECTION_TIME_MS = 8000   # 单轮最大耗时（毫秒）
DEFAULT_NO_IMPROVEMENT_STOP = 1         # Phase 6: 1 轮无改进即停止

REFLECTION_PROMPT = """You are a strict product quality auditor. Evaluate two dimensions simultaneously:

## Dimension 1: Does each retrieved product chunk match the customer needs?

Customer needs:
{requirement}

Product chunk list:
{products}

Score each product chunk (0-10), evaluating how well it matches the customer needs:
- 10 = perfect match, specs and features directly satisfy the needs
- 7-9 = mostly matches, 80%+ of needs met
- 4-6 = partially matches, significant gaps remain
- 1-3 = mostly not a match, severely diverges from needs
- 0 = completely irrelevant

## Dimension 2: Recommendation text quality

Recommendation content:
{recommendation}

Evaluation criteria (0-10):
1. Completeness — does it address all customer needs?
2. Accuracy — are the product specs accurate?
3. Helpfulness — is it practical and actionable?
4. Naturalness — does it read like natural conversation?

## Output format (must be strict JSON):

{{
    "chunk_scores": [
        {{"index": 0, "score": 8, "reason": "brief reason"}},
        {{"index": 1, "score": 3, "reason": "brief reason"}}
    ],
    "recommendation_score": 7,
    "recommendation_notes": "brief quality notes",
    "needs_refine": true/false,
    "suggestions": "improvement suggestions, or null"
}}"""


def _format_requirement(req: Dict[str, Any]) -> str:
    """Human-readable requirement summary."""
    parts = [f"{k}: {v}" for k, v in req.items() if v and str(v).strip()]
    return "; ".join(parts) if parts else "not specified"


def _format_products(products: List[Dict[str, Any]], top_k: int = 6) -> str:
    """Format product chunks for the prompt."""
    lines = []
    for i, chunk in enumerate(products[:top_k]):
        text = chunk.get("text", "")[:600].replace("\n", " ").strip()
        chunk_id = chunk.get("id", f"chunk_{i}")
        lines.append(f"[块{i} ID:{chunk_id}]\n{text}")
    return "\n\n".join(lines)


def reflection_node(state: SolutionState) -> SolutionState:
    """Phase 11：Reflection 从"让 LLM 重新决策"改为"确定性校验"。

    校验项见 ``src/rag/validation.py``：型号真实性、canonical 来源、环境/安装方式、
    点间距、箱体与模组数据、箱体与模组数量、以及回复是否出现虚构参数。

    校验通过 → 直接结束（不再调用 LLM）；发现问题 → 剔除无效产品并把问题写进
    ``validation_report``，由回复层决定是否追问，而不是让 LLM 换一个产品。
    """
    from ....rag.validation import validate_recommendation

    products = state.get("products") or []
    recommendation = state.get("recommendation") or ""
    profile = state.get("requirement_profile")
    selection = state.get("recommendation_result") or {}
    calculation = state.get("screen_calculation")
    reflection_count = int(state.get("reflection_count") or 0) + 1

    if not recommendation:
        return {
            **state,
            "reflection_score": 0.0,
            "reflection_notes": "没有可校验的推荐内容",
            "needs_refine": False,
            "next_action": "end",
            "reflection_count": reflection_count,
            "best_score": state.get("best_score", 0.0),
        }

    report = validate_recommendation(
        recommendation=recommendation,
        products=products,
        profile=profile,
        selection=selection,
        calculation=calculation,
    )

    validated_products = report.get("valid_products") or []
    if validated_products:
        products = validated_products

    logger.info(
        "Reflection(validation): score=%.1f checks=%s",
        report["score"], {k: v for k, v in report["checks"].items() if not v} or "all passed",
    )

    return {
        **state,
        "products": products,
        "reflection_score": report["score"],
        "reflection_notes": report["summary"],
        "validation_report": report,
        "needs_refine": False,
        "next_action": "end",
        "reflection_count": reflection_count,
        "best_score": max(float(state.get("best_score") or 0.0), report["score"]),
        "consecutive_no_improve": 0,
    }
