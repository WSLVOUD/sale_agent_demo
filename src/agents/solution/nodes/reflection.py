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
    """Evaluate retrieved chunks and the recommendation with budget control."""
    import time

    requirement = state.get("requirement", {})
    recommendation = state.get("recommendation", "")
    products = state.get("products", [])

    # ── 预算提取 ─────────────────────────────────────────────────────────
    max_rounds     = state.get("reflection_max_rounds", DEFAULT_MAX_REFLECTION_ROUNDS)
    max_tokens     = state.get("reflection_max_tokens", DEFAULT_MAX_REFLECTION_TOKENS)
    max_time_ms    = state.get("reflection_max_time_ms", DEFAULT_MAX_REFLECTION_TIME_MS)
    no_improve_stop = state.get("reflection_no_improve_stop", DEFAULT_NO_IMPROVEMENT_STOP)

    reflection_count  = state.get("reflection_count", 0)
    best_score       = state.get("best_score", 0)
    consecutive_no_improve = state.get("consecutive_no_improve", 0)

    # ── 预算检查：已达最大轮数 ───────────────────────────────────────────
    if reflection_count >= max_rounds:
        logger.info("Reflection budget: max rounds %d reached, stopping.", max_rounds)
        return {
            **state,
            "reflection_score": best_score,
            "reflection_notes": f"Max reflection rounds ({max_rounds}) reached, stopping",
            "needs_refine": False,
            "next_action": "end",
            "reflection_count": reflection_count,
            "best_score": best_score,
            "consecutive_no_improve": 0,
        }

    if not recommendation or recommendation.startswith("Sorry") or not products:
        return {
            **state,
            "reflection_score": 0,
            "reflection_notes": "No recommendation content, skipping evaluation",
            "needs_refine": False,
            "next_action": "end",
            "reflection_count": reflection_count,
            "best_score": best_score,
            "consecutive_no_improve": 0,
        }

    req_text = _format_requirement(requirement)
    products_text = _format_products(products)
    prompt = REFLECTION_PROMPT.format(
        requirement=req_text,
        products=products_text,
        recommendation=recommendation,
    )

    # Token 估算（粗略：prompt 长度 / 4）
    estimated_tokens = len(prompt) // 4
    total_estimated = (reflection_count + 1) * estimated_tokens
    if total_estimated > max_tokens * (reflection_count + 1):
        # 超 token 预算，但允许最后一轮
        if reflection_count + 1 >= max_rounds:
            logger.info("Reflection: token budget exceeded at round %d, stopping.", reflection_count + 1)
            return {
                **state,
                "reflection_score": best_score,
                "reflection_notes": "Token budget exhausted, stopping reflection",
                "needs_refine": False,
                "next_action": "end",
                "reflection_count": reflection_count,
                "best_score": best_score,
                "consecutive_no_improve": 0,
            }

    round_start = time.time()
    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        elapsed_ms = (time.time() - round_start) * 1000

        # 时间预算检查
        if elapsed_ms > max_time_ms:
            logger.warning("Reflection round %d took %.0fms (> budget %dms), stopping.", 
                          reflection_count + 1, elapsed_ms, max_time_ms)
            return {
                **state,
                "reflection_score": best_score,
                "reflection_notes": f"Round took {elapsed_ms:.0f}ms, exceeded time budget, stopping",
                "needs_refine": False,
                "next_action": "end",
                "reflection_count": reflection_count + 1,
                "best_score": best_score,
                "consecutive_no_improve": 0,
            }

        content = response.content if hasattr(response, "content") else str(response)

        content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE).strip()
        result = json.loads(content)

        chunk_scores = result.get("chunk_scores", [])
        rec_score = float(result.get("recommendation_score", 5))
        rec_needs_refine = bool(result.get("needs_refine", False))

        # Filter chunks below threshold
        filtered_products = []
        dropped_indices = []
        for item in chunk_scores:
            idx = int(item.get("index", -1))
            score = float(item.get("score", 0))
            if score >= CHUNK_MIN_SCORE and 0 <= idx < len(products):
                filtered_products.append(products[idx])
            else:
                dropped_indices.append(idx)
                logger.info("Reflection dropped chunk[%d] score=%.1f", idx, score)

        # ── 早停逻辑 ───────────────────────────────────────────────────
        # 计算本轮是否比上轮有改进
        this_round_score = rec_score
        is_improvement = this_round_score > best_score
        new_consecutive_no_improve = 0 if is_improvement else consecutive_no_improve + 1

        # 连续 N 轮无改进 → 停止
        if new_consecutive_no_improve >= no_improve_stop:
            logger.info(
                "Reflection early stop: %d consecutive rounds without improvement.",
                new_consecutive_no_improve,
            )
            return {
                **state,
                "reflection_score": best_score,
                "reflection_notes": f"{new_consecutive_no_improve} consecutive rounds without improvement, stopping",
                "needs_refine": False,
                "next_action": "end",
                "reflection_count": reflection_count + 1,
                "best_score": best_score,
                "consecutive_no_improve": new_consecutive_no_improve,
            }

        should_refine = bool(dropped_indices) or (
            rec_needs_refine and is_improvement and reflection_count + 1 < max_rounds
        )

        new_best = max(best_score, this_round_score)
        new_count = reflection_count + 1

        notes = result.get("recommendation_notes", "")
        if dropped_indices:
            notes = f"[{len(dropped_indices)} mismatched chunks dropped] {notes}"
        if not is_improvement:
            notes = f"[score={this_round_score:.1f}, no improvement] {notes}"

        logger.info(
            "Reflection #%d: rec_score=%.1f best=%.1f improve=%s "
            "kept=%d dropped=%d refine=%s time=%.0fms",
            new_count,
            this_round_score,
            new_best,
            is_improvement,
            len(filtered_products),
            len(dropped_indices),
            should_refine,
            elapsed_ms,
        )

        return {
            **state,
            "reflection_score": this_round_score,
            "reflection_notes": notes,
            "needs_refine": should_refine,
            "reflection_count": new_count,
            "best_score": new_best,
            "products": filtered_products if filtered_products else products,
            "next_action": "recommend" if should_refine else "end",
            "consecutive_no_improve": new_consecutive_no_improve,
        }

    except json.JSONDecodeError as e:
        logger.error("Reflection JSON parse error: %s", e)
        return {
            **state,
            "reflection_score": 5,
            "reflection_notes": "Evaluation parse failed, keeping existing results",
            "needs_refine": False,
            "next_action": "end",
            "reflection_count": reflection_count,
            "best_score": best_score,
            "consecutive_no_improve": 0,
        }
    except Exception as e:
        logger.error("Reflection error: %s", e)
        return {
            **state,
            "reflection_score": 5,
            "reflection_notes": f"Evaluation error: {e}",
            "needs_refine": False,
            "next_action": "end",
            "reflection_count": reflection_count,
            "best_score": best_score,
            "consecutive_no_improve": 0,
        }
