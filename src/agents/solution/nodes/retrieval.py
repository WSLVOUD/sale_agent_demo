"""Retrieval node for hybrid search."""
import logging
from typing import List, Dict, Any

from ..state import SolutionState

logger = logging.getLogger(__name__)


def _last_user_message(state: SolutionState) -> str:
    """取本轮用户消息（用于硬约束解析）。"""
    current = state.get("current_message")
    if current:
        return str(current)
    for msg in reversed(state.get("messages", []) or []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in ("user", "human"):
            return str(msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", ""))
    return ""


def _normalize_history(messages: List) -> List[Dict[str, str]]:
    """Normalize messages to role/content dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        else:
            result.append({"role": getattr(msg, "type", "user"), "content": getattr(msg, "content", "")})
    return result


def _build_search_query(state: SolutionState) -> str:
    """Build the search query from requirement and context.
    
    Uses both Chinese and English keywords to maximize vector search matching.
    Combines display_type, purpose, and interaction keywords.
    """
    req = state.get("requirement", {})
    
    parts = []
    
    # Display type - add both Chinese and English signals
    display_type = req.get("display_type", "")
    if display_type:
        parts.append(display_type)  # e.g. "IFP"
        dt_en_map = {
            "IFP": "interactive flat panel touchscreen",
            "LCD": "LCD commercial display",
            "LED": "LED display",
        }
        if display_type in dt_en_map:
            parts.append(dt_en_map[display_type])
    
    # Purpose - add Chinese signal (primary) and English signal (boost)
    purpose = req.get("purpose", "")
    if purpose:
        parts.append(purpose)  # Chinese: "会议室"
        purpose_en_map = {
            "会议室": "meeting room", "会议": "meeting room conference",
            "教室": "classroom education", "培训": "training",
            "商场": "retail store", "展厅": "exhibition hall",
            "医院": "hospital medical", "监控": "control room monitoring",
            "指挥": "command center", "舞台": "stage performance",
            "演唱会": "concert", "体育": "sports stadium",
            "广告": "advertising digital signage", "幕墙": "building facade",
            "租赁": "rental event",
        }
        for cn, en in purpose_en_map.items():
            if cn in purpose:
                parts.append(en)  # English boost
                break
    
    # Interaction keywords for IFP scenarios
    if display_type == "IFP":
        user_text = " ".join(
            msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            for msg in state.get("messages", [])
            if (isinstance(msg, dict) and msg.get("role") in ("user", "human"))
            or (not isinstance(msg, dict) and getattr(msg, "type", "") in ("user", "human"))
        )
        if any(kw in user_text for kw in ["手写", "书写", "触摸", "触控", "交互", "白板"]):
            parts.extend(["手写", "whiteboard", "touchscreen", "交互"])
    
    query = " ".join(parts) if parts else "LED LCD display product"
    logger.info(f"Search query built: {query}")
    return query


def retrieval_node(state: SolutionState) -> SolutionState:
    """Retrieve relevant products using hybrid search + hard metadata filter.

    Phase 3：明确硬条件（产品类型 / 室内外 / 固装租赁 / 客户指定的点间距或亮度 /
    特殊功能）在检索前通过 metadata filter 完成过滤，绝不允许把违规产品交给 LLM
    去"解释为什么也可以"。

    观看距离推导出的点间距与亮度属于**软条件**：不参与过滤，交给排序与
    Reflection 校验（Phase 8 / Phase 11）。
    """
    hybrid_search = state.get("hybrid_search")
    
    if not hybrid_search:
        logger.warning("No hybrid_search available in state")
        return {**state, "products": []}
    
    from ....rag.hard_filter import build_hard_constraints
    from ....rag.query_understanding import understand_query

    requirement = dict(state.get("requirement", {}) or {})
    message = _last_user_message(state)
    history = _normalize_history(state.get("messages", []) or [])

    # Phase 4：先把自然语言改写成结构化槽位 + 标准化英文检索式，再进 RAG
    profile = state.get("requirement_profile")
    understanding = understand_query(message, history=history, profile=profile)
    query = understanding.retrieval_query or _build_search_query(state)
    logger.info(
        "Query understanding: lang=%s slots=%s → query=%r",
        understanding.language, understanding.slots, query[:120],
    )

    # Phase 6：需求档案 —— 以"客户原话的确定性解析"为准；
    # legacy requirement（LLM 提取）只补 purpose（场景原话），
    # 其它工程参数一律不采用，避免 AI 自行补全/覆盖客户确认的事实。
    merged_profile = understanding.profile
    if merged_profile is not None and not merged_profile.purpose and requirement:
        from ....models.requirement import RequirementProfile

        legacy_profile = RequirementProfile.from_legacy(requirement)
        if legacy_profile.purpose:
            merged_profile.purpose = legacy_profile.purpose
            merged_profile.sources["purpose"] = "confirmed"
    if merged_profile is not None:
        logger.info(
            "Requirement profile: completeness=%.2f sufficient=%s missing=%s",
            merged_profile.completeness(), merged_profile.is_sufficient(),
            merged_profile.missing_slots(),
        )

    constraints = build_hard_constraints(
        merged_profile if merged_profile is not None else requirement,
        message=message,
    )
    logger.info("Hard filters: %s", constraints.describe())

    try:
        # 硬约束 → metadata filter；不做点间距/亮度的推断值过滤
        products = hybrid_search.search(
            query=query,
            top_k=20,  # Retrieve more for reranking
            filters=constraints.chroma_where() or None,
            **constraints.search_kwargs(),
        )

        # 兜底：任何违反硬约束的条目都不得进入后续推荐
        filtered_products = constraints.apply(products)
        dropped = len(products) - len(filtered_products)

        logger.info(
            "Retrieval: %d products (%d dropped by hard filter) for query: %s",
            len(filtered_products), dropped, query[:80],
        )
        return {
            **state,
            "products": filtered_products,
            # 软条件留给排序 / Reflection 使用
            "soft_pitch_min_mm": state.get("inferred_pixel_pitch_min_mm"),
            "soft_pitch_max_mm": state.get("inferred_pixel_pitch_max_mm"),
            "soft_brightness_min_nit": state.get("inferred_brightness_min_nit"),
            "hard_constraints": constraints.to_dict(),
            "hard_filter_dropped": dropped,
            "understood_slots": understanding.slots,
            "understood_language": understanding.language,
            "retrieval_query": query,
            "requirement_profile": merged_profile,
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {**state, "products": []}
