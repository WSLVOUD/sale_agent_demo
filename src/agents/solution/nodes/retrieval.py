"""Retrieval node for hybrid search."""
import logging
from typing import List, Dict, Any

from ..state import SolutionState

logger = logging.getLogger(__name__)


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
    """Retrieve relevant products using hybrid search."""
    hybrid_search = state.get("hybrid_search")
    
    if not hybrid_search:
        logger.warning("No hybrid_search available in state")
        return {**state, "products": []}
    
    # Build search query
    query = _build_search_query(state)
    
    # Get inferred parameters for filtering
    brightness_min = state.get("inferred_brightness_min_nit")
    brightness_max = state.get("inferred_brightness_max_nit")
    pitch_max = state.get("inferred_pixel_pitch_max_mm")
    pitch_min = state.get("inferred_pixel_pitch_min_mm")
    is_rental = state.get("inferred_is_rental")
    display_type = state.get("requirement", {}).get("display_type")
    
    try:
        # Build filter including display_type
        search_filters = {}
        if display_type:
            search_filters["display_type"] = display_type
        
        # Perform hybrid search
        products = hybrid_search.search(
            query=query,
            top_k=20,  # Retrieve more for reranking
            filters=search_filters if search_filters else None,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            pitch_max=pitch_max,
            pitch_min=pitch_min,
            is_rental=is_rental,
        )
        
        logger.info(f"Retrieval: retrieved {len(products)} products for query: {query[:80]}")
        return {**state, "products": products}
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {**state, "products": []}
