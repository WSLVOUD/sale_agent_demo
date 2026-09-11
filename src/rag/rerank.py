"""
重排序模块
根据需求对检索结果进行重排序
"""
from typing import Any, Dict, List, Optional
import json
import logging
import re

from src.core.llm import get_llm

logger = logging.getLogger(__name__)

# Internal disclosure patterns
_INTERNAL_DISCLOSURE_RE = re.compile(
    r"(?:我手头|现有|当前|提供的|召回的)?.{0,10}"
    r"(?:产品)?资料(?:里|中|库)?(?:没有|未|显示|提到|说明|包含|找到)|"
    r"(?:根据|基于).{0,8}资料|数据库|查询(?:到|结果)?|检索(?:到|结果)?|"
    r"没有专门针对.{0,20}(?:说明|推荐|资料)",
    re.IGNORECASE,
)

# Indoor product patterns
_INDOOR_PRODUCT_RE = re.compile(
    r"TW(?:11|21|31)-(?:IRHD|IR|3216)[\w.-]*|"
    r"(?:室内用|室内产品|室内系列|室内租赁)",
    re.IGNORECASE,
)

_PIXEL_PITCH_RE = re.compile(r"pixel\s*pitch\s*(?P<pitch>\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_BRIGHTNESS_RE = re.compile(r"brightness\s*(?:=\s*)?(?P<value>\d{2,6})\s*(?:nit|cd|nits|cds)?", re.IGNORECASE)

RERANK_PROMPT = """
你是一名显示产品顾问。请根据客户需求评估已检索到的产品候选，并为每个产品生成推荐理由。

客户需求：
{requirement}

候选产品：
{candidates}

重要规则：
1. 每个候选包含原始产品数据。你必须自行识别型号名称。
2. 环境匹配优先：如果需求是户外使用，只推荐户外适用的LED产品；如果需求是室内使用，只推荐室内产品。
3. 不要拒绝任何产品，只排序并给出推荐理由。
4. 推荐产品时只说正面理由，不要提及否定表述。

只输出 JSON：
{{"selected_indices": [1, 2, 3], "reason": "推荐理由"}}
"""


def _get_product_text(product) -> str:
    """Extract text from a product dict or Document object."""
    if hasattr(product, "page_content"):
        return product.page_content
    return product.get("text", "")


def _get_product_metadata(product) -> Dict[str, Any]:
    """Extract metadata from a product dict or Document object."""
    if hasattr(product, "metadata"):
        return product.metadata or {}
    return product.get("metadata", {})


def is_display_candidate(product: Dict[str, Any]) -> bool:
    """Return True only when the product is a genuine display."""
    metadata = _get_product_metadata(product)
    text = _get_product_text(product).lower()
    
    mount_keywords = ("floor stand", "standing bracket", "wall mount", "bracket", "支架", "吊架")
    if any(kw in text[:400] for kw in mount_keywords):
        if re.search(r"\b(interactive flat panel|digital display|led display|lcd display)\b", text[:400]):
            return True
        return False
    
    display_keywords = ("display", "screen", "panel", "led", "lcd", "cob", "oled", "会议一体机", "交互平板")
    if any(kw in text[:400] for kw in display_keywords):
        return True
    
    if re.search(r"\b(TW\d+|T\d{2}Omni|HG\d+)\b", text[:400], re.IGNORECASE):
        return True
    
    return True


def has_environment_conflict(product: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
    """Reject products whose environment conflicts with requirement."""
    metadata = _get_product_metadata(product)
    text = _get_product_text(product).lower()
    
    product_is_outdoor = metadata.get("outdoor") is True
    product_is_indoor = metadata.get("indoor") is True
    text_has_outdoor = bool(re.search(r"\boutdoor\b|户外", text))
    text_has_indoor = bool(re.search(r"\bindoor\b|室内", text))
    
    product_outdoor = product_is_outdoor or text_has_outdoor
    product_indoor = product_is_indoor or text_has_indoor
    
    if requirement.get("outdoor") and product_indoor and not product_outdoor:
        return True
    if requirement.get("indoor") and product_outdoor and not product_indoor:
        return True
    
    return False


def fallback_rerank(products: List[Dict[str, Any]], requirement: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep retrieval order while removing non-displays and conflicts."""
    display_only = [p for p in products if is_display_candidate(p)]
    result = [p for p in display_only if not has_environment_conflict(p, requirement)]
    return result


def rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to rerank retrieved chunks based on requirements."""
    requirement = state.get("requirement", {})
    raw_products = state.get("products", [])
    
    logger.info(f"Rerank received {len(raw_products)} products from retrieval")
    
    # Filter to displays only
    display_products = [p for p in raw_products if is_display_candidate(p)]
    logger.info(f"Filtered to {len(display_products)} display candidates")
    
    if not display_products:
        return {**state, "products": [], "next_action": "recommend"}
    
    # Filter environment conflicts
    candidates = [p for p in display_products if not has_environment_conflict(p, requirement)]
    
    if not candidates:
        candidates = display_products
    
    # Build candidates text
    lines = []
    for i, product in enumerate(candidates[:8], 1):
        text = _get_product_text(product)
        truncated = text[:600] if len(text) > 600 else text
        lines.append(f"{i}. 【产品资料】\n{truncated}")
    candidates_text = "\n\n".join(lines)
    
    req_parts = [f"{k}: {v}" for k, v in requirement.items() if v]
    requirement_text = ", ".join(req_parts) or "No specific requirements"
    
    try:
        response = get_llm(temperature=0).invoke(
            RERANK_PROMPT.format(requirement=requirement_text, candidates=candidates_text)
        )
        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0]
        
        parsed = json.loads(content.strip())
        selected_indices = parsed.get("selected_indices", []) if isinstance(parsed, dict) else []
        
        selected = []
        seen = set()
        for index in selected_indices:
            if not isinstance(index, int) or index in seen:
                continue
            if 1 <= index <= len(candidates):
                selected.append(candidates[index - 1])
                seen.add(index)
            if len(selected) == 3:
                break
        
        if not selected:
            selected = candidates[:3]
        
        return {**state, "products": selected, "next_action": "recommend"}
        
    except Exception as error:
        logger.warning(f"Reranking failed: {error}")
        return {
            **state,
            "products": fallback_rerank(candidates, requirement),
            "next_action": "recommend",
        }


def sanitize_customer_response(text: str, *, outdoor: bool = False) -> str:
    """Remove internal-data disclosures and environment-conflicting sentences."""
    if not text:
        return text

    kept_lines = []
    for line in str(text).splitlines():
        sentences = re.split(r"(?<=[。！？!?])", line)
        kept_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if _INTERNAL_DISCLOSURE_RE.search(sentence):
                continue
            if outdoor and _INDOOR_PRODUCT_RE.search(sentence):
                continue
            kept_sentences.append(sentence)
        cleaned_line = "".join(kept_sentences).strip()
        if cleaned_line:
            kept_lines.append(cleaned_line)
    return "\n".join(kept_lines).strip()


def get_product_category(product: Dict[str, Any]) -> Optional[str]:
    """Extract product category from product metadata."""
    metadata = _get_product_metadata(product)
    return metadata.get("category") or metadata.get("type") or metadata.get("display_type")


def is_ifp_product(product: Dict[str, Any]) -> bool:
    """Check if the product is an IFP (Interactive Flat Panel) product."""
    metadata = _get_product_metadata(product)
    text = _get_product_text(product).lower()
    
    # Check metadata
    category = get_product_category(product) or ""
    if "ifp" in category.lower() or "interactive" in category.lower():
        return True
    
    # Check text
    ifp_indicators = ["ifp", "interactive flat panel", "交互式平板", "会议平板", "触控一体机"]
    for indicator in ifp_indicators:
        if indicator in text:
            return True
    
    # Check model pattern (IFP products often have specific naming)
    text_content = _get_product_text(product)
    if re.search(r"\bIFP[-_]?\d", text_content, re.IGNORECASE):
        return True
    
    return False
