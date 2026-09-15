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


def _model_of(product) -> str:
    """从证据条目里取出型号（metadata.model → metadata.product_id → 文本解析）。"""
    metadata = _get_product_metadata(product)
    name = metadata.get("model") or metadata.get("product_id")
    if name:
        return str(name)
    text = _get_product_text(product)
    match = re.search(
        r"TW\d{2}-(?:IRHD|HOD|COB|3216|IR|OD)-P\d+(?:\.\d+)?(?:[HE])?(?:\(GOB\))?",
        text, re.IGNORECASE,
    )
    return match.group(0) if match else ""


def rank_evidence(
    products: List[Dict[str, Any]],
    selection: Optional[Dict[str, Any]] = None,
    profile: Any = None,
    limit: int = 3,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """v2.0 Phase 7：确定性的证据排序 + 冲突检测。

    职责边界（v2.0 明确要求）：
      - **不决定卖哪个产品**：顺序完全跟随 Recommendation Engine 的选型结果
      - 只做"证据排序"（把选中型号的 RAG 证据排到前面）
      - 只做"冲突检测"（丢弃环境冲突 / 非显示器证据）
      - **绝不突破硬约束**：冲突证据直接丢弃，不"复活"、不交给 LLM 解释

    Returns:
        (kept, dropped)：按引擎顺序排列的证据，以及被丢弃的条目
    """
    engine_order: List[str] = [
        str(rec.get("model"))
        for rec in (selection or {}).get("recommendations") or []
        if rec.get("model")
    ]
    requirement: Dict[str, Any] = {}
    if profile is not None:
        requirement = profile.to_facts() if hasattr(profile, "to_facts") else dict(profile)

    ranked: List[tuple[int, Dict[str, Any]]] = []
    dropped: List[Dict[str, Any]] = []

    for index, product in enumerate(products or []):
        name = _model_of(product)
        if not is_display_candidate(product):
            dropped.append({"model": name, "reason": "非显示产品"})
            continue
        if requirement and has_environment_conflict(product, requirement):
            dropped.append({"model": name, "reason": "环境与需求冲突"})
            continue
        position = engine_order.index(name) if name in engine_order else len(engine_order) + index
        ranked.append((position, product))

    ranked.sort(key=lambda item: item[0])
    kept = [product for _, product in ranked[:limit]]
    if dropped:
        logger.info(
            "Evidence rerank: dropped %d item(s): %s",
            len(dropped), [item["reason"] for item in dropped],
        )
    return kept, dropped


def rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """v2.0 Phase 7：Rerank 节点 = 证据排序 / 冲突检测。

    相比旧实现（把候选丢给 LLM 打分并"选 3 个"），现在：
      - 不调用 LLM，结果完全确定
      - 顺序跟随 Recommendation Engine，LLM 不再参与产品选择
      - 冲突证据直接丢弃，硬约束不可被突破
    """
    products = state.get("products") or []
    selection = state.get("recommendation_result") or {}
    profile = state.get("requirement_profile")

    kept, dropped = rank_evidence(products, selection, profile, limit=3)
    logger.info("Rerank(evidence): kept=%d dropped=%d", len(kept), len(dropped))
    return {
        **state,
        "products": kept,
        "evidence_dropped": dropped,
        "next_action": state.get("next_action", "recommend"),
    }


def sanitize_customer_response(text: str, *, outdoor: bool = False) -> str:
    """Remove internal-data disclosures and environment-conflicting sentences."""
    if not text:
        return text

    # 【客户口径】不允许对客户说"找不到 / 没有匹配的产品"：
    # 一旦出现这类话术，整段替换成"能不能放宽某个参数"的邀请（多种说法轮换）。
    try:
        from src.rag.reply_composer import has_no_product_phrase, relaxation_answer

        if has_no_product_phrase(text):
            logger.info("Rewriting 'no matching product' reply into a relaxation request")
            return relaxation_answer()
    except Exception:  # pragma: no cover - 防御式
        pass

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
