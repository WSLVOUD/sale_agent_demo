"""Recommendation node - generates product recommendations."""
from typing import Dict, Any, List
import logging
import re

from ..state import SolutionState
from ....core.llm import get_llm
from ....rag.rerank import is_display_candidate, get_product_category, has_environment_conflict, is_ifp_product
from ....utils.ifp_intent import has_ifp_intent, user_messages_text

logger = logging.getLogger(__name__)


RECOMMEND_PROMPT = """Customer needs: {requirement}

Product data:
{product}

{additional_context}

Generate one natural, conversational recommendation line that includes:
1. Complete product model name
2. 1-2 core selling points / reasons to recommend
3. If necessary, 1-2 key specs (e.g. pixel pitch, dimensions)

Requirements:
- Conversational, like recommending to a friend — do not recite the manual
- Greeting is only needed on the first recommendation; subsequent ones start directly with the product
- Strictly no more than 60 characters, shorter is better
- No markdown, no lists, no bold
- Never reveal internal info: do not say "I have the records", "the product data shows", "not in the data", "from the database", "I queried", etc.
- If a parameter is missing, do not explain the data gap or say "no specific info for this distance"; only recommend products you can confirm match, or briefly ask for one missing parameter
- Do not mention budget, price ranges, or cost
- 【Full model name rule】Always output the complete model name — never just the suffix
- Do not repeat or restate what the customer already stated (e.g. if they've already said outdoor, stage, or indoor, don't repeat those words)
- Only mention positive reasons — never say "not suitable", "not recommended", "doesn't match", or any negative phrasing
- Always answer based on the product data — do not make up content not in the data
- 【Hard environment rule】Do not repeat the customer's stated environment (outdoor, indoor, stage, etc.) in your reply. Output the model and core selling points directly. If the product data genuinely has no matching product for the customer's environment, say "No matching products found in the database" — do not substitute indoor products for outdoor needs or vice versa.

Output the recommendation directly (2-3 sentences max):"""


FOLLOW_UP_PROMPT = """Based on the recommended products, generate 1 short follow-up sentence.

Already recommended: {recommendations}

Requirements:
1. 1 sentence, conversational, like chatting with a friend
2. No emoji, no bold, no lists
3. No more than 15 words
4. Never ask about budget, price, or cost
5. Generate only from the content of already-recommended products
6. Never make up price ranges, cost tiers, or value comparisons not in the data.

Output directly:"""


def _get_product_text(product) -> str:
    """Extract text from a product dict or Document object."""
    if hasattr(product, "page_content"):
        return product.page_content
    if isinstance(product, dict):
        return product.get("text", "")
    return str(product)


def _get_product_metadata(product) -> Dict[str, Any]:
    """Extract metadata from a product dict or Document object."""
    if hasattr(product, "metadata"):
        return product.metadata or {}
    if isinstance(product, dict):
        return product.get("metadata", {}) or {}
    return {}


def _normalize_text(text: str) -> str:
    """Normalize common punctuation variants for regex matching."""
    return (text
        .replace("\uff1a", ":")
        .replace("\u3000", " ")
        .replace("\u2011", "-")
        .replace("\u2010", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-"))


def product_identifier(product, user_asked_model: str = "") -> str:
    """Extract the real product model name from a retrieved chunk."""
    if hasattr(product, "page_content"):
        text = _normalize_text(product.page_content)
        metadata = product.metadata or {}
    else:
        metadata = _get_product_metadata(product)
        text = _normalize_text(_get_product_text(product))

    # Check if user asked about a specific model
    if user_asked_model:
        asked_normalized = _normalize_text(user_asked_model)
        if asked_normalized in text or asked_normalized.upper() in text.upper():
            return user_asked_model

    # Try various patterns
    for pattern in [
        r"(?mi)^Model:\s*(.+?)$",
        r"(?mi)^Product internal models:\s*(.+?)$",
        r"(?mi)^Product:\s*(.+?)$",
    ]:
        for match in re.finditer(pattern, text, re.MULTILINE):
            candidate = match.group(1).strip().split(";")[0].split("|")[0].split(",")[0].strip()
            candidate = re.sub(r"\s+Series\s*$", "", candidate, flags=re.IGNORECASE)
            if candidate:
                return candidate

    # LED file pattern: TW21-COB-P0.6
    m = re.search(r"(?m)^(TW\d+[\w.-]*)\b", text)
    if m:
        return m.group(1).strip()

    # IFP pattern: TxxOmni-Xn
    m = re.search(r"(?m)^(T\d{2}Omni[\w\u2011\u2010\u2012\u2013\u2014\-\.]+\d[\w\u2011\u2010\u2012\u2013\u2014\-\.]*)\b", text)
    if m:
        model = m.group(1).strip()
        for hyphen in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2015", "\ufe63", "\uff0d"):
            model = model.replace(hyphen, "-")
        return model

    # Last resort: uppercase token with digits and hyphen
    m = re.search(r"(?:^|\s)([A-Z][A-Z0-9/\-'.\u2013\u2014]*\d[A-Z0-9/\-'.\u2013\u2014]*)\b", text)
    if m:
        return m.group(1).strip()

    return metadata.get("id") or ""


def _normalize_model(model: str) -> str:
    """Normalize a model identifier for deduplication."""
    if not model:
        return ""
    raw = model.split(",")[0].strip()
    for hyphen in ("\u2011", "\u2010", "\u2012", "\u2013", "\u2014", "\u2015", "\ufe63", "\uff0d"):
        raw = raw.replace(hyphen, "-")
    return raw.upper()


def named_products(products: list, limit: int = 3, user_asked_model: str = "") -> list:
    """Keep the highest-ranked unique product models, up to ``limit`` entries."""
    named = []
    seen_keys = set()
    for product in products:
        model = product_identifier(product, user_asked_model=user_asked_model)
        key = _normalize_model(model)
        if not model:
            metadata = _get_product_metadata(product)
            text = _get_product_text(product)
            fallback_key = _normalize_model(metadata.get("id") or text[:80])
            if fallback_key in seen_keys:
                continue
            seen_keys.add(fallback_key)
            named.append(product)
        else:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            named.append(product)
        if len(named) == limit:
            break
    return named


def _generate_follow_up(llm, recommendations_text: str, requirement_text: str) -> str:
    """Generate a natural conversational follow-up after the recommendation list."""
    try:
        prompt = FOLLOW_UP_PROMPT.format(
            requirement=requirement_text or "未明确",
            recommendations=recommendations_text,
        )
        response = llm.invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        text = text.strip("\"'`").strip()
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        if len(text) > 60:
            text = text[:60].rstrip("，,;；") + "。"
        return text if len(text) <= 200 else ""
    except Exception as error:
        logger.warning("Follow-up generation failed: %s", error)
        return ""


def recommend_node(state: SolutionState) -> SolutionState:
    """Generate a recommendation with full model name, specs and usage scenario."""
    requirement = state.get("requirement", {})
    raw_products = state.get("products", [])
    customer_text = user_messages_text(state.get("messages", []))
    if not customer_text:
        customer_text = str(state.get("current_message", ""))
    
    ifp_allowed = has_ifp_intent(requirement, user_text=customer_text if customer_text else None)
    logger.info(f"Recommend received {len(raw_products)} products from state")

    # Defensive filter: only keep display-category products
    display_products = [p for p in raw_products if is_display_candidate(p)]
    logger.info("Recommend: dropped %d non-display products", len(raw_products) - len(display_products))

    # Hard environment filter
    if requirement.get("outdoor"):
        display_products = [p for p in display_products if not has_environment_conflict(p, requirement)]
    elif requirement.get("indoor"):
        display_products = [p for p in display_products if not has_environment_conflict(p, requirement)]

    # IFP filter - only filter if user explicitly doesn't want IFP
    # If user explicitly said "don't want IFP" or "no need for touch", filter IFP
    if not ifp_allowed:
        # Check if user explicitly rejects IFP
        explicit_no_ifp = False
        no_ifp_patterns = [
            r"no ifp", r"no interactive", r"no touch", r"no whiteboard",
            r"don't.*ifp", r"don't.*interactive", r"don't.*touch",
            r"only.*led", r"only.*lcd",
        ]
        for pattern in no_ifp_patterns:
            if re.search(pattern, (customer_text or "").lower()):
                explicit_no_ifp = True
                break

        if explicit_no_ifp:
            pre_ifp = len(display_products)
            display_products = [p for p in display_products if not is_ifp_product(p)]
            logger.info("Recommend: dropped %d IFP products — user explicitly doesn't want IFP", pre_ifp - len(display_products))
        else:
            # User hasn't explicitly rejected IFP — keep IFP products
            logger.info("Recommend: keeping IFP products (user wants interaction features)")

    products = named_products(display_products)
    logger.info(f"Recommend: {len(products)} named products after filtering")

    if not products:
        # Relax filter: if it's a conference scenario without IFP, try LCD products
        purpose = str(requirement.get("purpose", "")).lower()
        is_conference = any(kw in purpose for kw in ["meeting", "conference", "classroom", "training", "exhibition"])
        has_interaction = any(kw in (customer_text or "") for kw in ["touch", "interactive", "whiteboard", "annotation"])

        if is_conference and has_interaction:
            logger.info("Conference with interaction but no IFP found, trying LCD fallback")
            lcd_products = [p for p in raw_products if "lcd" in _get_product_text(p).lower() or "LCD" in _get_product_metadata(p).get("display_type", "")]
            if lcd_products:
                products = named_products(lcd_products)
                logger.info(f"LCD fallback: {len(products)} LCD products found")

        if not products:
            all_displays = [p for p in raw_products if is_display_candidate(p)]
            if all_displays:
                products = named_products(all_displays)
                logger.info(f"Fallback: using all {len(products)} display products")
            else:
                # No products at all — return a friendly message
                return {
                    "recommendation": "Based on your needs, I can match standard products for meeting rooms and classrooms. Could you share more specific size requirements, or tell me how many people the space needs to accommodate? That'll help me narrow it down.",
                    "next_action": "reflect"
                }

    llm = get_llm(temperature=0.3)
    req_str = ", ".join([f"{k}: {v}" for k, v in requirement.items() if v])
    
    additional_reqs = state.get("additional_requirements", [])
    if additional_reqs:
        additional_context = f"\nCustomer additional requirements: {', '.join(additional_reqs)}\nPlease recommend product features that specifically address these requirements."
    else:
        additional_context = ""

    recommendations = []
    for rank, product in enumerate(products, 1):
        model = product_identifier(product)
        prompt = RECOMMEND_PROMPT.format(
            product=_get_product_text(product)[:600],
            requirement=req_str,
            additional_context=additional_context,
        )

        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip().replace("**", "").replace("__", "")
            content = re.sub(r"```[a-zA-Z]*", "", content).replace("```", "").strip()
            
            if rank > 1:
                content = re.sub(r"^您好[，！。\s]+", "", content)
                content = re.sub(r"^你好[，！。\s]+", "", content)
            
            if len(content) > 80:
                cut = re.search(r"[。！？!?]", content)
                if cut and cut.end() <= 80:
                    content = content[: cut.end()]
                else:
                    content = content[:80].rstrip("，,;；") + "。"
            
            recommendations.append(content)
        except Exception as error:
            logger.error("Recommendation generation failed for %s: %s", model, error)
            recommendations.append(f"Alternative {rank}: {model}.")

    recommendations_text = "\n".join(recommendations)
    follow_up = _generate_follow_up(llm, recommendations_text, req_str)
    if follow_up:
        recommendations_text = f"{recommendations_text}\n{follow_up}"

    return {
        "products": products,
        "recommendation": recommendations_text,
        "next_action": "reflect"
    }
