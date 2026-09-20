"""Others node - handle any user question outside the main workflow."""
from typing import Any, Dict, List
import json
import logging
import re

from ..state import SolutionState
from ....core.llm import get_llm

logger = logging.getLogger(__name__)

# Regex to extract model-like tokens
_MODEL_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)+)\b"
    r"|\b(TW\d+[\w.-]*\d[\w.-]*)\b",
)

OTHERS_PROMPT = """You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face. The customer asked a question — please answer directly.

Customer question: {question}
Model mentioned by customer (if any): {user_model}
Historical requirements (for context only): {requirement}
Product data (your internal knowledge — do not say "based on my records"):
{products}

Response rules:
- Conversational and natural, do not list things mechanically
- Answer only based on the provided product data — do not make up information not in the data
- If the data doesn't cover the topic, just say "I don't have specific info on that right now"
- Never output "top pick", "alternative", or any recommendation list format
- Professional yet friendly tone, like an experienced sales advisor chatting face-to-face
- Do not repeat what the customer has already stated
- Plain text only: no markdown at all
- Never say "based on my records", "I queried", "from the database" — do not reveal internal processes
- 【LANGUAGE RULE】{language_rule}
- 【Hallucination prevention】Every model name and every technical spec you mention must appear in the "Product data" below
- 【Full model name rule】Always output the complete model name, e.g. write "T65Omni-N4" or "TW21-COB-P0.9", never just the suffix
- 【Delivery rules】If the customer asks about delivery or lead time: counting from order and payment
  our delivery normally takes about 15–30 days. If they need it faster, we can ship by air — that
  shortens the delivery time but adds shipping cost. Never promise a specific date.
- 【Price question rules】Never quote or compare prices, and never say which model is cheaper
  or better value — pricing is handled separately by the sales team.
  If the customer asks about price or cost: say the exact price depends on the final model,
  configuration and quantity, and that we will prepare a formal quotation — then help them
  narrow down the technical requirements instead (brightness, pitch, size, installation).

Adjust answer length based on the question."""


def others_node(state: SolutionState) -> SolutionState:
    """Handle any user question outside the standard workflow."""
    messages = state.get("messages", [])
    question = ""
    
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in ("user", "human"):
            question = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
            break

    # Extract any model name the user mentioned
    user_asked_model = ""
    if question:
        m = _MODEL_RE.search(question)
        if m:
            user_asked_model = m.group(0)

    # Check if this is a price-related question
    price_keywords = ["价格", "多少钱", "便宜", "性价比", "贵", "价", "预算", "报价", "cost", "price"]
    is_price_question = any(kw in question.lower() for kw in price_keywords)

    # Retrieve from product knowledge base
    product_chunks = []
    hybrid_search = state.get("hybrid_search")

    if hybrid_search and question:
        try:
            # For price questions, search with price context
            if is_price_question:
                # Get recommended products from state
                recommended_products = state.get("products", [])
                product_ids = []
                for p in recommended_products:
                    mid = _MODEL_RE.search(_get_product_text(p))
                    if mid:
                        product_ids.append(mid.group(0))
                
                # Search specifically for price information
                search_query = f"{question} 价格 报价 性价比"
                product_chunks = hybrid_search.search(search_query, top_k=10)
                
                # If we have specific product IDs, try to find price info for those
                if product_ids:
                    for pid in product_ids:
                        price_chunks = hybrid_search.search(f"{pid} 价格", top_k=3)
                        for pc in price_chunks:
                            if pc not in product_chunks:
                                product_chunks.append(pc)
                logger.info(f"Others node price search retrieved {len(product_chunks)} chunks")
            else:
                context = " ".join(
                    message.get("content", "")
                    for message in messages[-4:]
                    if isinstance(message, dict) and message.get("role") == "assistant"
                )
                search_query = f"{context} {question}".strip()
                product_chunks = hybrid_search.search(search_query, top_k=6)
                logger.info(f"Others node retrieved {len(product_chunks)} chunks")
        except Exception as error:
            logger.warning("Others node retrieval failed: %s", error)

    # Format the retrieved products
    if product_chunks:
        products_text = "\n\n".join(
            f"[产品资料 {idx + 1}]\n{chunk.get('text', '')[:1200]}"
            for idx, chunk in enumerate(product_chunks)
        )
    else:
        products_text = "（未检索到相关产品资料）"

    # Generate answer
    try:
        from ....rag.query_understanding import response_language_rule
        from ....rag.reply_composer import reply_language

        response = get_llm(temperature=0.7).invoke(
            OTHERS_PROMPT.format(
                question=question,
                user_model=user_asked_model or "（未提及具体型号）",
                requirement=state.get("requirement", {}),
                products=products_text,
                # 默认策略下必须是纯英文（客户口径：不能出现任何一句中文）
                language_rule=response_language_rule(reply_language(question)),
            )
        )
        answer = response.content if hasattr(response, "content") else str(response)

        if "```" in answer:
            try:
                answer = answer.split("```", 2)[1]
                answer = answer.removeprefix("json").strip()
            except Exception:
                pass

        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = parsed["answer"]
        except json.JSONDecodeError:
            pass

    except Exception as error:
        logger.error("Others node answering failed: %s", error)
        answer = "Sorry, I had some trouble understanding that question. Could you try rephrasing it?"

    return {
        **state,
        "products": product_chunks,
        "recommendation": answer.strip(),
        "next_action": "end",
    }
