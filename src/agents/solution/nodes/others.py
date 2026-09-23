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

Recent conversation (oldest first; the customer is labelled 客户, you are labelled AI).
This is the context: read it to understand what the customer means — especially when
their message is short (yes / ok / a typo / one or two words) or when it answers
something you asked.
{recent_dialogue}

Customer question: {question}
Model mentioned by customer (if any): {user_model}
Historical requirements (for context only): {requirement}
Confirmed customer requirements (must NOT be contradicted):
{confirmed}
Product data (your internal knowledge — do not say "based on my records"):
{products}

Response rules:
- Conversational and natural, do not list things mechanically
- Read the "Recent conversation" first. If the customer is answering the question you
  asked last, treat their reply as that answer (even a short "yes" or a typo) and move
  the conversation forward accordingly. Never re-ask something already answered.
- Anything listed under "Confirmed customer requirements" is known: do not ask the
  customer for it again and do not ask them to confirm it a second time.
- If part of the picture is still missing, ask about that missing part only — never
  re-confirm what is already confirmed above.
- Answer only based on the provided product data — do not make up information not in the data
- If the data doesn't cover the topic, just say "I don't have specific info on that right now"
- Never output "top pick", "alternative", or any recommendation list format
- Never contradict the confirmed requirements above (for example: if the customer is
  indoor, do not talk about outdoor cabinets; if it is a fixed install, do not
  talk about rental panels)
- Do not name any product model unless the customer explicitly asked about that
  model — models are only given in a formal recommendation step
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
  configuration and quantity, and that we will prepare a formal quotation. Only mention
  what is still unknown (never size / installation / environment if they are already in
  "Confirmed customer requirements"); if nothing is missing, simply confirm you are
  preparing the quotation.

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

    # ── 客户口径（2026-09-22）：自由问答也必须"看得见语境" ────────────────
    # 实测 bug：客户回 "yes" 是在确认上一句 AI 的"要不要出报价"，但这条路径
    # 拿不到任何历史 → 只能按字面理解（"go ahead"），接着又把已经答过的
    # 尺寸 / 安装方式问了一遍。现在把最近 50 条对话（带 role）喂给模型，
    # 与 Sales 侧的 classify / requirement / 话术出口用同一个窗口。
    try:
        from ....memory.history_window import dialogue_window_text

        recent_dialogue = dialogue_window_text(
            str(state.get("session_id") or ""), max_items=50
        )
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Others node history window failed: %s", exc)
        recent_dialogue = ""
    if not recent_dialogue:
        recent_dialogue = "（暂无历史对话）"

    # Retrieve from product knowledge base
    product_chunks = []
    hybrid_search = state.get("hybrid_search")

    # 客户口径（2026-09-21）：自由问答也要遵守"已确认需求"——
    # 实测 bug：客户是室内教堂固装，回答里却出现 outdoor 型号（TW21-OD-P10）。
    from ....rag.model_guard import (
        drop_conflicting_chunks,
        requirement_lines,
        retrieval_filters,
        strip_environment_contradictions,
        strip_model_mentions,
    )

    requirements = state.get("requirement") or {}
    # 已确认需求的权威来源是 Sales 的 RequirementProfile；没有才退回 legacy 字典
    # （两个来源 requirement_summary 都认，见 src/rag/model_guard.py）。
    profile = state.get("requirement_profile")
    guard_source = profile if profile is not None else requirements
    guard_filters = retrieval_filters(guard_source)

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
                product_chunks = hybrid_search.search(
                    search_query, top_k=10, filters=guard_filters or None
                )
                
                # If we have specific product IDs, try to find price info for those
                if product_ids:
                    for pid in product_ids:
                        price_chunks = hybrid_search.search(
                            f"{pid} 价格", top_k=3, filters=guard_filters or None
                        )
                        for pc in price_chunks:
                            if pc not in product_chunks:
                                product_chunks.append(pc)
                logger.info(f"Others node price search retrieved {len(product_chunks)} chunks")
            else:
                # 检索关键词也要带语境：客户一句 "yes" 单靠自己检索不到任何东西，
                # 必须靠"刚才在聊什么"（最近几轮，用户 + AI 都算）。
                recent_turns = [
                    str(item.get("content") or "")
                    for item in (state.get("messages") or [])[-6:]
                    if isinstance(item, dict) and item.get("content")
                ]
                context = " ".join(recent_turns)
                search_query = f"{context} {question}".strip()
                product_chunks = hybrid_search.search(
                    search_query, top_k=6, filters=guard_filters or None
                )
                logger.info(f"Others node retrieved {len(product_chunks)} chunks")
        except Exception as error:
            logger.warning("Others node retrieval failed: %s", error)

    # 与已确认需求矛盾的片段直接丢掉（不让模型看见）
    product_chunks, dropped = drop_conflicting_chunks(
        list(product_chunks or []), guard_source
    )
    if dropped:
        logger.info("[ModelGuard] 自由问答丢掉 %d 条与需求相反的片段", dropped)

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

        # ── 计划 v2.9.1 §十一~§十三（Step 8/9/10）─────────────────────────
        # Others 不再自己拼 Prompt + 自己调 LLM + 自己格式化，而是把"问答所需
        # 的业务上下文"交给统一的 ResponseGenerator（内部还会过 Semantic Validator）：
        #
        #     Free Question → ResponseContext → ResponseGenerator → Validator
        #
        # RAG / ModelGuard / 需求约束（Step 7）保持不变，仍然在这一层完成。
        from ....dialogue import ResponseContext
        from ....dialogue.response_generator import generate_response
        from ....dialogue.turn_kind import UNKNOWN as _UNKNOWN_DOMAIN

        free_context = ResponseContext(
            action="FREE_QUESTION",
            customer_message=question,
            business_goal=(
                "answer the customer's question using ONLY the provided product data; "
                "do not recommend a model, do not mention models unless the customer "
                "named one, do not ask requirement questions"
            ),
            known_facts=list(requirement_lines(guard_source)),
            recent_dialogue=recent_dialogue,
            product_domain=str(state.get("product_domain") or _UNKNOWN_DOMAIN),
            language=reply_language(question),
            restrictions=[
                "do_not_invent_facts",
                "do_not_recommend_models",
                "do_not_ask_requirement_questions",
                "do_not_contradict_confirmed_requirements",
            ],
        )
        # 检索到的产品资料 = 这一轮要传达的事实（facts only，措辞交给 LLM）
        free_context.answer = products_text
        answer = generate_response(free_context, llm=get_llm(temperature=0.7))
        if not answer:
            # 兜底不能把内部标记（"（未检索到相关产品资料）"）发给客户
            answer = (
                products_text
                if product_chunks
                else "I don't have the specifics on that right now - let me check with "
                "our team and come back to you."
            )
        # 客户没点名型号 → 自由问答里不给型号（型号只由推荐链路给出）
        answer, removed_models = strip_model_mentions(
            answer, allow=[user_asked_model] if user_asked_model else ()
        )
        if removed_models:
            logger.info("[ModelGuard] 自由问答去掉型号：%s", removed_models)
        answer, removed_conflicts = strip_environment_contradictions(answer, requirements)
        if removed_conflicts:
            logger.info("[ModelGuard] 自由问答去掉与需求矛盾的说法：%s", removed_conflicts)

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
