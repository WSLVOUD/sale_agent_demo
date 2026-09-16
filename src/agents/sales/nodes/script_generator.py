"""script_generator node - generate the sales response text."""
import logging
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....core.llm import get_llm
from ....rag.reply_composer import (
    compose_requirement_reply,
    is_price_question,
    price_policy_answer,
    reply_language,
)

logger = logging.getLogger(__name__)


def _is_company_question(message: str) -> bool:
    """公司 / 办事处 / 地址类提问（必须按 company_profile.txt 照实回答）。"""
    from ....rag.company_info import is_company_question

    return is_company_question(message)


GREETING_PROMPT = """You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face.

Requirements:
1. Simple greeting, one sentence
2. Friendly tone
3. Naturally ask about their needs
4. Plain text only, no markdown

Output the reply directly:"""


def _strip_markdown(text: str) -> str:
    """Remove common markdown artifacts so the chat reply is plain text."""
    if not text:
        return text
    cleaned = text.replace("**", "").replace("__", "")
    cleaned = re.sub(r"`+([^`]*?)`+", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    return cleaned.strip()


def _turn_seed(state: SalesState) -> int:
    """提问 / 回应话术的轮换种子：客户说过几句话 + 会话标识（保证同一会话内逐轮换）。"""
    session_id = str(state.get("session_id") or "")
    user_turns = sum(
        1
        for msg in (state.get("messages") or [])
        if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", ""))
        in ("user", "human")
    )
    return user_turns + (sum(ord(ch) for ch in session_id) % 7)


def _answer_objection_like(state: SalesState, message: str) -> str:
    """异议 / 行业类问题的答复（话术库检索 + LLM 生成，只回复不推荐）。"""
    sales_search = state.get("sales_search")
    if not sales_search:
        return "Got it, let me learn more about your needs."
    try:
        docs = sales_search.similarity_search(message, k=2)
        reference_content = "\n\n".join([d.page_content for d in docs[:1]])
        prompt = SystemMessage(content=f"""You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face.

Reference talking points (for reference only):
{reference_content}

Customer says: {message}

Requirements:
1. Conversational and natural, like chatting with a friend
2. Short, 1-2 sentences
3. Plain text only, no markdown
4. Never invent specifications, prices or model names

Reply directly:""")
        response = get_llm(temperature=0.3).invoke([prompt])
        return response.content.strip()
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Objection answer failed: %s", exc)
        return "Got it, let me learn more about your needs."


def _answer_company_question(state: SalesState) -> str:
    """公司 / 办事处 / 地址类提问 → 按 company_profile.txt 照实回答 + 接回需求问题。"""
    from ....rag.company_info import company_answer

    current_message = str(state.get("current_message") or "")
    answer = company_answer(
        current_message,
        language=reply_language(current_message),
        seed=_turn_seed(state),
    ) or ""
    return _strip_markdown(
        compose_requirement_reply(
            answer=answer,
            question=str(state.get("pending_question") or ""),
            slot=str(state.get("pending_slot") or ""),
            message=current_message,
            seed=_turn_seed(state),
            requirement=state.get("requirements") or {},
            vision_confirmation=_vision_confirmation(state),
        )
    )


def _vision_confirmation(state: SalesState) -> str:
    """带图的那一轮：把"图片里看到了什么"跟客户核一遍（客户口径：识别完要确认）。

    客户说"对"→ 这些字段记为**客户确认**；说别的 → 以客户说的为准并记录纠正。
    见 vision.integration.resolve_vision_confirmation。
    """
    if not state.get("vision_applied"):
        return ""
    profile = state.get("requirement_profile")
    if profile is None:
        return ""
    try:
        from ....rag.reply_composer import reply_language, vision_confirmation_sentence

        return vision_confirmation_sentence(
            profile,
            reply_language(str(state.get("current_message") or "")),
            _turn_seed(state),
        )
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Vision confirmation sentence failed: %s", exc)
        return ""


def script_generator(state: SalesState) -> SalesState:
    """Produce the final user-facing response."""
    # Check if router has already processed
    if state.get("solutions") and state.get("response"):
        logger.info("Router has generated response with solutions, keeping it")
        # 带图那一轮即使直接给了推荐，也要把"图片里看到什么"跟客户核一遍
        confirmation = _vision_confirmation(state)
        if confirmation and confirmation not in str(state["response"]):
            state["response"] = f"{state['response']} {confirmation}".strip()
        state["next_action"] = "trigger_solution"  # Keep trigger signal for orchestrator
        return state
    
    intent = state["intent"]
    
    # 检查是否需要抑制问候语（首次接待刚完成后）
    suppress_greeting = state.get("suppress_greeting", False)

    # ── 交付时间 / 安装档期（客户口径）──────────────────────────────────────
    # 客户问交期 → 从下单付款开始计算，常规交付约 15–30 天；
    # 客户要求加快 → 可以走空运，能提前但成本会增加；
    # 客户说"想 11 月安装"这类档期 → 先接住他的话，再把交期说清楚（不承诺具体日期）。
    current_message_text = str(state.get("current_message") or "")
    from ....rag.delivery_info import delivery_answer, install_timing_note

    delivery_reply = delivery_answer(
        current_message_text,
        language=reply_language(current_message_text),
        seed=_turn_seed(state),
    ) or install_timing_note(
        current_message_text,
        language=reply_language(current_message_text),
        seed=_turn_seed(state),
    )
    if delivery_reply:
        pending = str(state.get("pending_question") or "")
        state["response"] = _strip_markdown(
            compose_requirement_reply(
                answer=delivery_reply,
                question=pending,
                slot=str(state.get("pending_slot") or ""),
                message=current_message_text,
                seed=_turn_seed(state),
                requirement=state.get("requirements") or {},
                # 交付口径本身就是"接住客户这句话"（含档期复述），不再叠 LLM 客套，
                # 否则会出现 "Got it… Got it, 11月 is your target…" 这种重复。
                include_ack=False,
                vision_confirmation=_vision_confirmation(state),
            )
        )
        state["next_action"] = "ask"
        logger.info("Delivery / lead-time reply: %s", state["response"])
        return state

    # ── Phase 7：需求采集 —— 一次只问一个高价值问题 ─────────────────────────
    # 由 question_planner / Ready Gate 决定"问什么"；
    # 回复由 reply_composer 合成："先接住客户这句话 + 再追问"，
    # 避免销售只会重复问问题（不调用 LLM，措辞按轮次轮换）。
    # 注意：product_question / others 不在这一支 —— 它们要先让 Solution Agent
    # 用 RAG 回答客户的问题，再由 orchestrator 把追问接在答复后面。
    pending_question = state.get("pending_question")
    if (
        pending_question
        and intent in ("need_query", "greeting", "industry")
        and not state.get("should_generate_solution")
        and not state.get("response")
    ):
        state["response"] = _strip_markdown(
            compose_requirement_reply(
                question=pending_question,
                slot=str(state.get("pending_slot") or ""),
                message=str(state.get("current_message") or ""),
                seed=_turn_seed(state),
                requirement=state.get("requirements") or {},
                llm_ack=str(state.get("acknowledgement") or ""),
                # 需求重置时 runner 已经加过"我们重新来一遍"的确认语，这里不重复
                include_ack=not state.get("requirements_reset"),
                vision_confirmation=_vision_confirmation(state),
            )
        )
        state["next_action"] = "ask"
        logger.info(
            "Requirement mining continues — reply: %s", state["response"]
        )
        return state
    
    response_text = ""
    
    logger.info(f"script_generator: intent={intent}, should_generate_solution={state.get('should_generate_solution')}, requirements={state.get('requirements')}, suppress_greeting={suppress_greeting}")
    
    # Greeting - 如果首次接待刚完成则跳过
    if intent == "greeting":
        if suppress_greeting:
            # 首次接待已完成，不需要再次问候，也不要再问姓名（首次接待已问过）
            # 直接询问场景用途，进入需求挖掘流程
            response_text = "What display scenario are you looking into? Meeting room, classroom, retail, advertising...?"
            state["next_action"] = "ask"
            logger.info("Greeting suppressed (first contact completed), asking about scenario directly")
        else:
            prompt = SystemMessage(content=GREETING_PROMPT)
            response = get_llm(temperature=0.3).invoke([prompt, HumanMessage(content=state["current_message"])])
            response_text = response.content.strip()
            state["next_action"] = "ask"
    
    # Need query
    elif intent == "need_query":
        # 【重要】是否推荐**只由 Ready Gate 决定**。
        # 旧版这里用 "有 usage 就推荐"，还会从人数/面积推算视距 —— 那正是
        # "客户只说了场景就被推荐"以及"AI 自己猜参数"的根因，已删除。
        # Gate 放行时由 graph 直接走 router，不会走到这里；
        # 走到这里只可能是"Gate 未就绪 + 没有待问项"的兜底 → 只问基础场景，绝不推荐。
        if state.get("should_generate_solution", False):
            state["next_action"] = "trigger_solution"
            logger.info("need_query: Ready Gate 已放行 → 触发推荐")
        else:
            response_text = (
                "What kind of scenario will this display be used in? "
                "Meeting room, classroom, exhibition, advertising, etc.?"
            )
            state["next_action"] = "ask"
            state["response"] = _strip_markdown(response_text)
            logger.info("need_query: Gate 未就绪且无待问项 → 追问基础场景（不推荐）")

    # Objection / Industry
    elif intent in ["objection", "industry"]:
        requirements = state.get("requirements", {})
        current_message = str(state.get("current_message") or "")

        # 是否推荐只看 Ready Gate（旧版这里是 `or requirements.get("usage")`，
        # 导致"客户只说了场景 + 问个价格"也会被拉去推荐，实测日志出现过）
        if state.get("should_generate_solution", False):
            if not state.get("response"):
                state["response"] = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("Objection/industry: Ready Gate 已放行 → 触发推荐")
        elif _is_company_question(current_message):
            # 公司 / 办事处 / 地址类提问：按公司信息照实回答，再接回需求问题，
            # 不走没有公司资料的自由问答（否则会答"我没有相关信息"或乱答 Yes）
            state["response"] = _answer_company_question(state)
            state["next_action"] = "ask"
            logger.info("Company question — answered from company profile: %s", state["response"])
        else:
            # 需求还没问清时问价格 → 先说明"要确认产品才能报价"，紧接着继续问需求
            if is_price_question(current_message):
                answer = price_policy_answer(
                    language=reply_language(current_message), seed=_turn_seed(state)
                )
                logger.info(
                    "Price question while collecting requirements — quote policy + next question"
                )
            else:
                answer = _answer_objection_like(state, current_message)

            pending_question = str(state.get("pending_question") or "")
            state["response"] = _strip_markdown(
                compose_requirement_reply(
                    answer=answer,
                    question=pending_question,
                    slot=str(state.get("pending_slot") or ""),
                    message=current_message,
                    seed=_turn_seed(state),
                    requirement=requirements,
                    llm_ack=str(state.get("acknowledgement") or ""),
                    vision_confirmation=_vision_confirmation(state),
                )
            )
            state["next_action"] = "ask"
            logger.info("Objection/industry without ready gate — reply: %s", state["response"])

    # Product question
    elif intent == "product_question":
        # Check if requirements are sufficient to trigger solution
        if state.get("should_generate_solution", False):
            response_text = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("product_question with sufficient requirements, triggering solution")
        elif _is_company_question(state.get("current_message", "")):
            state["response"] = _answer_company_question(state)
            state["next_action"] = "ask"
            logger.info("Company question (product_question) — answered from company profile")
        else:
            response_text = "Sure."
            state["next_action"] = "product_question"

    # Others
    elif intent == "others":
        # Check if requirements are sufficient to trigger solution
        if state.get("should_generate_solution", False):
            response_text = "Sure, let me find the right products for you..."
            state["next_action"] = "trigger_solution"
            logger.info("others with sufficient requirements, triggering solution")
        elif _is_company_question(state.get("current_message", "")):
            state["response"] = _answer_company_question(state)
            state["next_action"] = "ask"
            logger.info("Company question (others) — answered from company profile")
        else:
            response_text = "Sure."
            state["next_action"] = "others"

    # Closing
    elif intent == "closing":
        response_text = "Alright, I'll put together a detailed proposal and quote for you as soon as possible. Feel free to ask if you have any other questions."
        state["next_action"] = "end"

    else:
        response_text = "What display scenario are you looking into?"
        state["next_action"] = "ask"
    
    # Only set response if it hasn't been set by router
    if not state.get("response"):
        state["response"] = _strip_markdown(response_text)
    
    logger.info(f"Generated response for intent={intent}, next_action={state['next_action']}")
    return state
