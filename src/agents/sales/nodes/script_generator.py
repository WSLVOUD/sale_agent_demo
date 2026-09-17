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


_QUESTION_POLISH_PROMPT = """你是 LED 显示屏产品的销售，正在微信上和客户聊天。
下面这行"草稿"是系统已经决定要问客户的问题（问什么由系统定，不能改）。
请把它说得自然、口语化，并让它和"接住客户这句话"自然地连成一段（最多两句），
像真人销售一口气说出来的话。

草稿：{draft}
客户刚说：{message}
已知需求（仅供判断语气，不要复述、不要新增）：{requirement}
最近已经发出去的话（**不要**再用它们的说法、开头和过渡词）：
{recent}

硬性规则：
1. 询问的意思必须和草稿**完全一致**：不能换成别的问题，不能多问，也不能少问。
2. 整段话里只能有**一个问句**（问号最多一个）。
3. **必须有自然的过渡**：先接住客户这句话，再用一个过渡（例如 So / By the way / That said /
   Now / 那么 / 顺便 / 话说回来）接到这一问上，读起来是**一段连贯的话**，
   不能像两句话硬拼在一起。
4. **必须换一种说法**（这是重点）：不要照抄草稿的句式和用词 ——
   用同义词替换、把否定/疑问换个说法、把语序调一调，
   也可以换成同样意思的另一种问法（例如 "Will it be indoors or outdoors?"
   → "Is this an indoor job or an outdoor one?" / "Are we going indoors or outdoors?"）。
   意思一模一样，但读起来不能是同一句话。
5. **过渡词也要换着用**：不要每轮都用同一个（例如老用 "So" / "By the way"）；
   最近用过的开头和过渡必须避开，从第 1 条里挑别的或者用别的自然说法。
6. **不要**新增任何参数、型号、价格、方案、承诺或建议；不要解释任何知识；
   不要提"数据库/资料/检索"之类内部说法。
7. 不要保留草稿里那种固定铺垫（例如 "That's okay —"、
   "While we're at it,"、"Meanwhile —"），换成人话过渡；但不要跑题。
8. 语言要求：{language_rule}
9. 直接输出这段话，不要 JSON、不要引号、不要解释。"""


def _recent_question_texts(state: SalesState, limit: int = 3) -> str:
    """最近几轮已经发给客户的话（给改写作"不要重复"的参考）。"""
    lines: list[str] = []
    for item in reversed(state.get("messages") or []):
        role = ""
        content = ""
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "")
            content = str(item.get("content") or "")
        else:  # pragma: no cover - LangChain 消息对象
            role = str(getattr(item, "type", "") or "")
            content = str(getattr(item, "content", "") or "")
        if role not in ("assistant", "ai"):
            continue
        text = content.strip()
        if not text:
            continue
        lines.append(f"- {text[:160]}")
        if len(lines) >= limit:
            break
    return "\n".join(reversed(lines)) or "（暂无）"


def _question_temperature() -> float:
    from ....config import config as _config

    try:
        return float(getattr(_config, "QUESTION_TEMPERATURE", 0.2))
    except (TypeError, ValueError):  # pragma: no cover - 防御式
        return 0.2


_MODEL_CODE_RE = re.compile(r"\bTW\s*\d{2}\s*-", re.IGNORECASE)
_PRICE_WORD_RE = re.compile(r"price|cost|报价|价格|多少钱|美元|\$", re.IGNORECASE)


def _polish_question_message(
    draft: str,
    *,
    state: SalesState,
) -> str:
    """把"要问的问题"改写成一段自然的话（不照抄模板），失败则返回空串。

    问什么完全由 Gate/模板决定；这里只改措辞，并保证：
      - 与"接住客户这句话"衔接成一段（不是两个画风）；
      - 只有一个问句；
      - 不新增参数 / 型号 / 价格 / 建议。
    """
    draft = str(draft or "").strip()
    if not draft:
        return ""
    try:
        from ....rag.query_understanding import response_language_rule
    except Exception:  # pragma: no cover - 防御式
        return ""
    try:
        message = str(state.get("current_message") or "")
        language = reply_language(message)
        llm = get_llm(temperature=_question_temperature())
        prompt = _QUESTION_POLISH_PROMPT.format(
            draft=draft,
            message=message[:200],
            requirement=state.get("requirements") or {},
            recent=_recent_question_texts(state),
            language_rule=response_language_rule(language),
        )
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = re.sub(r"```[a-zA-Z]*", "", str(text)).replace("```", "").strip()
        text = text.strip('"\'“”').replace("**", "").replace("__", "").strip()

        if not text or "{" in text or "}" in text:
            return ""
        if len(text) > 320:
            return ""
        if text.count("?") + text.count("？") != 1:
            # 多问 / 没问 → 说明改跑偏了，退回模板
            logger.info("Polished question dropped (question count != 1): %r", text[:80])
            return ""
        if _MODEL_CODE_RE.search(text) or _PRICE_WORD_RE.search(text):
            logger.info("Polished question dropped (mentions model/price): %r", text[:80])
            return ""
        return text
    except Exception as exc:
        logger.warning("Question polish failed: %s", exc)
        return ""


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

    # ── 客户说了与需求无关的话：只"接住这句" + 继续问需求 ─────────────────
    # （接话话术由较高温度单独生成；这里绝不去回答无关问题、也不倒产品）
    if state.get("offtopic_turn"):
        ack = str(state.get("acknowledgement") or "")
        pending = str(state.get("pending_question") or "")
        if pending:
            draft = compose_requirement_reply(
                question=pending,
                slot=str(state.get("pending_slot") or ""),
                message=str(state.get("current_message") or ""),
                seed=_turn_seed(state),
                requirement=state.get("requirements") or {},
                llm_ack=ack,
            )
            polished = _polish_question_message(draft, state=state)
            state["response"] = _strip_markdown(polished or draft)
        else:
            state["response"] = _strip_markdown(ack)
        state["next_action"] = "ask"
        logger.info("Off-topic turn — reply: %s", state["response"])
        return state

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
        # 先按模板合成一版（问什么由 Gate 决定，这版是"意思基准"，也是兜底）
        draft = compose_requirement_reply(
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
        # 客户口径：不要原封不动发模板，也不要是"接话 + 提问"两个画风 ——
        # 用低温度（默认 0.2）围绕这版草稿改写成一段自然的话（意思不变）。
        polished = _polish_question_message(draft, state=state)
        state["response"] = _strip_markdown(polished or draft)
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
