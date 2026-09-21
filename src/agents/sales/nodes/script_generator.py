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
5. {language_rule}

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
        from ....rag.query_understanding import response_language_rule

        docs = sales_search.similarity_search(message, k=2)
        reference_content = "\n\n".join([d.page_content for d in docs[:1]])
        prompt = SystemMessage(content=f"""You are a sales advisor for LED and LCD display products, speaking with a customer face-to-face.

Reference talking points (for reference only):
{reference_content}

Customer says: {message}

Requirements:
1. Conversational and natural, like chatting with a friend
2. Natural and conversational, 2-4 sentences (do not answer with one bare line)
3. Plain text only, no markdown
4. Never invent specifications, prices or model names
5. {response_language_rule(reply_language(message))}

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


# ⚠️ 已弃用（2026-09-21，客户口径："明确授权 LLM 自己组织接话和问法"）：
# 这段提示词要求"必须用一个过渡（So / By the way / That said）把接话和问句连起来"，
# 与新一代 NATIVE 提示词（明确禁止这些套路化开头、允许 LLM 自己组织说法）方向相反。
# 现在**没有任何调用点**（唯一使用者 _polish_question_message 也已停用）；
# 保留在此仅作历史记录，不要重新接回主链路。
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
5b. **不要举例**：不要罗列"会议室 / 教室 / 商场 / 广告位"这类例子，也不要写
   "比如 / 例如 / such as / for example / like a …" 的开头——直接问问题本身。
6. **不要**新增任何参数、型号、价格、方案、承诺或建议；不要解释任何知识；
   不要提"数据库/资料/检索"之类内部说法。
7. 不要保留草稿里那种固定铺垫（例如 "That's okay —"、
   "While we're at it,"、"Meanwhile —"），换成人话过渡；但不要跑题。
8. 语言要求：{language_rule}
9. 结构要求：{plan_rules}
10. 直接输出这段话，不要 JSON、不要引号、不要解释。"""


def _plan_rules_for_prompt(state: SalesState) -> str:
    """v2.3 §19~§22：把 Response Planner 的表达结构翻译成给 LLM 的规则。"""
    plan = state.get("response_plan") or {}
    if not isinstance(plan, dict):
        return "先自然接住客户这句话，再问这一个问题；不要像问卷，不要复述需求清单"
    blocks = " → ".join(str(item) for item in (plan.get("blocks") or []))
    parts = []
    if blocks:
        parts.append(f"按「{blocks}」的顺序组织这一段")
    why = str(plan.get("why") or "").strip()
    if why:
        parts.append(f"用半句话说明为什么问它有用（{why}）")
    parts.append("不要复述整份需求清单")
    parts.append("不要像问卷，也不要连着问第二个问题")
    return "；".join(parts)


def _recent_question_list(state: SalesState, limit: int = 3) -> list:
    """最近几轮已经发给客户的话（最近的在前）。"""
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
        lines.append(text[:160])
        if len(lines) >= limit:
            break
    return lines


def _recent_question_texts(state: SalesState, limit: int = 3) -> str:
    """最近几轮已经发给客户的话（给改写作"不要重复"的参考，字符串形态）。"""
    lines = _recent_question_list(state, limit=limit)
    return "\n".join(f"- {item}" for item in reversed(lines)) or "（暂无）"


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
        plan_rules = _plan_rules_for_prompt(state)
        prompt = _QUESTION_POLISH_PROMPT.format(
            draft=draft,
            message=message[:200],
            requirement=state.get("requirements") or {},
            recent=_recent_question_texts(state),
            language_rule=response_language_rule(language),
            plan_rules=plan_rules,
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


# ── v2.5++（僵硬话术优化 §9）：话术唯一出口 ─────────────────────────────
# 业务决策（问什么 / 答什么 / 推什么）由 Python 定；措辞由 LLM 从结构化上下文
# 原生生成（DialogueAction → ResponseContext → ResponseGenerator → Validator）。
# 旧模板链路只在"生成失败 / 特殊轮次"时兜底（§16：先建新链路，旧链路留作回退）。


def _dialogue_llm():
    """有凭据才拿 LLM；没有就返回 None（走结构化兜底，不发请求）。"""
    try:
        import os as _os

        if not _os.getenv("DEEPSEEK_API_KEY"):
            from ....config import config as _config

            if not getattr(_config, "DEEPSEEK_API_KEY", ""):
                return None
        return get_llm(temperature=_question_temperature())
    except Exception:  # pragma: no cover - 防御式
        return None


def _known_facts(state: SalesState) -> list:
    """客户已经确认的事实（给 LLM 当上下文，不让它自己猜）。"""
    profile = state.get("requirement_profile")
    if profile is None:
        requirements = state.get("requirements") or {}
        return [f"{key}={value}" for key, value in requirements.items() if value]
    items = []
    for slot, field in (
        ("display_type", "display_type"),
        ("environment", "environment"),
        ("purpose", "purpose"),
        ("installation", "installation"),
        ("viewing_distance", "viewing_distance_m"),
        ("size", "target_size"),
        ("pixel_pitch", "pixel_pitch_mm"),
        ("budget", "budget_level"),
    ):
        try:
            if not profile.slot_is_confirmed(slot):
                continue
        except Exception:  # pragma: no cover - 防御式
            continue
        if slot == "size":
            value = f"{profile.target_width_m}m x {profile.target_height_m}m"
        else:
            value = getattr(profile, field, None)
        if value in (None, "", [], {}):
            continue
        items.append(f"{slot}={value}")
    return items


# ── v2.5++（僵硬话术优化 §9/§11）：只保留"有内容"的接话 ──────────────────
# reply_composer.acknowledge 仍然是有用的 Fact / Format 工具（公司办事处、产品有没有
# 某规格这类**事实回答**都从它来）。但它是"接话 + 复述 + 客套"三合一的旧写法：
#   · 泛客套（"Got it" / "Thanks"）→ 不再作为固定开场（LLM 自己决定要不要客套）
#   · 机械复述客户刚说的话        → 不再带上（客户口径：不要每轮复述）
# 只有"事实回答"（既不是客套、也不是复述）才作为可选开场交给 LLM / 兜底拼装。
def _opening_for(state: SalesState, current_message: str, *, allow_ack: bool = True) -> str:
    if not allow_ack:
        return ""
    try:
        from ....dialogue import echo_ratio
        from ....dialogue.response_validator import GENERIC_ACK_RE
        from ....rag.reply_composer import acknowledge, reply_language

        acknowledgement = str(state.get("acknowledgement") or "").strip()
        if not acknowledgement:
            acknowledgement = str(
                acknowledge(
                    current_message,
                    requirement=state.get("requirements") or {},
                    language=reply_language(current_message),
                    seed=_turn_seed(state),
                    llm_ack="",
                    slot=str(state.get("pending_slot") or ""),
                )
                or ""
            ).strip()
        if not acknowledgement:
            return ""
        if GENERIC_ACK_RE.search(acknowledgement):
            return ""
        if echo_ratio(acknowledgement, current_message) >= 0.5:
            return ""
        return acknowledgement
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("[Dialogue] opening line failed: %s", exc)
        return str(state.get("acknowledgement") or "")


def _natural_reply(
    state: SalesState,
    *,
    answer: str = "",
    question: str = "",
    slot: str = "",
    allow_ack: bool = True,
    business_goal: str = "",
) -> str:
    """唯一话术出口：结构化上下文 → LLM 原生生成 → 校验 → 结构化拼装 → 旧模板兜底。"""
    from ....dialogue import (
        ANSWER_AND_ASK,
        ASK,
        DIRECT_ANSWER,
        build_context,
        generate_response,
    )
    from ....dialogue.grounded_facts import build_grounded_facts
    from ....rag.readiness import question_intent

    current_message = str(state.get("current_message") or "")
    slot = slot or str(state.get("pending_slot") or "")
    if answer and question:
        action = ANSWER_AND_ASK
    elif answer:
        action = DIRECT_ANSWER
    else:
        action = ASK
    # v2.5+++（计划 §5）：把"有来源的事实"整理好交给 LLM —— 它只能用这些
    profile = state.get("requirement_profile")
    grounded = build_grounded_facts(
        profile=profile,
        recommendation=state.get("recommendation") or None,
        calculations=state.get("screen_calculation") or None,
    )
    context = build_context(
        action=action,
        customer_message=current_message,
        question=question,
        # v2.7 修订（客户口径）：问句以"槽位 + 意图"交给 LLM，由它自己组织说法；
        # 系统给的成句只当意思锚点（prompt 里明确要求不要照抄）。
        question_slot=slot,
        question_intent=question_intent(slot) if slot else "",
        recent_questions=_recent_question_list(state),
        answer=answer,
        grounded_facts=grounded,
        pitch_resolution=state.get("pitch_resolution") or None,
        # 系统已经生成的"接话"（例如对自我介绍的回应）：LLM 可自行决定要不要用，
        # 无 LLM 时结构化拼装会带上它（不丢内容，也不再强制每轮都接话）
        opening=_opening_for(state, current_message, allow_ack=allow_ack),
        newly_confirmed={},
        missing_fields=[slot] if slot else [],
        known_facts=_known_facts(state),
        business_goal=business_goal,
        required_question=slot,
        language="en",          # 客户口径：对客户始终说英文
        restrictions=[
            "ask_only_one_question",
            "phrasing_is_yours_do_not_copy_canned_lines",
            "you_may_react_naturally_in_your_own_words",
            "do_not_invent_facts",
            "do_not_repeat_customer_unnecessarily",
        ],
        style="natural_b2b_sales",
    )
    context.allow_ack = allow_ack
    context.allow_connector = allow_ack
    text = ""
    try:
        text = generate_response(context, llm=_dialogue_llm(), seed=_turn_seed(state))
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("[Dialogue] generate_response failed: %s", exc)
        text = ""
    if text:
        # 记录来源：LLM 写的句子不再被"去僵硬"清洗回头改（那是模板才需要的）
        state["response_source"] = "llm"
        return _strip_markdown(text)
    # 兜底：旧模板链路（保证一定有话可说）
    state["response_source"] = "template"
    return _strip_markdown(
        compose_requirement_reply(
            answer=answer,
            question=question,
            slot=slot,
            message=current_message,
            seed=_turn_seed(state),
            requirement=state.get("requirements") or {},
            include_ack=allow_ack,
            llm_ack=str(state.get("acknowledgement") or ""),
            vision_confirmation="",
        )
    )


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
            confirmation = _vision_confirmation(state)
            draft = compose_requirement_reply(
                question=pending,
                slot=str(state.get("pending_slot") or ""),
                message=str(state.get("current_message") or ""),
                seed=_turn_seed(state),
                requirement=state.get("requirements") or {},
                llm_ack=ack,
                # 本轮带了图片 → 先把"图片里看到什么"跟客户核一遍，再问需求
                # （实测 bug：只发了图片 + "i need this"，系统直接跳到问点间距）
                vision_confirmation=confirmation,
            )
            # 图片确认句是系统精心写好的，不要让 LLM 改写（改写了也不好核对）
            if confirmation:
                # 图片确认句是系统写好的事实核对，保持原样
                state["response"] = _strip_markdown(draft)
            else:
                # v2.5++：唯一话术出口（结构化上下文 → LLM 原生生成 → 校验 → 回退）
                state["response"] = _natural_reply(
                    state,
                    question=pending,
                    business_goal="acknowledge the customer, then keep collecting requirements",
                )
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
        # v2.5++++（计划 §16.5）：客户问交期 → 先答交期，**不得无理由追加**需求问题
        try:
            from ....dialogue.policy import should_append_requirement_question

            if not should_append_requirement_question(
                current_message_text, state.get("requirement_profile")
            ):
                pending = ""
                logger.info("[DialoguePolicy] answer_only：交期问题本轮不追加需求问题")
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[DialoguePolicy] 判定失败：%s", exc)
        state["response"] = _natural_reply(
            state,
            answer=delivery_reply,
            question=pending,
            slot=str(state.get("pending_slot") or ""),
            allow_ack=False,
            business_goal="answer the delivery or schedule question, then continue",
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
        confirmation = _vision_confirmation(state)
        draft = compose_requirement_reply(
            question=pending_question,
            slot=str(state.get("pending_slot") or ""),
            message=str(state.get("current_message") or ""),
            seed=_turn_seed(state),
            requirement=state.get("requirements") or {},
            llm_ack=str(state.get("acknowledgement") or ""),
            # 需求重置时 runner 已经加过"我们重新来一遍"的确认语，这里不重复
            include_ack=not state.get("requirements_reset"),
            vision_confirmation=confirmation,
        )
        # 客户口径：不要原封不动发模板，也不要是"接话 + 提问"两个画风 ——
        # 用低温度（默认 0.2）围绕这版草稿改写成一段自然的话（意思不变）。
        # 需求重置时 runner 已经加了"我们重新来一遍"的确认语；这里再让 LLM 改写
        # 容易又叠一句客套（实测："Sure, let's collect… Sure, happy to help…"）
        # 带图的这一轮也不改写：图片确认句要保持原样，客户才好核对。
        if state.get("requirements_reset") or confirmation:
            # 需求重置时 runner 已经加过确认语；带图那一轮要保持图片核对句原样
            state["response"] = _strip_markdown(draft)
        else:
            # v2.5++：唯一话术出口（结构化上下文 → LLM 原生生成 → 校验 → 回退）
            state["response"] = _natural_reply(
                state,
                question=pending_question,
                business_goal="collect the missing hard requirement",
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
            # 直接询问场景用途，进入需求挖掘流程（客户口径：不举例，直接问）
            response_text = "What will the screen mainly be used for?"
            state["next_action"] = "ask"
            logger.info("Greeting suppressed (first contact completed), asking about scenario directly")
        else:
            from ....rag.query_understanding import response_language_rule

            prompt = SystemMessage(
                content=GREETING_PROMPT.format(
                    language_rule=response_language_rule(reply_language(state["current_message"]))
                )
            )
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
            # 客户口径：问场景不举例，直接问问题
            response_text = "What will this display mainly be used for?"
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
            # v2.5++++（计划 §9）：客户问价格 → 先答价格口径，**不**硬塞需求问题
            try:
                from ....dialogue.policy import should_append_requirement_question

                if not should_append_requirement_question(
                    current_message, state.get("requirement_profile")
                ):
                    pending_question = ""
                    logger.info("[DialoguePolicy] answer_only：价格/交期问题本轮不追加需求问题")
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[DialoguePolicy] 判定失败：%s", exc)
            state["response"] = _natural_reply(
                state,
                answer=answer,
                question=pending_question,
                slot=str(state.get("pending_slot") or ""),
                allow_ack=False,
                business_goal="answer the customer's question, then continue collecting",
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
