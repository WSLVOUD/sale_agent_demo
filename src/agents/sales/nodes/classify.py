"""classify node - intent classification for the Sales Agent."""
import logging
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import SalesState
from ....config import config
from ....rag.parameter_inference import detect_intent

logger = logging.getLogger(__name__)

# ── 真正的"结束对话"说法（只有这些才允许判成 closing）────────────────────────
# 教训（实测日志）：客户回答"close"（意思是"近"），被分类成 closing，
# 于是整轮被强制不推荐、只回了一句"我会准备报价"。短回答绝不能被当成结束。
_CLOSING_STOP_RE = re.compile(
    r"\b(?:bye|goodbye|see you|that'?s all|that is all|that'?s everything|"
    r"no thanks|no thank you|we'?re done|we are done|nothing else|end the chat|stop here)\b|"
    r"不用了|就这样|先这样|结束|再见|拜拜|没有其他|没有了|不需要了",
    re.IGNORECASE,
)

# "像在回答需求"的短回答 / 典型答法
_ANSWER_LIKE_RE = re.compile(
    r"^\s*(?:yes|no|ok|okay|sure|fine|yep|nope|maybe|"
    r"close|closer|closest|near|nearby|far|farther|medium|middle|"
    r"permanent|fixed|rental|indoor|outdoor|"
    r"近|远|很近|比较近|固定|租赁|室内|室外)\s*[.!。！]?\s*$|"
    r"\b(?:i (?:do ?n[o']t|do not) know|not sure|no idea|unsure)\b|"
    r"不知道|不清楚|不确定|没量过|"
    r"\d+(?:[.,]\d+)?\s*(?:m|meter|metre|ft|feet|inch|mm|cm|%|nit|瓦|寸|米|英尺|厘米|毫米)",
    re.IGNORECASE,
)


def looks_like_requirement_answer(message: str) -> bool:
    """客户这句话是不是"在回答需求问题"（而不是要结束对话）。

    短回答（"close" / "5m" / "permanent" / "室内"）也算 —— 这正是被误判的根源。
    """
    text = str(message or "").strip()
    if not text:
        return False
    words = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text)
    if len(words) <= 4:
        return True
    return bool(_ANSWER_LIKE_RE.search(text))


def is_explicit_closing(message: str) -> bool:
    """"我要结束了"的明确说法（只有这种才真的走收尾流程）。"""
    return bool(_CLOSING_STOP_RE.search(str(message or "")))


def _asked_product_type_question(session_id: str) -> tuple[bool, str]:
    """我们上一轮是不是在问"要 LED 还是 LCD"（读对话状态，不做关键词匹配）。

    返回 (是否问过类型, 上一轮问的原话)。
    """
    if not session_id:
        return False, ""
    try:
        from ....dialogue import get_conversation_state

        conversation_state = get_conversation_state(session_id)
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("Conversation state unavailable for product type: %s", exc)
        return False, ""
    if conversation_state is None:
        return False, ""
    slot = str(
        getattr(conversation_state, "last_question_slot", "")
        or getattr(conversation_state, "last_slot", "")
        or ""
    )
    question = str(getattr(conversation_state, "last_ai_question", "") or "")
    return slot == "display_type", question


def classify(state: SalesState) -> SalesState:
    """Classify the user's intent.

    Possible intents:
      - "greeting": user is just saying hello
      - "product_question": user asks about product specs, LED vs LCD comparison,
        capability (4K/HDR support), differences, specific model features
      - "need_query": user is describing requirements or asking for recommendations
      - "objection": user is raising an objection
      - "industry": user mentions a specific industry scenario
      - "closing": user wants to close / wrap up
      - "others": user is asking about anything OUTSIDE the standard sales flow
    """
    message = state["current_message"]

    # 【关键修复】先调用 LLM 分类，再检查 detect_intent。
    # 如果 detect_intent 检测到推荐意图，即使 LLM 分类为 others，也强制走推荐流程。
    detected = detect_intent(message)
    logger.info("detect_intent result: %r", detected)
    
    # 注意：不要在 LLM 调用前 return，否则 detect_intent 的返回值无法覆盖 LLM 的结果
    # 正确做法：先 LLM，再 override

    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=0,
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    prompt = SystemMessage(content="""你是销售意图分类器。分析用户消息，返回以下意图之一：
- greeting: 问候、打招呼（你好、hi、hello、谢谢、再见）—— 注意：「好的」「知道了」「明白了」等确认词不属于greeting，属于need_query
- product_question: 用户在问产品参数、功能、支持情况、LED和LCD的区别、某款产品能否做某事、显示效果好不好等——任何不是「帮我推荐/采购/选哪款」的提问
- need_query: 用户描述需求或请求推荐（我要会议室用的屏、想采购广告屏、帮我选一个）或确认需求（好的、知道了、明白了、对的、是的）
- objection: 提出针对销售方案的异议或询问销售政策（价格太贵、能不能便宜、报价有效期、付款方式、采购流程、售后政策、安装调试）
- industry: 提到特定行业场景（零售、教育、会议室、体育场馆等）
- closing: 准备下单或询问下一步（那就这样定了、下单、签合同）
- others: 项目正常流程无法处理的问题，比如公司信息（你们公司在哪里、联系方式）、工作时间（几点上班、节假日）、产品保修政策（保修多久、保修范围）、商务流程疑问（怎么签约、什么时候能交付）、闲聊话题等。

判断规则：
1. 只有当用户在同一句话里同时包含业务问题和其他问题时，才选择其他更精确的类别（如 product_question）。否则默认走 others。
2. 「价格能便宜点吗」「能再优惠点吗」「报价怎么样」属于 objection，不是 others。
3. 「保修多久」「保修几年」「质保范围」属于 objection，不是 others。
4. 「你们公司在哪里」「怎么联系你们」「几点上班」「工厂在哪」属于 others。
5. 「好的」「知道了」「明白了」「对的」「是的」等确认词属于 need_query，不属于 greeting。

只返回意图类型，不要其他内容。""")

    # 客户口径（2026-09-21）：不能只看当前这一句 —— 带上最近 50 条对话，
    # 让模型结合上下文判断"这句话到底在做什么"（回答？提问？要推荐？）。
    conversation = ""
    try:
        from ....memory.history_window import dialogue_window_text

        conversation = dialogue_window_text(str(state.get("session_id") or ""))
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("History window unavailable for classify: %s", exc)

    human_content = (
        f"最近的对话（越靠下越新）：\n{conversation}\n\n客户最新一句：{message}"
        if conversation
        else message
    )
    response = llm.invoke([prompt, HumanMessage(content=human_content)])
    intent = response.content.strip().lower()

    logger.info(f"Classified intent: {intent}")

    # 【关键修复】客户在回答需求（哪怕是"close"这种短回答）时，不能被判成"结束对话"。
    # 否则 requirement_mining 会强制不推荐，只回一句"我会准备报价"，
    # 既不推荐产品也不问尺寸（实测日志出现过）。
    if intent == "closing" and not is_explicit_closing(message) and looks_like_requirement_answer(message):
        logger.info(
            "Overriding LLM intent 'closing' → 'need_query' (message looks like a requirement answer: %r)",
            message[:60],
        )
        intent = "need_query"

    # 【关键修复】客户**明确要推荐**（推荐 / 帮我选 / 换一款 / recommend / quote…）时，
    # 不管 LLM 把它判成什么（others / industry / product_question / objection），
    # 都必须走推荐链路 —— 实测 bug（2026-09-21）：客户说"给我推荐"，LLM 判成
    # product_question → 被当成"提问"走去自由问答，只回了一句"我这就给你准备"，
    # 一个产品都没推荐。
    explicit_reco = False
    if intent != "need_query":
        try:
            from .requirement import _EXPLICIT_RECO_REQUEST_RE

            explicit_reco = bool(_EXPLICIT_RECO_REQUEST_RE.search(str(message or "")))
        except Exception:  # pragma: no cover - 防御式
            explicit_reco = False
    if intent != "need_query" and (
        explicit_reco or (detected == "recommendation" and intent in ("others", "industry"))
    ):
        if intent == "closing" and is_explicit_closing(message):
            pass  # 客户在明确结束对话 → 不强行改成推荐
        else:
            logger.info(
                "Overriding LLM intent '%s' → 'need_query' (explicit_reco=%s, detect_intent=%s)",
                intent, explicit_reco, detected or "-",
            )
            intent = "need_query"
    # 【兜底】如果 detect_intent 返回非推荐非空，且 LLM 分类为 others，但消息包含推荐关键词，也强制覆盖
    elif detected == "" and intent == "others":
        # 再次检查消息是否包含推荐关键词（独立于 detect_intent 的正则）
        recommendation_keywords = ["会议室", "教室", "培训", "广告", "屏", "LED", "LCD", "IFP", "手写", "触控",
                                   "舞台", "演出", "租赁", "户外", "室内", "需要", "采购", "推荐"]
        if any(kw in message for kw in recommendation_keywords):
            logger.info(f"Overriding LLM intent 'others' → 'need_query' (fallback keyword check)")
            intent = "need_query"
    
    # 【首次接待后修复】如果首次接待刚完成(suppress_greeting=True)且消息包含场景关键词，
    # 强制分类为 need_query 而不是 greeting，避免进入问候流程
    if state.get("suppress_greeting") and intent == "greeting":
        scenario_keywords = [
            "会议室", "教室", "培训", "会议", "学校", "培训室",
            "零售", "店铺", "商场", "商店", "超市",
            "广告", "传媒", "宣传",
            "舞台", "演出", "演唱会", "表演", "剧场",
            "体育", "赛场", "场馆", "球场",
            "展厅", "展览", "博物馆",
            "幕墙", "外墙", "建筑",
            "租赁", "活动", "临展",
            "LED", "LCD", "IFP", "显示屏", "屏幕",
            "屏", "大屏", "显示器",
            "室内", "室外", "户外",
            "meeting", "classroom", "retail", "store", "advertising", "stage", "LED", "LCD", "display",
            "screen", "outdoor", "indoor"
        ]
        if any(kw in message.lower() for kw in scenario_keywords):
            logger.info(f"Overriding LLM intent 'greeting' → 'need_query' (suppress_greeting + scenario keywords detected)")
            intent = "need_query"

    state["intent"] = intent

    # 计划 v2.9.2 §四/§七：统一回合理解 → 产品域路由（纯接口，不做 LCD/IFP 策略）
    try:
        from ....dialogue.product_type_router import (
            UNKNOWN as _DT_UNKNOWN,
            load_decision as _load_dt_decision,
            route_display_type as _route_display_type,
        )
        from ....dialogue.product_router import route_product_domain
        from ....dialogue.turn_understanding import (
            RequirementBook,
            understand_turn,
        )
        from ....memory.store import memory as _memory

        session_id = str(state.get("session_id") or "")
        # ── 计划 v2.9.3 §四/§五：第一层产品类型判断（LED / LCD / UNKNOWN）──
        # 优先级：客户明确 LED/LCD → IFP 特征(归 LCD) → 图片 → 用途推断 → 解释 → 询问；
        # 客户明确确认过就锁定（locked=True），只有客户明确改口才重新路由。
        previous_decision = (
            _load_dt_decision(_memory.get_display_type_decision(session_id))
            if session_id
            else None
        )
        vision_type = ""
        vision = state.get("vision")
        if isinstance(vision, dict):
            vision_type = str(vision.get("display_type") or "")

        # ── 客户口径（2026-09-24）：类型判断要读**语境**，不许靠关键字触发 ──────
        # 只有"我们上一轮确实在问'要 LED 还是 LCD'"且类型还没定的时候，才让模型
        # 结合语境判断客户这句话在干什么（chose / rejects / does_not_know /
        # asks_meaning / delegates / unrelated）。拿不到信号时路由会自动退回规则解析。
        reply_signal: dict = {}
        try:
            type_settled = (
                previous_decision is not None
                and previous_decision.display_type in ("LED", "LCD")
                and (
                    previous_decision.locked
                    or previous_decision.status == "CONFIRMED"
                )
            )
            asked_type, asked_question = _asked_product_type_question(session_id)
            if not type_settled and asked_type:
                from ....dialogue.product_type_understanding import (
                    understand_product_type_reply,
                )

                reply_signal = understand_product_type_reply(
                    message,
                    session_id=session_id,
                    conversation=conversation,
                    asked_question=asked_question,
                    decision=previous_decision,
                )
                if reply_signal:
                    logger.info(
                        "Product type reply (context): reply=%s type=%s reason=%s",
                        reply_signal.get("reply"), reply_signal.get("display_type"),
                        reply_signal.get("reason"),
                    )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Product type reply understanding failed: %s", exc)
            reply_signal = {}

        decision = _route_display_type(
            message,
            profile=state.get("requirement_profile"),
            vision_display_type=vision_type,
            vision_reason=str(
                (vision or {}).get("reason") if isinstance(vision, dict) else ""
            ),
            current=previous_decision,
            reply_signal=reply_signal,
        )
        state["display_type_decision"] = decision.to_dict()

        book = (
            RequirementBook.from_payload(_memory.get_requirement_book(session_id))
            if session_id
            else RequirementBook()
        )
        understanding = understand_turn(
            message,
            intent=intent,
            profile=state.get("requirement_profile"),
            book=book,
        )
        # IFP 属于 LCD（计划 §四/§十四）：第一层只记 LED / LCD；子类型单独留痕
        if decision.display_type in ("LED", "LCD"):
            understanding.product_domain = decision.display_type
            understanding.product_domains = [decision.display_type]
        elif decision.display_type == _DT_UNKNOWN:
            understanding.product_domain = _DT_UNKNOWN
            understanding.product_domains = []
        changes = book.apply(understanding)
        understanding.active_requirement = (
            str(changes.get("active_after") or "") or understanding.active_requirement
        )
        state["turn_understanding"] = understanding.to_dict()
        state["product_domain"] = understanding.product_domain
        state["product_subtype"] = decision.subtype
        state["turn_kind"] = understanding.conversation_type
        state["product_entry"] = route_product_domain(
            understanding.product_domain, comparison=understanding.comparison
        )
        state["product_switch"] = changes
        if session_id:
            _memory.set_requirement_book(session_id, {"requirements": book.snapshot()})
            _memory.set_display_type_decision(session_id, decision.to_dict())
        logger.info(
            "Turn understanding: intent=%s domain=%s subtype=%s status=%s locked=%s "
            "kind=%s entry=%s active=%s",
            intent, state["product_domain"], state["product_subtype"],
            decision.status, decision.locked, state["turn_kind"],
            state["product_entry"], understanding.active_requirement,
        )
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("turn understanding failed: %s", exc)

    # Set next_action so orchestrator can route appropriately
    if intent == "product_question":
        state["next_action"] = "product_question"
    elif intent == "others":
        state["next_action"] = "others"
    elif intent == "greeting":
        state["next_action"] = "ask"
    elif intent in ("objection", "industry"):
        state["next_action"] = "ask"
    elif intent == "closing":
        state["next_action"] = "end"
    else:
        state["next_action"] = "ask"  # need_query falls through to requirement_mining

    return state
