"""
Orchestrator - Dual Agent Coordination
Coordinates between Sales Agent, Solution Agent, and First Contact Flow
"""
import logging
import uuid
from typing import Any, Dict, List, Optional

from .agents.sales.runner import SalesAgentRunner
from .agents.solution.runner import SolutionAgentRunner
from .memory.store import memory as shared_memory
from .first_contact.handler import first_contact_handler
from .first_contact.profile import load_profile

logger = logging.getLogger(__name__)


def _turn_product_family(profile: Any) -> str:
    """这一轮属于哪条产品链路（led / lcd / ifp）—— 给售后 / 交期 / 闲聊口径分家。"""
    try:
        from .utils.product_family import product_family_of

        return product_family_of(profile)
    except Exception:  # pragma: no cover - 防御式
        return ""

# 注：确认 / 纠正这类"与需求相关"的 SpeechAct 口径已迁到对话层
# （`dialogue.continuation_budget.REQUIREMENT_RELATED_SPEECH_ACTS`）。
# ── v2.3.1 Phase 2：业务逻辑已迁出，这里只做编排与兼容 ────────────────────────
from .observability.perf import PerfTracker  # noqa: E402,F401  (按性能埋点，从本模块迁出)
from .vision.pipeline import (  # noqa: E402,F401  (Vision 接入，从本模块迁出)
    _merge_vision_into_stored_profile,
    _vision_enabled,
    vision_display_type_payload,
)


# ── 语言配置（v2.0 Phase 14）────────────────────────────────────────────────
# 由 config.RESPONSE_LANGUAGE_POLICY 控制：
#   "en"   → 所有 AI 回复强制英语（系统既有策略，默认）
#   "auto" → 跟随客户语言回复（v2.0 的 Original Language Response）
try:
    from .config import config as _config

    RESPONSE_LANGUAGE = _config.RESPONSE_LANGUAGE_POLICY
except Exception:  # pragma: no cover - 防御式
    RESPONSE_LANGUAGE = "en"
    
class DualAgentOrchestrator:
    """
    Orchestrates between Sales Agent and Solution Agent.
    
    Flow:
    1. Sales Agent handles initial dialogue, requirement gathering
    2. When requirements are sufficient, triggers Solution Agent
    3. Solution Agent returns product recommendations
    4. Sales Agent wraps up with recommendation + follow-up
    """
    
    def __init__(
        self,
        sales_agent: SalesAgentRunner,
        solution_agent: SolutionAgentRunner
    ):
        """
        Initialize orchestrator with both agents.
        
        Args:
            sales_agent: Sales Agent instance
            solution_agent: Solution Agent instance
        """
        self.sales_agent = sales_agent
        self.solution_agent = solution_agent
        # 重要：所有模块必须引用同一个 memory 实例
        self.memory_store = shared_memory

        # Inject dependencies into sales agent
        self.sales_agent.memory_store = self.memory_store
        self.sales_agent.solution_runner = self.solution_agent
        
        logger.info("Dual Agent Orchestrator initialized")
    
    def process_message(
        self,
        message: str,
        session_id: str,
        history: List[Dict[str, Any]] = None,
        images: Optional[List[Any]] = None,
        message_count: int = 1,
        aggregated: bool = False,
        turn_id: str = "",
        turn_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process user message through appropriate agent(s).

        Args:
            message: User's message
            session_id: Session identifier
            history: Conversation history (optional)
            images: 客户随消息发送的图片（可选，最多 VISION_MAX_IMAGES 张）
            turn_id: v2.7：本次执行所属的 Turn（TurnExecutor 分配；重放/并发时
                保证所有日志、LLM 记账、重复提问判定都挂在同一个 turn 上）
            turn_context: v2.7：Turn 引擎带上来的上下文（message_ids /
                previous_question / source …）

        Returns:
            Dict with response and metadata
        """
        perf = PerfTracker(session_id, message)
        # v2.6 §24/§27：这一轮的 turn_id（FinalResponse / 日志 / DecisionAudit 共用）
        # v2.7：TurnExecutor 传来的 turn_id 优先（一个 Turn = 一次业务决策）
        turn_context = dict(turn_context or {})
        turn_id = str(turn_id or turn_context.get("turn_id") or "").strip() or uuid.uuid4().hex[:12]
        perf.turn_id = turn_id
        # ── v2.6 §5~§10：客户这一句在回答上一轮哪个问题 ─────────────────────
        self._note_customer_turn(session_id, message, turn_id=turn_id)
        # v2.5++++（计划 §15）：一轮的 LLM 调用统一记账（这一轮里所有 LLM 都算上）
        try:
            from .observability.llm_tracker import get_llm_tracker

            get_llm_tracker().begin_turn(
                session_id,
                message_count=message_count,
                aggregated=aggregated,
                turn_id=turn_id,
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("LLM tracker begin_turn failed: %s", exc)

        # ── 纯图片消息（计划第二十四阶段 Case 1）────────────────────────────
        # 客户只发了图片、没有文字时，用一句"照片已收到"的话驱动需求采集，
        # 这样 Sales Agent 会正常回答 + 继续问缺失项，而不是拿空字符串走自由问答。
        # 注意：这里不编造任何需求，只是描述"客户发了一张照片"这个事实。
        customer_text = message
        if images and not str(message or "").strip():
            message = "I've sent a photo of the screen I'm interested in."

        # ── 视觉需求提取（《智谱视觉需求提取接入实施计划》第九/十四阶段）──────
        # Orchestrator 只负责"协调"：把图片交给 Vision 模块，拿到结构化结果；
        # 视觉失败绝不影响主流程（第十九阶段）。
        vision_results: List[Any] = []
        vision_metrics: Dict[str, Any] = {}
        if images and _vision_enabled():
            from .vision import extract_vision_for_turn

            try:
                vision_results, vision_metrics = extract_vision_for_turn(
                    images, session_id=session_id, customer_text=customer_text
                )
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[%s] Vision pipeline failed: %s", session_id, exc)
                vision_results, vision_metrics = [], {"error": str(exc)}
            logger.info(
                "[%s] Vision metrics: images=%s latency_ms=%s explicit=%s inferred=%s null=%s success=%s error=%s",
                session_id,
                vision_metrics.get("images"),
                vision_metrics.get("vision_latency_ms"),
                vision_metrics.get("fields_extracted"),
                vision_metrics.get("fields_inferred"),
                vision_metrics.get("fields_null"),
                vision_metrics.get("vision_success"),
                vision_metrics.get("error") or "-",
            )
            perf.vision_metrics = vision_metrics
        elif images:
            logger.info("[%s] Vision disabled — images ignored", session_id)

        # ── 计划 v2.9.4 §六：图片识别出的产品类型要接进第一层判断 ──────────────
        # 以前只传 has_vision 布尔量，图片判断出的 LED/LCD/IFP 到不了 classify，
        # Product Type Router 的"③ 图片判断"分支在真实链路上是死代码。
        vision_payload: Dict[str, Any] = {}
        if vision_results:
            vision_payload = vision_display_type_payload(vision_results)
            if vision_payload:
                logger.info(
                    "[%s] Vision display type → 第一层判断：type=%s source=%s",
                    session_id, vision_payload.get("display_type"), vision_payload.get("source"),
                )

        # ── First Contact 检查 ────────────────────────────────────────────────
        # 客户第一次发送消息时，执行固定接待流程
        # 注意：首次接待流程必须完整执行，不允许任何其他 Agent 介入
        if not self.memory_store.is_first_contact_done(session_id):
            logger.info("[%s] First contact detected, running fixed flow...", session_id)

            # 执行首次接待流程（强制使用英语）
            # v2.0 Phase 14：策略为 auto 时按客户语言回复，否则保持英语
            if RESPONSE_LANGUAGE == "auto":
                try:
                    from .rag.query_understanding import detect_language

                    fc_language = detect_language(message)
                except Exception:  # pragma: no cover - 防御式
                    fc_language = "en"
            else:
                fc_language = "en"

            fc_result = first_contact_handler.run(
                session_id=session_id,
                customer_message=message,
                language=fc_language,
            )

            # ── 记住触发首次接待的那句话里的需求 ──────────────────────────
            # 首次接待仍然完整执行（不跳过、不进入 Sales Agent）。
            # 用规则槽位固化客户已经说过的事实（LED / 8x6 feet / shop …），
            # 下一轮追问时不再把这些当成没说过。
            initial_requirements: Dict[str, Any] = {}
            try:
                from .models.requirement import RequirementProfile
                from .rag.query_understanding import extract_slots

                first_slots = {
                    key: value
                    for key, value in (extract_slots(message) or {}).items()
                    if not str(key).startswith("_") and value not in (None, "", [], {})
                }
                if first_slots or vision_results:
                    profile = RequirementProfile.from_slots(
                        first_slots, explicit_keys=set(first_slots)
                    )
                    # 首轮就带图片：图片里的需求同样要保存（第十五阶段：
                    # 不能因为图片消息而绕过 First Contact，但也不能丢掉图片信息）
                    if vision_results:
                        from .vision import apply_vision_to_profile

                        for result in vision_results:
                            profile, vision_stats = apply_vision_to_profile(profile, result)
                            logger.info(
                                "[%s] First-contact vision merged: %s", session_id, vision_stats
                            )
                        vision_metrics["merge_success"] = True
                    if hasattr(self.memory_store, "set_requirement_profile"):
                        self.memory_store.set_requirement_profile(session_id, profile)
                    from .models.legacy_adapter import profile_to_legacy

                    initial_requirements = profile_to_legacy(profile)
                    self.memory_store.set_requirements(session_id, initial_requirements)
                    logger.info(
                        "[%s] Remembered first-contact requirements: %s",
                        session_id, initial_requirements,
                    )
            except Exception as exc:
                logger.warning("[%s] Failed to remember first-contact requirements: %s", session_id, exc)
                initial_requirements = {}

            # 记录客户的首条消息 + 首次接待生成的消息到 memory
            # （否则后续轮次的历史里会缺失客户的第一句话）
            self.memory_store.add(session_id, "user", message)
            fc_messages = fc_result.to_messages()
            for msg in fc_messages:
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
                if role and content:
                    self.memory_store.add(session_id, role, content)

            # 标记首次接待完成
            self.memory_store.mark_first_contact_done(session_id)

            # 将首次接待结果存入 perf，供调用方返回
            perf.first_contact_messages = fc_messages
            perf.first_contact_intro = fc_result.intro_text
            logger.info(
                "[%s] First contact completed: intro_ok=%s all_assets_success=%s",
                session_id, fc_result.intro_success, fc_result.all_success
            )

            # 【关键修复】首次接待完成后，直接返回，不运行 Sales Agent
            # 首次接待是固定流程，必须完整执行，不允许 Sales Agent 在同一轮介入
            # 下一轮客户消息才会正常进入 Sales Agent
            perf.mark("sales_done")
            perf.route = "first_contact"
            perf.intent = "first_contact"
            result = {
                "response": fc_result.intro_text,
                "agent": "first_contact",
                "route": "first_contact",
                "complexity": "fixed_flow",
                "requirements": initial_requirements,
                "products": [],
                "next_action": "first_contact_done",
                # v2.6 §4/§24：素材通道单独记录（不是第二条对话回复）
                "first_contact_messages": fc_messages,
            }
            self._finalize_turn_response(result, session_id, message, turn_context)
            result["_perf"] = perf.summary()
            if vision_metrics:
                result["vision"] = vision_metrics
            return result

        # ── Sales Agent（仅在首次接待完成后执行）────────────────────────────
        # 图片识别出的需求先并入 memory 里的需求档案，Sales Agent 下一行就能看到
        _merge_vision_into_stored_profile(
            self.memory_store, session_id, vision_results, vision_metrics
        )

        # ── 一个项目多条屏（客户口径 2026-09-18）─────────────────────────────
        #  ① 一句话里给了两块屏的规格（"4m x2.5 indoor and 3m x2m outdoor"）→ 分别记录
        #  ② 客户指明"改那一块"（"把室外那块改成 3m x 2m"）→ 切到那一块再改
        #  ③ 客户自己提到另一块屏 → 开一条新记录（没指明差异时两块记成一样的）
        multi = self._multi_screen()
        multi_specs = multi.split_and_apply_screen_specs(session_id, message)
        if not multi_specs and multi.maybe_target_screen(session_id, message) is None:
            multi.maybe_start_new_item(session_id, message)

        # Step 1: Sales Agent processes message
        # v2.7 §18：记下"这一轮之前"已有哪些需求 → 之后 diff 出 newly_filled_slots
        from .dialogue import newly_filled_slots, profile_slot_map

        profile_before = profile_slot_map(self._stored_profile(session_id))
        sales_result = self.sales_agent.run(
            session_id=session_id,
            message=message,
            has_vision=bool(vision_results),
            vision=vision_payload,
        )
        perf.mark("sales_done")
        # 客户口径：同一个问题全项目最多问两次 —— 客户答过的共有项要同步到其他屏，
        # 否则两块屏会各问一遍（实测被连问 4 次）。
        # 只填空值，绝不覆盖：能用 P3/P5、固装/租赁 等参数区分两块屏时，这些差异保留。
        multi.share_common_facts(session_id)
        perf.intent = sales_result.get("intent", "")
        result_newly_filled = newly_filled_slots(
            profile_before, profile_slot_map(self._stored_profile(session_id))
        )

        # 多屏拆分的那一轮：Sales 的需求抽取只看到"整句话"，会把两块屏的参数
        # 混到当前这条档案里 —— 这里按拆分结果把每块屏的参数重新写回去。
        if multi_specs:
            multi.split_and_apply_screen_specs(session_id, message)
        # 客户口径：多块屏时"没说清哪块要什么"就两块记成一样的 —— 所以客户
        # 答过一次的共有项（视频/图片、安装方式、视距、价格取向…）要同步到
        # 其他屏，绝不能因为另一块"还没答过"就把同一个问题再问一遍。
        multi.share_common_facts(session_id)

        next_action = sales_result.get("next_action")
        requirements = sales_result.get("requirements", {})

        # Step 2: Check if Solution Agent should be triggered
        if next_action == "trigger_solution":
            perf.route = "trigger_solution"
            logger.info(
                "[%s] Solution triggered: response length=%d products=%d",
                session_id,
                len(sales_result.get("response", "")),
                len(sales_result.get("products", [])),
            )
            perf.final_products = len(sales_result.get("products", []))
            perf.solution_route = sales_result.get("route", "agent")
            result = {
                "response": sales_result.get("response", ""),
                "agent": "dual",
                "requirements": requirements,
                "products": sales_result.get("products", []),
                "next_action": "follow_up",
            }

        # Step 2.5: Free-form product question
        elif next_action == "product_question":
            perf.route = "product_question"
            logger.info("[%s] Product question: routing to Solution Agent for free-form RAG", session_id)
            history = self._load_history(session_id)
            solution_result = self.solution_agent.run(
                message=message,
                history=history,
                session_id=session_id,
                profile=self._stored_profile(session_id),
                # 意图由 Sales 定（客户是在问问题，不是要重新推荐）
                intent=str(sales_result.get("intent") or ""),
            )
            perf.solution_route = solution_result.get("route", "agent")
            perf.llm_calls += 1
            answer = solution_result.get("answer", sales_result.get("response", ""))
            result = {
                # 先回答客户这个问题，再接着问还缺的需求（不能只会反问）
                "response": self._response_coordinator().compose_with_requirement_question(
                    answer=answer,
                    sales_result=sales_result,
                    message=message,
                    seed=len(history),
                    session_id=session_id,
                ),
                "agent": "solution_question",
                # §28（Phase 14）：这条路径走的是旧 compose_requirement_reply
                # → 明确标成 FALLBACK，便于统计"还有多少回复走旧逻辑"
                "response_mode": "FALLBACK",
                "requirements": requirements,
                "products": solution_result.get("products", []),
                "next_action": "follow_up",
            }

        # Step 2.6: Catch-all others question
        elif next_action == "others":
            perf.route = "others"
            # ── 跑题 / 闲聊**不进产品 RAG**（客户口径 2026-10）──────────────────
            # 实测链：客户问 "do u like watching TV"（需求已推荐完）→ 这里无条件送
            # 产品 RAG → 检索回来 6 条**产品**片段 → 模型只能聊型号 →
            # 校验器判 'customer_question_not_answered' → ModelGuard 又把型号全抹掉
            # → 正文变空 → api 空回复兜底把型号说成"符合你需求的最接近型号"：
            #     "Based on your requirements, the closest match is TW21-3216-P2.5."
            # 所以：不是业务/产品问题 → 按语境接话并委婉拉回产品，且**不带产品**。
            from .agents.sales.nodes.requirement import _is_product_or_business_question

            # 只把**真正的跑题**挡下来：不是业务问题，且明显是"在聊别的"
            # （成句的闲话 / 问句）。
            # 短回答（"yes" / "ok" / "no"）不算跑题 —— 那是客户在接上一问，
            # 仍要走正常回答路径（自由问答的提示词专门写了"短句要靠上下文理解"）。
            _text = str(message or "").strip()
            _words = len(_text.split())
            _text_is_question = _text.endswith("?")
            # ① 客户"同意推进"——**语义判定**，不是关键词/词数。
            #    销售层用 LLM（带最近对话）理解客户这一整句话在做什么，结果放在
            #    quote_confirmation 里透传过来（客户口径 2026-10：不能只靠关键词猜）。
            #    客户口径：这时候应该是"我去准备报价单，请稍等"。
            #    以前这种轮次被丢进自由问答，正文被清空后落到"放宽条件"的兜底话术：
            #      "Let's take a slightly different angle, if one of the requirements can
            #       be relaxed …" —— 和客户那句 "yes" 完全不搭（客户实测反馈）。
            _affirm = bool(sales_result.get("quote_confirmation"))
            if not _affirm:
                # 【客户口径 2026-10】销售层没给出结论（或那条分支没走到）时，**在这里再判一次**。
                # 教训：只依赖上游一个标志，上游漏一次就会退回"放宽条件"兜底，
                # 客户会再次看到 "if one of the requirements can be relaxed …"（实测复现）。
                # 只要本会话**已经给过推荐**，就把"客户这句话在做什么"交给模型结合上下文理解
                # —— 不是关键词、不是词数，判不出来（None）就保守当"不是同意"。
                _already_recommended = bool(
                    getattr(self.memory_store, "has_recommendation", lambda *_: False)(session_id)
                )
                if _already_recommended:
                    _approved = None
                    try:
                        from .dialogue.approval import understand_approval

                        _approved = understand_approval(_text, session_id=session_id)
                    except Exception as exc:  # pragma: no cover - 防御式
                        logger.warning("Approval understanding failed: %s", exc)
                    _affirm = _approved is True
                    logger.info(
                        "[%s] 语义判定『客户是否在同意推进』= %s（本轮 input=%r）",
                        session_id, _approved, _text[:40],
                    )
            # ② 真正的跑题/闲聊（成句的闲话 / 问句）—— 不进产品 RAG。
            #    短回答（"yes" / "ok" / "no"）不算跑题：那是客户在接上一问。
            _chitchat = not _is_product_or_business_question(_text) and (
                _words > 3 or _text_is_question
            )
            if _affirm or _chitchat:
                _purpose = "quote_confirmation" if _affirm else "chitchat"
                logger.info(
                    "[%s] %s → 按语境%s（不进产品 RAG）",
                    session_id,
                    "客户确认推进报价" if _affirm else "Off-topic/others",
                    "回'报价单在准备'" if _affirm else "接话并拉回产品",
                )
                chat_reply = ""
                try:
                    from .dialogue.chat_reply import generate_chat_reply

                    chat_reply = generate_chat_reply(
                        _text,
                        session_id=session_id,
                        state=sales_result,
                        purpose=_purpose,
                    )
                except Exception as exc:  # pragma: no cover - 防御式
                    logger.warning("Chat reply failed: %s", exc)
                if not chat_reply:
                    # 模型不可用时的确定性兜底（多句轮换，绝不承诺价格/交期）
                    from .rag.reply_composer import (
                        off_topic_steer_answer,
                        quote_confirmation_answer,
                    )

                    _seed = len(self._load_history(session_id))
                    _lang = str(sales_result.get("language") or "") or None
                    chat_reply = (
                        quote_confirmation_answer(_lang, _seed)
                        if _affirm
                        else off_topic_steer_answer(_lang, _seed)
                    )
                result = {
                    "response": chat_reply or sales_result.get("response", ""),
                    "agent": "sales_quote_confirm" if _affirm else "sales_offtopic",
                    "response_mode": "FALLBACK",
                    "requirements": requirements,
                    # 【关键】这两种轮次都**不带产品**：杜绝拿型号糊弄客户
                    "products": [],
                    "offtopic_turn": not _affirm,
                    "next_action": "follow_up",
                }
            else:
                logger.info("[%s] Others question: routing to Solution Agent for free-form RAG", session_id)
                history = self._load_history(session_id)
                solution_result = self.solution_agent.run(
                    message=message,
                    history=history,
                    session_id=session_id,
                    # 【关键】把本会话已经收集到的需求档案一起带过去：
                    # 否则 Solution 会只拿这一句话重建需求，又回头问"室内还是室外"（实测出现过）。
                    profile=self._stored_profile(session_id),
                    intent=str(sales_result.get("intent") or ""),
                )
                perf.solution_route = solution_result.get("route", "agent")
                perf.llm_calls += 1
                answer = solution_result.get("answer", sales_result.get("response", ""))
                result = {
                    "response": self._response_coordinator().compose_with_requirement_question(
                        answer=answer,
                        sales_result=sales_result,
                        message=message,
                        seed=len(history),
                        session_id=session_id,
                    ),
                    "agent": "solution_others",
                    "response_mode": "FALLBACK",
                    "requirements": requirements,
                    "products": solution_result.get("products", []),
                    "next_action": "follow_up",
                }

        # Step 3: Sales Agent handles alone (greeting, objection, need_query)
        else:
            perf.route = next_action
            result = {
                "response": sales_result.get("response", ""),
                "agent": "sales",
                "requirements": requirements,
                "products": [],
                "next_action": next_action,
            }

        # ── 最后一道闸门：**已经推荐过，就绝不允许出现"放宽条件"话术**（2026-10）──────
        # 客户原话：「推荐完后，还是会出现这句话，帮我彻底的解决」。
        # 为什么必须在收口处堵：它可能从多条内部路径漏出来 ——
        #   · solution/runner 的空回答兜底（本轮没选出型号时）
        #   · trigger_solution 分支（实测 "pls prepare the quotation for me" 走的就是它）
        #   · api 的空回复兜底
        # 逻辑上也讲不通：型号早就匹配过了，还说"放宽某个条件我就能匹配"
        # 等于告诉客户没匹配上。所以在这里按**会话事实**（是否已推荐过）统一拦掉。
        _resp_text = str(result.get("response") or "")
        if _resp_text and (
            bool(getattr(self.memory_store, "has_recommendation", lambda *_: False)(session_id))
        ):
            from .rag.reply_composer import is_relaxation_answer, quote_confirmation_answer

            if is_relaxation_answer(_resp_text):
                logger.warning(
                    "[%s] 已推荐过却产出了'放宽条件'话术 → 换成推进报价的说法（input=%r）",
                    session_id, str(message or "")[:40],
                )
                result["response"] = quote_confirmation_answer(
                    str(sales_result.get("language") or "") or None,
                    len(self._load_history(session_id)),
                )

        # Emit structured performance log
        # v2.6 §16/§17：Dialogue Policy 的结果（SpeechAct + 唯一 Action）透传给收口层，
        # FinalResponse / 日志 / DecisionAudit 都用同一份判定，不再各猜一次。
        result.setdefault("speech_act", sales_result.get("speech_act") or {})
        result.setdefault("dialogue_action", sales_result.get("dialogue_action") or {})
        # v2.6 §4.3：待问项也要带出来 —— 否则收口层不知道"这一轮问的是哪一项"，
        # 日志里就会出现 question_slot=-、FinalResponse.question_slot 为空。
        result.setdefault("pending_question", sales_result.get("pending_question") or "")
        result.setdefault("pending_slot", sales_result.get("pending_slot") or "")
        result.setdefault("acknowledgement", sales_result.get("acknowledgement") or "")
        # 2026-09-22（客户口径）：这一轮客户是不是在说"与需求无关的话"（闲聊）。
        # 判定由销售节点在语境里做（LLM 语义理解 + 规则，不靠关键词），
        # 这里只透传 —— 收口层据此决定"只承接"还是"接住 + 追问"。
        result.setdefault("offtopic_turn", bool(sales_result.get("offtopic_turn")))
        # 计划 v2.9.1 §八/§九：产品域 + 轮次类型（Others / 未来 LCD-IFP 策略共用）
        result.setdefault("product_domain", str(sales_result.get("product_domain") or ""))
        result.setdefault("turn_kind", str(sales_result.get("turn_kind") or ""))
        # 计划 v2.9.3：类型判断 + 入口名（收口层的类型闸门要用）
        result.setdefault(
            "display_type_decision", dict(sales_result.get("display_type_decision") or {})
        )
        # 计划 v2.9.4：销售层这一轮是不是**已经把类型问题/类型说明**发出去了 ——
        # 收口层的类型闸门据此只对齐槽位、不再把同一段话追加一遍。
        result.setdefault(
            "product_type_gate", dict(sales_result.get("product_type_gate") or {})
        )
        result.setdefault("product_subtype", str(sales_result.get("product_subtype") or ""))
        result.setdefault("product_entry", str(sales_result.get("product_entry") or ""))
        # LCD / IFP 需求链的决策结果（下一问 / 缺什么 / 分支）——收口层据此隔离 LED 链路
        result.setdefault("lcd_action", dict(sales_result.get("lcd_action") or {}))
        from .dialogue import dialogue_action_candidates

        result.setdefault("action_candidates", dialogue_action_candidates(sales_result))
        # v2.7 §18/§19：这一轮新填的槽位 + Turn 上下文（覆盖率与闸门的依据）
        result.setdefault("newly_filled_slots", result_newly_filled)
        result.setdefault("turn_context", turn_context)
        result.setdefault("turn_id", turn_id)
        # §25/§28：正常路径由 ResponseGenerator/单一出口产出客户文本
        result.setdefault("response_mode", "NATURAL")
        # v2.7 修订：LLM 写的句子 vs 模板拼的句子（只有模板才需要去机械话术）
        result.setdefault("response_source", sales_result.get("response_source") or "template")
        # 2026-09-21：全字段理解留痕（哪个字段、依据客户哪句话、是否入档）
        result.setdefault("understanding", sales_result.get("understanding") or {})
        # 服务口径（安装/说明书/质保）已经由销售这一轮答过 → 收口层不再拼标准口径
        result.setdefault(
            "service_faq_answered", sales_result.get("service_faq_answered") or ""
        )
        logger.info(
            "[%s] PERF route=%s solution_route=%s intent=%s "
            "total_ms=%.1f sales_ms=%.1f solution_ms=%.1f "
            "llm_calls=%d final_products=%d",
            session_id,
            perf.route,
            perf.solution_route or "-",
            perf.intent,
            perf.total_ms,
            perf._span("_start", "sales_done"),
            perf.latency("sales_done"),
            # v2.6 §27：这一行在 end_turn 之前打，必须向 tracker 要实时计数，
            # 否则会出现 "PERF llm_calls=0" 与 "[Turn] llm_calls=2" 自相矛盾
            perf.live_llm_calls(),
            perf.final_products,
        )

        result["_perf"] = perf.summary()
        # 客户口径（2026-09-18）：**不再**在推荐后追问"个人还是公司 / 姓名 / 邮箱"。
        # 实测：客户后面问"我能定制产品吗""交付日期多久"时，这些话被联系方式收集
        # 当成回答吞掉，回了 "Got it <客户原话>, thanks — I've passed your details on…"，
        # 客户真正的问题没有任何回答。联系信息改由销售人工在后端获取。
        # ── 一个项目多条屏：几块屏就按量给几个型号（客户口径 2026-09-18）────────
        # 单屏会话仍然"只报一个型号"；多屏会话不能省 —— 每块屏各推一个。
        if perf.route == "trigger_solution":
            multi_reply = self._multi_screen().recommend_all_screens(session_id, message)
            if multi_reply:
                result["response"] = multi_reply
                self._finalize_turn_response(result, session_id, message, turn_context)
                # 多屏回复里每块屏各自有环境（室内那块本来就该出现室内型号）；
                # API 层据此**不**用"当前这块屏的环境"整段过滤回复。
                result["multi_screen"] = True
                result["_perf"] = perf.summary()
                if vision_metrics:
                    result["vision"] = vision_metrics
                return result

        # ── 一个项目多条屏：第二块（及以后）的推荐要标清楚是哪一块 ────────────
        # v2.3.1 Phase 2：这段业务已搬到 rag.multi_screen（这里只调用）。
        if perf.route == "trigger_solution" and result.get("response"):
            result["response"] = self._multi_screen().prefix_active_screen_label(
                session_id, message, str(result["response"])
            )

        # ── 一个项目多条屏：记录这一块屏的推荐 + 追问"还有其他位置吗" ────────
        multi_extra = self._multi_screen().multi_item_follow_up(session_id, result, message)
        if multi_extra:
            result.setdefault("extra_messages", []).append(multi_extra)
        self._finalize_turn_response(result, session_id, message, turn_context)
        if vision_metrics:
            result["vision"] = vision_metrics
        return result

    # ── 售后 / 服务类固定口径（说明书图纸 / 现场安装 / 质保）───────────────
    def _response_coordinator(self):
        """v2.3 §13：回复组装交给 ResponseCoordinator（Orchestrator 只做编排）。"""
        coordinator = getattr(self, "_response_coordinator_instance", None)
        if coordinator is None:
            from .dialogue import ResponseCoordinator

            coordinator = ResponseCoordinator(
                store=getattr(self, "memory_store", None),
                profile_lookup=self._stored_profile,
            )
            self._response_coordinator_instance = coordinator
        return coordinator

    def _finalize_turn_response(
        self,
        result: Dict[str, Any],
        session_id: str,
        message: str,
        turn_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """本轮回复的最后两道加工（顺序固定）：

          1. 客户问到的售后口径（说明书图纸 / 现场安装 / 质保）—— 必须回答；
          2. 本轮带了图片 → **先把"图片里看到什么"跟客户核一遍**，再继续问需求。

        实测 bug（客户日志）：客户只发了图片 + "i need this"，系统直接跳到问点间距，
        既没跟客户核对图片识别结果，客户也没机会纠正判错的"固定/租赁"。

        v2.6 §4（ONE TURN → ONE ACTION → ONE RESPONSE）：收口不再只是"把多出来的
        问句删掉"，而是**只剩一条客户可见回复** —— 追加气泡并入正文，最终由
        ``FinalResponseCoordinator`` 产出唯一的 ``FinalResponse``。
        """
        # ── v2.7 Phase 7~14 的对话层依赖（延迟导入，避免模块级循环依赖）──
        from .dialogue import facts_to_dicts

        # 决策段（覆盖率 → 闸门 → Policy 对齐 → 类型闸门 → Momentum → Action →
        # 承接额度）全部在对话层：ResponseCoordinator.prepare_final
        plan = self._response_coordinator().prepare_final(
            result, session_id=session_id, message=message, turn_context=turn_context
        )

        # ① 售后口径 + 图片核对 + Guard 收口（原有链路，先算出"想说的话"）
        # 这一轮如果是"问需求"（ask_only / answer_then_ask）且已经定了要问哪一项，
        # 正文里必须真的出现那一问（LLM 漏写时由 finalize 补回）。
        # 客户口径（2026-09-28）：**LCD / IFP 会话**里"已经定好要问的那一项"
        # 必须真的出现在正文里（实测：LCD 入口轮 LLM 只写了一句 ack，问题被吞掉，
        # 客户等不到下一问）。LED 侧口径保持不变（那一条链有自己的护栏与测试）。
        lcd_turn_active = bool(result.get("lcd_action")) or str(
            result.get("product_domain") or ""
        ).upper() in ("LCD", "IFP")
        require_question = (
            lcd_turn_active
            and bool(plan.questions)
            and not result.get("products")
            # 只要这一轮**确实定了要问哪一项**，那一问就必须真的出现在正文里。
            # 不再看"动作名"白名单 —— 那是 LED 侧的口径：客户答一个 "no" 被通用层
            # 标成 clarify_only 时，LCD 算好的下一问曾被整条吞掉（实测 2026-09-30）。
            # LED 走不到这里（lcd_turn_active=False），口径不受影响。
            and bool(str(plan.question_slot or "").strip())
        )
        text = self._response_coordinator().finalize(
            plan.text,
            session_id=session_id,
            message=message,
            questions=plan.questions,
            service_faq_answered=str(result.get("service_faq_answered") or ""),
            require_question=require_question,
            # LCD / IFP：售后与交期问题只回答，不跟产品话术；闲聊也只承接一句
            product_family=_turn_product_family(self._stored_profile(session_id)),
            offtopic_turn=bool(result.get("offtopic_turn")),
        )
        # ② v2.7 §15/§16 + 型号闸门：正文加工统一在回复层（ResponseCoordinator）
        text = self._response_coordinator().postprocess_final(
            text, result=result, density=plan.density
        )

        # ③ v2.6：合并追加气泡 → 只保留一条回复 + 最多一个问题
        final = self._final_response_coordinator().build(
            text=text or str(result.get("response") or ""),
            extras=plan.extras,
            questions=plan.questions,
            action=plan.action,
            question_slot=plan.question_slot,
            turn_id=str(result.get("turn_id") or getattr(self, "_current_turn_id", "") or ""),
            facts=facts_to_dicts(result.get("grounded_facts") or []),
            first_contact_messages=result.get("first_contact_messages"),
        )

        result["response"] = final.text
        result["final_response"] = final.to_dict()
        result["response_count"] = final.response_count
        result["question_count"] = final.question_count
        result["question_slot"] = final.question_slot
        result["action"] = final.action
        result["turn_id"] = final.turn_id
        # 计划 §4.2：追加气泡不再单独发给客户（已并入唯一回复）
        result.pop("extra_messages", None)
        # 计划 §8：把"这一轮 AI 问了什么 / 最终说了什么"记进 ConversationState
        state_service = self._turn_state_service()
        state_service.note_ai_turn(result, session_id, final)
        # 客户口径（2026-09-22）：others / product_question 这一轮，Sales 只写了占位符
        # （"Sure."），真正发给客户的是 Solution 的答复 —— 把它写回历史，下一轮的
        # "最近 50 条"才看得到 AI 自己说过什么（否则客户回 "yes" 时无从判断在问什么）。
        state_service.replace_placeholder_history(result, session_id, final.text)
        result["conversation_state_before"] = plan.conversation_before
        self._finish_llm_turn(result, session_id)
        return result

    def _final_response_coordinator(self):
        coordinator = getattr(self, "_final_response_instance", None)
        if coordinator is None:
            from .dialogue import FinalResponseCoordinator

            coordinator = FinalResponseCoordinator()
            self._final_response_instance = coordinator
        return coordinator

    def _turn_state_service(self):
        from .orchestrator_state import TurnStateService

        return TurnStateService(
            memory_store=getattr(self, "memory_store", None),
            profile_lookup=self._stored_profile,
        )

    def _note_customer_turn(self, session_id: str, message: str, *, turn_id: str = "") -> None:
        """Record the customer turn while keeping the assigned ID on the orchestrator."""
        self._current_turn_id = turn_id
        self._turn_state_service().note_customer_turn(
            session_id, message, turn_id=turn_id
        )

    def _finish_llm_turn(self, result: Dict[str, Any], session_id: str) -> None:
        self._turn_state_service().finish_llm_turn(result, session_id)

    # ── 一个项目多条屏（客户口径 2026-09-18）─────────────────────────────



    # 多块屏之间**默认共享**的字段（客户没说"这块要什么、那块要什么"时就一样）：
    # ── v2.3.1 Phase 4：兼容层 —— 旧调用继续可用，实现已迁到对应模块 ────────
    def _multi_screen(self):
        """多屏业务（`src/rag/multi_screen.py`）：Orchestrator 只调用它。"""
        manager = getattr(self, "_multi_screen_instance", None)
        if manager is None:
            from .rag.multi_screen import MultiScreenManager

            manager = MultiScreenManager(
                store=self.memory_store,
                profile_lookup=self._stored_profile,
                history_lookup=self._load_history,
                solution_agent=self.solution_agent,
            )
            self._multi_screen_instance = manager
        return manager

    def _stored_profile(self, session_id: str):
        """取本会话已收集的需求档案（转发给 Solution Agent，避免它重新问一遍）。"""
        if not self.memory_store or not hasattr(self.memory_store, "get_requirement_profile"):
            return None
        try:
            stored = self.memory_store.get_requirement_profile(session_id)
            if not stored:
                return None
            from .models.requirement import RequirementProfile

            return (
                stored if isinstance(stored, RequirementProfile)
                else RequirementProfile.model_validate(stored)
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[%s] Load stored profile failed: %s", session_id, exc)
            return None

    def _load_history(self, session_id: str) -> List[Dict[str, str]]:
        """从共享 memory 实例读取历史消息。"""
        if not self.memory_store:
            return []
        session_data = self.memory_store.get(session_id, {})
        if isinstance(session_data, dict):
            return session_data.get("messages", []) or []
        if isinstance(session_data, list):
            return session_data
        return []
