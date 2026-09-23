"""
Orchestrator - Dual Agent Coordination
Coordinates between Sales Agent, Solution Agent, and First Contact Flow
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .agents.sales.runner import SalesAgentRunner
from .agents.solution.runner import SolutionAgentRunner
from .memory.store import memory as shared_memory
from .first_contact.handler import first_contact_handler
from .first_contact.profile import load_profile

logger = logging.getLogger(__name__)

# ── v2.3.1 Phase 2：业务逻辑已迁出，这里只做编排与兼容 ────────────────────────
from .observability.perf import PerfTracker  # noqa: E402,F401  (按性能埋点，从本模块迁出)
from .vision.pipeline import (  # noqa: E402,F401  (Vision 接入，从本模块迁出)
    _merge_vision_into_stored_profile,
    _vision_enabled,
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
        multi_specs = self._split_and_apply_screen_specs(session_id, message)
        if not multi_specs and self._maybe_target_screen(session_id, message) is None:
            self._maybe_start_new_item(session_id, message)

        # Step 1: Sales Agent processes message
        # v2.7 §18：记下"这一轮之前"已有哪些需求 → 之后 diff 出 newly_filled_slots
        profile_before = self._profile_slot_map(session_id)
        sales_result = self.sales_agent.run(
            session_id=session_id,
            message=message,
            has_vision=bool(vision_results),
        )
        perf.mark("sales_done")
        # 客户口径：同一个问题全项目最多问两次 —— 客户答过的共有项要同步到其他屏，
        # 否则两块屏会各问一遍（实测被连问 4 次）。
        # 只填空值，绝不覆盖：能用 P3/P5、固装/租赁 等参数区分两块屏时，这些差异保留。
        self._share_common_facts(session_id)
        perf.intent = sales_result.get("intent", "")
        result_newly_filled = self._newly_filled_slots(profile_before, session_id)

        # 多屏拆分的那一轮：Sales 的需求抽取只看到"整句话"，会把两块屏的参数
        # 混到当前这条档案里 —— 这里按拆分结果把每块屏的参数重新写回去。
        if multi_specs:
            self._split_and_apply_screen_specs(session_id, message)
        # 客户口径：多块屏时"没说清哪块要什么"就两块记成一样的 —— 所以客户
        # 答过一次的共有项（视频/图片、安装方式、视距、价格取向…）要同步到
        # 其他屏，绝不能因为另一块"还没答过"就把同一个问题再问一遍。
        self._share_common_facts(session_id)

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
                "response": self._compose_with_requirement_question(
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
                "response": self._compose_with_requirement_question(
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
        result.setdefault("action_candidates", self._action_candidates(sales_result))
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
            multi_reply = self._recommend_all_screens(session_id, message)
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
        multi_extra = self._multi_item_follow_up(session_id, result, message)
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
        from .dialogue import (
            DETAILED,
            MINIMAL,
            build_natural_continuation,
            compute_answer_coverage,
            compute_momentum,
            configured_ack_streak_limit,
            decide_continuation,
            decide_turn_action,
            next_candidate_slot,
            render_minimal,
            strip_mechanical_phrases,
        )
        from .dialogue.duplicate_firewall import FIREWALL

        turn_context = dict(turn_context or result.get("turn_context") or {})
        questions = self._question_candidates(result)
        # §28：审计要记"做决定之前"的对话状态，所以先拍一张快照
        conversation_before = self._conversation_snapshot(session_id)
        # 计划 §4.4/§4.5：候选 Action → 唯一 Action（其余记进 discarded_actions）
        selected_action, discarded_actions = self._select_turn_action(result)
        action = (
            selected_action.action
            if selected_action is not None
            else self._dialogue_action_label(result)
        )
        result["discarded_actions"] = [item.to_dict() for item in discarded_actions]
        if selected_action is not None:
            result["selected_action"] = selected_action.to_dict()
            result["action_candidates"] = [
                item.to_dict() for item in ([selected_action] + list(discarded_actions))
            ]
        question_slot = str(result.get("pending_slot") or "")
        extras = [str(item) for item in (result.get("extra_messages") or []) if item]
        result.setdefault("customer_input", str(message or ""))

        # ══ v2.7 Phase 7~11：覆盖率 → 闸门 → 唯一 Action → 承接上下文 ══════
        previous_question = turn_context.get("previous_question") or {}
        previous_slot = str(previous_question.get("question_slot") or "")
        newly_filled = [str(item) for item in (result.get("newly_filled_slots") or [])]
        coverage = compute_answer_coverage(
            asked_slot=previous_slot,
            match=self._answer_match_object(session_id),
            newly_filled_slots=newly_filled,
        )
        result["answer_coverage"] = coverage.to_dict()

        question_slot, duplicate_check = self._apply_duplicate_firewall(
            result,
            session_id=session_id,
            coverage=coverage,
            previous_question=previous_question,
            newly_filled=newly_filled,
            firewall=FIREWALL,
            next_candidate=next_candidate_slot,
        )
        if question_slot:
            result["pending_slot"] = question_slot
        result["duplicate_check"] = duplicate_check
        questions = self._question_candidates(result)

        # ── Phase 1（计划 §16.2）：最终问的必须是 DialoguePolicy 定的那一项 ──
        # 销售层准备好的问句只能"提候选"：如果 Policy 定的槽位与它不一致，
        # 以 Policy 为准（日志里问的槽位和客户看到的问句必须一致）。
        # 例外：重复提问闸门刚刚**故意改问别的槽位**（duplicate_question*）——
        # 那是同一决策层的防重放行，不能反过来被覆盖。
        policy_slot = str(getattr(selected_action, "question_slot", "") or "")
        firewall_rerouted = str(duplicate_check or "").startswith("duplicate_question")
        if (
            not firewall_rerouted
            and policy_slot
            and question_slot
            and policy_slot != question_slot
        ):
            logger.warning(
                "[ActionConsistency] Policy=ASK(%s) 与销售层准备的问句(%s)不一致 → 以 Policy 为准",
                policy_slot, question_slot,
            )
            overridden_slot = question_slot
            aligned = self._question_text_for_slot(policy_slot)
            question_slot = policy_slot if aligned else ""
            result["pending_slot"] = question_slot
            result["pending_question"] = aligned
            if aligned:
                # 正文里那句"问错槽位"的问句必须换掉：先去掉旧问句，再补 Policy 槽位的问句
                guard = self._response_coordinator()._guard()
                kept = guard.strip_questions(str(result.get("response") or "")).strip()
                result["response"] = f"{kept} {aligned}".strip() if kept else aligned
            result["action_consistency"] = {
                "policy_slot": policy_slot,
                "overridden_slot": overridden_slot,
                "question_realigned": bool(aligned),
            }
            questions = self._question_candidates(result)

        conversation = self._conversation_state(session_id)
        momentum = compute_momentum(
            newly_filled_slots=newly_filled,
            answered_slots=coverage.answered_slots,
            last_question_slot=str(getattr(conversation, "last_question_slot", "") or ""),
        )
        result["momentum"] = momentum.to_dict()
        speech_act = result.get("speech_act") or {}
        customer_question = bool(
            speech_act.get("customer_questions") or speech_act.get("is_customer_question")
        )
        question_kind = str(speech_act.get("question_kind") or "")
        turn_action = decide_turn_action(
            customer_question=customer_question,
            question_kind=question_kind,
            ready_to_recommend=bool(
                (result.get("recommendation_gate") or {}).get("ready")
            ),
            recommend_requested=bool(result.get("recommend_requested")),
            conflicts=(result.get("requirements") or {}).get("conflicts"),
            newly_filled_slots=newly_filled,
            missing_slots=result.get("missing_slots") or [],
            question_candidates=self._ranked_question_candidates(session_id),
            momentum_slot=momentum.slot,
            previous_question_slot=previous_slot,
            blocked_slot=str(previous_question.get("question_slot") or "")
            if duplicate_check.startswith("duplicate_question")
            else "",
            hard_gate_slot="environment" if result.get("pending_slot") == "environment" else "",
        )
        result["turn_action"] = turn_action.to_dict()
        # §20/§21：Question Planner/Flow 产出候选，Dialogue Policy 决定"这一轮做不做、
        # 做的优先级"；**具体问哪一项仍以已落地的候选为准** —— 实测教训：在这里
        # 事后改槽位会和已经组好的正文脱节（日志里 last_question 与 last_response
        # 不一致，客户看到的是另一个问题）。Policy 的选择记录在 turn_action 里。
        result["policy_preferred_slot"] = str(getattr(turn_action, "target_slot", "") or "")
        # §14：Conversation State 的"当前话题"（momentum）只用于对话规划
        if conversation is not None:
            conversation.current_topic = momentum.slot
        continuation = build_natural_continuation(
            customer_message=message,
            speech_act=speech_act,
            newly_filled_slots=newly_filled,
            momentum=momentum,
            next_required_slot=question_slot,
            customer_question=customer_question,
            question_kind=question_kind,
            conflicts=(result.get("requirements") or {}).get("conflicts"),
            has_recommendation=bool(result.get("products")),
            is_first_contact=bool(result.get("first_contact_messages")),
        )
        result["response_density"] = continuation.density
        result["natural_continuation"] = continuation.to_dict()

        # ── 客户口径（2026-09-21 → 2026-09-22 最终版）：不要连续提问 ─────
        # 判定依据**不是**"客户有没有回答上一问"，而是"这句话跟需求有没有关系"：
        #   · 与需求有关（给参数 / 答别的一项 / 问业务问题）→ 直接"接住 + 追问缺项"，
        #     绝不做"只承接"（实测 bug：客户答 "maybe 5m"，AI 只寒暄一句就停了）；
        #   · 与需求无关（闲聊）→ 只承接 1 条；第 2 条闲聊必须"接住 + 提问"
        #     写在同一条消息里（客户口径："第二条……在同一条消息询问需求"）。
        # 判定由销售节点在语境里给出（offtopic_turn，LLM + 规则，不看关键词）；
        # 同一轮里客户连发多条消息按**一次**算（一个 turn 只决策一次）。
        conversation_state = conversation
        ack_streak = int(getattr(conversation_state, "ack_streak", 0) or 0)
        ack_limit = configured_ack_streak_limit()
        off_topic = bool(result.get("offtopic_turn"))
        # 档案里有值 = 客户确实给过这一项 → 问题登记簿同步成 ANSWERED，
        # 并把"档案里有值"也算成"客户答过"（实测：客户答"只在意质量"，
        # 档案已记 price_preference=quality，但登记簿还是 ASKED → 误判成没回答）。
        profile_slots = self._profile_slot_map(session_id)
        if conversation_state is not None:
            for _slot in profile_slots:
                conversation_state.note_answered(str(_slot))
        has_question = bool(question_slot and result.get("pending_question"))
        decision = decide_continuation(
            has_question=has_question,
            off_topic=off_topic,
            ack_streak=ack_streak,
            max_ack_streak=ack_limit,
        )
        result["continuation"] = decision.to_dict()
        result["ack_streak"] = decision.next_streak
        if conversation_state is not None:
            conversation_state.ack_streak = decision.next_streak
        if decision.suppress_question and has_question:
            logger.info(
                "[Continuation] 客户说的是与需求无关的话 → 本轮先承接、不问问题"
                "（承接 %d/%d，下一轮接住 + 提问）",
                decision.next_streak, ack_limit,
            )
            result["suppressed_question"] = {
                "slot": question_slot,
                "question": str(result.get("pending_question") or ""),
                "ack_streak": decision.next_streak,
            }
            result["pending_question"] = ""
            result["pending_slot"] = ""
            question_slot = ""
            questions = []
        if (
            continuation.density == MINIMAL
            and str(result.get("response_mode") or "") != "FALLBACK"
        ):
            result["response_mode"] = "MINIMAL"

        # ① 售后口径 + 图片核对 + Guard 收口（原有链路，先算出"想说的话"）
        text = self._response_coordinator().finalize(
            str(result.get("response") or ""),
            session_id=session_id,
            message=message,
            questions=questions,
            service_faq_answered=str(result.get("service_faq_answered") or ""),
        )
        # ② v2.7 §15 原先是"短回答模式：客户只给一个参数 → 直接甩一个问题"。
        # 客户口径（2026-09-21）：太短了，要 3~4 句 —— 所以这里**不再**把回复
        # 砍成裸问句；长度交给 LLM 提示词（NATIVE prompt 里给了 3~4 句的要求）。
        # 只在"客户只有一个词确认 + 没有 LLM 文本"时保留极短兜底。
        # 承接轮：正文里不许再留下问句（只接住客户的话）
        if result.get("suppressed_question"):
            text = self._continuation_only_text(text, result)
        # ③ v2.7 §16：去掉机械确认开头（"Got it / Thanks / Based on that…"）
        # 只管模板拼出来的句子；LLM 自己写的开场（客户口径：明确授权它自己组织）
        # 不再被回头清洗，否则会把自然的话削成半句。
        text, removed_mechanical = strip_mechanical_phrases(
            text,
            allow=(
                continuation.density == DETAILED
                or str(result.get("response_source") or "") == "llm"
                or bool(result.get("suppressed_question"))
            ),
        )
        if removed_mechanical:
            result["mechanical_phrases_removed"] = removed_mechanical
        # ② v2.6：合并追加气泡 → 只保留一条回复 + 最多一个问题
        final = self._final_response_coordinator().build(
            text=text or str(result.get("response") or ""),
            extras=extras,
            questions=questions,
            action=action,
            question_slot=question_slot,
            turn_id=str(result.get("turn_id") or getattr(self, "_current_turn_id", "") or ""),
            facts=self._facts_for_turn(result),
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
        self._note_ai_turn(result, session_id, final)
        # 客户口径（2026-09-22）：others / product_question 这一轮，Sales 只写了占位符
        # （"Sure."），真正发给客户的是 Solution 的答复 —— 把它写回历史，下一轮的
        # "最近 50 条"才看得到 AI 自己说过什么（否则客户回 "yes" 时无从判断在问什么）。
        self._replace_placeholder_history(result, session_id, final.text)
        result["conversation_state_before"] = conversation_before
        self._finish_llm_turn(result, session_id)
        return result

    def _replace_placeholder_history(
        self, result: Dict[str, Any], session_id: str, text: str
    ) -> None:
        """Solution 答复的路径：用真正发出去的那条替换历史里的 Sales 占位符。"""
        if not str(result.get("agent") or "").startswith("solution"):
            return
        if not text or not self.memory_store:
            return
        replace = getattr(self.memory_store, "replace_last_assistant", None)
        if not callable(replace):  # pragma: no cover - 防御式（别家 store 实现）
            return
        try:
            if replace(session_id, text):
                logger.info(
                    "[History] 用最终答复替换占位符（session=%s，%d 字）",
                    session_id, len(text),
                )
        except Exception as exc:  # pragma: no cover - 留痕失败不影响业务
            logger.warning("[History] 替换占位符失败：%s", exc)

    # ── v2.6 §4：把本轮"想说的话"整理成候选 ──────────────────────────────
    @staticmethod
    def _question_candidates(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """本轮想问的问题（可能多个）；交给 Guard / FinalResponse 收成一个。"""
        candidates: List[Dict[str, Any]] = []
        pending = str(result.get("pending_question") or "")
        if pending:
            candidates.append({
                "text": pending,
                "slot": str(result.get("pending_slot") or ""),
                "source": "sales",
            })
        for item in result.get("question_candidates") or []:
            if isinstance(item, dict) and item.get("text"):
                candidates.append(dict(item))
            elif item:
                candidates.append({"text": str(item)})
        return candidates

    @staticmethod
    def _dialogue_action_label(result: Dict[str, Any]) -> str:
        """这一轮的业务动作（Dialogue Policy 选的唯一 Action）。

        v2.6 §4.4/§4.5：候选可以有多个，最终只能留一个 —— 由
        ``select_single_action`` 挑，其余进 ``discarded_actions``。
        """
        decision = result.get("dialogue_action") or {}
        if isinstance(decision, dict) and decision.get("action"):
            return str(decision["action"])
        if result.get("pending_question"):
            return "ask_only"
        return str(result.get("next_action") or "")

    @staticmethod
    def _select_turn_action(result: Dict[str, Any]):
        """把本轮出现的候选 Action 收成唯一一个（计划 §4.4/§4.5）。

        Returns:
            ``(selected, discarded)``；没有候选时返回 ``(None, [])``。
        """
        try:
            from .dialogue import (
                DialogueDecision,
                QUESTION_PRIORITY_SALES_PREFERENCE,
                question_priority,
                select_single_action,
            )
        except Exception:  # pragma: no cover - 防御式
            return None, []
        candidates = []
        decision = result.get("dialogue_action") or {}
        if isinstance(decision, dict) and decision.get("action"):
            slot = str(decision.get("target_slot") or "") or str(
                result.get("pending_slot") or ""
            )
            candidates.append(DialogueDecision(
                action=str(decision.get("action")),
                reason=str(decision.get("reason") or ""),
                question_slot=slot if decision.get("question") else "",
                question_target=slot if decision.get("question") else "",
                question_priority=int(
                    decision.get("priority") or QUESTION_PRIORITY_SALES_PREFERENCE
                ),
                question_count=1 if decision.get("question") else 0,
            ))
        # ── Phase 1（架构收口）：DialoguePolicy 定了槽位就以它为准 ──────────
        # 计划 §16.1/§16.2：QuestionPlanner / QuestionFlow / script_generator
        # 只能"提候选"，不能改最终决定。以前这里无条件把销售层准备的
        # pending_slot 也当成候选，于是它可能压过 Policy 的选择
        # （实测：Policy=ASK(viewing_distance) 最终却问了 pixel_pitch）。
        policy_decided_slot = (
            str((result.get("dialogue_action") or {}).get("target_slot") or "")
            if isinstance(result.get("dialogue_action"), dict)
            else ""
        )
        if result.get("pending_question") and not policy_decided_slot:
            slot = str(result.get("pending_slot") or "")
            # 注意：这里必须用 **Dialogue Policy 的动作词表**（ask_only），
            # 不能混进 DialogueDecision 的 ASK —— 否则日志里同一件事会有两个名字。
            candidates.append(DialogueDecision(
                action="ask_only",
                reason="question_flow",
                question_slot=slot,
                question_target=slot,
                question_priority=question_priority(slot) if slot else 99,
                question_count=1,
            ))
        if not candidates:
            return None, []
        return select_single_action(candidates)

    @staticmethod
    def _action_candidates(sales_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计划 §4.4：列出本轮出现过的**候选** Action（最终只执行一个）。

        候选可以有多个（Policy 选的、QuestionFlow 想追问的…），
        但"候选 ≠ 最终"：最终动作由 ``FinalResponse.action`` 唯一确定。
        """
        candidates: List[Dict[str, Any]] = []
        decision = (sales_result or {}).get("dialogue_action") or {}
        if isinstance(decision, dict) and decision.get("action"):
            candidates.append(dict(decision))
        if (sales_result or {}).get("pending_question"):
            candidates.append({
                "action": "ask_only",
                "target_slot": str(sales_result.get("pending_slot") or ""),
                "question": str(sales_result.get("pending_question") or ""),
                "source": "question_flow",
            })
        return candidates

    @staticmethod
    def _question_text_for_slot(slot: str) -> str:
        """按槽位取标准问句（Phase 1：Policy 定槽位、模板给句子，措辞仍由 LLM 重写）。

        只在"销售层准备的问句与 Policy 定的槽位不一致"时兜底用。
        """
        target = str(slot or "").strip()
        if not target:
            return ""
        try:
            from .rag.readiness import question_for

            return str(question_for(target, "en", 0, easier=False) or "")
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ActionConsistency] 取标准问句失败 slot=%s: %s", target, exc)
            return ""

    @staticmethod
    def _facts_for_turn(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """这一轮用到的业务事实（GroundedFact），供 FinalResponse 留痕。"""
        facts = result.get("grounded_facts") or []
        out: List[Dict[str, Any]] = []
        for item in facts if isinstance(facts, (list, tuple)) else []:
            if isinstance(item, dict):
                out.append(dict(item))
            elif hasattr(item, "to_dict"):
                try:
                    out.append(dict(item.to_dict()))
                except Exception:  # pragma: no cover - 防御式
                    continue
        return out

    def _final_response_coordinator(self):
        coordinator = getattr(self, "_final_response_instance", None)
        if coordinator is None:
            from .dialogue import FinalResponseCoordinator

            coordinator = FinalResponseCoordinator()
            self._final_response_instance = coordinator
        return coordinator

    def _conversation_snapshot(self, session_id: str) -> Dict[str, Any]:
        """这一轮做决定**之前**的对话状态（给 §28 的审计用）。"""
        try:
            from .dialogue import get_conversation_state

            return get_conversation_state(session_id).to_dict()
        except Exception:  # pragma: no cover - 防御式
            return {}

    def _continuation_only_text(self, text: str, result: Dict[str, Any]) -> str:
        """承接轮：把正文里的问句去掉，只留"接住客户那句话"的内容。

        客户口径：客户没回答时不要再抛问题；先顺着客户的消息聊一句，
        最多一条之后（由 continuation_budget 控制）必须拉回需求。
        """
        guarded = self._response_coordinator()._guard()
        stripped = guarded.strip_questions(str(text or "")).strip()
        if stripped:
            return stripped
        # 去掉问句后没内容了 → 用这一轮的"接话"（LLM 生成的 acknowledgement）
        acknowledgement = str(result.get("acknowledgement") or "").strip()
        if acknowledgement:
            return guarded.strip_questions(acknowledgement).strip() or acknowledgement
        # 兜底：把客户刚说的话接住（不提问）
        customer = " ".join(str(result.get("customer_input") or "").split())
        if customer:
            # 客户口径：不要"回执腔"复读客户原话（"3*5 — noted." 很僵硬），
            # 用一句简短的人话接住即可（真正的接话由 LLM 的 acknowledgement 负责）。
            return self._neutral_continuations(seed=len(customer)) 
        return "Got it."

    @staticmethod
    def _neutral_continuations(seed: int = 0) -> str:
        """没拿到 LLM 接话时的中性兜底（短、像人、不复读客户原话）。"""
        options = (
            "That makes sense.",
            "Right, I follow you.",
            "Good to know.",
            "Makes sense so far.",
        )
        return options[int(seed or 0) % len(options)]

    def _conversation_state(self, session_id: str):
        """取当前会话的对话状态对象（拿不到就返回 None）。"""
        try:
            from .dialogue import get_conversation_state

            return get_conversation_state(session_id)
        except Exception:  # pragma: no cover - 防御式
            return None

    def _answer_match_object(self, session_id: str):
        """v2.7 §18：客户这一句与上一轮问题的匹配结果（对象形态）。"""
        try:
            from types import SimpleNamespace

            state = self._conversation_state(session_id)
            data = dict(getattr(state, "last_answer_match", {}) or {})
            return SimpleNamespace(**data) if data else None
        except Exception:  # pragma: no cover - 防御式
            return None

    def _profile_slot_map(self, session_id: str) -> Dict[str, Any]:
        """当前需求档案里"已经有值"的槽位（用于 diff 出 newly_filled_slots）。

        实测 bug（2026-09-22）：这里原来用 ``profile.to_facts()``，而它**不包含**
        ``price_preference`` / ``content_type`` / ``budget_level`` → 客户答了
        "只在意质量"，档案里其实已经有值，但 newly_filled_slots 里看不到 →
        「承接上限」逻辑误判成"客户一直没回答那个问题" → 每轮都把问题压下去，
        结果既不问缺的 `installation`、也不推荐，对话卡死。

        v2.3.1 边界（``tests/test_v231_orchestrator_boundary.py``）：Orchestrator
        不持有"档案字段 → 槽位名"的业务映射，映射表放在对话层
        (:mod:`src.dialogue.profile_slots`)，这里只做委托。
        """
        from .dialogue.profile_slots import profile_slot_map

        return profile_slot_map(self._stored_profile(session_id))

    def _newly_filled_slots(self, before: Dict[str, Any], session_id: str) -> List[str]:
        """v2.7 §18：这一轮新填进来的槽位（Answer Coverage 的核心输入）。"""
        after = self._profile_slot_map(session_id)
        return [
            key for key, value in after.items()
            if key not in before or before.get(key) != value
        ]

    def _ranked_question_candidates(self, session_id: str) -> List[str]:
        """Dialogue Policy 的候选问题（按业务价值排序）—— 闸门拦下时换问用。"""
        try:
            from .dialogue import question_candidates

            profile = self._stored_profile(session_id)
            return [
                str(slot)
                for slot, _score in (question_candidates(profile, session_id=session_id) or [])
            ]
        except Exception:  # pragma: no cover - 防御式
            return []

    def _apply_duplicate_firewall(
        self,
        result: Dict[str, Any],
        *,
        session_id: str,
        coverage: Any,
        previous_question: Dict[str, Any],
        newly_filled: List[str],
        firewall: Any,
        next_candidate: Any,
    ) -> "tuple[str, str]":
        """v2.7 §19.1：发送前的重复提问闸门。

        Returns:
            ``(最终的 question_slot, duplicate_check 结论)``
        """
        slot = str(result.get("pending_slot") or "")
        if not result.get("pending_question") or not slot:
            return slot, "no_question"

        conversation = self._conversation_state(session_id)
        registry = getattr(conversation, "registry", None)
        question_state = ""
        if registry is not None:
            record = registry.get(slot)
            question_state = str(getattr(record, "question_state", "") or "") if record else ""

        decision = firewall.check(
            current_slot=slot,
            previous_slot=str(previous_question.get("question_slot") or ""),
            current_turn_id=str(result.get("turn_id") or ""),
            previous_turn_id=str(previous_question.get("turn_id") or ""),
            answered_slots=coverage.answered_slots,
            newly_filled_slots=newly_filled,
            question_state=question_state,
        )
        if decision.allowed:
            return slot, "pass"

        alternative = next_candidate(
            self._ranked_question_candidates(session_id),
            blocked_slot=decision.blocked_slot or slot,
            answered_slots=coverage.answered_slots,
        )
        if alternative:
            try:
                from .rag.readiness import question_for

                question = question_for(alternative, "en", 0, easier=False) or ""
            except Exception:  # pragma: no cover - 防御式
                question = ""
            if question:
                logger.info(
                    "[Firewall] %s → 改问 %s（不再重复 %s）",
                    decision.reason, alternative, slot,
                )
                result["pending_question"] = question
                result["pending_slot"] = alternative
                return alternative, f"{decision.reason}_rerouted"

        logger.info("[Firewall] %s → 本轮不再重复提问（slot=%s）", decision.reason, slot)
        result["pending_question"] = ""
        result["pending_slot"] = ""
        return "", decision.reason

    # ── v2.6 §8：AI 侧留痕 ───────────────────────────────────────────────
    def _note_ai_turn(self, result: Dict[str, Any], session_id: str, final: Any) -> None:
        try:
            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            state.note_ai_turn(
                action=final.action,
                question=str(result.get("pending_question") or ""),
                slot=final.question_slot or "",
                response=final.text,
                turn_id=final.turn_id,
                # 计划 §27：这一轮的 SpeechAct 也要能落进日志
                speech_act=str((result.get("speech_act") or {}).get("speech_act") or ""),
            )
            result["conversation_state"] = state.to_dict()
        except Exception as exc:  # pragma: no cover - 留痕失败不影响业务
            logger.warning("[ConversationState] note_ai_turn failed: %s", exc)
        self._emit_decision_audit(result, session_id, final)

    # ── v2.6 §28：Decision Audit ─────────────────────────────────────────
    def _emit_decision_audit(self, result: Dict[str, Any], session_id: str, final: Any) -> None:
        """记录"为什么这一轮问了这个问题"（只写日志，不影响业务）。"""
        try:
            import json

            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            payload = {
                "turn_id": final.turn_id,
                "session_id": session_id,
                "input": str(result.get("customer_input") or "")[:200],
                "speech_act": state.current_speech_act,
                # §28：决定之前的输入（需求档案 + 对话状态）
                "requirement_state_before": self._requirement_summary(session_id),
                # 2026-09-21：这一轮"听懂了什么、依据客户哪句话"
                "understanding": result.get("understanding") or {},
                "conversation_state_before": (
                    result.get("conversation_state_before") or state.to_dict()
                ),
                "conversation_state_after": state.to_dict(),
                "candidate_actions": list(result.get("action_candidates") or []),
                "selected_action": final.action,
                "discarded_actions": list(result.get("discarded_actions") or []),
                "final_response": final.text[:400],
                "question_count": final.question_count,
                "question_slot": final.question_slot,
                "validation_result": final.validation_result,
            }
            logger.info("[DecisionAudit] %s", json.dumps(payload, ensure_ascii=False))
            result["decision_audit"] = payload
        except Exception as exc:  # pragma: no cover - 审计失败不影响业务
            logger.warning("[DecisionAudit] emit failed: %s", exc)

    def _requirement_summary(self, session_id: str) -> Dict[str, Any]:
        """当前已知需求（拿不到就留空，绝不让审计影响业务）。"""
        try:
            profile = self._stored_profile(session_id)
            if profile is None:
                return {}
            facts = profile.to_facts() if hasattr(profile, "to_facts") else {}
            return {str(key): value for key, value in dict(facts or {}).items()}
        except Exception:  # pragma: no cover - 防御式
            return {}

    # ── v2.6 §5~§10：把客户这一句记进 ConversationState ──────────────────
    def _note_customer_turn(self, session_id: str, message: str, *, turn_id: str = "") -> None:
        """客户说完一句 → 判断"在回答哪一项"，并记账（不改变既有业务判定）。"""
        self._current_turn_id = turn_id
        if not str(message or "").strip():
            return
        try:
            from .dialogue import get_conversation_state

            state = get_conversation_state(session_id)
            match = state.answer_to(message)
            # v2.7 §18：把匹配结果留在会话状态里，收口时算 Answer Coverage 用
            state.last_answer_match = match.to_dict()
            state.note_customer_turn(
                text=message,
                answer_slot=(
                    match.slot
                    if match.kind in ("ANSWER_PREVIOUS_QUESTION", "ANSWER_WRONG_SLOT")
                    else ""
                ),
                turn_id=turn_id,
            )
            # 答非所问也不能丢信息（§10）：这一句明确给出的槽位都记 ANSWERED
            for slot in match.covered_slots:
                if slot == match.slot:
                    continue
                state.note_answered(str(slot))
            if match.kind in ("ANSWER_PREVIOUS_QUESTION", "ANSWER_WRONG_SLOT"):
                logger.info(
                    "[ConversationState] turn=%s %s answer_slot=%s expected=%s slots=%s",
                    turn_id, match.kind, match.slot, match.expected_slot, list(match.slots),
                )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("[ConversationState] note_customer_turn failed: %s", exc)

    # ── v2.5++++（计划 §15）：一轮的 LLM 统计收口 ────────────────────────
    def _finish_llm_turn(self, result: Dict[str, Any], session_id: str) -> None:
        """结束本轮记账，并把 messages / aggregated / llm_calls / latency 打到日志里。

        实测问题：同一轮实际打了好几次 DeepSeek，日志却是 `llm_calls=0`。
        现在无论走哪条分支（首轮接待 / Sales / Solution / 多屏），都会在这里收口。
        """
        try:
            import time as _time

            from .observability.llm_tracker import get_llm_tracker

            tracker = get_llm_tracker()
            context = tracker.current_turn()
            stats = tracker.end_turn()
            if stats is None:
                return
            total_ms = ((_time.time() - context.started_at) * 1000) if context else 0.0
            # v2.6 §27：一行日志能回答"为什么问了两个问题 / 这一轮到底做了什么"
            action = str(result.get("action") or "")
            question_slot = str(result.get("question_slot") or "")
            response_count = int(result.get("response_count") or 1)
            question_count = int(result.get("question_count") or 0)
            final_response = result.get("final_response") or {}
            speech_act = ""
            try:
                from .dialogue import get_conversation_state

                speech_act = str(get_conversation_state(session_id).current_speech_act or "")
            except Exception:  # pragma: no cover - 防御式
                speech_act = ""
            logger.info(
                "[Turn] session=%s messages=%s aggregated=%s llm_calls=%s "
                "llm_latency_ms=%s llm_tokens=%s total_ms=%s "
                "turn_id=%s action=%s speech_act=%s question_slot=%s "
                "response_count=%s question_count=%s validation=%s",
                session_id,
                context.message_count if context else 1,
                context.aggregated if context else False,
                stats.calls,
                round(stats.latency_ms, 1),
                stats.total_tokens,
                round(total_ms, 1),
                result.get("turn_id") or (context.turn_id if context else "-"),
                action or "-",
                speech_act or "-",
                question_slot or "-",
                response_count,
                question_count,
                str(final_response.get("validation_result") or "-"),
            )
            result["_llm_stats"] = stats.to_dict()
            perf_summary = result.get("_perf")
            if isinstance(perf_summary, dict):
                # 收口发生在 perf.summary() 之后 → 这里把 v2.6 的观测字段补进去
                perf_summary.update({
                    "turn_id": result.get("turn_id") or (context.turn_id if context else ""),
                    "action": action,
                    "speech_act": speech_act,
                    "question_slot": question_slot,
                    "response_count": response_count,
                    "question_count": question_count,
                })
            if context is not None:
                result["_turn"] = {
                    "message_count": context.message_count,
                    "aggregated": context.aggregated,
                    "turn_id": context.turn_id,
                    # v2.6 §27：统一可观测口径
                    "action": action,
                    "speech_act": speech_act,
                    "question_slot": question_slot,
                    "response_count": response_count,
                    "question_count": question_count,
                    "llm_calls": stats.calls,
                    "llm_latency_ms": round(stats.latency_ms, 1),
                    "total_latency_ms": round(total_ms, 1),
                }
        except Exception as exc:  # pragma: no cover - 统计失败不影响业务
            logger.warning("LLM tracker end_turn failed: %s", exc)



    
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

    def _split_and_apply_screen_specs(self, session_id: str, message: str) -> list:
        return self._multi_screen()._split_and_apply_screen_specs(session_id, message)

    def _maybe_target_screen(self, session_id: str, message: str):
        return self._multi_screen()._maybe_target_screen(session_id, message)

    def _maybe_start_new_item(self, session_id: str, message: str) -> str:
        return self._multi_screen()._maybe_start_new_item(session_id, message)

    def _screen_pending_block(self, index: int, profile_data: dict, language: str) -> str:
        return self._multi_screen()._screen_pending_block(index, profile_data, language)

    def _share_common_facts(self, session_id: str) -> None:
        return self._multi_screen()._share_common_facts(session_id)

    def _model_matches_screen(self, model_name: str, profile_data: dict) -> bool:
        return self._multi_screen()._model_matches_screen(model_name, profile_data)

    def _recommend_all_screens(self, session_id: str, message: str):
        return self._multi_screen()._recommend_all_screens(session_id, message)

    def _multi_item_follow_up(self, session_id: str, result, message: str):
        return self._multi_screen()._multi_item_follow_up(session_id, result, message)

    # 回复层（`src/dialogue/response_coordinator.py`）：只做委托
    def _attach_service_faq(self, response: str, message: str) -> str:
        return self._response_coordinator().attach_service_faq(response, message)

    def _attach_vision_confirmation(self, response: str, session_id: str, message: str) -> str:
        return self._response_coordinator().attach_vision_confirmation(
            response, session_id, message
        )

    def _vision_confirmation_sentence(self, session_id: str, message: str) -> str:
        return self._response_coordinator().vision_confirmation_sentence(session_id, message)

    def _compose_with_requirement_question(
        self,
        *,
        answer: str,
        sales_result: Dict[str, Any],
        message: str,
        seed: int = 0,
        session_id: str = "",
    ) -> str:
        return self._response_coordinator().compose_with_requirement_question(
            answer=answer,
            sales_result=sales_result,
            message=message,
            seed=seed,
            session_id=session_id,
        )

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
