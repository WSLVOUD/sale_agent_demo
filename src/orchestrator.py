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
    ) -> Dict[str, Any]:
        """
        Process user message through appropriate agent(s).

        Args:
            message: User's message
            session_id: Session identifier
            history: Conversation history (optional)
            images: 客户随消息发送的图片（可选，最多 VISION_MAX_IMAGES 张）

        Returns:
            Dict with response and metadata
        """
        perf = PerfTracker(session_id, message)
        # v2.6 §24/§27：这一轮的 turn_id（FinalResponse / 日志 / DecisionAudit 共用）
        turn_id = uuid.uuid4().hex[:12]
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
            self._finalize_turn_response(result, session_id, message)
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
        result.setdefault("action_candidates", self._action_candidates(sales_result))
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
                self._finalize_turn_response(result, session_id, message)
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
        self._finalize_turn_response(result, session_id, message)
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

    def _finalize_turn_response(self, result: Dict[str, Any], session_id: str, message: str) -> Dict[str, Any]:
        """本轮回复的最后两道加工（顺序固定）：

          1. 客户问到的售后口径（说明书图纸 / 现场安装 / 质保）—— 必须回答；
          2. 本轮带了图片 → **先把"图片里看到什么"跟客户核一遍**，再继续问需求。

        实测 bug（客户日志）：客户只发了图片 + "i need this"，系统直接跳到问点间距，
        既没跟客户核对图片识别结果，客户也没机会纠正判错的"固定/租赁"。

        v2.6 §4（ONE TURN → ONE ACTION → ONE RESPONSE）：收口不再只是"把多出来的
        问句删掉"，而是**只剩一条客户可见回复** —— 追加气泡并入正文，最终由
        ``FinalResponseCoordinator`` 产出唯一的 ``FinalResponse``。
        """
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

        # ① 售后口径 + 图片核对 + Guard 收口（原有链路，先算出"想说的话"）
        text = self._response_coordinator().finalize(
            str(result.get("response") or ""),
            session_id=session_id,
            message=message,
            questions=questions,
        )
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
        result["conversation_state_before"] = conversation_before
        self._finish_llm_turn(result, session_id)
        return result

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
        if result.get("pending_question"):
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
