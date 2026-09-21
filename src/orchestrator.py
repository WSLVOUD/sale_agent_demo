"""
Orchestrator - Dual Agent Coordination
Coordinates between Sales Agent, Solution Agent, and First Contact Flow
"""
import logging
import time
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
            perf.llm_calls,
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
        """
        # v2.5+++（计划 §3）：把本轮"想要问的那一项"作为候选问题交给 FinalResponseGuard，
        # 由它按优先级（硬性 Gate > 工程必要 > 推荐优化 > 销售偏好）只保留一个。
        questions = []
        if result.get("pending_question"):
            questions.append({
                "text": str(result.get("pending_question") or ""),
                "slot": str(result.get("pending_slot") or ""),
                "source": "sales",
            })
        text = self._response_coordinator().finalize(
            str(result.get("response") or ""),
            session_id=session_id,
            message=message,
            questions=questions,
        )
        if text:
            result["response"] = text
        # v2.5+++（计划 §3）：附加气泡（extra_messages）也要收口 ——
        # 否则会出现"主回复问一个问题、附加气泡又冒出一个问题"两个气泡连着问。
        extras = [str(item) for item in (result.get("extra_messages") or []) if item]
        if extras:
            guarded = self._response_coordinator().guard_extras(
                str(result.get("response") or ""), extras
            )
            if guarded:
                result["extra_messages"] = guarded
            else:
                result.pop("extra_messages", None)
        return result



    
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
