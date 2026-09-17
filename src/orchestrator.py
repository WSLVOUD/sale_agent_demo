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

# ── 语言配置（v2.0 Phase 14）────────────────────────────────────────────────
# 由 config.RESPONSE_LANGUAGE_POLICY 控制：
#   "en"   → 所有 AI 回复强制英语（系统既有策略，默认）
#   "auto" → 跟随客户语言回复（v2.0 的 Original Language Response）
try:
    from .config import config as _config

    RESPONSE_LANGUAGE = _config.RESPONSE_LANGUAGE_POLICY
except Exception:  # pragma: no cover - 防御式
    RESPONSE_LANGUAGE = "en"


def _vision_enabled() -> bool:
    """视觉开关（默认开；配置里 VISION_ENABLED=false 可整体关掉）。"""
    try:
        return bool(getattr(_config, "VISION_ENABLED", True))
    except Exception:  # pragma: no cover - 防御式
        return True


def _merge_vision_into_stored_profile(
    memory_store: Any,
    session_id: str,
    vision_results: List[Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """把图片识别出的需求并进"memory 里的需求档案"。

    为什么放在 Orchestrator 而不是 Sales Agent：
      - 需求档案是 Sales Agent 的输入，图片在 Sales 之前就要合并好；
      - Sales Agent 保持不动（计划第十六阶段：不新建 Vision Sales Agent）。

    合并结果写回 memory；下一轮 Sales Agent 从 memory 读取时即可看到图片信息。
    任何异常都不往上抛（图片不能拖垮主流程）。
    """
    if not vision_results or not memory_store:
        return
    try:
        from .models.requirement import RequirementProfile
        from .vision import apply_vision_to_profile

        stored = None
        if hasattr(memory_store, "get_requirement_profile"):
            stored = memory_store.get_requirement_profile(session_id)
        profile = None
        if isinstance(stored, RequirementProfile):
            profile = stored
        elif isinstance(stored, dict) and stored:
            try:
                profile = RequirementProfile.model_validate(stored)
            except Exception:  # pragma: no cover - 防御式
                profile = None
        if profile is None:
            profile = RequirementProfile()

        for result in vision_results:
            profile, stats = apply_vision_to_profile(profile, result)
            logger.info(
                "[%s] Vision merged into profile: fields=%s conflicts=%s size_hint=%s",
                session_id, stats.get("merged_fields"), stats.get("conflict_count"),
                stats.get("size_hint"),
            )

        if hasattr(memory_store, "set_requirement_profile"):
            memory_store.set_requirement_profile(session_id, profile)
        if isinstance(metrics, dict):
            metrics["merge_success"] = True
        # 旧的 requirements 视图由 Sales Agent 下一轮从 Profile 重建
    except Exception as exc:  # pragma: no cover - 防御式
        logger.warning("[%s] Vision merge failed: %s", session_id, exc)


class PerfTracker:
    """Lightweight performance tracker for one request cycle."""

    def __init__(self, session_id: str, message: str):
        self.session_id = session_id
        self.message = message
        self._t0: float = time.time()
        self._markers: Dict[str, float] = {"_start": time.time()}
        self.route: str = "unknown"
        self.solution_route: Optional[str] = None   # "fast" | "agent"
        self.intent: str = ""
        self.llm_calls: int = 0
        self.final_products: int = 0

    def mark(self, name: str) -> None:
        self._markers[name] = time.time()

    @property
    def total_ms(self) -> float:
        return (time.time() - self._t0) * 1000

    def latency(self, after: str) -> float:
        """Milliseconds between marker 'after' and now."""
        t = self._markers.get(after)
        return (time.time() - t) * 1000 if t else 0.0

    def _span(self, a: str, b: str) -> float:
        """Milliseconds between two markers."""
        ta = self._markers.get(a)
        tb = self._markers.get(b)
        if ta is None or tb is None:
            return 0.0
        return (tb - ta) * 1000

    def summary(self) -> Dict[str, Any]:
        """Build a structured perf summary dict."""
        return {
            "session_id": self.session_id,
            "route": self.route,
            "solution_route": self.solution_route,
            "intent": self.intent,
            "total_latency_ms": round(self.total_ms, 1),
            "sales_latency_ms": round(self._span("_start", "sales_done"), 1),
            "solution_latency_ms": round(self.latency("sales_done"), 1),
            "llm_calls": self.llm_calls,
            "final_products": self.final_products,
            "first_contact_intro": getattr(self, "first_contact_intro", ""),
            "first_contact_messages": getattr(self, "first_contact_messages", []),
        }


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
                result["_perf"] = perf.summary()
                if vision_metrics:
                    result["vision"] = vision_metrics
                return result

        # ── 一个项目多条屏：第二块（及以后）的推荐要标清楚是哪一块 ────────────
        # 实测：客户开了第二块屏之后，推荐话术里仍会出现 "For your church indoor
        # screen …"（历史里的第一块屏），客户分不清在说哪一块。
        if perf.route == "trigger_solution" and result.get("response"):
            from .rag.project_items import screen_label
            from .rag.reply_composer import reply_language

            active_index = (
                self.memory_store.get_active_item_index(session_id)
                if self.memory_store and hasattr(self.memory_store, "get_active_item_index")
                else 0
            )
            if active_index >= 1 and self._stored_profile(session_id) is not None:
                label = screen_label(
                    active_index, self._stored_profile(session_id), reply_language(message)
                )
                if label and not str(result["response"]).lstrip().startswith(label.strip()):
                    result["response"] = f"{label}{result['response']}"

        # ── 一个项目多条屏：记录这一块屏的推荐 + 追问"还有其他位置吗" ────────
        multi_extra = self._multi_item_follow_up(session_id, result, message)
        if multi_extra:
            result.setdefault("extra_messages", []).append(multi_extra)
        if vision_metrics:
            result["vision"] = vision_metrics
        return result
    
    # ── 一个项目多条屏（客户口径 2026-09-18）─────────────────────────────
    def _split_and_apply_screen_specs(self, session_id: str, message: str) -> list:
        """一句话里给了多块屏的规格 → 每块屏各存一份需求。

        客户口径：一句里能看出两块屏（"4m wide x2.5 high for indoor and 3m x2m
        for the outdoor"）就**分别记录**；分不清哪组参数属于哪块屏时两块记成一样的。
        返回拆分结果（空列表表示这一轮不是多屏描述）。
        """
        from .models.requirement import RequirementProfile
        from .rag.project_items import split_multi_screen_specs

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return []

        specs = split_multi_screen_specs(message)
        if len(specs) < 2:
            return []

        base = self._stored_profile(session_id)
        base_slots = base.model_dump() if base is not None else {}
        shared_skip = (
            "sources", "ask_counts", "unknown_reasons", "conflicts", "conflict_slots",
            "last_asked_slot", "model", "series_id", "pixel_pitch_mm",
        )

        items = memory_store.get_project_items(session_id)
        while len(items) < len(specs):
            items.append({})

        for index, spec in enumerate(specs):
            slots = {key: value for key, value in base_slots.items() if key not in shared_skip}
            # 这一块屏**自己已经有**的值优先（否则第二句话只说了点间距时，
            # 会把另一块屏的安装方式/尺寸覆盖成 base 的值 —— 实测踩过）
            existing = dict((items[index] or {}).get("profile") or {})
            for key, value in existing.items():
                if key == "sources" or value in (None, "", [], {}):
                    continue
                slots[key] = value
            if spec.get("environment"):
                slots["environment"] = spec["environment"]
            if spec.get("width_m"):
                slots["target_width_m"] = spec["width_m"]
            if spec.get("height_m"):
                slots["target_height_m"] = spec["height_m"]
            # 安装方式与点间距也要按屏幕分开（"permanent for indoor, rental for
            # outdoor" / "p3 for indoor p5 for outdoor" 实测会被混到一块上）
            if spec.get("installation"):
                slots["installation"] = spec["installation"]
            if spec.get("pixel_pitch_mm"):
                slots["pixel_pitch_mm"] = spec["pixel_pitch_mm"]
            try:
                profile = RequirementProfile.model_validate(slots)
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[%s] Multi-spec profile build failed: %s", session_id, exc)
                continue
            # 这些值是客户这句话里明说的 → 记为 explicit，直接可用于选型/计算
            for field in (
                "environment", "target_width_m", "target_height_m",
                "installation", "pixel_pitch_mm",
            ):
                profile.sources[field] = "explicit"
            record = dict(items[index] or {})
            record["profile"] = profile.model_dump()
            # 规格变了 → 这一块之前的推荐作废，重新推荐
            record.pop("model", None)
            items[index] = record

        memory_store.set_project_items(session_id, items)
        memory_store.set_active_item_index(session_id, len(specs) - 1)
        last = memory_store.get_project_items(session_id)[len(specs) - 1].get("profile")
        if last:
            memory_store.set_requirement_profile(
                session_id, RequirementProfile.model_validate(last)
            )
        logger.info("[%s] Multi-item: 按一句话拆出 %d 块屏的规格", session_id, len(specs))
        return specs

    def _maybe_target_screen(self, session_id: str, message: str) -> Optional[int]:
        """客户指明"改哪一块屏" → 把当前档案切到那一块，本轮就改它。"""
        from .models.requirement import RequirementProfile
        from .rag.project_items import detect_screen_target

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        items = memory_store.get_project_items(session_id)
        index = detect_screen_target(message, items)
        if index is None:
            return None

        current = memory_store.get_active_item_index(session_id)
        if index == current:
            return index

        profile = self._stored_profile(session_id)
        if profile is not None and current < len(items):
            record = dict(items[current] or {})
            record["profile"] = profile.model_dump()
            items[current] = record
            memory_store.set_project_items(session_id, items)

        target = (items[index] or {}).get("profile")
        if target:
            try:
                memory_store.set_requirement_profile(
                    session_id, RequirementProfile.model_validate(target)
                )
            except Exception as exc:  # pragma: no cover - 防御式
                logger.warning("[%s] Screen switch failed: %s", session_id, exc)
                return None
        memory_store.set_active_item_index(session_id, index)
        logger.info("[%s] Multi-item: 客户指定改第 %d 块屏", session_id, index + 1)
        return index

    def _maybe_start_new_item(self, session_id: str, message: str) -> str:
        """客户这句话如果在说"另一块屏" → 归档当前这块，开一条新需求档案。

        客户口径："教堂里一块室内屏，门口再来一块室外屏" —— 一个项目、两块屏，
        各自收集需求、各自推荐、各自算箱体，最后给一份汇总。
        """
        from .rag.project_items import detect_new_item

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return ""

        profile = self._stored_profile(session_id)
        if profile is None:
            return ""

        items = memory_store.get_project_items(session_id)
        index = memory_store.get_active_item_index(session_id)
        already = bool(
            getattr(memory_store, "has_recommendation", lambda *_: False)(session_id)
        ) or bool((items[index] if index < len(items) else {}).get("model"))

        should_start, reason = detect_new_item(
            message, profile, already_recommended=already
        )
        if not should_start:
            return ""

        # 归档第 N 块屏（保留需求档案，最后汇总要用）
        while len(items) <= index:
            items.append({})
        archived = dict(items[index] or {})
        archived["profile"] = profile.model_dump()
        items[index] = archived
        memory_store.set_project_items(session_id, items)
        memory_store.set_active_item_index(session_id, index + 1)

        # 客户口径：客户没说"这块要什么、那块要什么"时，两块**记成一样的** ——
        # 所以新条目先复制当前这一份需求，本轮这句话里明说的差异（"室外的"）再覆盖上去。
        from .models.requirement import RequirementProfile

        seeded = RequirementProfile.model_validate(profile.model_dump())
        items = memory_store.get_project_items(session_id)
        while len(items) <= index + 1:
            items.append({})
        items[index + 1] = {"profile": seeded.model_dump()}
        memory_store.set_project_items(session_id, items)
        memory_store.set_requirement_profile(session_id, seeded)
        # 旧 requirements 是 Profile 的只读投影，下一轮由 Sales 重建
        memory_store.clear_requirements(session_id)
        if hasattr(memory_store, "clear_recommendation"):
            memory_store.clear_recommendation(session_id)
        logger.info(
            "[%s] Multi-item: 开始收集第 %d 块屏的需求（%s）", session_id, index + 2, reason
        )
        return reason

    # 多块屏之间**默认共享**的字段（客户没说"这块要什么、那块要什么"时就一样）：
    # 内容类型 / 安装方式 / 视距 / 价格取向 / 屏类型 / 场景。
    # 环境与尺寸**不共享** —— 那正是区分两块屏的东西。
    _SHARED_FACT_FIELDS = (
        "display_type", "purpose", "content_type", "installation",
        "price_preference", "budget_level", "viewing_distance_m",
        # 没说清就两块一样：点间距也一样（"P3 for the indoor P5 for the outdoor"
        # 这种明确说了的才各自不同）
        "pixel_pitch_mm",
    )

    def _share_common_facts(self, session_id: str) -> None:
        """把当前这块屏已确认的共有项同步给其他屏，避免同一个问题被问第二遍。

        实测：两块屏时，客户答了 "both"（视频/图片）之后，系统又为另一块屏
        把同一个问题问了一遍 —— 客户体验就是"回答过了还问"。
        """
        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return
        items = memory_store.get_project_items(session_id)
        if len(items) < 2:
            return
        active = memory_store.get_active_item_index(session_id)
        if active >= len(items):
            return
        # 当前这块的**实时**档案（items 里的可能还没同步过来）
        live = self._stored_profile(session_id)
        source = live.model_dump() if live is not None else dict(
            (items[active] or {}).get("profile") or {}
        )
        if not source:
            return
        sources = dict(source.get("sources") or {})
        changed = False
        for index, item in enumerate(items):
            merged = dict(item or {})
            if index == active:
                # 当前这块：把实时档案写回条目，保证"每条记录都是完整的"
                merged["profile"] = source
                items[index] = merged
                continue
            profile = dict((item or {}).get("profile") or {})
            if not profile:
                continue
            for field in self._SHARED_FACT_FIELDS:
                value = source.get(field)
                if value in (None, "", [], {}) or profile.get(field) not in (None, "", [], {}):
                    continue
                profile[field] = value
                if sources.get(field):
                    profile.setdefault("sources", {})[field] = sources[field]
                changed = True
            if changed:
                merged = dict(item or {})
                merged["profile"] = profile
                items[index] = merged
        if changed:
            memory_store.set_project_items(session_id, items)

    @staticmethod
    def _model_matches_screen(model_name: str, profile_data: Dict[str, Any]) -> bool:
        """型号的使用环境是否和这块屏一致（避免"室外屏推室内型号"）。"""
        environment = str((profile_data or {}).get("environment") or "").strip().lower()
        if environment not in ("indoor", "outdoor"):
            return True
        try:
            from .config import config
            from .rag.json_loader import canonical_model_index

            record = canonical_model_index(config.DATA_DIR).get(model_name)
            if record is None:
                return True
            if environment == "indoor":
                return bool(getattr(record, "indoor", False))
            return bool(getattr(record, "outdoor", False))
        except Exception:  # pragma: no cover - 防御式
            return True

    def _recommend_all_screens(self, session_id: str, message: str) -> Optional[str]:
        """多块屏：**每块屏各跑一次推荐**，合成一条回复（一块屏一个型号）。

        实测问题：客户一次给了两块屏的规格时，只有"当前那块"会跑推荐，
        另一块没有型号 → 回复里只出现一个型号、另一个屏被漏掉。
        """
        from .models.requirement import RequirementProfile
        from .rag.project_items import product_model, screen_label
        from .rag.reply_composer import reply_language

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        items = memory_store.get_project_items(session_id)
        if len(items) < 2:
            return None

        history = self._load_history(session_id)
        language = reply_language(message)
        blocks: list = []
        changed = False

        # 【修复串台】先把**当前这块屏的实时档案**写回它的条目：
        # 之前用 items[active] 里的旧拷贝（会少掉刚说的点间距/视距），
        # 导致那一块屏拿旧档案去推荐 → 0 候选 → 兜底话术。
        live = self._stored_profile(session_id)
        if live is not None and 0 <= memory_store.get_active_item_index(session_id) < len(items):
            active_index = memory_store.get_active_item_index(session_id)
            merged_active = dict(items[active_index] or {})
            merged_active["profile"] = live.model_dump()
            items[active_index] = merged_active
            memory_store.set_project_items(session_id, items)

        for index, item in enumerate(items):
            profile_data = item.get("profile") or {}
            model_name = str(item.get("model") or "").strip()
            text = str(item.get("reply") or "").strip()
            if not model_name:
                if not profile_data:
                    continue
                try:
                    outcome = self.solution_agent.run(
                        message=message,
                        history=history,
                        session_id=session_id,
                        profile=RequirementProfile.model_validate(profile_data),
                        intent="need_query",
                    )
                except Exception as exc:  # pragma: no cover - 防御式
                    logger.warning("[%s] Per-screen recommend failed: %s", session_id, exc)
                    continue
                products = outcome.get("products") or []
                if not products:
                    continue
                model_name = product_model(products[0])
                if not model_name:
                    continue
                # 【修复串台】型号必须和这块屏的环境一致：
                # 实测室外那块屏被推了室内租赁型号 TW11-IR-P4.8 —— 那种宁可不出，
                # 也不能把不同环境的型号写给客户。
                if not self._model_matches_screen(model_name, profile_data):
                    logger.warning(
                        "[%s] 屏 %d 环境=%s 但推荐出 %s（环境不符）→ 跳过",
                        session_id, index + 1, profile_data.get("environment"), model_name,
                    )
                    continue
                text = str(outcome.get("answer") or "").strip()
                item["model"] = model_name
                item["reply"] = text
                changed = True

            label = screen_label(index, profile_data, language)
            if not text:
                text = label + model_name
            elif not text.lstrip().startswith(label.strip()):
                text = label + text
            blocks.append(text)

        if changed:
            memory_store.set_project_items(session_id, items)
        if len(blocks) < 2:
            return None
        logger.info("[%s] Multi-item: 按 %d 块屏分别推荐", session_id, len(blocks))
        return "\n\n".join(blocks)

    def _multi_item_follow_up(
        self, session_id: str, result: Dict[str, Any], message: str = ""
    ) -> Optional[str]:
        """推荐完之后：记录这一块屏的结果；第二块（及以后）推荐完就给整份汇总。

        客户口径（2026-09-18 二次确认）：**不要**在推荐后主动追问
        "By the way — is this the only screen in the project…"。
        多屏仍然支持 —— 客户自己提到第二块屏（"门口再来一块室外的屏" /
        "another screen for the entrance"）时照常开新条目，最后出汇总。
        """
        from .rag.project_items import product_model, screen_label
        from .rag.reply_composer import reply_language

        memory_store = self.memory_store
        if memory_store is None or not hasattr(memory_store, "get_project_items"):
            return None

        products = result.get("products") or []
        profile = self._stored_profile(session_id)
        index = memory_store.get_active_item_index(session_id)

        # 只有真的走了推荐路径才算"这一块屏推荐完了"（自由问答里的检索片段不算）
        route = str((result.get("_perf") or {}).get("route") or "")
        if route != "trigger_solution":
            return None

        customer_message = str(message or "")

        # 这一轮没有推荐出产品 → 不改动项目状态
        if not products:
            return None

        model = product_model(products[0])
        if not model:
            return None

        memory_store.record_item_recommendation(
            session_id,
            {
                "model": model,
                "profile": profile.model_dump() if profile is not None else {},
                # 这一块屏自己的推荐话术（拼"多屏一起给"的时候要用）
                "reply": str(result.get("response") or "").strip(),
            },
        )
        items = memory_store.get_project_items(session_id)
        language = reply_language(customer_message)

        # 客户口径（2026-09-18）多块屏时要**按屏幕数量**推荐：
        # 每块屏给一个型号，不再只报一块屏。这里把每块屏自己的推荐话术拼成一条，
        # 每条前面带 "Screen N (…)" 标签，客户一眼能看出哪块屏对应哪个型号。
        blocks: list = []
        for position, item in enumerate(items, start=1):
            model_name = str(item.get("model") or "").strip()
            if not model_name:
                continue
            text = str(item.get("reply") or "").strip()
            label = screen_label(position - 1, item.get("profile") or {}, language)
            if not text:
                text = label + model_name
            elif not text.lstrip().startswith(label.strip()):
                # 第一块屏当初是单屏推荐、没带标签 → 这里补上，客户才分得清哪块是哪块
                text = label + text
            blocks.append(text)
        if len(blocks) >= 2:
            result["response"] = "\n\n".join(blocks)
            logger.info("[%s] Multi-item: 按 %d 块屏分别给出推荐", session_id, len(blocks))
            return None

        return None

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

    def _compose_with_requirement_question(
        self,
        *,
        answer: str,
        sales_result: Dict[str, Any],
        message: str,
        seed: int = 0,
    ) -> str:
        """把"答复客户"与"继续追问需求"合成一句自然的销售回复。

        需求还没问清时（Ready Gate 未放行），客户的问题照样要答 —— 但答完必须
        把还缺的那个关键问题接上，否则销售就变成"只会问问题"或"答完就断线"。
        """
        question = str(sales_result.get("pending_question") or "")
        if not question:
            return answer
        try:
            from .rag.reply_composer import compose_requirement_reply, reply_language

            return compose_requirement_reply(
                answer=answer,
                question=question,
                slot=str(sales_result.get("pending_slot") or ""),
                message=message,
                language=reply_language(message),
                seed=seed,
                requirement=sales_result.get("requirements") or {},
                include_ack=not sales_result.get("requirements_reset", False),
                llm_ack=str(sales_result.get("acknowledgement") or ""),
            )
        except Exception as exc:  # pragma: no cover - 防御式
            logger.warning("Compose requirement reply failed: %s", exc)
            return answer or question
