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
        history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user message through appropriate agent(s).

        Args:
            message: User's message
            session_id: Session identifier
            history: Conversation history (optional)

        Returns:
            Dict with response and metadata
        """
        perf = PerfTracker(session_id, message)

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
                if first_slots:
                    profile = RequirementProfile.from_slots(
                        first_slots, explicit_keys=set(first_slots)
                    )
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
            return result

        # ── Sales Agent（仅在首次接待完成后执行）────────────────────────────
        # Step 1: Sales Agent processes message
        sales_result = self.sales_agent.run(
            session_id=session_id,
            message=message
        )
        perf.mark("sales_done")
        perf.intent = sales_result.get("intent", "")

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
        return result
    
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
