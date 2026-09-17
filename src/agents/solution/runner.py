"""
Solution Agent Runner - main entry point for executing the RAG agent.
Supports Fast Path (structured) and Agent Path (full RAG).
"""
import re
from typing import Dict, Any, List, Optional
import logging

from .state import SolutionState
from .graph import build_solution_graph
from ...rag.retriever import retrieve
from ...rag.sparse import get_sparse_search
from ...rag.fusion import HybridSearch
from ...rag.rerank import is_ifp_product
from ...rag.router import classify_complexity, QueryRoute
from ...rag.fast_path import fast_path_handle
from ...utils.ifp_intent import has_ifp_intent, remove_unsupported_ifp_text, user_messages_text

logger = logging.getLogger(__name__)


def _strip_markdown(text: str) -> str:
    """Remove common markdown artifacts so the chat reply stays plain text."""
    if not text:
        return text
    cleaned = text.replace("**", "").replace("__", "")
    cleaned = re.sub(r"`+([^`]*?)`+", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    return cleaned.strip()


def _normalize_role(role: str) -> str:
    """Map message types to canonical role names."""
    return {"human": "user", "ai": "assistant"}.get(role, role)


def _normalize_history(history: List) -> List[Dict[str, str]]:
    """Normalize conversation history to role/content dicts."""
    result = []
    for entry in history:
        if isinstance(entry, dict):
            result.append({
                "role": _normalize_role(entry.get("role", "")),
                "content": entry.get("content", ""),
            })
        else:
            result.append({
                "role": _normalize_role(getattr(entry, "type", "unknown")),
                "content": getattr(entry, "content", ""),
            })
    return [m for m in result if m.get("content")]


def _normalize_index_documents(documents: List = None) -> List[Dict[str, Any]]:
    """Convert LangChain Document objects into dicts expected by Sparse/BM25."""
    if not documents:
        return []
    normalized = []
    for i, doc in enumerate(documents):
        if isinstance(doc, dict) and "text" in doc:
            item = dict(doc)
            item.setdefault("id", str(i))
            item.setdefault("metadata", {})
            normalized.append(item)
            continue
        page_content = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", None)
        if page_content is not None:
            normalized.append({
                "id": str(i),
                "text": page_content,
                "metadata": metadata or {},
            })
            continue
        if isinstance(doc, dict):
            normalized.append({
                "id": str(doc.get("id", i)),
                "text": doc.get("text") or doc.get("page_content") or "",
                "metadata": doc.get("metadata") or {},
            })
    return normalized


def _extract_requirements(message: str, existing: Dict = None) -> Dict[str, Any]:
    """Extract requirements from a message."""
    from ...rag.parameter_inference import extract_requirements
    return extract_requirements(message, existing or {})


def merge_fast_path_constraints(
    inferred: Dict[str, Any] | None,
    requirements: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """把 Sales 累加需求里的硬约束并进 fast path 的约束里。

    实测 bug：客户前面已经说了"室外 + 租赁"，后面只回一句 "P3" 时，fast path
    只用"本条消息"抽到的点间距约束 → 给室外租赁推荐了室内型号
    （TW11-3216 / TW21-3216），随后又被回复清洗器删掉，最后退化成
    "请放宽某个条件"的兜底话术。
    """
    constraints = dict(inferred or {})
    accumulated = requirements or {}

    if accumulated.get("display_type") and not constraints.get("display_type"):
        constraints["display_type"] = accumulated["display_type"]

    location = str(accumulated.get("location_type") or "")
    if accumulated.get("outdoor") is True or location in ("室外", "户外", "露天"):
        constraints.setdefault("outdoor", True)
        constraints.setdefault("indoor", False)
    elif accumulated.get("indoor") is True or location in ("室内", "户内"):
        constraints.setdefault("indoor", True)
        constraints.setdefault("outdoor", False)

    if accumulated.get("is_rental") is True:
        constraints.setdefault("is_rental", True)
    elif accumulated.get("is_rental") is False:
        constraints.setdefault("is_rental", False)

    return constraints


def routing_requirements(
    requirements: Dict[str, Any] | None,
    profile: Any = None,
) -> Dict[str, Any] | None:
    """给路由层用的需求上下文。

    Orchestrator 只传 ``profile``、不传 ``requirements``；路由层因此看不到
    "本会话已经说了场景 / 室内外 / 安装方式"，会把一句长得像参数查询的话
    （例如 "video mainly. we care about price"）判成 FAST，从而绕过确定性选型
    引擎与屏体尺寸追问（实测：室内 5m 教堂被推成 P0.7H）。

    这里补一份 legacy 视图；调用方已经传了 requirements 就原样使用。
    """
    if requirements:
        return requirements
    if profile is None:
        return requirements

    from ...models.legacy_adapter import profile_to_legacy

    return profile_to_legacy(profile) or requirements


class SolutionAgentRunner:
    """Runner class for the Solution Agent.
    
    Handles initialization and execution of the RAG agent for
    product recommendations and question answering.
    """
    
    def __init__(self, vectorstore, documents: List[Dict[str, Any]] = None):
        """Initialize the agent runner.
        
        Args:
            vectorstore: Chroma vector store instance
            documents: List of product documents for BM25 index
        """
        self.vectorstore = vectorstore

        from ...rag.bm25 import BM25Search
        from ...rag.sparse import get_sparse_search
        from ...rag.fusion import HybridSearch
        from ...config import config

        sparse = None
        bm25 = None
        documents = _normalize_index_documents(documents)
        if documents:
            sparse = get_sparse_search(documents, config.VECTORSTORE_DIR)
            logger.info(f"BGE-M3 sparse index loaded for {len(documents)} documents")
            bm25 = BM25Search(documents)
            logger.info(f"BM25 index built with {len(documents)} documents")

        # Initialize hybrid search with both retrievers
        self.hybrid_search = HybridSearch(vectorstore, sparse=sparse, bm25=bm25)

        # Compile the graph
        self.graph = build_solution_graph()
        logger.info("Solution Agent runner initialized")

    def _build_initial_state(
        self,
        message: str,
        history: List[Dict[str, Any]],
        requirements: Dict[str, Any] = None,
        additional_requirements: List[str] = None,
        profile: Any = None,
        session_id: str = "",
        intent: str = "",
    ) -> SolutionState:
        """Build the initial state for the agent."""
        # Normalize history
        normalized_history = _normalize_history(history)
        
        # Drop tail if caller already prepended current message
        if (normalized_history and normalized_history[-1].get("role") == "user" 
                and normalized_history[-1].get("content") == message):
            normalized_history = normalized_history[:-1]

        # ── M7：优先使用 Sales 传下来的 RequirementProfile ─────────────────────
        # Solution 不再自己从对话重建需求；旧 requirement 字典只是 Profile 的投影。
        from ...models.legacy_adapter import profile_to_solution_requirement

        if profile is not None:
            merged_requirement = profile_to_solution_requirement(profile)
            if not merged_requirement and requirements:
                merged_requirement = dict(requirements)
            logger.info(
                "Solution: 使用 Sales 的 RequirementProfile（%s 个字段），不再重建需求",
                len(merged_requirement),
            )
        elif requirements:
            location = requirements.get("location_type", "")
            is_indoor = location in ("室内", "户内", "室内使用")
            is_outdoor = location in ("户外", "室外", "外面", "露天", "全户外", "半户外", "户外使用", "室外使用")
            
            # Infer location from usage if not explicit
            outdoor_usages = ("演唱会", "音乐会", "体育", "足球", "篮球", "田径", "广告", "户外", "露天", "舞台", "演出")
            indoor_usages = ("会议", "教室", "培训", "医院", "商场", "展厅", "博物馆", "展示", "会议室", "报告厅")
            usage = requirements.get("usage", "")
            if not is_indoor and not is_outdoor and usage:
                if any(kw in usage for kw in outdoor_usages):
                    is_outdoor = True
                elif any(kw in usage for kw in indoor_usages):
                    is_indoor = True

            merged_requirement = {
                "indoor": is_indoor,
                "outdoor": is_outdoor,
                "distance": requirements.get("viewing_distance", ""),
                "purpose": requirements.get("usage", ""),
                "size": requirements.get("size", ""),
                "display_type": requirements.get("display_type"),
            }
        else:
            # Extract requirements from history
            merged_requirement = {}
            for entry in normalized_history:
                if entry.get("role") != "user":
                    continue
                parsed = _extract_requirements(entry.get("content", ""), merged_requirement)
                merged_requirement = {**merged_requirement, **(parsed.get("requirement") or {})}

            current_parsed = _extract_requirements(message, merged_requirement)
            merged_requirement = {**merged_requirement, **(current_parsed.get("requirement") or {})}

        # Determine intent
        if intent:
            # 上游（Sales / Orchestrator）已经定好了这一轮是什么意图，别再自己判一遍，
            # 否则"客户在问别的事"会被重新判成 recommendation → 又走推荐/反问。
            current_intent = intent
        elif requirements:
            current_intent = "recommendation"
        else:
            from .nodes.intent import detect_intent
            current_intent = detect_intent(message)

        return {
            "session_id": session_id,
            "messages": history + [{"role": "user", "content": message}],
            "requirement": merged_requirement,
            # 【M7】Sales 的 RequirementProfile 直接进入 Solution state，
            # recommendation_gate_node 会优先使用它（不再自己重建）
            "requirement_profile": profile,
            "intent": current_intent,
            "current_message": message,
            "info_sufficient": False,
            "missing_info": [],
            "pending_question": "",
            "products": [],
            "last_products": [],
            "recommendation": "",
            "reflection_score": 0,
            "reflection_notes": "",
            "needs_refine": False,
            "reflection_count": 0,
            "best_score": 0,
            "next_action": "",
            "hybrid_search": self.hybrid_search,
            "search_keywords": [],
            "additional_requirements": additional_requirements or [],
            "inferred_brightness_min_nit": None,
            "inferred_brightness_max_nit": None,
            "inferred_pixel_pitch_min_mm": None,
            "inferred_pixel_pitch_max_mm": None,
            "inferred_screen_size": None,
            "inferred_is_rental": None,
            "is_greeting": False,
            "greeting_reply": "",
            "has_conflict": False,
            "conflict_reason": "",
            "clarification_question": "",
            "conflict_message": "",
            "awaiting_confirmation": False,
            "display_type": "",
        }

    def run(
        self,
        message: str,
        history: List[Dict[str, Any]] = None,
        requirements: Dict[str, Any] = None,
        additional_requirements: List[str] = None,
        profile: Any = None,
        session_id: str = "",
        intent: str = "",
    ) -> Dict[str, Any]:
        """Run the agent with a user message.

        先做复杂度路由：
        - Fast Path（trivial / parameter / simple）→ 结构化过滤 + 模板
        - Agent Path（complex）→ 完整 LangGraph Agent + RAG + Reflection

        Args:
            message: User's message
            history: Conversation history
            requirements: Pre-extracted requirements from Sales Agent (optional)
            additional_requirements: Extra requirements for specialized analysis (optional)
            profile: Sales Agent 的 RequirementProfile（M7：唯一需求来源）

        Returns:
            Dict with answer and metadata
        """
        history = history or []

        # ── Step 1: 三层业务路由 ────────────────────────────────────────
        from src.rag.router import QueryRoute

        # 【修复】调用方（Orchestrator）只传了 profile，没传 requirements ——
        # 这里补一份 legacy 视图给路由与 fast path 用（详见 routing_requirements）
        requirements = routing_requirements(requirements, profile)
        routing = classify_complexity(message, existing_requirements=requirements)
        logger.info(
            "Routing: route=%s reason=%s inferred=%s",
            routing.route.value,
            routing.reason,
            routing.inferred_constraints,
        )

        # ── Step 2a: FAST ──────────────────────────────────────────────
        # 纯参数查询，不进 Agent，直接结构化过滤 + 模板
        if routing.route == QueryRoute.FAST:
            from src.config import config as _cfg

            # 【修复】fast path 不能只用"本条消息里抽到的约束"，必须并上 Sales
            # 已经收集的环境 / 安装方式 / 屏类型（见 merge_fast_path_constraints 注释）
            fast_constraints = merge_fast_path_constraints(
                routing.inferred_constraints, requirements
            )

            fast_result = fast_path_handle(
                query=message,
                constraints=fast_constraints,
                template_type=None,
                data_dir=_cfg.DATA_DIR,
            )
            return {
                "answer": fast_result["answer"],
                "requirement": routing.inferred_constraints or {},
                "reflection_score": 0,
                "reflection_notes": f"FAST ({routing.reason})",
                "products": fast_result.get("products", []),
                "route": "fast",
                "complexity": routing.complexity,
            }

        # ── Step 2b: Agent Path ────────────────────────────────────────
        initial_state = self._build_initial_state(
            message, history, requirements, additional_requirements, profile, session_id, intent
        )

        # 将路由层提取的约束注入 agent state（避免 LLM 重复推理）
        if routing.inferred_constraints and not initial_state.get("requirement"):
            for k, v in routing.inferred_constraints.items():
                if v is not None and initial_state.get("requirement", {}).get(k) is None:
                    initial_state["requirement"][k] = v

        # 注入 Reflection 预算参数（来自 config 或默认值）
        from ...config import config as _cfg

        initial_state["reflection_max_rounds"]       = getattr(_cfg, "REFLECTION_MAX_ROUNDS", 3)
        initial_state["reflection_max_tokens"]       = getattr(_cfg, "REFLECTION_MAX_TOKENS", 2000)
        initial_state["reflection_max_time_ms"]      = getattr(_cfg, "REFLECTION_MAX_TIME_MS", 8000)
        initial_state["reflection_no_improve_stop"] = getattr(_cfg, "REFLECTION_NO_IMPROVE_STOP", 2)

        try:
            result = self.graph.invoke(initial_state)

            # Build answer
            conflict_msg = result.get("conflict_message", "")
            searching_msg = result.get("searching_message", "")
            answer = result.get("recommendation", "")

            if conflict_msg:
                answer = conflict_msg
            elif searching_msg and answer and searching_msg not in answer:
                answer = f"{searching_msg}\n{answer}"

            answer = _strip_markdown(answer)

            # IFP safety net
            customer_text = user_messages_text(initial_state.get("messages", []))
            response_products = result.get("products", [])
            if not has_ifp_intent(user_text=customer_text):
                answer = remove_unsupported_ifp_text(answer)
                response_products = [p for p in response_products if not is_ifp_product(p)]

            # Remove internal implementation wording
            final_requirement = result.get("requirement", initial_state.get("requirement", {})) or {}
            from ...rag.rerank import sanitize_customer_response
            answer = sanitize_customer_response(answer, outdoor=bool(final_requirement.get("outdoor")))

            if not answer:
                # 【客户口径】不说"找不到"，改成邀请客户放宽某个条件
                from ...rag.reply_composer import relaxation_answer

                answer = relaxation_answer()

            # Keep compact format
            answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())

            # Keep response concise - max 800 chars
            if len(answer) > 800:
                truncated = answer[:800]
                last_period = max(truncated.rfind('。'), truncated.rfind('！'), truncated.rfind('？'))
                if last_period > 200:
                    answer = truncated[:last_period + 1]
                else:
                    answer = truncated + "..."

            return {
                "answer": answer,
                "requirement": result.get("requirement", {}),
                "reflection_score": result.get("reflection_score", 0),
                "reflection_notes": result.get("reflection_notes", ""),
                "products": response_products[:1],
                "route": routing.route.value,
                "complexity": routing.complexity,
            }
        except Exception as e:
            logger.error(f"Agent run error: %s", e)
            return {
                "answer": "抱歉，遇到了错误，请重试。",
                "requirement": {},
                "reflection_score": 0,
                "reflection_notes": str(e),
                "products": [],
                "route": "agent",
                "complexity": routing.complexity,
            }

    def run_stream(
        self,
        message: str,
        history: List[Dict[str, Any]] = None,
        profile: Any = None,
        session_id: str = "",
    ):
        """Stream the agent response.
        
        Args:
            message: User's message
            history: Conversation history
            
        Yields:
            Response chunks
        """
        history = history or []
        initial_state = self._build_initial_state(
            message, history, profile=profile, session_id=session_id
        )

        for event in self.graph.stream(initial_state):
            for node_name, node_result in event.items():
                if node_name == "recommend":
                    yield node_result.get("recommendation", "")
                elif node_name == "intent":
                    conflict_msg = node_result.get("conflict_message", "")
                    if conflict_msg:
                        yield conflict_msg
                        return
