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
from ...rag.router import has_structured_requirement
from ...rag.structured_product_query import structured_product_query_handle
from ...utils.ifp_intent import remove_unsupported_ifp_text, user_messages_text

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


def cap_answer_length(answer: str, *, max_length: int = 2000) -> str:
    """回复的"防跑飞"长度上限（推荐/交付轮不能因为长度被砍半句）。

    客户口径（2026-09-28）：推荐时要完整交付"型号 + 实测参数 + 两种排布"，
    多屏时**每一块屏**都要完整。原实现把答案硬切在 800 字符，而且只认中文标点
    —— 英文回复会被切在句子中间并补上 "..."。现在上限放宽到 2000 字符（仍然
    防 LLM 跑飞），并且统一在**句子边界**收尾（中英文标点都认，见
    ``src/utils/text.truncate_text``）。
    """
    from ...utils.text import truncate_text

    return truncate_text(str(answer or ""), max_length=max_length)


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


def _profile_from(result: Any, initial_state: Any) -> Any:
    """从 result / initial_state 里取需求档案（取到就返回）。"""
    profile = None
    for source in (result, initial_state):
        if isinstance(source, dict):
            profile = source.get("requirement_profile") or profile
    return profile


def _profile_family(result: Any, initial_state: Any) -> str:
    """这一轮属于哪条产品链路（led / lcd / ifp）—— 给兜底话术选口径。"""
    try:
        from ...utils.product_family import product_family_of

        return product_family_of(_profile_from(result, initial_state))
    except Exception:  # pragma: no cover - 防御式
        return ""


def _is_lcd_like_profile(profile: Any) -> bool:
    """档案是不是 LCD / IFP 会话（这两条走自己的选型链路）。

    客户口径（2026-09-30）："完全把两个链路隔离开"—— 结构化产品查询那条路是按
    LED 的字段（点间距 / 亮度 / 防护等级）找型号的，LCD / IFP 会话不能走进去。
    LED 侧行为不变（这里只对 LCD/IFP 返回 True）。
    """
    if profile is None:
        return False
    display_type = str(
        getattr(profile, "display_type", "")
        or (profile.get("display_type") if isinstance(profile, dict) else "")
        or ""
    ).upper()
    return display_type in ("LCD", "IFP")


def _profile_needs_ifp(result: Any, initial_state: Any) -> bool:
    """需求档案是不是明确要交互平板（IFP）。

    这是 IFP 安全网的**主路径**：客户自己说了要手写 / 触控（且是会议教育场景），
    或者档案里的产品类型就是 IFP。关键词判定只作为兜底 —— 客户把 "meeting"
    拼成 "metting" 不该让系统把自己选出来的交互平板删掉（实测 2026-09-30）。
    """
    try:
        from ...dialogue.lcd_decision import is_ifp_requirement
    except Exception:  # pragma: no cover - 防御式
        return False
    profile = _profile_from(result, initial_state)
    if profile is None:
        return False
    try:
        return bool(is_ifp_requirement(profile))
    except Exception:  # pragma: no cover - 防御式
        return False


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
        already_recommended: bool = False,
        previous_models: Optional[List[str]] = None,
        multi_screen_brief: str = "",
    ) -> SolutionState:
        """Build the initial state for the agent."""
        # Normalize history
        normalized_history = _normalize_history(history)
        
        # Drop tail if caller already prepended current message
        if (normalized_history and normalized_history[-1].get("role") == "user" 
                and normalized_history[-1].get("content") == message):
            normalized_history = normalized_history[:-1]

        # Phase 12-1：没有 RequirementProfile 时明确标记未就绪（见下面的分支）
        requirement_not_ready = False

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
        else:
            # ── Phase 12-1（计划 2.0）步骤 3/4：关闭 Solution 自己的需求重建 ──────
            # 以前这里有两条"第二套需求逻辑"：
            #   (a) 用中文关键词表重新推断室内外（outdoor_usages / indoor_usages）；
            #   (b) 逐条扫 history 调 _extract_requirements 重新抽一遍需求。
            # 现在都不做了：没有 RequirementProfile 就**只原样带上调用方给的旧字典**，
            # 并标记 REQUIREMENT_NOT_READY，交由上层（Sales / Orchestrator）处理。
            merged_requirement = dict(requirements or {})
            requirement_not_ready = True
            logger.warning(
                "Solution: 没有 RequirementProfile → REQUIREMENT_NOT_READY"
                "（不再从 history / 关键词重建需求）"
            )

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
            # "另外推荐一款" = 换一个没给过的型号（不是重新采集需求）
            "already_recommended": bool(already_recommended),
            "previous_recommended_models": list(previous_models or []),
            "messages": history + [{"role": "user", "content": message}],
            "requirement": merged_requirement,
            # 【M7】Sales 的 RequirementProfile 直接进入 Solution state，
            # recommendation_gate_node 会优先使用它（不再自己重建）
            "requirement_profile": profile,
            # Phase 12-1：没有 profile → 明确告诉上层"需求还没准备好"
            "requirement_not_ready": bool(requirement_not_ready),
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
            # 多屏逐屏推荐：这一段只写哪一块屏（由 MultiScreenManager 传）
            "multi_screen_brief": str(multi_screen_brief or ""),
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
        already_recommended: bool = False,
        previous_models: Optional[List[str]] = None,
        multi_screen_brief: str = "",
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

        # ── Step 1/2：结构化产品查询（计划 update_v2.9.10 P2-⑨⑩⑪）───────────
        # 原来的 fast / normal / agent 三层业务路由已删除：这里只有一个判断 ——
        # "客户是不是在问结构化的产品事实（型号 / 点间距 / 亮度 / 防护等级）"。
        # 是 → 走结构化产品查询（Product Query → Structured Filter → Product Result）；
        # 否 → 直接进下面的统一 Agent 链路（RequirementProfile → Gate → RAG…）。
        requirements = routing_requirements(requirements, profile)
        from src.rag.router import has_structured_requirement
        from src.rag.structured_product_query import (
            looks_like_structured_product_query,
            structured_product_query_handle,
        )

        if (
            looks_like_structured_product_query(message)
            # 会话里已经采集到场景级需求时不走这条 —— 那时要复用上下文做确定性选型
            # （原 Router 的同款保护：实测"教堂+室内+5m"后回一句价格偏好会被
            # 判成参数查询，绕过"环境+视距→点间距"规则表）
            and not has_structured_requirement(requirements)
            # LCD / IFP 会话**永远**不走这条：结构化查询是按 LED 字段找型号的，
            # 走进去就会拿 LED 的候选去回答 LCD 客户（客户口径 2026-09-30：两条链路完全隔离）。
            and not _is_lcd_like_profile(profile)
        ):
            from src.config import config as _cfg
            from src.rag.query_understanding import extract_slots

            constraints = merge_fast_path_constraints(
                dict(extract_slots(message) or {}), requirements
            )
            product_result = structured_product_query_handle(
                query=message,
                constraints=constraints,
                template_type=None,
                data_dir=_cfg.DATA_DIR,
            )
            logger.info(
                "Structured product query: constraints=%s products=%d",
                constraints or "{}", len(product_result.get("products") or []),
            )
            return {
                "answer": product_result["answer"],
                "requirement": constraints,
                "reflection_score": 0,
                "reflection_notes": "structured product query",
                "products": product_result.get("products", []),
                "route": "product_query",
                "complexity": "parameter",
            }

        # ── Step 2b: Agent Path ────────────────────────────────────────
        initial_state = self._build_initial_state(
            message, history, requirements, additional_requirements, profile, session_id, intent,
            already_recommended=already_recommended, previous_models=previous_models,
            multi_screen_brief=multi_screen_brief,
        )

        # 把"客户这句话里的结构化槽位"注入 agent state（避免 LLM 重复推理）。
        # 计划 update_v2.9.10 P2：这里原来是 `routing.inferred_constraints`
        # （三层路由的产物），现在直接取统一抽取器的规则槽位，不依赖路由。
        from ...rag.query_understanding import extract_slots as _extract_slots

        inferred_constraints = dict(_extract_slots(message) or {})
        if inferred_constraints and not initial_state.get("requirement"):
            for k, v in inferred_constraints.items():
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
            #
            # 客户口径（整改计划 §八"IFP 判断必须统一"）：
            # **只能由 LCD Decision Center 决定 IFP**。以前这里还有第二道关键词判定
            # （has_ifp_intent），它会让"已经判成 IFP 的需求链"和"关键词判定"两个系统
            # 同时做 IFP 决策 —— 客户把 "meeting" 打成 "metting" 时关键词判 False，
            # 系统自己按需求选出来的交互平板就被连型号一起删掉。
            #
            # 现在只认决策中心：档案里明确要手写/触控（``is_ifp_requirement``）→ 放行；
            # 否则才清掉 IFP 型号/话术（防止模型凭空推荐交互平板）。
            customer_text = user_messages_text(initial_state.get("messages", []))
            response_products = result.get("products", [])
            if not _profile_needs_ifp(result, initial_state):
                answer = remove_unsupported_ifp_text(answer)
                response_products = [p for p in response_products if not is_ifp_product(p)]

            # Remove internal implementation wording
            final_requirement = result.get("requirement", initial_state.get("requirement", {})) or {}
            from ...rag.rerank import sanitize_customer_response
            family = _profile_family(result, initial_state)
            answer = sanitize_customer_response(
                answer,
                outdoor=bool(final_requirement.get("outdoor")),
                product_family=family,
            )

            if not answer:
                # 【客户口径 2026-10】"能不能放宽某个条件"这句兜底**只属于"确实匹配不到"**。
                # 客户原话：「推荐完后，还是会出现这句话，帮我彻底解决」。
                # 型号都已经选出来了还说"放宽条件我就能匹配"，等于告诉客户没匹配上。
                from ...rag.reply_composer import (
                    quote_confirmation_answer,
                    relaxation_answer,
                )

                if response_products:
                    logger.warning(
                        "Answer emptied while %d products were selected → proceed reply, "
                        "never the relaxation line",
                        len(response_products),
                    )
                    answer = quote_confirmation_answer()
                else:
                    # 真的一个型号都没匹配到，才允许邀请客户放宽条件；
                    # 口径跟链路走（LCD / IFP 不说点间距 / 观看距离）。
                    answer = relaxation_answer(product_family=family)

            # Keep compact format
            answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())

            # 客户口径（2026-09-28）：推荐/交付要完整，不做"看起来简短"的截断。
            # 这里只保留一个防跑飞的上限，并且统一在句子边界收尾。
            answer = cap_answer_length(answer)

            return {
                "answer": answer,
                "requirement": result.get("requirement", {}),
                "reflection_score": result.get("reflection_score", 0),
                "reflection_notes": result.get("reflection_notes", ""),
                "products": response_products[:1],
                "route": "agent",
                "complexity": "agent",
            }
        except Exception as e:
            logger.error(f"Agent run error: %s", e)
            return {
                # 对外话术一律英文（客户口径）
                "answer": "Sorry, something went wrong on my side. Could you try that again?",
                "requirement": {},
                "reflection_score": 0,
                "reflection_notes": str(e),
                "products": [],
                "route": "agent",
                "complexity": "agent",
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
