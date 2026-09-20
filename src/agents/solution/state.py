"""Solution Agent state definition."""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add_messages


class SolutionState(TypedDict):
    """State definition for the Solution Agent (LED Product RAG)."""

    # 会话 ID（语义缓存按会话隔离；缺失时不复用任何跨会话结果）
    session_id: str

    # 本会话是否已经给过推荐（"另外推荐一款" → 换一个没给过的型号）
    already_recommended: bool
    # 已经推荐过的型号
    previous_recommended_models: Optional[List[str]]
    
    # Conversation history (auto-managed by add_messages)
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # Extracted user requirements
    requirement: Dict[str, Any]

    # Whether user just greeted (LLM-detected)
    is_greeting: bool

    # Friendly reply when user greeted
    greeting_reply: str

    # Conflict detection (user requirement may not match their environment)
    has_conflict: bool
    conflict_reason: str
    clarification_question: str
    conflict_message: str
    awaiting_confirmation: bool

    # Whether we have enough info to recommend
    info_sufficient: bool
    
    # Missing info to ask
    missing_info: List[str]

    # The requirement currently being requested from the customer
    pending_question: str
    
    # Determined display type: "LED", "LCD", or "BOTH"
    display_type: str
    
    # Retrieved products from hybrid search
    products: List[Dict[str, Any]]
    
    # Generated recommendation
    recommendation: str
    
    # Reflection evaluation
    reflection_score: float
    reflection_notes: str
    
    # Whether recommendation needs refinement
    needs_refine: bool

    # Reflection tracking
    reflection_count: int  # Number of reflections done
    best_score: float  # Best score so far

    # Next action to take
    next_action: str  # "ask" / "recommend" / "product_question" / "end" / "classify"

    # Intent of the current user turn
    intent: str  # "recommendation" / "product_question" / "conversation"

    # The exact message classified at the start of this turn
    current_message: str

    # Retrieved products shown in the previous turn
    last_products: List[Dict[str, Any]]

    # Hybrid search instance
    hybrid_search: Optional[Any]

    # External industry specifications scraped from the web, used to enrich the
    # vector query and to validate candidates in rerank.
    search_keywords: List[str]

    # Additional/custom requirements not in the predefined list
    # These will be analyzed for pixel pitch, brightness, rental considerations
    additional_requirements: List[str]

    # ── Inferred technical parameters (added before retrieval) ──────────────
    # These are produced by the parameter_inference node from the customer's
    # natural-language requirement, so the retriever and reranker can apply
    # numeric filters (e.g. outdoor needs ≥4500nit, near-viewing needs ≤P2).
    inferred_brightness_min_nit: Optional[int]      # e.g. 4500 for outdoor
    inferred_brightness_max_nit: Optional[int]      # e.g. 800  for indoor
    inferred_pixel_pitch_min_mm: Optional[float]     # e.g. 1.2  for ≤3m viewing
    inferred_pixel_pitch_max_mm: Optional[float]     # e.g. 10.0 for ≥30m viewing
    inferred_screen_size: Optional[str]              # e.g. "65-75英寸" for 3m viewing
    inferred_is_rental: Optional[bool]             # True/False/None(don't care)

    # ── Phase 3：硬约束过滤（Hard Metadata Filter）──────────────────────────
    # 明确硬条件在检索前过滤；推断出来的点间距/亮度作为软条件参与排序与校验
    hard_constraints: Optional[Dict[str, Any]]
    hard_filter_dropped: int

    # ── Phase 8：软条件（用于推荐打分的期望区间）────────────────────────────
    soft_pitch_min_mm: Optional[float]
    soft_pitch_max_mm: Optional[float]
    soft_brightness_min_nit: Optional[int]

    # ── Phase 4：Query Understanding（结构化槽位 + 标准化检索式）──────────────
    understood_slots: Optional[Dict[str, Any]]
    understood_language: Optional[str]
    retrieval_query: Optional[str]

    # ── Phase 5：统一工程推断结果（Python 规则产出，LLM 不得修改）────────────
    technical_parameters: Optional[Dict[str, Any]]

    # ── Phase 6：结构化需求档案（RequirementProfile）─────────────────────────
    requirement_profile: Optional[Any]

    # ── Phase 8/9/10：确定性选型结果 + 工程计算结果 ─────────────────────────
    recommendation_result: Optional[Dict[str, Any]]
    screen_calculation: Optional[Dict[str, Any]]

    # ── Phase 11：确定性校验报告 ────────────────────────────────────────────
    validation_report: Optional[Dict[str, Any]]

    # ── v2.0 Phase 4/9：两个 Gate 的判定结果 ────────────────────────────────
    recommendation_gate: Optional[Dict[str, Any]]
    calculation_gate: Optional[Dict[str, Any]]
