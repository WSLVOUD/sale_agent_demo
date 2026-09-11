"""Sales Agent state definition."""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import add_messages


class SalesState(TypedDict):
    """State for the Sales Agent.

    The Sales Agent runs alongside the existing Solution Agent and shares
    the same `session_id` / `memory_store` so that both agents see the
    same conversation history.
    """

    # Conversation
    messages: Annotated[List[Dict], add_messages]
    session_id: str
    current_message: str  # The user's current message being processed

    # Intent classification: "greeting" | "need_query" | "objection" | "closing" | "industry"
    intent: str

    # Requirements extracted from the user so far
    requirements: Dict[str, Any]
    # {
    #   "location_type": "indoor" | "outdoor" | None,
    #   "usage": str | None,
    #   "size": str | None,
    #   "brightness": str | None,
    #   "resolution": str | None,
    # }

    # Mining progress
    required_met: List[str]        # 已确认的关键项
    required_missing: List[str]   # 还差的关键项（按优先级排序）
    turn_count: int

    # Additional/custom requirements not in the predefined list
    # These will be passed to AI for analysis (pixel pitch, brightness, rental, etc.)
    additional_requirements: List[str]

    # Trigger flag
    should_generate_solution: bool

    # Sales script RAG
    retrieved_scripts: List[Dict[str, Any]]

    # Outputs
    response: str
    solutions: Optional[List[Dict[str, Any]]]   # 由 Solution Agent 填充

    # Routing: "ask" | "answer" | "trigger_solution" | "end"
    next_action: str

    # Shared references (set by the runner)
    sales_search: Optional[Any]
    solution_runner: Optional[Any]

    # 首次接待刚完成后抑制销售问候语（因为已经自我介绍过了）
    suppress_greeting: bool
