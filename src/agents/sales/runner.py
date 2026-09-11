"""SalesAgentRunner - entry point for the Sales Agent."""
import logging
import re
from typing import Dict, Any, Optional

from .state import SalesState
from .graph import build_sales_graph

logger = logging.getLogger(__name__)


def _normalize_role(role: str) -> str:
    """Map LangChain message types to the canonical dict role names."""
    mapping = {"human": "user", "ai": "assistant", "system": "system"}
    return mapping.get(role, role)


def _to_dict(msg) -> dict:
    """Normalize any message format (LangChain message or dict) to a plain dict."""
    if isinstance(msg, dict):
        return {"role": _normalize_role(msg.get("role", "")), "content": msg.get("content", "")}
    return {
        "role": _normalize_role(getattr(msg, "type", "unknown")),
        "content": getattr(msg, "content", str(msg)),
    }


def _dedupe_messages(messages):
    """Remove consecutive duplicate (role, content) entries."""
    deduped = []
    for msg in messages or []:
        item = _to_dict(msg)
        if deduped and deduped[-1].get("role") == item.get("role") and deduped[-1].get("content") == item.get("content"):
            continue
        deduped.append(item)
    return deduped


# Screen type keywords for detection
DISPLAY_TYPE_PATTERNS = {
    "LED": re.compile(r"(?<![a-zA-Z])LED(?![a-zA-Z])", re.IGNORECASE),
    "LCD": re.compile(r"(?<![a-zA-Z])LCD(?![a-zA-Z])", re.IGNORECASE),
    "IFP": re.compile(
        r"(?<![a-zA-Z])(IFP|会议一体机|触摸一体机|交互平板|交互式平板|电子白板)(?![a-zA-Z])",
        re.IGNORECASE,
    ),
}


def _extract_display_type_from_message(message: str) -> Optional[str]:
    """Extract explicit display type from user message.

    Returns "LED", "LCD", "IFP", or None if no explicit type mentioned.
    """
    if not message:
        return None
    text = message.lower()

    for dtype in ("IFP", "LED", "LCD"):
        if DISPLAY_TYPE_PATTERNS[dtype].search(text):
            return dtype
    return None


class SalesAgentRunner:
    """Runs the Sales Agent for a given session.

    The runner is responsible for:
      - Loading shared dependencies (sales_search, solution_runner).
      - Initializing per-session state from `memory_store`.
      - Invoking the compiled graph.
      - Persisting the response back into `memory_store`.
    """

    def __init__(self, sales_search=None, solution_runner=None, memory_store=None):
        self.sales_search = sales_search
        self.solution_runner = solution_runner
        self.memory_store = memory_store
        self.graph = None

    def setup(self):
        """Build the graph. Call once at startup."""
        self.graph = build_sales_graph()
        logger.info("Sales Agent graph compiled")

    def run(self, session_id: str, message: str) -> dict:
        """Process a single user turn and return the response payload.
        
        Args:
            session_id: Unique session identifier
            message: User's current message
            
        Returns:
            Dict with keys: response, products, requirements, intent, next_action
        """
        if not self.graph:
            raise RuntimeError("Graph not built. Call setup() first.")

        # Load conversation history and requirements from memory store
        history = []
        accumulated_requirements = {}
        if self.memory_store:
            if hasattr(self.memory_store, "get_history"):
                history = self.memory_store.get_history(session_id)
                accumulated_requirements = self.memory_store.get_requirements(session_id)
            else:
                session_data = self.memory_store.get(session_id, {})
                if isinstance(session_data, dict):
                    raw_messages = session_data.get("messages", [])
                    accumulated_requirements = session_data.get("requirements", {})
                else:
                    raw_messages = session_data if isinstance(session_data, list) else []
                history = [_to_dict(m) for m in raw_messages]

            history = [_to_dict(m) for m in history]

        # Detect display type change
        requirements_cleared = False
        current_display_type = _extract_display_type_from_message(message)
        if current_display_type and self.memory_store and hasattr(self.memory_store, "get_previous_display_type"):
            previous_display_type = self.memory_store.get_previous_display_type(session_id)
            if previous_display_type and previous_display_type != current_display_type:
                logger.info(
                    "[%s] Display type changed: %s → %s, clearing accumulated requirements",
                    session_id, previous_display_type, current_display_type,
                )
                accumulated_requirements = {}
                requirements_cleared = True

        # Build initial state
        # 检查是否需要抑制销售问候语（首次接待刚完成后）
        suppress_greeting = False
        if self.memory_store and hasattr(self.memory_store, "should_suppress_sales_greeting"):
            suppress_greeting = self.memory_store.should_suppress_sales_greeting(session_id)
            if suppress_greeting:
                logger.info("[%s] Sales greeting suppressed (first contact just completed)", session_id)

        state: SalesState = {
            "messages": history + [{"role": "user", "content": message}],
            "current_message": message,
            "intent": "",
            "requirements": accumulated_requirements.copy(),
            "additional_requirements": [],
            "required_met": False,
            "required_missing": [],
            "should_generate_solution": False,
            "solutions": [],
            "response": "",
            "next_action": "ask",
            "solution_runner": self.solution_runner,
            "sales_search": self.sales_search,
            "turn_count": 0,
            "suppress_greeting": suppress_greeting,  # 首次接待刚完成后抑制问候语
        }
        
        # Invoke the graph
        logger.info(f"[{session_id}] Processing: {message[:50]}...")
        result = self.graph.invoke(state)

        # Persist updated conversation AND requirements
        if self.memory_store:
            final_messages = _dedupe_messages(result.get("messages", []))

            response_text = result.get("response", "")
            if (
                response_text
                and (
                    not final_messages
                    or final_messages[-1].get("role") != "assistant"
                    or final_messages[-1].get("content") != response_text
                )
            ):
                final_messages.append({"role": "assistant", "content": response_text})
                final_messages = _dedupe_messages(final_messages)

            if hasattr(self.memory_store, "extend"):
                # 保存关键状态字段，避免被 clear() 删除
                preserved_first_contact_sent = False
                preserved_suppress_greeting = False
                if hasattr(self.memory_store, "is_first_contact_done"):
                    preserved_first_contact_sent = self.memory_store.is_first_contact_done(session_id)
                if hasattr(self.memory_store, "should_suppress_sales_greeting"):
                    preserved_suppress_greeting = self.memory_store.should_suppress_sales_greeting(session_id)

                self.memory_store.clear(session_id)
                self.memory_store.extend(session_id, final_messages)
                self.memory_store.set_requirements(session_id, result.get("requirements", {}))

                # 恢复关键状态字段
                if preserved_first_contact_sent:
                    self.memory_store.mark_first_contact_done(session_id)
                    # 如果之前是抑制状态，保持抑制直到被消费
                    if preserved_suppress_greeting and not suppress_greeting:
                        # 只有当本次没有消费抑制标记时，才恢复抑制状态
                        if hasattr(self.memory_store, "_sessions") and session_id in self.memory_store._sessions:
                            self.memory_store._sessions[session_id]["suppress_sales_greeting"] = True

                if current_display_type and hasattr(self.memory_store, "set_previous_display_type"):
                    self.memory_store.set_previous_display_type(session_id, current_display_type)
            else:
                self.memory_store[session_id] = {
                    "messages": final_messages,
                    "requirements": result.get("requirements", {}),
                }

        # Build notification if requirements were cleared
        response_text = result.get("response", "")
        if requirements_cleared:
            type_change_notice = (
                f"\n\n※ 您已切换到{current_display_type}屏幕，之前收集的需求已清除，请重新告诉我您的 {current_display_type} 场景需求（如室内/室外、尺寸、用途等）。"
            )
            response_text = response_text + type_change_notice if response_text else type_change_notice

        # 消费抑制标记（如果已使用）
        if suppress_greeting and self.memory_store and hasattr(self.memory_store, "consume_suppress_sales_greeting"):
            self.memory_store.consume_suppress_sales_greeting(session_id)

        return {
            "response": response_text,
            "products": result.get("solutions", []),
            "requirements": result.get("requirements", {}),
            "additional_requirements": result.get("additional_requirements", []),
            "intent": result.get("intent", ""),
            "next_action": result.get("next_action", "ask"),
        }
