"""SalesAgentRunner - entry point for the Sales Agent."""
import logging
import re
from typing import Dict, Any, Optional

from .state import SalesState
from .graph import build_sales_graph
from ...rag.reply_composer import reply_language
from ...rag.session_switch import detect_requirement_reset, reset_acknowledgement

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

    def run(self, session_id: str, message: str, has_vision: bool = False) -> dict:
        """Process a single user turn and return the response payload.
        
        Args:
            session_id: Unique session identifier
            message: User's current message
            has_vision: 本轮客户是否带了图片（图片识别结果要跟客户确认一次）
            
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

        # ── 会话内需求重置：换产品 / 换项目 / 改需求 ─────────────────────────
        # 客户在拿到推荐之后要换产品时，前面采集的需求必须清空，
        # 否则 Gate 会拿旧需求再次推荐（既没有重新采集，也推荐得不对）。
        # 纯规则检测，不调用 LLM，避免"AI 自己推测客户想换什么"。
        display_type_change = None
        current_display_type = _extract_display_type_from_message(message)
        if current_display_type and self.memory_store and hasattr(self.memory_store, "get_previous_display_type"):
            previous_display_type = self.memory_store.get_previous_display_type(session_id)
            if previous_display_type and previous_display_type != current_display_type:
                display_type_change = (previous_display_type, current_display_type)

        existing_profile = None
        recommended_before = False
        if self.memory_store:
            if hasattr(self.memory_store, "get_requirement_profile"):
                profile_data = self.memory_store.get_requirement_profile(session_id)
                # memory 返回的是 dict，需要转换成 RequirementProfile 对象
                if profile_data:
                    from ...models.requirement import RequirementProfile
                    if isinstance(profile_data, dict):
                        try:
                            existing_profile = RequirementProfile.model_validate(profile_data)
                        except Exception as e:
                            logger.warning("[%s] Failed to load profile from memory: %s", session_id, e)
                            existing_profile = None
                    else:
                        existing_profile = profile_data
            if hasattr(self.memory_store, "has_recommendation"):
                recommended_before = self.memory_store.has_recommendation(session_id)

        reset = detect_requirement_reset(
            message,
            requirements=accumulated_requirements,
            profile=existing_profile,
            recommended=recommended_before,
            display_type_change=display_type_change,
        )
        if reset.should_reset:
            logger.info(
                "[%s] Requirement reset (%s, evidence=%r): %s",
                session_id, reset.reason, reset.evidence, reset.detail,
            )
            accumulated_requirements = {}
            existing_profile = None
            # 这是一轮全新的咨询：之前的"已推荐"标记不能带过来
            recommended_before = False
            if self.memory_store and hasattr(self.memory_store, "reset_requirement_state"):
                self.memory_store.reset_requirement_state(session_id)
            # 语义缓存也要一起清：重置后是全新需求，不能再用旧上下文理解同一条消息
            try:
                from ...core.requirement_extractor import get_requirement_extractor

                get_requirement_extractor().clear_session_semantics(session_id)
            except Exception as error:  # pragma: no cover - 防御式
                logger.warning("[%s] Clear semantic cache failed: %s", session_id, error)

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
            "session_id": session_id,
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
            "requirement_profile": existing_profile,
            "pending_question": "",
            "pending_slot": "",
            "requirements_reset": bool(reset.should_reset),
            "reset_reason": reset.reason,
            # 本会话是否已经给过推荐：决定"客户后续提问时要不要再推荐一遍"
            "already_recommended": bool(recommended_before),
            # 本轮是否带图片：带图的这一轮要把"图片里看到什么"跟客户核一遍
            "vision_applied": bool(has_vision),
        }
        
        # Invoke the graph
        logger.info(f"[{session_id}] Processing: {message[:50]}...")
        result = self.graph.invoke(state)

        # 需求重置：给客户一句口语确认，再问重新采集的第一个问题
        # （先拼接再持久化，保证 memory 里的历史与客户实际看到的一致）
        if reset.should_reset:
            ack = reset_acknowledgement(
                language=reply_language(message),
                seed=len(history) + (sum(ord(ch) for ch in str(session_id)) % 5),
            )
            base_response = result.get("response", "")
            result["response"] = f"{ack} {base_response}".strip() if base_response else ack

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

                # "本会话已经给过推荐"的标记同样要在 clear() 前后保留：
                # 否则客户推荐后随便问一句（本轮不出产品），标记就被清掉，
                # 下一轮又会被当成"还没推荐过"→ 再推荐一遍（实测出现过）。
                preserved_recommendation = {}
                if hasattr(self.memory_store, "get_recommendation"):
                    preserved_recommendation = self.memory_store.get_recommendation(session_id) or {}

                # 【修复】"一个项目下多条屏"的状态同样必须跨轮保留：
                # clear() 之后如果丢掉 project_items / active_item_index，
                # 多屏流程每轮都会回到"第 1 块屏"，第二块屏永远开不出来。
                preserved_project_items = (
                    self.memory_store.get_project_items(session_id)
                    if hasattr(self.memory_store, "get_project_items")
                    else []
                )
                preserved_active_item = (
                    self.memory_store.get_active_item_index(session_id)
                    if hasattr(self.memory_store, "get_active_item_index")
                    else 0
                )

                self.memory_store.clear(session_id)
                self.memory_store.extend(session_id, final_messages)
                self.memory_store.set_requirements(session_id, result.get("requirements", {}))
                # Phase 6：结构化需求档案随会话持久化（供下一轮 / 方案 Agent 复用）
                if hasattr(self.memory_store, "set_requirement_profile"):
                    self.memory_store.set_requirement_profile(
                        session_id, result.get("requirement_profile")
                    )

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

                # 恢复"一个项目下多条屏"的状态（见上面 preserved_project_items）
                if preserved_project_items and hasattr(
                    self.memory_store, "set_project_items"
                ):
                    self.memory_store.set_project_items(session_id, preserved_project_items)
                if preserved_active_item and hasattr(
                    self.memory_store, "set_active_item_index"
                ):
                    self.memory_store.set_active_item_index(session_id, preserved_active_item)

                # 记录"本轮已经给过推荐"：下一轮客户说"想换个产品"时，
                # 系统据此判断需要在同一会话里清空旧需求、重新采集。
                # 注意：只有真的给出产品才算（空触发的 trigger_solution 不算，
                # 否则会在客户还没看到任何产品时就被标记成"已推荐"）。
                products = result.get("solutions") or []
                if products and hasattr(
                    self.memory_store, "mark_recommendation_done"
                ):
                    self.memory_store.mark_recommendation_done(session_id, products)
                elif (
                    preserved_recommendation.get("delivered")
                    and hasattr(self.memory_store, "mark_recommendation_done")
                ):
                    # 本轮不是推荐轮 → 把之前的推荐记录放回去（型号列表一并保留）
                    self.memory_store.mark_recommendation_done(
                        session_id,
                        [{"model": model} for model in (preserved_recommendation.get("models") or [])],
                    )
            else:
                self.memory_store[session_id] = {
                    "messages": final_messages,
                    "requirements": result.get("requirements", {}),
                }

        response_text = result.get("response", "")

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
            # 需求采集追问：orchestrator 会把它接在"回答客户问题"的后面
            "pending_question": result.get("pending_question", ""),
            "pending_slot": result.get("pending_slot", ""),
            "requirements_reset": bool(reset.should_reset),
            "acknowledgement": result.get("acknowledgement", ""),
        }
