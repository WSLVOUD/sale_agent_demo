"""router node - invoke the Solution Agent and collect results."""
import logging
from ..state import SalesState

logger = logging.getLogger(__name__)


def router(state: SalesState) -> SalesState:
    """When `should_generate_solution` is True, call the Solution Agent 
    and store its output into `state["solutions"]`.
    """
    if not state.get("should_generate_solution"):
        logger.info("Skipping solution generation (not triggered)")
        return state
    
    solution_runner = state.get("solution_runner")
    if not solution_runner:
        logger.warning("Solution runner not available")
        state["solutions"] = []
        return state
    
    # Build requirement query from extracted requirements
    req = state["requirements"]
    query_parts = []
    if req.get("location_type"):
        query_parts.append(req["location_type"])
    if req.get("usage"):
        query_parts.append(req["usage"])
    if req.get("display_type"):
        query_parts.append(req["display_type"])
    if req.get("size"):
        query_parts.append(f"尺寸{req['size']}")
    if req.get("brightness"):
        query_parts.append(f"亮度{req['brightness']}")
    if req.get("resolution"):
        query_parts.append(f"分辨率{req['resolution']}")

    # Include additional requirements for specialized analysis
    additional_reqs = state.get("additional_requirements", [])
    if additional_reqs:
        query_parts.append(f"额外需求：{', '.join(additional_reqs)}")

    query = " ".join(query_parts) if query_parts else "LED 或 LCD 显示产品推荐"
    customer_message = str(state.get("current_message", "")).strip() or query

    logger.info("Calling Solution Agent with customer message: %s", customer_message)

    try:
        # Convert messages to proper format
        history = []
        for msg in state.get("messages", []):
            if isinstance(msg, dict):
                history.append(msg)
            else:
                history.append({
                    "role": getattr(msg, "type", "user"),
                    "content": getattr(msg, "content", str(msg))
                })

        # Call the Solution Agent
        result = solution_runner.run(
            message=customer_message,
            history=history,
            requirements=req,
            additional_requirements=additional_reqs,
            # 【M7】把 Sales 的 RequirementProfile 直接交给 Solution，
            # 让它消费同一份需求，而不是自己再从对话重建一遍
            profile=state.get("requirement_profile"),
            # 客户是否已经拿到过推荐 + 已经给过哪些型号：
            # "另外推荐一款"要换一个没给过的型号，而不是重新问需求
            already_recommended=bool(state.get("already_recommended")),
            previous_models=list(state.get("previous_recommended_models") or []),
        )

        # Extract products from result
        products = result.get("products", [])
        recommendation = result.get("answer", "")

        state["solutions"] = products
        if recommendation:
            state["response"] = recommendation
        else:
            # 对外话术一律英文（客户口径：回复里不能出现任何一句中文）
            state["response"] = (
                "Based on what you described, I've shortlisted some suitable screens. "
                "Share a bit more detail, such as the target size or budget, and I'll narrow it down "
                "to the best model for you."
            )

        # Signal the orchestrator to proceed to Solution Agent
        state["next_action"] = "trigger_solution"

        logger.info(f"Solution Agent returned {len(products)} products, response length={len(state['response'])}")
    except TimeoutError:
        logger.error("Solution Agent timed out")
        state["solutions"] = []
        state["response"] = (
            "The search took too long. Could you tell me your requirements again? "
            "I'll match the right screen for you."
        )
    except Exception as e:
        logger.error(f"Error calling Solution Agent: {e}")
        state["solutions"] = []
        state["response"] = (
            "Something went wrong while matching the models. Could you repeat your requirements? "
            "I'll look again for you."
        )

    return state
