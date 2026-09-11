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
            additional_requirements=additional_reqs
        )

        # Extract products from result
        products = result.get("products", [])
        recommendation = result.get("answer", "")

        state["solutions"] = products
        if recommendation:
            state["response"] = recommendation
        else:
            state["response"] = "根据您的需求，我找到了几款合适的屏幕。您可以告诉我更具体的尺寸或预算，我来帮您筛选最合适的型号。"

        # Signal the orchestrator to proceed to Solution Agent
        state["next_action"] = "trigger_solution"

        logger.info(f"Solution Agent returned {len(products)} products, response length={len(state['response'])}")
    except TimeoutError:
        logger.error("Solution Agent timed out")
        state["solutions"] = []
        state["response"] = "检索超时了，您可以再说一次您的需求，我重新帮您匹配。"
    except Exception as e:
        logger.error(f"Error calling Solution Agent: {e}")
        state["solutions"] = []
        state["response"] = "匹配过程出了点问题，您可以再说一次您的需求，我重新帮您找。"

    return state
