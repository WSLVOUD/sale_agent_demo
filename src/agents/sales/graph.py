"""graph - compile the Sales Agent graph."""
from langgraph.graph import StateGraph, END

from .state import SalesState
from .nodes.classify import classify
from .nodes.requirement import requirement_mining
from .nodes.router import router
from .nodes.script_generator import script_generator


def should_route_to_solution(state: SalesState) -> str:
    """Conditional edge after classify: skip requirement_mining for product_question/others."""
    intent = state.get("intent")
    if state.get("should_generate_solution"):
        return "router"
    if intent in ("product_question", "others"):
        return "script_generator"
    return "script_generator"


def should_continue(state: SalesState) -> str:
    """Conditional edge: decide if conversation ends."""
    next_action = state.get("next_action", "ask")
    if next_action == "end":
        return END
    return "script_generator"


def build_sales_graph():
    """Compile and return the Sales Agent graph.

    Flow:
        START -> classify -> requirement_mining 
              -> (router if triggered, else script_generator)
              -> END
    """
    graph = StateGraph(SalesState)
    
    # Add nodes
    graph.add_node("classify", classify)
    graph.add_node("requirement_mining", requirement_mining)
    graph.add_node("router", router)
    graph.add_node("script_generator", script_generator)
    
    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "requirement_mining")
    
    # Conditional routing after requirement_mining
    graph.add_conditional_edges(
        "requirement_mining",
        should_route_to_solution,
        {
            "router": "router",
            "script_generator": "script_generator"
        }
    )
    
    # After router, go to script_generator
    graph.add_edge("router", "script_generator")
    
    # Script generator ends
    graph.add_edge("script_generator", END)
    
    return graph.compile()
