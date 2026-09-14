"""Solution Agent graph definition."""
from langgraph.graph import StateGraph, END

from .state import SolutionState
from .nodes.intent import intent_node, conversation_node, route_by_intent
from .nodes.recommend import recommend_node
from .nodes.reflection import reflection_node
from .nodes.others import others_node
from .nodes import requirement as req_node
from .nodes import retrieval as ret_node


def _route_after_reflect(state: SolutionState) -> str:
    """Route after reflection node."""
    # If waiting for clarification, stop here and ask the question
    if state.get("waiting_for_clarification"):
        return "clarify"
    # If info is not sufficient, go to clarify to ask for missing requirements
    if not state.get("info_sufficient", True):
        return "clarify"
    if state.get("needs_refine"):
        return "recommend"
    return "END"


def _route_after_recommendation_gate(state: SolutionState) -> str:
    """v2.0 Phase 4：Gate 决定"继续推荐"还是"先追问"。"""
    if (state.get("recommendation_gate") or {}).get("ready") is False:
        return "clarify"
    return "retrieve"


def build_solution_graph():
    """Compile and return the Solution Agent graph.

    Flow:
        START -> intent_recognition -> understand -> infer_parameters -> retrieve
              -> recommend -> reflect -> (recommend if needs_refine)
              -> END
              
        Intent routes:
        - recommendation -> understand -> ...
        - product_question -> product_question_node -> END
        - conversation -> conversation_node -> END
        - others -> others_node -> END
    """
    graph = StateGraph(SolutionState)
    
    # Core nodes
    graph.add_node("intent_recognition", intent_node)
    graph.add_node("understand", req_node.understand_node)
    graph.add_node("infer_parameters", req_node.infer_parameters_node)
    # 注意：节点名不能与 state key 同名（LangGraph 会报
    # "'recommendation_gate' is already being used as a state key"），
    # 因此节点用 *_check 后缀，state 里保存判定结果的 key 仍是 recommendation_gate。
    graph.add_node("recommendation_gate_check", req_node.recommendation_gate_node)
    graph.add_node("retrieve", ret_node.retrieval_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("reflect", reflection_node)
    graph.add_node("clarify", req_node.clarify_node)
    
    # Branch nodes
    graph.add_node("conversation", conversation_node)
    graph.add_node("product_question", req_node.product_question_node)
    graph.add_node("others", others_node)
    
    # Entry point
    graph.set_entry_point("intent_recognition")
    
    # Intent routing
    graph.add_conditional_edges(
        "intent_recognition",
        route_by_intent,
        {
            "recommendation": "understand",
            "product_question": "product_question",
            "conversation": "conversation",
            "others": "others",
            "requirement": "understand",
        }
    )
    
    # Recommendation flow: understand -> infer_parameters -> recommendation_gate
    #                      -> (retrieve | clarify) -> recommend -> reflect -> END
    graph.add_edge("understand", "infer_parameters")
    graph.add_edge("infer_parameters", "recommendation_gate_check")
    graph.add_conditional_edges(
        "recommendation_gate_check",
        _route_after_recommendation_gate,
        {
            "retrieve": "retrieve",
            "clarify": "clarify",
        },
    )
    graph.add_edge("retrieve", "recommend")
    graph.add_edge("recommend", "reflect")
    
    # Reflect routing
    graph.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {
            "clarify": "clarify",
            "recommend": "recommend",
            "END": END,
        }
    )
    
    # Clarify node ends the current turn, waiting for user input
    graph.add_edge("clarify", END)
    
    # Branch nodes end
    graph.add_edge("conversation", END)
    graph.add_edge("product_question", END)
    graph.add_edge("others", END)
    
    return graph.compile()
