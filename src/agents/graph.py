"""
LangGraph Multi-Agent Workflow Assembly for LearnTube AI.

Builds and compiles the stateful cyclic multi-agent graph:
START -> Router Node -> Retriever Node -> Tutor Node -> Evaluator Node -> (Conditional Edge: END or Loop back)
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.nodes import (
    router_node,
    retriever_node,
    tutor_node,
    evaluator_node
)


def should_revisit_query(state: AgentState) -> str:
    """
    Conditional edge decision function:
    - If Evaluator detected hallucinations AND retries < 1: loops back to 'router'
    - Otherwise: terminates graph execution at END
    """
    grade = state.get("grounding_grade", "GROUNDED")
    retry_count = state.get("retry_count", 0)

    # Only allow 1 retry if genuinely hallucinated, otherwise terminate immediately
    if grade == "HALLUCINATED" and retry_count <= 1:
        return "router"
    
    return END


def create_agent_workflow():
    """
    Constructs and compiles the LangGraph StateGraph.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Agent Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("evaluator", evaluator_node)

    # 2. Define Execution Edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "tutor")
    workflow.add_edge("tutor", "evaluator")

    # 3. Add Conditional Guardrail Routing
    workflow.add_conditional_edges(
        "evaluator",
        should_revisit_query,
        {
            "router": "router",
            END: END
        }
    )

    return workflow.compile()


# Singleton compiled graph instance
agent_graph = create_agent_workflow()


def run_agent_chat(
    question: str,
    video_id: str,
    chat_history: List[Dict[str, Any]] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    High-level entry point to execute the Multi-Agent LangGraph workflow.
    
    Returns:
    {
        "answer": str,
        "sources": list of dicts,
        "thought_steps": list of AgentThought dicts,
        "grounding_grade": str,
        "intent": str
    }
    """
    initial_state: AgentState = {
        "question": question,
        "chat_history": chat_history or [],
        "video_id": video_id,
        "metadata": metadata or {},
        "reformulated_query": None,
        "intent": None,
        "retrieved_chunks": [],
        "retrieval_confidence": 0.0,
        "retry_count": 0,
        "draft_answer": None,
        "grounding_grade": None,
        "evaluation_feedback": None,
        "final_answer": None,
        "sources": [],
        "thought_steps": []
    }

    final_state = agent_graph.invoke(initial_state, config={"recursion_limit": 6})

    return {
        "answer": final_state.get("final_answer") or final_state.get("draft_answer") or "I couldn't generate an answer from the video context.",
        "sources": final_state.get("sources", []),
        "thought_steps": final_state.get("thought_steps", []),
        "grounding_grade": final_state.get("grounding_grade", "GROUNDED"),
        "intent": final_state.get("intent", "concept")
    }

