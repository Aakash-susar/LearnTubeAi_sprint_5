"""
Agent State Schema for LearnTube AI LangGraph Workflow.

Defines the shared data structure (AgentState) passed between all agent nodes:
- Router Agent
- Video RAG Specialist Agent
- Socratic Pedagogical Tutor Agent
- Hallucination / Grounding Evaluator Node
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator


class AgentThought(TypedDict):
    """Represents a single thought or action taken by an agent for UI transparency."""
    agent_name: str
    action: str
    detail: str
    timestamp: Optional[float]


class AgentState(TypedDict):
    """
    The shared state dictionary passed across all nodes in the LangGraph graph.
    """
    # 1. User Input & Conversation Context
    question: str                                    # Original user question
    chat_history: List[Dict[str, Any]]              # Previous conversation messages
    video_id: str                                    # YouTube Video ID being studied
    metadata: Optional[Dict[str, Any]]              # Video title, channel, duration

    # 2. Agent Reasoning & Query Transformation
    reformulated_query: Optional[str]               # Query rewritten by Router Agent for semantic vector search
    intent: Optional[str]                            # Intent classification: "concept", "timestamp", "comparison", "summary"

    # 3. Retrieval Artifacts
    retrieved_chunks: List[Dict[str, Any]]          # Transcript chunks with timestamp intervals
    retrieval_confidence: float                     # Quality score of retrieved chunks (0.0 to 1.0)
    retry_count: int                                # Number of query reformulation attempts

    # 4. Pedagogical Generation & Verification
    draft_answer: Optional[str]                     # Initial explanation crafted by Tutor Agent
    grounding_grade: Optional[str]                  # Evaluator decision: "GROUNDED", "HALLUCINATED", "INSUFFICIENT_CONTEXT"
    evaluation_feedback: Optional[str]              # Explanation if answer needs correction
    
    # 5. Final Output & Step-by-Step Transparency
    final_answer: Optional[str]                     # Verified, student-facing response with formatted timestamps
    sources: List[Dict[str, Any]]                   # Clickable timestamp sources for video jumping
    thought_steps: Annotated[List[AgentThought], operator.add]  # Append-only list of agent thoughts for UI display
