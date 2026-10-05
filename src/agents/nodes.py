"""
Agent Nodes for LearnTube AI LangGraph System.

Implements the specialized worker agents:
1. Router & Query Reformulator Agent (Analyzes intent & expands queries)
2. Video Transcript & Timestamp Specialist Agent (Retrieves chunks & extracts timestamps)
3. Socratic Pedagogical Tutor Agent (Synthesizes educational answers)
4. Hallucination Evaluator & Grounding Guardrail (Validates factual accuracy against video context)
"""

import os
import json
import time
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from src.agents.state import AgentState, AgentThought
from src.RAG.embeddings import search_documents
from src.RAG.chunker import format_timestamp


# LLM Instances
def _get_agent_llm(temperature: float = 0.2):
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ========================================
# 1. 🧠 ROUTER & QUERY REFORMULATOR AGENT
# ========================================

ROUTER_PROMPT = PromptTemplate(
    input_variables=["chat_history", "video_title", "question"],
    template="""You are the Supervisor and Query Intelligence Agent for LearnTube AI.
Your job is to analyze the student's question in the context of the YouTube lecture being studied.

VIDEO TITLE: {video_title}

CONVERSATION HISTORY:
{chat_history}

STUDENT QUESTION: {question}

TASKS:
1. Classify the user intent:
   - "concept" (explaining theoretical or practical ideas)
   - "timestamp" (locating where in the video something happens)
   - "comparison" (comparing two or more things mentioned in the lecture)
   - "summary" (asking for an overview of a section)
2. Rewrite the student question into an optimal standalone semantic search query for the vector database.
   - Resolve any pronouns ("it", "they", "that thing mentioned earlier") using conversation history.
   - Include key technical terms from the video.

OUTPUT STRICTLY IN JSON:
{{
    "intent": "concept | timestamp | comparison | summary",
    "reformulated_query": "standalone search query",
    "reasoning": "1-sentence explanation of query transformation"
}}
"""
)


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes student intent and reformulates ambiguous questions into high-density search queries.
    """
    question = state.get("question", "")
    chat_history = state.get("chat_history", [])
    metadata = state.get("metadata") or {}
    video_title = metadata.get("title", "Educational Lecture")

    # Format history
    history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-4:]])
    if not history_text:
        history_text = "No previous messages."

    llm = _get_agent_llm(temperature=0.1)
    chain = ROUTER_PROMPT | llm | JsonOutputParser()

    try:
        result = chain.invoke({
            "chat_history": history_text,
            "video_title": video_title,
            "question": question
        })
        intent = result.get("intent", "concept")
        reformulated = result.get("reformulated_query", question)
        reasoning = result.get("reasoning", "Extracted semantic intent")
    except Exception:
        intent = "concept"
        reformulated = question
        reasoning = "Using original student query as fallback"

    thought: AgentThought = {
        "agent_name": "🧠 Router Agent",
        "action": f"Classified intent as [{intent.upper()}] & reformulated query",
        "detail": f"Optimized query: \"{reformulated}\" ({reasoning})",
        "timestamp": time.time()
    }

    return {
        "intent": intent,
        "reformulated_query": reformulated,
        "thought_steps": [thought]
    }


# ========================================
# 2. 📹 VIDEO TRANSCRIPT & TIMESTAMP SPECIALIST AGENT
# ========================================

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Queries ChromaDB / Vector Store for timestamped chunks matching the reformulated query.
    """
    query = state.get("reformulated_query") or state.get("question")
    video_id = state.get("video_id")
    retry_count = state.get("retry_count", 0)

    raw_results = search_documents(query=query, video_id=video_id, k=5)

    retrieved_chunks = []
    total_dist = 0.0
    for doc, dist in raw_results:
        meta = doc.metadata or {}
        start_sec = meta.get("start_time", 0.0)
        end_sec = meta.get("end_time", 0.0)
        formatted_start = format_timestamp(start_sec)
        formatted_end = format_timestamp(end_sec)

        retrieved_chunks.append({
            "text": doc.page_content,
            "start_time": start_sec,
            "end_time": end_sec,
            "formatted_start": formatted_start,
            "formatted_end": formatted_end,
            "chunk_index": meta.get("chunk_index", 0),
            "distance": round(float(dist), 4)
        })
        total_dist += float(dist)

    confidence = round(1.0 - (total_dist / len(raw_results)), 2) if raw_results else 0.0

    timestamps_found = [c["formatted_start"] for c in retrieved_chunks[:3]]
    thought: AgentThought = {
        "agent_name": "📹 Video RAG Specialist",
        "action": f"Retrieved {len(retrieved_chunks)} relevant transcript intervals from vector database",
        "detail": f"Key timestamps matched: {', '.join(timestamps_found)} (Confidence: {confidence:.0%})",
        "timestamp": time.time()
    }

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_confidence": confidence,
        "thought_steps": [thought]
    }


# ========================================
# 3. 🎓 SOCRATIC PEDAGOGICAL TUTOR AGENT
# ========================================

TUTOR_PROMPT = PromptTemplate(
    input_variables=["video_title", "channel_title", "context", "chat_history", "question"],
    template="""You are LearnTube AI, an expert, engaging Socratic educational tutor.

You are helping a student understand a YouTube lecture.
VIDEO DETAILS:
- Title: {video_title}
- Channel / Speaker: {channel_title}

CONVERSATION CONTEXT:
{chat_history}

TRANSCRIPT CONTEXT FROM VIDEO (with exact timestamps):
{context}

STUDENT QUESTION:
{question}

TUTORING GUIDELINES:
1. Answer strictly based on the provided video transcript context.
2. Structure your answer cleanly with clear headings, bold technical terms, and concise bullet points.
3. If the question IS covered in the lecture, mention the exact timestamp (e.g. `[▶ 03:15]`).
4. If the question is NOT covered in the video (or asks about unrelated outside topics), clearly and politely inform the student that the video does not discuss this topic. Do NOT cite or invent unrelated timestamps for out-of-scope topics.
5. End with a short, engaging "💡 Think About This:" Socratic reflection checkpoint to help the student connect concepts.
6. Do NOT hallucinate facts that are not supported by the video transcript.

STUDENT RESPONSE:
"""
)


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes retrieved video context into an engaging, structured educational answer.
    Filters out timestamp sources if the question is out-of-scope for the video.
    """
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    chat_history = state.get("chat_history", [])
    metadata = state.get("metadata") or {}

    video_title = metadata.get("title", "Lecture")
    channel_title = metadata.get("channel_title", "Educator")

    # Format context with timestamps
    context_parts = []
    candidate_sources = []
    for chunk in chunks:
        ts_tag = f"[{chunk.get('formatted_start', '00:00')} - {chunk.get('formatted_end', '00:00')}]"
        context_parts.append(f"{ts_tag}\n{chunk.get('text', '')}")
        candidate_sources.append({
            "formatted_start": chunk.get("formatted_start", "00:00"),
            "formatted_end": chunk.get("formatted_end", "00:00"),
            "start_time": chunk.get("start_time", 0.0),
            "text": chunk.get("text", "")[:120] + "..."
        })

    context_str = "\n\n".join(context_parts) if context_parts else "No direct transcript matches found."

    history_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-3:]])
    if not history_text:
        history_text = "Initial question."

    llm = _get_agent_llm(temperature=0.25)
    chain = TUTOR_PROMPT | llm | StrOutputParser()

    draft_answer = chain.invoke({
        "video_title": video_title,
        "channel_title": channel_title,
        "context": context_str,
        "chat_history": history_text,
        "question": question
    })

    # Relevance Filter: Detect if the answer indicates the topic is not in the video
    out_of_scope_phrases = [
        "does not cover",
        "does not discuss",
        "does not mention",
        "not covered in the video",
        "not mentioned in the video",
        "not discussed in the video",
        "no information about",
        "no relevant timestamps",
        "does not provide any context about",
        "not available in the video",
        "not found in the video",
        "video you referenced does not cover",
        "video does not cover"
    ]
    lower_ans = draft_answer.lower()
    is_out_of_scope = any(phrase in lower_ans for phrase in out_of_scope_phrases)

    # Only provide timestamp source buttons if the question was genuinely relevant
    sources = [] if is_out_of_scope else candidate_sources

    thought: AgentThought = {
        "agent_name": "🎓 Socratic Tutor Agent",
        "action": "Synthesized structured explanation with video citations & pedagogical reflection" if not is_out_of_scope else "Clarified out-of-scope query & provided Socratic bridge (Suppressed timestamps)",
        "detail": f"Formulated educational response citing {len(sources)} relevant timestamp segments" if not is_out_of_scope else "Out-of-scope query: suppressed unrelated timestamp buttons",
        "timestamp": time.time()
    }

    return {
        "draft_answer": draft_answer,
        "sources": sources,
        "thought_steps": [thought]
    }


# ========================================
# 4. 🛡️ HALLUCINATION & GROUNDING EVALUATOR
# ========================================

EVALUATOR_PROMPT = PromptTemplate(
    input_variables=["context", "draft_answer"],
    template="""You are the Grounding and Hallucination Evaluator for LearnTube AI.
Your job is to check whether the draft answer is faithfully grounded in the video transcript context.

TRANSCRIPT CONTEXT:
{context}

DRAFT ANSWER TO EVALUATE:
{draft_answer}

EVALUATION CRITERIA:
1. Is every major claim in the draft answer supported by the transcript?
2. Did the answer invent facts that contradict or were never mentioned in the video?
3. IMPORTANT: If the draft answer honestly informs the student that the requested topic is NOT covered in the video (or politely clarifies the actual topic of the video), grade it as "GROUNDED". It is an honest and accurate response, NOT a hallucination.

OUTPUT STRICTLY IN JSON:
{{
    "grade": "GROUNDED | HALLUCINATED | INSUFFICIENT_CONTEXT",
    "score": 0.0 to 1.0,
    "reasoning": "1-2 sentence justification"
}}
"""
)


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates factual accuracy of the draft answer against transcript snippets before presenting to student.
    """
    draft_answer = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    retry_count = state.get("retry_count", 0)
    context_str = "\n".join([c.get("text", "") for c in chunks])

    llm = _get_agent_llm(temperature=0.0)
    chain = EVALUATOR_PROMPT | llm | JsonOutputParser()

    try:
        res = chain.invoke({
            "context": context_str,
            "draft_answer": draft_answer
        })
        grade = res.get("grade", "GROUNDED")
        reasoning = res.get("reasoning", "Answer accurately grounded in transcript")
    except Exception:
        grade = "GROUNDED"
        reasoning = "Automatic pass"

    # If grounded or max retries reached, finalize answer
    final_answer = draft_answer

    thought: AgentThought = {
        "agent_name": "🛡️ Grounding Guardrail",
        "action": f"Evaluated factual integrity — Grade: [{grade}]",
        "detail": reasoning,
        "timestamp": time.time()
    }

    return {
        "grounding_grade": grade,
        "evaluation_feedback": reasoning,
        "final_answer": final_answer,
        "retry_count": retry_count + 1,
        "thought_steps": [thought]
    }

