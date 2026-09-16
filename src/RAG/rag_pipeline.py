import os
import warnings
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Modular imports from RAG sub-components
from src.RAG.chunker import (
    format_timestamp,
    create_timestamped_chunks,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from src.RAG.embeddings import (
    LearnTubeEmbeddings,
    get_embedding_model,
    get_vector_store,
    is_video_indexed,
    delete_video_chunks,
    store_documents,
    search_documents,
    DEFAULT_DISTANCE_THRESHOLD
)
from src.RAG.prompts import (
    PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    QUIZ_PROMPT_TEMPLATE
)

# LLM CONFIGURATION

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

_openai_client = None


def get_openai_client():
    """
    Initialize and cache the OpenAI client.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set OPENAI_API_KEY in your .env file."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# INGESTION PIPELINE

def ingest_transcript(transcript: list, video_id: str, force_reindex: bool = False):
    """
    Complete ingestion pipeline with instant ChromaDB cache hit for previously indexed videos.
    """
    # 1. Fast Cache Hit (0.01s)
    if not force_reindex and is_video_indexed(video_id):
        vector_store = get_vector_store()
        try:
            existing_records = vector_store.get(where={"video_id": video_id})
            count = len(existing_records.get("ids", []))
        except Exception:
            count = 1
        return {
            "success": True,
            "video_id": video_id,
            "chunks_created": count,
            "cached": True
        }

    # 2. Ingest if not in cache
    documents = create_timestamped_chunks(transcript=transcript, video_id=video_id)
    if not documents:
        return None

    delete_video_chunks(video_id)
    vector_store = store_documents(documents=documents, video_id=video_id)
    if vector_store is None:
        return None

    return {
        "success": True,
        "video_id": video_id,
        "chunks_created": len(documents),
        "cached": False
    }


# CONTEXT BUILDER

def build_rag_context(search_results: list) -> str:
    """
    Convert retrieved search results into a clean context string for OpenAI LLM.
    """
    if not search_results:
        return ""

    context_parts = []
    for index, (document, distance) in enumerate(search_results, start=1):
        start_time = document.metadata.get("start_time", 0.0)
        end_time = document.metadata.get("end_time", 0.0)
        start_fmt = format_timestamp(start_time)
        end_fmt = format_timestamp(end_time)

        context_parts.append(
            f"SOURCE {index}\n"
            f"TIMESTAMP: {start_fmt} - {end_fmt}\n"
            f"CONTENT:\n{document.page_content}\n"
        )

    return "\n".join(context_parts)


# QUESTION ANSWERING PIPELINE


def answer_question(
    question: str,
    video_id: str,
    k: int = 5,
    metadata: dict = None
) -> dict:
    """
    Complete RAG Question Answering pipeline with video metadata injection.
    Query -> Retrieve filtered chunks -> Format Prompt with Metadata -> OpenAI LLM -> Final Answer + Timestamps.
    """
    search_results = search_documents(query=question, video_id=video_id, k=k)

    # Extract metadata fields
    video_title = "Not available"
    channel_title = "Not available"
    video_description = "Not available"
    if metadata and isinstance(metadata, dict):
        video_title = metadata.get("title") or "Not available"
        channel_title = metadata.get("channel_title") or "Not available"
        desc = metadata.get("description") or ""
        video_description = desc[:1200] if desc else "Not available"

    if not search_results:
        context = "No specific transcript excerpts matched the query."
    else:
        context = build_rag_context(search_results)

    prompt = PROMPT_TEMPLATE.format(
        video_title=video_title,
        channel_title=channel_title,
        video_description=video_description,
        context=context,
        question=question
    )

    client = get_openai_client()

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=OPENAI_TEMPERATURE
        )

        response_text = completion.choices[0].message.content or ""
        answer = response_text.strip()

        if not answer:
            answer = "I couldn't find this information in the video."

    except Exception as error:
        print(f"OpenAI API Error: {error}")
        return {
            "answer": "Sorry, I was unable to generate an answer at this time.",
            "sources": []
        }

    # If the answer is a refusal / not found, do NOT show irrelevant sources
    is_refusal = (
        "couldn't find this information" in answer.lower()
        or "could not find this information" in answer.lower()
        or "unable to find" in answer.lower()
        or "not mentioned in the video" in answer.lower()
    )

    sources = []
    if not is_refusal and search_results:
        best_distance = search_results[0][1]
        for document, distance in search_results:
            # Include top relevant matches
            if distance <= best_distance + 0.20:
                start_time = document.metadata.get("start_time", 0.0)
                end_time = document.metadata.get("end_time", 0.0)
                sources.append({
                    "chunk_index": document.metadata.get("chunk_index"),
                    "start_time": start_time,
                    "end_time": end_time,
                    "formatted_start": format_timestamp(start_time),
                    "formatted_end": format_timestamp(end_time),
                    "distance": round(distance, 4)
                })

    return {
        "answer": answer,
        "sources": sources
    }


# VIDEO SUMMARY (LANGCHAIN LCEL PIPELINE)

def generate_video_summary(transcript: list, metadata: dict = None) -> str:
    """
    Generate structured study notes and summary for a video using LangChain LCEL:
    SUMMARY_PROMPT_TEMPLATE | ChatOpenAI | StrOutputParser
    """
    if not transcript:
        return "No transcript available to generate a summary."

    transcript_text = "\n".join(
        snippet.get("text", "") for snippet in transcript if snippet.get("text")
    )
    transcript_text = transcript_text[:40000]

    video_title = metadata.get("title", "Educational Video") if metadata else "Educational Video"
    channel_title = metadata.get("channel_title", "Instructor") if metadata else "Instructor"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY is not configured in your .env file."

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=api_key
    )

    summary_chain = SUMMARY_PROMPT_TEMPLATE | llm | StrOutputParser()

    try:
        summary_result = summary_chain.invoke({
            "video_title": video_title,
            "channel_title": channel_title,
            "transcript_text": transcript_text
        })
        return summary_result.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"


# VIDEO QUIZ GENERATOR (LANGCHAIN LCEL)

def generate_video_quiz(transcript: list, metadata: dict = None, num_questions: int = 4) -> list[dict]:
    """
    Generate an interactive multiple-choice quiz using LangChain LCEL:
    QUIZ_PROMPT_TEMPLATE | ChatOpenAI | JsonOutputParser
    """
    if not transcript:
        return []

    transcript_text = "\n".join(
        snippet.get("text", "") for snippet in transcript if snippet.get("text")
    )
    transcript_text = transcript_text[:40000]

    video_title = metadata.get("title", "Educational Video") if metadata else "Educational Video"
    channel_title = metadata.get("channel_title", "Instructor") if metadata else "Instructor"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=api_key
    )

    quiz_chain = QUIZ_PROMPT_TEMPLATE | llm | JsonOutputParser()

    try:
        quiz_result = quiz_chain.invoke({
            "video_title": video_title,
            "channel_title": channel_title,
            "transcript_text": transcript_text,
            "num_questions": num_questions
        })
        if isinstance(quiz_result, list):
            return quiz_result
        return []
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return []