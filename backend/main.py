import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.database import (
    test_database_connection,
    get_cached_video,
    save_cached_video,
    get_all_videos,
    delete_cached_video,
    get_cached_summary,
    save_cached_summary,
    get_cached_quiz,
    save_cached_quiz,
    get_cached_visualization,
    save_cached_visualization,
    save_chat_message,
    get_chat_history
)
from src.youtube.video_id import extract_video_id
from src.youtube.pipeline import process_youtube_video
from src.transcription.whisper_transcriber import transcribe_media_file, generate_upload_video_id
from src.Validation.educational_validator import validate_educational_content
from src.RAG.rag_pipeline import (
    ingest_transcript,
    answer_question,
    generate_video_summary,
    generate_video_quiz,
    generate_video_visualization
)
from src.agents import run_agent_chat

app = FastAPI(
    title="LearnTube AI API",
    description="Backend API for LearnTube AI - Video Processing, RAG QA, Summaries & Quizzes with MongoDB Persistence",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Models ----

class VideoRequest(BaseModel):
    url: str
    auto_ingest: bool = True  # Automatically validate and store in ChromaDB

class ChatRequest(BaseModel):
    question: str
    video_id: str
    k: Optional[int] = 5
    metadata: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, Any]]] = None

class ValidateRequest(BaseModel):
    metadata: Dict[str, Any]
    transcript: List[Dict[str, Any]]
    video_id: Optional[str] = None

class IngestRequest(BaseModel):
    transcript: List[Dict[str, Any]]
    video_id: str
    force_reindex: Optional[bool] = False

class ContentRequest(BaseModel):
    transcript: List[Dict[str, Any]]
    video_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    style: Optional[str] = "study_notes"
    visual_type: Optional[str] = "mindmap"
    num_questions: Optional[int] = 4
    force_regenerate: Optional[bool] = False


# ---- Endpoints ----

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "LearnTube AI API is running smoothly 🎓",
        "mongodb": test_database_connection()
    }


@app.get("/videos")
def list_videos(limit: Optional[int] = 50):
    """Retrieve all processed videos stored in MongoDB for the library sidebar."""
    try:
        videos = get_all_videos(limit=limit or 50)
        return {
            "success": True,
            "count": len(videos),
            "videos": videos
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/video/{video_id}")
def get_video(video_id: str):
    """Retrieve full video data (metadata, transcript, validation) from MongoDB by video_id."""
    try:
        cached_video = get_cached_video(video_id)
        if not cached_video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found in database.")
        video_data = {
            "video_id": cached_video["video_id"],
            "metadata": cached_video.get("metadata", {}),
            "transcript": cached_video.get("transcript", [])
        }
        return {
            "success": True,
            "data": {
                "video_data": video_data,
                "validation": cached_video.get("validation")
            }
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.delete("/video/{video_id}")
def delete_video(video_id: str):
    """Delete a video and all associated cached documents from MongoDB."""
    try:
        deleted = delete_cached_video(video_id)
        return {
            "success": deleted,
            "message": f"Video {video_id} deleted successfully" if deleted else "Failed to delete video"
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/process")
def process_video(request: VideoRequest):
    """Fetch video metadata and transcript (with MongoDB caching), and optionally validate & index."""
    try:
        extracted_id = extract_video_id(request.url)
        
        # 1. Check MongoDB cache first
        if extracted_id:
            cached_video = get_cached_video(extracted_id)
            if cached_video and cached_video.get("transcript"):
                video_data = {
                    "video_id": cached_video["video_id"],
                    "metadata": cached_video.get("metadata", {}),
                    "transcript": cached_video.get("transcript", [])
                }
                validation = cached_video.get("validation")
                response_data = {
                    "video_data": video_data,
                    "cached": True,
                    "source": "mongodb"
                }

                if request.auto_ingest:
                    if not validation:
                        validation = validate_educational_content(video_data["metadata"], video_data["transcript"])
                        save_cached_video(extracted_id, video_data["metadata"], video_data["transcript"], validation)
                    
                    response_data["validation"] = validation

                    if validation.get("is_educational", False):
                        ingestion = ingest_transcript(video_data["transcript"], video_data["video_id"])
                        response_data["ingestion"] = ingestion

                return {"success": True, "data": response_data}

        # 2. Scrape from YouTube if not in cache
        result = process_youtube_video(request.url)
        if result is None:
            raise HTTPException(status_code=400, detail="Unable to process the YouTube video. Please check URL or captions.")

        response_data = {"video_data": result, "cached": False, "source": "youtube"}
        validation = None

        if request.auto_ingest:
            validation = validate_educational_content(result["metadata"], result["transcript"])
            response_data["validation"] = validation
            
            if validation.get("is_educational", False):
                ingestion = ingest_transcript(result["transcript"], result["video_id"])
                response_data["ingestion"] = ingestion
                # Only cache in MongoDB if verified educational
                save_cached_video(result["video_id"], result["metadata"], result["transcript"], validation)

        return {"success": True, "data": response_data}

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...), auto_ingest: bool = True):
    """
    Accepts an uploaded video/audio file (.mp4, .mkv, .mov, .mp3, .wav, .m4a),
    transcribes it locally using faster-whisper on CPU, validates educational content,
    ingests to ChromaDB, and caches in MongoDB.
    """
    try:
        filename = file.filename or "uploaded_media.mp4"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        video_id = generate_upload_video_id(filename, content)

        # 1. Check MongoDB cache first
        cached_video = get_cached_video(video_id)
        if cached_video and cached_video.get("transcript"):
            video_data = {
                "video_id": cached_video["video_id"],
                "metadata": cached_video.get("metadata", {}),
                "transcript": cached_video.get("transcript", [])
            }
            validation = cached_video.get("validation")
            response_data = {
                "video_data": video_data,
                "cached": True,
                "source": "mongodb"
            }

            if auto_ingest:
                if not validation:
                    validation = validate_educational_content(video_data["metadata"], video_data["transcript"])
                    save_cached_video(video_id, video_data["metadata"], video_data["transcript"], validation)
                
                response_data["validation"] = validation

                if validation.get("is_educational", False):
                    ingestion = ingest_transcript(video_data["transcript"], video_data["video_id"])
                    response_data["ingestion"] = ingestion

            return {"success": True, "data": response_data}

        # 2. Save to temporary file for PyAV / faster-whisper decoding
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            transcript = transcribe_media_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not transcript:
            raise HTTPException(status_code=400, detail="No speech or dialogue could be detected in the uploaded media file.")

        # Calculate estimated duration
        total_duration = round(transcript[-1]["start"] + transcript[-1]["duration"], 2) if transcript else 0.0

        # Construct clean metadata
        clean_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        metadata = {
            "title": clean_title,
            "channel_title": "Uploaded Media",
            "source": "uploaded_file",
            "filename": filename,
            "duration": total_duration,
            "upload_id": video_id
        }

        result = {
            "video_id": video_id,
            "metadata": metadata,
            "transcript": transcript
        }

        response_data = {"video_data": result, "cached": False, "source": "uploaded_file"}
        validation = None

        if auto_ingest:
            validation = validate_educational_content(result["metadata"], result["transcript"])
            response_data["validation"] = validation
            
            if validation.get("is_educational", False):
                ingestion = ingest_transcript(result["transcript"], result["video_id"])
                response_data["ingestion"] = ingestion
                # Only cache in MongoDB if verified educational
                save_cached_video(result["video_id"], result["metadata"], result["transcript"], validation)

        return {"success": True, "data": response_data}

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/validate")
def validate_video(request: ValidateRequest):
    """Validate whether the video content is educational."""
    try:
        result = validate_educational_content(request.metadata, request.transcript)
        
        # If video_id provided, persist to MongoDB ONLY if verified educational, otherwise delete
        if request.video_id:
            if result.get("is_educational", False):
                save_cached_video(request.video_id, request.metadata, request.transcript, result)
            else:
                delete_cached_video(request.video_id)

        return {"success": True, "data": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/ingest")
def ingest_video(request: IngestRequest):
    """Chunk and index transcript into ChromaDB vector store."""
    try:
        result = ingest_transcript(
            transcript=request.transcript,
            video_id=request.video_id,
            force_reindex=request.force_reindex or False
        )
        return {"success": True, "data": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/chat")
def chat(request: ChatRequest):
    """RAG Question Answering over video transcript chunks with metadata injection and MongoDB history logging."""
    try:
        # Resolve metadata if not passed directly in payload
        metadata = request.metadata
        if not metadata and request.video_id:
            cached_vid = get_cached_video(request.video_id)
            if cached_vid and "metadata" in cached_vid:
                metadata = cached_vid["metadata"]

        result = answer_question(
            question=request.question,
            video_id=request.video_id,
            k=request.k,
            metadata=metadata
        )
        
        # Persist conversation message in MongoDB
        save_chat_message(
            video_id=request.video_id,
            question=request.question,
            answer=result.get("answer", ""),
            sources=result.get("sources", [])
        )

        return {"success": True, "data": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/chat/agent")
def agent_chat(request: ChatRequest):
    """
    Multi-Agent Question Answering powered by LangGraph.
    Executes Router -> Retriever -> Tutor -> Grounding Evaluator cycle.
    Returns the final answer, timestamp sources, and step-by-step agent thought logs.
    """
    try:
        metadata = request.metadata
        if not metadata and request.video_id:
            cached_vid = get_cached_video(request.video_id)
            if cached_vid and "metadata" in cached_vid:
                metadata = cached_vid["metadata"]

        result = run_agent_chat(
            question=request.question,
            video_id=request.video_id,
            chat_history=request.chat_history or [],
            metadata=metadata
        )

        # Persist conversation message in MongoDB
        save_chat_message(
            video_id=request.video_id,
            question=request.question,
            answer=result.get("answer", ""),
            sources=result.get("sources", [])
        )

        return {"success": True, "data": result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/chat/{video_id}")
def get_history(video_id: str):
    """Retrieve persistent conversation history for a video session."""
    try:
        history = get_chat_history(video_id)
        return {"success": True, "history": history}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/video/summary/{video_id}")
def get_cached_summary_endpoint(video_id: str, style: Optional[str] = "study_notes"):
    """Fetch cached summary from MongoDB for a specific style if available."""
    try:
        summary_style = style or "study_notes"
        cached = get_cached_summary(video_id, style=summary_style)
        if cached:
            return {"success": True, "summary": cached, "style": summary_style, "cached": True}
        return {"success": False, "summary": None, "style": summary_style, "cached": False}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/video/quiz/{video_id}")
def get_cached_quiz_endpoint(video_id: str, num_questions: Optional[int] = None, style: Optional[str] = "quick_test"):
    """Fetch cached quiz from MongoDB if available."""
    try:
        quiz_style = style or "quick_test"
        cached = get_cached_quiz(video_id, num_questions=num_questions, style=quiz_style)
        if cached:
            return {"success": True, "quiz": cached, "style": quiz_style, "cached": True}
        return {"success": False, "quiz": None, "style": quiz_style, "cached": False}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/summary")
def get_summary(request: ContentRequest):
    """Generate structured summary in the requested style with MongoDB caching."""
    try:
        summary_style = request.style or "study_notes"

        # Check cache if not forcing regeneration
        if not request.force_regenerate and request.video_id:
            cached_summary = get_cached_summary(request.video_id, style=summary_style)
            if cached_summary:
                return {
                    "success": True,
                    "summary": cached_summary,
                    "style": summary_style,
                    "cached": True,
                    "source": "mongodb"
                }

        summary = generate_video_summary(
            transcript=request.transcript,
            metadata=request.metadata,
            style=summary_style
        )

        # Save to MongoDB
        if request.video_id and summary and not summary.startswith("Error:"):
            save_cached_summary(request.video_id, summary, style=summary_style)

        return {
            "success": True,
            "summary": summary,
            "style": summary_style,
            "cached": False,
            "source": "openai"
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/quiz")
def get_quiz(request: ContentRequest):
    """Generate interactive active-recall quiz with MCQs with MongoDB caching."""
    try:
        quiz_style = request.style or "quick_test"
        num_q = request.num_questions or 5

        # Check cache if not forcing regeneration
        if not request.force_regenerate and request.video_id:
            cached_quiz = get_cached_quiz(request.video_id, num_questions=num_q, style=quiz_style)
            if cached_quiz:
                return {
                    "success": True,
                    "quiz": cached_quiz,
                    "style": quiz_style,
                    "cached": True,
                    "source": "mongodb"
                }

        quiz = generate_video_quiz(
            transcript=request.transcript,
            metadata=request.metadata,
            num_questions=num_q,
            style=quiz_style
        )

        # Save to MongoDB
        if request.video_id and quiz:
            save_cached_quiz(request.video_id, quiz, num_questions=num_q, style=quiz_style)

        return {
            "success": True,
            "quiz": quiz,
            "style": quiz_style,
            "cached": False,
            "source": "openai"
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/video/visualization/{video_id}")
def get_cached_visualization_endpoint(video_id: str, type: Optional[str] = "mindmap"):
    """Fetch cached visualization diagram from MongoDB if available."""
    try:
        v_type = type or "mindmap"
        cached = get_cached_visualization(video_id, visual_type=v_type)
        if cached:
            return {"success": True, "diagram": cached, "type": v_type, "cached": True}
        return {"success": False, "diagram": None, "type": v_type, "cached": False}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/video/visualization")
def get_visualization(request: ContentRequest):
    """Generate interactive Mermaid.js diagram (Mindmap or Flowchart) with MongoDB caching."""
    try:
        v_type = request.visual_type or request.style or "mindmap"

        # Check cache if not forcing regeneration
        if not request.force_regenerate and request.video_id:
            cached_diagram = get_cached_visualization(request.video_id, visual_type=v_type)
            if cached_diagram:
                return {
                    "success": True,
                    "diagram": cached_diagram,
                    "type": v_type,
                    "cached": True,
                    "source": "mongodb"
                }

        diagram = generate_video_visualization(
            transcript=request.transcript,
            metadata=request.metadata,
            visual_type=v_type
        )

        # Save to MongoDB
        if request.video_id and diagram and not diagram.startswith("Error:"):
            save_cached_visualization(request.video_id, diagram, visual_type=v_type)

        return {
            "success": True,
            "diagram": diagram,
            "type": v_type,
            "cached": False,
            "source": "openai"
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
