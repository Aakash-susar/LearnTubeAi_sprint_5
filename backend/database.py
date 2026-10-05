import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "learntube")

_client = None
db = None

if MONGODB_URI:
    try:
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        db = _client[MONGODB_DATABASE]
    except Exception as err:
        print(f"⚠️ MongoDB Initialization Warning: {err}")
        _client = None
        db = None


def test_database_connection() -> bool:
    """Check if MongoDB is actively connected and reachable."""
    if _client is None:
        return False
    try:
        _client.admin.command("ping")
        return True
    except Exception as error:
        print(f"MongoDB connection ping error: {error}")
        return False


# ========================================
# 1. VIDEOS COLLECTION
# ========================================

def get_cached_video(video_id: str) -> dict | None:
    """Retrieve cached video metadata and transcript from MongoDB."""
    if db is None or not video_id:
        return None
    try:
        doc = db.videos.find_one({"video_id": video_id}, {"_id": 0})
        return doc
    except Exception as err:
        print(f"MongoDB get_cached_video error: {err}")
        return None


def save_cached_video(video_id: str, metadata: dict, transcript: list, validation: dict = None) -> bool:
    """Save or update video metadata, transcript, and validation in MongoDB."""
    if db is None or not video_id:
        return False
    try:
        db.videos.update_one(
            {"video_id": video_id},
            {
                "$set": {
                    "video_id": video_id,
                    "metadata": metadata,
                    "transcript": transcript,
                    "validation": validation,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        return True
    except Exception as err:
        print(f"MongoDB save_cached_video error: {err}")
        return False


def get_all_videos(limit: int = 50) -> list[dict]:
    """Retrieve list of all verified educational videos from MongoDB (excluding non-educational)."""
    if db is None:
        return []
    try:
        cursor = db.videos.find(
            {"validation.is_educational": True},
            {"_id": 0, "video_id": 1, "metadata": 1, "validation": 1, "created_at": 1, "updated_at": 1}
        ).sort("updated_at", -1).limit(limit)
        return list(cursor)
    except Exception as err:
        print(f"MongoDB get_all_videos error: {err}")
        return []


def delete_cached_video(video_id: str) -> bool:
    """Delete a video and all associated summaries, quizzes, chats, and visualizations from MongoDB."""
    if db is None or not video_id:
        return False
    try:
        db.videos.delete_one({"video_id": video_id})
        db.summaries.delete_many({"video_id": video_id})
        db.quizzes.delete_many({"video_id": video_id})
        db.chats.delete_many({"video_id": video_id})
        db.visualizations.delete_many({"video_id": video_id})
        return True
    except Exception as err:
        print(f"MongoDB delete_cached_video error: {err}")
        return False


# ========================================
# 2. SUMMARIES COLLECTION
# ========================================

def get_cached_summary(video_id: str, style: str = "study_notes") -> str | None:
    """Retrieve cached summary for a video in a specific style."""
    if db is None or not video_id:
        return None
    try:
        norm_style = style.lower().strip() if style else "study_notes"
        doc = db.summaries.find_one(
            {"video_id": video_id, "style": norm_style},
            {"_id": 0, "summary": 1}
        )
        if doc and "summary" in doc:
            return doc["summary"]

        # Fallback check for legacy records saved without style
        if norm_style in ["study_notes", "deep_dive"]:
            doc_legacy = db.summaries.find_one(
                {"video_id": video_id, "style": {"$exists": False}},
                {"_id": 0, "summary": 1}
            )
            if doc_legacy and "summary" in doc_legacy:
                return doc_legacy["summary"]

        return None
    except Exception as err:
        print(f"MongoDB get_cached_summary error: {err}")
        return None


def save_cached_summary(video_id: str, summary: str, style: str = "study_notes") -> bool:
    """Save generated summary in MongoDB tagged with style."""
    if db is None or not video_id or not summary:
        return False
    try:
        norm_style = style.lower().strip() if style else "study_notes"
        db.summaries.update_one(
            {"video_id": video_id, "style": norm_style},
            {
                "$set": {
                    "video_id": video_id,
                    "style": norm_style,
                    "summary": summary,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        return True
    except Exception as err:
        print(f"MongoDB save_cached_summary error: {err}")
        return False


# ========================================
# 3. QUIZZES COLLECTION
# ========================================

def get_cached_quiz(video_id: str, num_questions: int = None, style: str = "quick_test") -> list | None:
    """Retrieve cached quiz for a video matching style and optional question count."""
    if db is None or not video_id:
        return None
    try:
        norm_style = style.lower().strip() if style else "quick_test"
        query = {"video_id": video_id, "style": norm_style}
        if num_questions:
            doc = db.quizzes.find_one(
                {"video_id": video_id, "style": norm_style, "num_questions": num_questions},
                {"_id": 0, "quiz": 1}
            )
            if doc and "quiz" in doc:
                return doc["quiz"]

        # Fallback to most recent quiz for this video & style
        doc = db.quizzes.find_one(
            query,
            {"_id": 0, "quiz": 1},
            sort=[("updated_at", -1)]
        )
        if doc and "quiz" in doc:
            return doc["quiz"]

        # Legacy fallback if style is not present in document
        doc_legacy = db.quizzes.find_one(
            {"video_id": video_id, "style": {"$exists": False}},
            {"_id": 0, "quiz": 1},
            sort=[("updated_at", -1)]
        )
        if doc_legacy and "quiz" in doc_legacy:
            return doc_legacy["quiz"]

        return None
    except Exception as err:
        print(f"MongoDB get_cached_quiz error: {err}")
        return None


def save_cached_quiz(video_id: str, quiz: list, num_questions: int = None, style: str = "quick_test") -> bool:
    """Save generated quiz questions in MongoDB tagged with style and question count."""
    if db is None or not video_id or not quiz:
        return False
    try:
        norm_style = style.lower().strip() if style else "quick_test"
        q_count = num_questions or len(quiz)
        db.quizzes.update_one(
            {"video_id": video_id, "style": norm_style, "num_questions": q_count},
            {
                "$set": {
                    "video_id": video_id,
                    "style": norm_style,
                    "num_questions": q_count,
                    "quiz": quiz,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        return True
    except Exception as err:
        print(f"MongoDB save_cached_quiz error: {err}")
        return False


# ========================================
# 4. CHATS COLLECTION
# ========================================

def save_chat_message(video_id: str, question: str, answer: str, sources: list = None) -> bool:
    """Persist user question and AI answer in MongoDB."""
    if db is None or not video_id:
        return False
    try:
        db.chats.insert_one({
            "video_id": video_id,
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return True
    except Exception as err:
        print(f"MongoDB save_chat_message error: {err}")
        return False


def get_chat_history(video_id: str, limit: int = 50) -> list:
    """Retrieve recent chat history for a video."""
    if db is None or not video_id:
        return []
    try:
        cursor = db.chats.find(
            {"video_id": video_id},
            {"_id": 0}
        ).sort("timestamp", 1).limit(limit)
        return list(cursor)
    except Exception as err:
        print(f"MongoDB get_chat_history error: {err}")
        return []


# ========================================
# 5. VISUALIZATIONS COLLECTION
# ========================================

def get_cached_visualization(video_id: str, visual_type: str = "mindmap") -> str | None:
    """Retrieve cached Mermaid diagram string for a video matching visual_type."""
    if db is None or not video_id:
        return None
    try:
        norm_type = visual_type.lower().strip() if visual_type else "mindmap"
        doc = db.visualizations.find_one(
            {"video_id": video_id, "visual_type": norm_type},
            {"_id": 0, "diagram": 1}
        )
        if doc and "diagram" in doc:
            return doc["diagram"]
        return None
    except Exception as err:
        print(f"MongoDB get_cached_visualization error: {err}")
        return None


def save_cached_visualization(video_id: str, diagram: str, visual_type: str = "mindmap") -> bool:
    """Save generated Mermaid diagram in MongoDB tagged with visual_type."""
    if db is None or not video_id or not diagram:
        return False
    try:
        norm_type = visual_type.lower().strip() if visual_type else "mindmap"
        db.visualizations.update_one(
            {"video_id": video_id, "visual_type": norm_type},
            {
                "$set": {
                    "video_id": video_id,
                    "visual_type": norm_type,
                    "diagram": diagram,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )
        return True
    except Exception as err:
        print(f"MongoDB save_cached_visualization error: {err}")
        return False