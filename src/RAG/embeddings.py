import os
import warnings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "learntube_videos")
DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("MAX_DISTANCE_THRESHOLD", "1.60"))

_embedding_model = None


def get_embedding_model():
    """
    Load and cache the SentenceTransformer embedding model silently.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


# ========================================
# LANGCHAIN EMBEDDING ADAPTER
# ========================================

class LearnTubeEmbeddings:
    """
    LangChain-compatible wrapper for SentenceTransformer embeddings.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = get_embedding_model()
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()


# VECTOR STORE MANAGEMENT & CACHING

def get_vector_store():
    """
    Get persistent ChromaDB vector store.
    """
    embeddings = LearnTubeEmbeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )


def is_video_indexed(video_id: str) -> bool:
    """
    Check if chunks for this video already exist in ChromaDB.
    """
    try:
        vector_store = get_vector_store()
        existing = vector_store.get(where={"video_id": video_id}, limit=1)
        return bool(existing and existing.get("ids"))
    except Exception:
        return False


def delete_video_chunks(video_id: str):
    """
    Remove existing stored chunks for a specific video before re-ingestion.
    """
    vector_store = get_vector_store()
    try:
        vector_store.delete(where={"video_id": video_id})
    except Exception:
        pass


def store_documents(documents: list[Document], video_id: str):
    """
    Store transcript chunks into ChromaDB.
    """
    if not documents:
        return None

    vector_store = get_vector_store()
    document_ids = [f"{video_id}_chunk_{i}" for i in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=document_ids)

    return vector_store


def search_documents(
    query: str,
    video_id: str,
    k: int = 5,
    max_distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> list:
    """
    Retrieve relevant transcript chunks filtered by video_id.
    Applies candidate retrieval (k_fetch = max(k*2, 8)) and filters out weak matches (distance > max_distance_threshold).
    """
    if not query or not video_id:
        return []

    vector_store = get_vector_store()
    fetch_k = max(k * 2, 8)

    try:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=fetch_k,
            filter={"video_id": video_id}
        )
    except Exception:
        return []

    results.sort(key=lambda item: item[1])
    filtered_results = [(doc, dist) for doc, dist in results if dist <= max_distance_threshold]

    return filtered_results[:k]
