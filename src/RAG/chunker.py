import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def format_timestamp(seconds: float) -> str:
    """
    Format float seconds into human-readable HH:MM:SS string (e.g., 01:16:58).
    """
    if seconds is None:
        return "00:00:00"
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_timestamped_chunks(
    transcript: list,
    video_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Document]:
    """
    Convert raw transcript snippets into timestamp-aware LangChain Documents
    using LangChain's RecursiveCharacterTextSplitter while maintaining accurate timestamps.
    """
    if not transcript or not video_id:
        return []

    # 1. Build full text and record character spans for each snippet
    snippet_spans = []  # list of (start_char, end_char, start_time, end_time)
    text_parts = []
    current_char = 0

    for snippet in transcript:
        text = snippet.get("text", "").strip()
        if not text:
            continue

        start_time = float(snippet.get("start", 0.0))
        duration = float(snippet.get("duration", 0.0))
        end_time = start_time + duration

        start_char = current_char
        end_char = start_char + len(text)
        snippet_spans.append((start_char, end_char, start_time, end_time))

        text_parts.append(text)
        current_char = end_char + 1  # Account for space join

    if not text_parts or not snippet_spans:
        return []

    full_text = " ".join(text_parts)

    # 2. Split text recursively using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    split_chunks = text_splitter.split_text(full_text)

    # 3. Calculate timestamps for each split chunk and build Document objects
    documents = []
    search_pos = 0

    for chunk_index, chunk_text in enumerate(split_chunks):
        chunk_text_clean = chunk_text.strip()
        if not chunk_text_clean:
            continue

        # Find position of chunk in full_text
        pos = full_text.find(chunk_text_clean, search_pos)
        if pos == -1:
            pos = full_text.find(chunk_text_clean)

        if pos != -1:
            chunk_start_char = pos
            chunk_end_char = pos + len(chunk_text_clean)
            search_pos = max(0, pos + min(1, len(chunk_text_clean) - chunk_overlap))
        else:
            chunk_start_char = 0
            chunk_end_char = len(full_text)

        # Map character span back to snippet start and end timestamps
        start_time = None
        end_time = None

        for s_start, s_end, s_time, e_time in snippet_spans:
            if start_time is None and s_end > chunk_start_char:
                start_time = s_time
            if s_start < chunk_end_char:
                end_time = e_time

        if start_time is None:
            start_time = snippet_spans[0][2]
        if end_time is None:
            end_time = snippet_spans[-1][3]

        documents.append(
            Document(
                page_content=chunk_text_clean,
                metadata={
                    "video_id": video_id,
                    "chunk_index": chunk_index,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
        )

    return documents
