from src.youtube.pipeline import process_youtube_video
from src.Validation.educational_validator import validate_educational_content
from src.RAG.rag_pipeline import ingest_transcript, answer_question


def main():
    video_url = "https://www.youtube.com/watch?v=vXH-U5kpXIc"
    question = "What is the 90-day plan explained in this video?"

    print("\n" + "=" * 60)
    print("                LEARNTUBE AI - YOUTUBE RAG")
    print("=" * 60)

    # Step 1: YouTube Video Processing
    print(f"\nProcessing YouTube Video: {video_url}...")
    video_data = process_youtube_video(video_url)
    if not video_data:
        print("Failed to process YouTube video.")
        return

    metadata = video_data.get("metadata", {})
    transcript = video_data.get("transcript", [])
    video_id = video_data.get("video_id") or metadata.get("video_id") or "current_video"

    print(f"Title: {metadata.get('title', 'N/A')}")
    print(f"Video ID: {video_id} ({len(transcript)} transcript snippets)")

    # Step 2: Educational Content Validation
    validation_result = validate_educational_content(
        metadata=metadata,
        transcript=transcript
    )

    if not validation_result.get("is_educational", False):
        print("\n[REJECTED] Video content is not educational.")
        print(f"Reason: {validation_result.get('reason')}")
        return

    print("Status: Accepted as Educational")

    # Step 3: RAG Ingestion into ChromaDB
    ingestion_result = ingest_transcript(
        transcript=transcript,
        video_id=video_id
    )

    if not ingestion_result:
        print("[FAILED] RAG Ingestion failed.")
        return

    print(f"Index: Ingested {ingestion_result.get('chunks_created', 0)} chunks into ChromaDB")

    # Step 4: Ask Question & Get RAG Answer
    print("\n" + "-" * 60)
    print(f"QUESTION: \"{question}\"")
    print("-" * 60)

    rag_result = answer_question(
        question=question,
        video_id=video_id,
        k=3,
        metadata=metadata
    )

    # Step 5: Clean Output
    print("\nANSWER:")
    print(rag_result.get("answer"))

    sources = rag_result.get("sources", [])
    if sources:
        print("\nSOURCE TIMESTAMPS:")
        for index, source in enumerate(sources, start=1):
            fmt_start = source.get("formatted_start", "00:00:00")
            fmt_end = source.get("formatted_end", "00:00:00")
            print(f"  [{index}] {fmt_start} -> {fmt_end}")
    else:
        print("\nNo matching sources found.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()