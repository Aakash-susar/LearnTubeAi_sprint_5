from src.youtube.video_id import extract_video_id
from src.youtube.metadata import fetch_video_metadata
from src.youtube.transcript import fetch_transcript


def process_youtube_video(video_url: str) -> dict | None:
    """
    Process a YouTube video URL and return its video ID, metadata, and transcript.

    Returns:
        dict: Object containing video_id, metadata, and transcript list, or None on failure.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL provided.")
        return None

    metadata = fetch_video_metadata(video_id)
    if not metadata:
        print(f"Could not fetch metadata for video ID: {video_id}")
        return None

    transcript = fetch_transcript(video_id)
    if not transcript:
        print(f"Could not fetch transcript for video ID: {video_id}")
        return None

    return {
        "video_id": video_id,
        "metadata": metadata,
        "transcript": transcript
    }