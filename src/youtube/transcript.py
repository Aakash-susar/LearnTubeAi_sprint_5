from youtube_transcript_api import YouTubeTranscriptApi


def fetch_transcript(video_id: str) -> list[dict] | None:
    """
    Fetch the transcript of a YouTube video using priority:
    1. English ('en')
    2. Hindi ('hi')
    3. Any available transcript

    Returns:
        List of snippet dicts with 'text', 'start', and 'duration', or None if unavailable.
    """
    if not video_id:
        return None

    try:
        # Try fetching transcript list (supports classmethod and instance API)
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)

        selected_transcript = None

        # 1. Try English
        try:
            selected_transcript = transcript_list.find_transcript(["en"])
        except Exception:
            pass

        # 2. Try Hindi
        if selected_transcript is None:
            try:
                selected_transcript = transcript_list.find_transcript(["hi"])
            except Exception:
                pass

        # 3. Fallback to any transcript
        if selected_transcript is None:
            for transcript_item in transcript_list:
                selected_transcript = transcript_item
                break

        if selected_transcript is None:
            print(f"No transcript available for video: {video_id}")
            return None

        fetched_transcript = selected_transcript.fetch()

        clean_transcript = []
        for snippet in fetched_transcript:
            clean_transcript.append({
                "text": snippet.get("text") if isinstance(snippet, dict) else snippet.text,
                "start": float(snippet.get("start", 0) if isinstance(snippet, dict) else snippet.start),
                "duration": float(snippet.get("duration", 0) if isinstance(snippet, dict) else snippet.duration)
            })

        return clean_transcript

    except Exception as error:
        print(f"Error fetching transcript for video {video_id}: {error}")
        return None