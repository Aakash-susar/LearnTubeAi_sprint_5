import os
import requests
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def fetch_video_metadata(video_id: str) -> dict | None:
    """
    Fetch metadata for a YouTube video using YouTube Data API v3.

    Returns:
        dict: Clean video metadata, or None if error/video not found.
    """
    if not video_id:
        return None

    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY not found in environment variables. "
            "Please ensure it is set in your .env file."
        )

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])
        if not items:
            print(f"No video found on YouTube API for ID: {video_id}")
            return None

        video = items[0]
        snippet = video.get("snippet", {})
        content_details = video.get("contentDetails", {})
        statistics = video.get("statistics", {})

        return {
            "video_id": video.get("id"),
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_id": snippet.get("channelId"),
            "channel_title": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt"),
            "tags": snippet.get("tags", []),
            "category_id": snippet.get("categoryId"),
            "default_language": snippet.get("defaultLanguage"),
            "duration": content_details.get("duration"),
            "caption_available": content_details.get("caption"),
            "view_count": statistics.get("viewCount"),
            "like_count": statistics.get("likeCount"),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url")
        }

    except requests.exceptions.RequestException as error:
        print(f"Error fetching video metadata: {error}")
        return None