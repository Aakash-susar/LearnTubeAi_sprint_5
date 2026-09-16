from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str | None:
    """
    Extract a YouTube video ID from a valid YouTube URL.

    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/watch?v=VIDEO_ID&t=120
    """
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()

    # Ensure URL protocol prefix
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if not hostname:
        return None

    hostname = hostname.lower().replace("www.", "")

    # Standard YouTube URLs
    if hostname == "youtube.com":
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]

        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1]

        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1]

    # Shortened YouTube URL (youtu.be)
    if hostname == "youtu.be":
        video_id = parsed_url.path.strip("/")
        return video_id or None

    return None