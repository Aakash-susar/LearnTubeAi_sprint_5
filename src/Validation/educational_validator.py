import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def sample_transcript(transcript: list, samples_per_section: int = 5) -> str:
    """
    Create a representative text sample from the beginning, middle, and end of the transcript.
    """
    if not transcript:
        return ""

    total_snippets = len(transcript)

    if total_snippets <= samples_per_section * 3:
        sampled_text = "\n".join(snippet.get("text", "") for snippet in transcript)
        return sampled_text[:3000]

    beginning = transcript[:samples_per_section]
    middle_index = total_snippets // 2
    middle_start = max(0, middle_index - samples_per_section // 2)
    middle_end = middle_start + samples_per_section
    middle = transcript[middle_start:middle_end]
    end = transcript[-samples_per_section:]

    sampled_snippets = beginning + middle + end
    return "\n".join(snippet.get("text", "") for snippet in sampled_snippets)[:3000]


def rule_based_validation(metadata: dict) -> dict | None:
    """
    Perform fast rule-based validation using video title and strong intent signals.
    """
    title = metadata.get("title", "").lower()
    description = metadata.get("description", "").lower()[:1000]
    tags = metadata.get("tags", [])
    tags_text = " ".join(tags).lower()

    # Clear non-educational indicators in Title
    strong_non_educational_title_patterns = [
        r"\bofficial trailer\b", r"\bteaser trailer\b", r"\bmovie trailer\b",
        r"\bofficial music video\b", r"\bmusic video\b", r"\blyric video\b",
        r"\bstandup comedy\b", r"\bprank\b", r"\broast video\b",
        r"\bmotivational speech\b", r"\bgym motivation\b", r"\bsigma mindset\b",
        r"\bdaily vlog\b", r"\bfamily vlog\b", r"\bgaming stream\b",
        r"\bday in the life\b", r"\bday in my life\b", r"\bmorning routine\b",
        r"\bget ready with me\b", r"\bgrwm\b", r"\breaction video\b"
    ]

    for pattern in strong_non_educational_title_patterns:
        if re.search(pattern, title):
            return {
                "is_educational": False,
                "confidence": 0.95,
                "category": None,
                "topic": None,
                "reason": "Video title indicates entertainment, vlog, or non-educational content.",
                "tier": 1,
                "tier_name": "Tier 1: Rule-Based Check"
            }

    # Strong Educational keywords across title & description
    educational_keywords = [
        "tutorial", "course", "lesson", "lecture", "training", "workshop", "bootcamp", "masterclass",
        "crash course", "full course", "step by step", "how to code", "how to build", "architecture",
        "programming", "coding", "python", "javascript", "typescript", "react", "c++", "java", "sql",
        "machine learning", "deep learning", "data science", "artificial intelligence", "rag", "llm",
        "langchain", "llamaindex", "agentic", "agents", "vector database", "chromadb", "embeddings",
        "mathematics", "physics", "chemistry", "biology", "history", "engineering", "computer science",
        "system design", "algorithms", "data structures", "interview preparation", "academic"
    ]

    # Check title separately for strong signal
    title_edu_match = any(re.search(r"\b" + re.escape(kw) + r"\b", title) for kw in educational_keywords)

    if title_edu_match:
        # Title itself contains educational keyword → fast accept
        return {
            "is_educational": True,
            "confidence": 0.95,
            "category": "Technology & Education",
            "topic": title[:60],
            "reason": "Strong educational signal detected in video title.",
            "tier": 1,
            "tier_name": "Tier 1: Rule-Based Check"
        }

    # Check description & tags (weaker signal — needs higher threshold)
    combined_text = f"{tags_text} {description}"
    educational_score = sum(1 for kw in educational_keywords if re.search(r"\b" + re.escape(kw) + r"\b", combined_text))

    if educational_score >= 4:
        # Description/tags have 4+ educational keywords → likely educational
        return {
            "is_educational": True,
            "confidence": 0.85,
            "category": "Technology & Education",
            "topic": title[:60],
            "reason": "Multiple educational signals detected in description and tags.",
            "tier": 1,
            "tier_name": "Tier 1: Rule-Based Check"
        }

    # For ambiguous videos (0-3 keywords in desc/tags, no title match), defer to LLM
    return None


def build_validation_prompt(metadata: dict, transcript_sample: str) -> str:
    """
    Build a structured validation prompt to check relevance for an EdTech technical/academic platform.
    """
    title = metadata.get("title", "Not available")
    description = metadata.get("description", "Not available")[:1200]
    tags = metadata.get("tags", [])
    tags_text = ", ".join(tags[:15]) if tags else "Not available"

    return f"""You are a content evaluation engine for an EdTech technical and academic learning platform.

Evaluate whether this YouTube video is strictly RELEVANT and SUITABLE for an EdTech technical/academic learning platform.

CRITERIA FOR RELEVANCE:
- SUITABLE (is_educational: true): The video delivers structured learning, technical training, programming, computer science, mathematics, engineering, formal science, system design, academic curriculum, or technical skill building.
- NOT SUITABLE (is_educational: false): The video is a lifestyle vlog, "day in the life", casual conversation/podcast, personal opinion piece, pop-psychology advice, entertainment, gaming, reaction video, marketing promo, or general motivation without concrete technical/academic curriculum.

VIDEO TITLE:
{title}

VIDEO DESCRIPTION:
{description}

TAGS:
{tags_text}

TRANSCRIPT SAMPLE:
{transcript_sample}

Return ONLY valid JSON in this exact schema:
{{
    "is_educational": true,
    "confidence": 0.95,
    "category": "Technology | Computer Science | Mathematics | Science | Academic",
    "topic": "Concise topic title",
    "reason": "Clear explanation of why this video is or is not suitable for an EdTech learning platform"
}}
"""


def clean_json_response(response_text: str) -> str:
    """
    Clean markdown code block syntax from JSON response string.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^```json\s*", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"^```\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    return response_text.strip()


def validate_educational_content(metadata: dict, transcript: list) -> dict:
    """
    Validate whether a YouTube video is strictly educational using a 2-tier pipeline:
    1. Rule-based evaluation (fast path)
    2. Fast LLM evaluation via OpenAI gpt-4o-mini (with strict 8s timeout)
    """
    rule_result = rule_based_validation(metadata)
    if rule_result is not None:
        return rule_result

    openai_key = os.getenv("OPENAI_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")

    if openai_key:
        client = OpenAI(api_key=openai_key, timeout=8.0)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif nvidia_key:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key,
            timeout=8.0
        )
        model_name = "deepseek-ai/deepseek-v4-pro-0813"
    else:
        return {
            "is_educational": True,
            "confidence": 0.60,
            "category": None,
            "topic": None,
            "reason": "Fallback acceptance (no validation API key configured)."
        }

    transcript_sample_text = sample_transcript(transcript)
    prompt = build_validation_prompt(metadata, transcript_sample_text)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250
        )

        response_text = completion.choices[0].message.content or ""
        cleaned_text = clean_json_response(response_text)

        result = None
        try:
            result = json.loads(cleaned_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError(f"Failed to parse LLM response as JSON: {response_text}")

        result["tier"] = 2
        result["tier_name"] = "Tier 2: AI / LLM Evaluation"
        return result

    except Exception as error:
        return {
            "is_educational": True,
            "confidence": 0.60,
            "category": None,
            "topic": None,
            "reason": f"Fallback acceptance due to validator timeout/error: {error}",
            "tier": 2,
            "tier_name": "Tier 2: AI / LLM Evaluation (Fallback)"
        }