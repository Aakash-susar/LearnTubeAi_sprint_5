"""
Local Speech-to-Text Transcription Engine for LearnTube AI.

Uses faster-whisper (CTranslate2 + PyAV) to transcribe local uploaded audio/video files
into timestamped transcript snippets compatible with the LearnTube RAG pipeline.
"""

import os
import hashlib
import time
from typing import List, Dict, Any, Optional

_whisper_model_instance = None
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", "base")


def get_whisper_model(model_size: str = DEFAULT_WHISPER_MODEL, device: str = "cpu", compute_type: str = "int8"):
    """
    Initialize and cache the faster-whisper model singleton.
    Uses int8 quantization on CPU for ultra-fast Intel CPU AVX inference.
    """
    global _whisper_model_instance
    if _whisper_model_instance is None:
        from faster_whisper import WhisperModel
        num_threads = min(os.cpu_count() or 4, 8)
        print(f"[Whisper] Loading local faster-whisper '{model_size}' model on {device} ({compute_type}) with {num_threads} CPU threads...")
        _whisper_model_instance = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_threads,
            num_workers=1
        )
        print("[Whisper] Model loaded successfully.")
    return _whisper_model_instance


def generate_upload_video_id(filename: str, file_bytes: Optional[bytes] = None) -> str:
    """
    Generate a deterministic, unique video ID for an uploaded media file.
    Example: 'upload_a1b2c3d4e5f6'
    """
    if file_bytes:
        digest = hashlib.md5(file_bytes[:1024 * 1024]).hexdigest()[:12]
    else:
        digest = hashlib.md5(f"{filename}_{time.time()}".encode("utf-8")).hexdigest()[:12]
    return f"upload_{digest}"


def transcribe_media_file(
    file_path: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transcribes any video/audio file (.mp4, .mkv, .mov, .mp3, .wav, .m4a) into timestamped segments.

    Optimized for CPU performance:
    - beam_size=1 (Greedy search, 4-5x faster than beam_size=5 with virtually identical quality)
    - condition_on_previous_text=False (Prevents repetitive loops and excessive latency)
    - vad_filter=True (Skips silent and non-speech portions)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Media file not found at: {file_path}")

    start_time = time.time()
    model = get_whisper_model(model_size=model_size)

    print(f"[Whisper] Starting transcription on '{os.path.basename(file_path)}'...")

    # Transcribe directly from media container via PyAV
    segments, info = model.transcribe(
        file_path,
        beam_size=1,  # Fast greedy decoding
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,  # Prevents repetition loops
        language=language,
        vad_filter=True,  # Filter out silence/noise
        vad_parameters=dict(min_silence_duration_ms=400)
    )

    detected_lang = getattr(info, "language", "auto")
    detected_prob = getattr(info, "language_probability", 1.0)
    print(f"[Whisper] Detected language: {detected_lang} ({detected_prob:.0%})")

    transcript = []
    seg_idx = 0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start_sec = round(float(seg.start), 2)
        end_sec = round(float(seg.end), 2)
        duration = round(end_sec - start_sec, 2)
        if duration <= 0.0:
            duration = 1.0

        transcript.append({
            "start": start_sec,
            "duration": duration,
            "text": text
        })
        seg_idx += 1
        if seg_idx % 10 == 0:
            print(f"[Whisper] Processed {seg_idx} segments (Timestamp: {start_sec:.1f}s)...")

    elapsed = round(time.time() - start_time, 2)
    print(f"[Whisper] Completed {len(transcript)} segments in {elapsed}s.")
    return transcript
