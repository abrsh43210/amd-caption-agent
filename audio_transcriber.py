"""Speech-to-text transcription with Fireworks API and local Whisper fallback."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from openai import APIStatusError, OpenAI

from pipeline import _chat_completion_with_retry, build_client, resolve_fireworks_api_key
from video_processor import (
    cleanup_audio_file,
    extract_audio_from_video,
    extract_midpoint_frame,
    extract_sample_frames,
)

logger = logging.getLogger(__name__)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
WHISPER_MODEL = "whisper-v3"
LOCAL_WHISPER_MODEL_SIZE = "base"

# Fireworks vision model used for silent-video scene description.
VISION_MODEL = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

# Minimum character length below which a transcript is considered empty/silent.
_MIN_TRANSCRIPT_CHARS = 5

# Fallback transcript used when a video has no audible speech AND vision
# description also fails, so caption generation never crashes.
SILENT_VIDEO_BASELINE = (
    "[Silent video detected — no audible speech. "
    "Visual content only; captions based on contextual inference.]"
)


def _resolve_api_key(api_key: str | None) -> str:
    resolved = resolve_fireworks_api_key(api_key)
    if not resolved:
        raise ValueError("FIREWORKS_API_KEY is not configured.")
    if not resolved.startswith("fw_"):
        raise ValueError("FIREWORKS_API_KEY must start with 'fw_'.")
    return resolved


def build_transcription_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)


def _fireworks_audio_unavailable(exc: Exception) -> bool:
    """Return True for any failure that should route to the local Whisper fallback.

    Covers:
    - Fireworks deprecated serverless audio (June 2026) — HTTP 401/403/404/410
    - Network / connection failures (DNS, TCP, proxy, timeout)
    """
    if isinstance(exc, APIStatusError) and exc.status_code in {401, 403, 404, 410}:
        return True
    # httpx (used by the OpenAI SDK) raises httpx.ConnectError for network failures.
    # Catch by type name so we don't need a hard httpx import here.
    exc_type = type(exc).__name__
    if exc_type in {"ConnectError", "ConnectTimeout", "RemoteProtocolError", "ReadTimeout"}:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "unauthorized",
            "not found",
            "connection",
            "connect",
            "network",
            "timeout",
            "refused",
            "unreachable",
            "name or service not known",
            "failed to establish",
        )
    )


@lru_cache(maxsize=1)
def _get_local_whisper_model():
    from faster_whisper import WhisperModel

    logger.info("Loading local Whisper model (%s) on CPU", LOCAL_WHISPER_MODEL_SIZE)
    return WhisperModel(LOCAL_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_with_fireworks(audio_path: str, *, api_key: str) -> str:
    client = build_transcription_client(api_key)
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )
    return (response.text or "").strip()


def transcribe_with_local_whisper(audio_path: str) -> str:
    model = _get_local_whisper_model()
    segments, _info = model.transcribe(audio_path)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        logger.warning("Local Whisper returned an empty transcript for %s", audio_path)
    return text


def transcribe_audio(audio_path: str, *, api_key: str | None = None) -> tuple[str, str]:
    """
    Transcribe audio via Fireworks Whisper-v3, falling back to local Whisper on CPU.

    Returns (transcript, backend_label) where backend_label is
    ``"fireworks-whisper-v3"`` or ``"local-whisper-base"``.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    resolved_key = _resolve_api_key(api_key)

    try:
        text = transcribe_with_fireworks(path.as_posix(), api_key=resolved_key)
        if not text:
            raise ValueError("Fireworks Whisper returned an empty transcript.")
        return text, "fireworks-whisper-v3"
    except Exception as exc:
        if not _fireworks_audio_unavailable(exc):
            logger.exception("Fireworks transcription failed for %s", audio_path)
            raise RuntimeError(f"Speech-to-text failed: {exc}") from exc

        logger.warning(
            "Fireworks audio API unavailable (%s). Falling back to local Whisper.",
            exc,
        )
        try:
            text = transcribe_with_local_whisper(path.as_posix())
            if not text:
                raise ValueError("Local Whisper returned an empty transcript.")
            return text, "local-whisper-base"
        except Exception as local_exc:
            logger.exception("Local Whisper transcription failed for %s", audio_path)
            raise RuntimeError(
                "Fireworks audio is unavailable and local Whisper fallback failed: "
                f"{local_exc}"
            ) from local_exc


def _image_url_block(image_path: str) -> dict:
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}


def _describe_frame_with_vision(image_path: str, *, api_key: str) -> str | None:
    """Send a JPEG frame to the Fireworks vision model and return a scene description.

    Returns the description string on success, or None if anything goes wrong
    (the caller must then fall back to SILENT_VIDEO_BASELINE).
    """
    return _describe_frames_with_vision(
        [image_path],
        api_key=api_key,
        prompt="Describe what is happening in this video scene in 1-2 detailed, factual sentences.",
    )


def _describe_frames_with_vision(
    image_paths: list[str], *, api_key: str, prompt: str
) -> str | None:
    """Send one or more JPEG frames to the Fireworks vision model in a single call.

    Returns the description string on success, or None on any failure.
    """
    if not image_paths:
        return None
    try:
        client = build_client(api_key)
        content: list[dict] = [_image_url_block(path) for path in image_paths]
        content.append({"type": "text", "text": prompt})
        response = _chat_completion_with_retry(
            client,
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            temperature=0.3,
        )
        description = (response.choices[0].message.content or "").strip()
        if description:
            logger.info("Vision description obtained (%d chars, %d frame(s)).", len(description), len(image_paths))
            return description
        logger.warning("Vision model returned an empty description.")
        return None
    except Exception as exc:
        logger.warning("Vision description failed: %s", exc)
        return None


def get_visual_context(video_path: str, api_key: str, *, num_frames: int = 3) -> str | None:
    """
    Always-on visual grounding: sample a few frames spread across the video
    and ask the vision model to summarize what's actually shown on screen.

    This runs regardless of whether speech transcription succeeded, so
    captions can be grounded in real visual content (not just narration) —
    unlike a vision-only-on-silence fallback. Returns None on any failure;
    callers must treat this as best-effort and never let it block the pipeline.
    """
    frame_paths: list[str] = []
    try:
        frame_paths = extract_sample_frames(video_path, num_frames=num_frames)
        if not frame_paths:
            return None
        return _describe_frames_with_vision(
            frame_paths,
            api_key=api_key,
            prompt=(
                f"These {len(frame_paths)} frames are sampled evenly across a video, in "
                "chronological order. In 2-4 factual sentences, describe the visual "
                "content and how it changes over time (setting, subjects, on-screen "
                "text/UI, actions). Do not guess at audio or dialogue."
            ),
        )
    except Exception as exc:
        logger.warning("get_visual_context failed for %s: %s", video_path, exc)
        return None
    finally:
        for path in frame_paths:
            cleanup_audio_file(path)


def transcribe_with_vision_fallback(
    video_path: str, api_key: str, *, max_duration: float = 120.0
) -> tuple[str, str]:
    """
    Transcribe a video with the full 4-tier fallback chain:
    Fireworks Whisper-v3 -> local Whisper (CPU) -> vision-described midpoint frame ->
    static silent-video baseline.

    Returns (text, backend_label) where backend_label is one of
    "fireworks-whisper-v3", "local-whisper-base", "vision-fallback", "static-baseline".
    """
    audio_path: str | None = None
    try:
        try:
            audio_path = extract_audio_from_video(video_path, max_duration=max_duration)
        except ValueError:
            audio_path = None  # No audio track — treat as silent.
        except Exception as exc:
            logger.warning("Audio extraction failed for %s: %s", video_path, exc)
            audio_path = None

        transcript = ""
        backend = "static-baseline"
        if audio_path:
            try:
                transcript, backend = transcribe_audio(audio_path, api_key=api_key)
            except Exception as exc:
                logger.warning("Transcription failed for %s: %s", video_path, exc)
                transcript = ""

        if len(transcript.strip()) >= _MIN_TRANSCRIPT_CHARS:
            return transcript, backend

        logger.warning(
            "Transcript too short (%d chars) for %s; attempting vision fallback.",
            len(transcript.strip()),
            video_path,
        )
        frame_path: str | None = None
        try:
            fd, frame_tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            frame_path = frame_tmp

            saved_frame, frame_err = extract_midpoint_frame(video_path, frame_path)
            if frame_err or not saved_frame:
                logger.warning("Frame extraction failed for %s: %s", video_path, frame_err)
                return SILENT_VIDEO_BASELINE, "static-baseline"

            description = _describe_frame_with_vision(saved_frame, api_key=api_key)
            if description:
                return description, "vision-fallback"
            return SILENT_VIDEO_BASELINE, "static-baseline"
        finally:
            if frame_path:
                cleanup_audio_file(frame_path)
    finally:
        if audio_path:
            cleanup_audio_file(audio_path)
