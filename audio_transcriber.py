"""Speech-to-text transcription with Fireworks API and local Whisper fallback."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from openai import APIStatusError, OpenAI

from pipeline import resolve_fireworks_api_key

logger = logging.getLogger(__name__)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
WHISPER_MODEL = "whisper-v3"
LOCAL_WHISPER_MODEL_SIZE = "base"


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
