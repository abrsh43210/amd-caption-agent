"""Headless batch runner for the AMD caption-agent pipeline.

Reads tasks from /input/tasks.json, generates captions for each video, and
writes results atomically to /output/results.json after every task completes.

Usage:
    python run_headless.py

Environment:
    FIREWORKS_API_KEY  – Fireworks API key (loaded from .env or system env).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: load .env before any module that reads env vars
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)  # system env wins; .env fills missing vars
except ImportError:
    pass  # python-dotenv optional; rely on system env

# ---------------------------------------------------------------------------
# Project imports (after env is populated)
# ---------------------------------------------------------------------------
from audio_transcriber import transcribe_audio
from pipeline import (
    build_client,
    generate_raw_context,
    resolve_fireworks_api_key,
    run_caption_critic_loop,
)
from video_processor import cleanup_temp_files, extract_midpoint_frame, safe_extract_audio

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_headless")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
INPUT_TASKS_PATH = Path("/input/tasks.json")
OUTPUT_RESULTS_PATH = Path("/output/results.json")

# Fallback transcript used when a video has no audible speech AND vision
# description also fails, so caption generation never crashes.
SILENT_VIDEO_BASELINE = (
    "[Silent video detected — no audible speech. "
    "Visual content only; captions based on contextual inference.]"
)

# Fireworks vision model used for silent-video scene description.
VISION_MODEL = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

# Minimum character length below which a transcript is considered empty/silent.
_MIN_TRANSCRIPT_CHARS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    """Resolve the Fireworks API key or abort with a clear error."""
    key = resolve_fireworks_api_key()
    if not key:
        logger.error(
            "FIREWORKS_API_KEY is not set. "
            "Export it as an environment variable or add it to .env."
        )
        sys.exit(1)
    return key


def _load_tasks() -> list[dict[str, Any]]:
    """Load and return the task list from /input/tasks.json."""
    if not INPUT_TASKS_PATH.exists():
        logger.error("Task file not found: %s", INPUT_TASKS_PATH)
        sys.exit(1)

    try:
        with INPUT_TASKS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", INPUT_TASKS_PATH, exc)
        sys.exit(1)

    if not isinstance(data, list):
        logger.error(
            "Expected a JSON array in %s, got %s.", INPUT_TASKS_PATH, type(data).__name__
        )
        sys.exit(1)

    logger.info("Loaded %d task(s) from %s.", len(data), INPUT_TASKS_PATH)
    return data


def _extract_task_fields(task: dict[str, Any], index: int) -> tuple[str, str] | None:
    """Return (task_id, video_url) from a task dict using safe fallback key lookups.

    Supported schemas
    -----------------
    - {"id": "video1", "video_url": "https://..."}
    - {"task_id": "1", "url": "https://..."}
    - Any combination of the above key names.

    Returns None and logs a warning if the required fields cannot be resolved.
    """
    # Resolve task identifier
    task_id: str | None = (
        task.get("task_id")
        or task.get("id")
        or task.get("taskId")
    )
    if not task_id:
        # Fall back to positional index so the loop can continue
        task_id = str(index)
        logger.warning(
            "Task at index %d has no 'id' / 'task_id' field; using index '%s'.",
            index,
            task_id,
        )

    # Resolve video URL
    video_url: str | None = (
        task.get("video_url")
        or task.get("url")
        or task.get("videoUrl")
        or task.get("video_link")
    )
    if not video_url:
        logger.warning(
            "Task '%s' (index %d) has no video URL field; skipping.", task_id, index
        )
        return None

    return str(task_id), str(video_url)


def _download_video(url: str) -> str:
    """Download *url* to a temporary MP4 file and return its path.

    The caller is responsible for deleting the file when done.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    logger.info("Downloading video: %s -> %s", url, tmp_path)
    urllib.request.urlretrieve(url, tmp_path)
    return tmp_path


def _read_results() -> dict[str, Any]:
    """Read the existing results file, returning an empty dict if absent or corrupt."""
    if not OUTPUT_RESULTS_PATH.exists():
        return {}
    try:
        with OUTPUT_RESULTS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read existing %s (%s); starting from empty results.",
            OUTPUT_RESULTS_PATH,
            exc,
        )
        return {}


def _write_results_atomic(results: dict[str, Any]) -> None:
    """Write *results* to OUTPUT_RESULTS_PATH atomically via a sibling temp file."""
    OUTPUT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=OUTPUT_RESULTS_PATH.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        # Atomic rename — replaces the target if it already exists
        os.replace(tmp_path, OUTPUT_RESULTS_PATH)
    except Exception:
        # Clean up partial temp file on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _describe_frame_with_vision(image_path: str, *, api_key: str) -> str | None:
    """Send a JPEG frame to the Fireworks vision model and return a scene description.

    Returns the description string on success, or None if anything goes wrong
    (the caller must then fall back to SILENT_VIDEO_BASELINE).
    """
    try:
        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        from pipeline import build_client  # already imported at module level, re-use
        client = build_client(api_key)

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe what is happening in this video scene "
                                "in 1-2 detailed, factual sentences."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.3,
        )
        description = (response.choices[0].message.content or "").strip()
        if description:
            logger.info("Vision description obtained (%d chars).", len(description))
            return description
        logger.warning("Vision model returned an empty description.")
        return None
    except Exception as exc:
        logger.warning("Vision description failed: %s", exc)
        return None


def _merge_and_persist(task_id: str, caption_data: dict[str, str]) -> None:
    """Merge one task's result into the output file and save atomically."""
    results = _read_results()
    results[task_id] = caption_data
    _write_results_atomic(results)
    logger.info("Results for task '%s' written to %s.", task_id, OUTPUT_RESULTS_PATH)


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

def _process_task(task_id: str, video_url: str, *, api_key: str) -> dict[str, str]:
    """Download, transcribe, and caption one video.  Returns the caption dict."""
    video_path: str | None = None
    audio_path: str | None = None
    frame_path: str | None = None

    try:
        # 1. Download video
        video_path = _download_video(video_url)

        # 2. Extract audio
        logger.info("[%s] Extracting audio...", task_id)
        with open(video_path, "rb") as vf:
            video_bytes = vf.read()

        audio_path, audio_error = safe_extract_audio(video_bytes)
        if audio_error:
            logger.warning(
                "[%s] Audio extraction failed: %s. Using silent baseline.", task_id, audio_error
            )

        # 3. Transcribe
        transcript = ""
        if audio_path:
            logger.info("[%s] Transcribing audio...", task_id)
            try:
                transcript, backend = transcribe_audio(audio_path, api_key=api_key)
                logger.info(
                    "[%s] Transcription via '%s': %d chars.", task_id, backend, len(transcript)
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Transcription error: %s. Using silent baseline.", task_id, exc
                )

        # Gracefully handle silent / untranscribable video
        if len(transcript.strip()) < _MIN_TRANSCRIPT_CHARS:
            logger.warning(
                "[%s] Transcript too short (%d chars); attempting vision fallback.",
                task_id,
                len(transcript.strip()),
            )
            visual_description: str | None = None
            try:
                # Extract the midpoint frame from the downloaded video
                fd, frame_tmp = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                frame_path = frame_tmp

                saved_frame, frame_err = extract_midpoint_frame(video_path, frame_path)
                if frame_err or not saved_frame:
                    logger.warning(
                        "[%s] Frame extraction failed: %s", task_id, frame_err
                    )
                else:
                    visual_description = _describe_frame_with_vision(
                        saved_frame, api_key=api_key
                    )
            except Exception as vision_exc:
                logger.warning(
                    "[%s] Vision fallback raised an unexpected error: %s",
                    task_id,
                    vision_exc,
                )

            if visual_description:
                logger.info(
                    "[%s] Using vision-derived description as transcript baseline.",
                    task_id,
                )
                transcript = visual_description
            else:
                logger.warning(
                    "[%s] Vision fallback unavailable; using static silent baseline.",
                    task_id,
                )
                transcript = SILENT_VIDEO_BASELINE

        # 4. Context generation
        logger.info("[%s] Generating context summary...", task_id)
        client = build_client(api_key)
        context = generate_raw_context(transcript, client=client)
        logger.info("[%s] Context summary (%d chars) ready.", task_id, len(context))

        # 5. Self-correcting critic loop
        logger.info("[%s] Running caption critic loop...", task_id)
        evaluation = run_caption_critic_loop(transcript, context, client=client)
        logger.info(
            "[%s] Critic loop finished. Approved=%s, scores=%s",
            task_id,
            evaluation.approved,
            evaluation.tonal_scores,
        )

        # 6. Build output dict
        captions = evaluation.captions
        return {
            "formal": captions.formal,
            "sarcastic": captions.sarcastic,
            "humorous_tech": captions.humorous_tech,
            "humorous_non_tech": captions.humorous_non_tech,
        }

    finally:
        # 7. Cleanup temporary files
        logger.info("[%s] Cleaning up temporary files.", task_id)
        files_to_remove: list[str | None] = [audio_path, frame_path]
        if video_path:
            files_to_remove.append(video_path)
        cleanup_temp_files(*files_to_remove)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== AMD Caption Agent - Headless Batch Runner ===")

    api_key = _resolve_api_key()
    tasks = _load_tasks()

    total = len(tasks)
    succeeded = 0
    failed = 0

    for index, task in enumerate(tasks, start=1):
        fields = _extract_task_fields(task, index - 1)
        if fields is None:
            failed += 1
            continue

        task_id, video_url = fields
        logger.info(
            "-- Task %d/%d  id='%s'  url='%s'", index, total, task_id, video_url
        )

        try:
            caption_data = _process_task(task_id, video_url, api_key=api_key)
            _merge_and_persist(task_id, caption_data)
            succeeded += 1
            logger.info("[%s] Done.", task_id)

        except Exception as exc:
            logger.exception("[%s] Failed: %s", task_id, exc)
            failed += 1
            # Continue with remaining tasks; do not write partial data for this one.

    logger.info(
        "=== Batch complete: %d succeeded, %d failed out of %d total. ===",
        succeeded,
        failed,
        total,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
