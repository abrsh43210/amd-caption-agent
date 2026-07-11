"""Headless batch runner for the AMD caption-agent pipeline.

Reads tasks from /input/tasks.json, generates captions for each video, and
writes results atomically to /output/results.json after every task completes.

Usage:
    python run_headless.py

Environment:
    FIREWORKS_API_KEY  – Fireworks API key (loaded from .env or system env).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
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
from audio_transcriber import get_visual_context, transcribe_with_vision_fallback
from pipeline import (
    build_client,
    generate_raw_context,
    resolve_fireworks_api_key,
    run_caption_critic_loop,
)
from schemas import TelemetrySummary
from video_processor import cleanup_temp_files, validate_duration

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

# Maximum bytes to download for a single video (default 500MB, overridable).
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(500 * 1024 * 1024)))

# Soft wall-clock budget for the whole batch (default 9 min, leaving a buffer
# under the harness's 10-minute hard timeout). Once elapsed time exceeds this,
# remaining tasks are skipped rather than started, so the process can still
# exit cleanly with whatever results were already written instead of being
# killed mid-task.
BATCH_DEADLINE_SECONDS = float(os.getenv("BATCH_DEADLINE_SECONDS", "540"))


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

    Enforces a connect/read timeout and a maximum download size
    (MAX_DOWNLOAD_BYTES) to avoid hanging or unbounded downloads.
    The caller is responsible for deleting the file when done.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    logger.info("Downloading video: %s -> %s", url, tmp_path)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError(
                    f"Video at {url} declares {content_length} bytes, "
                    f"exceeding MAX_DOWNLOAD_BYTES ({MAX_DOWNLOAD_BYTES})."
                )

            downloaded = 0
            chunk_size = 1024 * 1024
            with open(tmp_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError(
                            f"Video at {url} exceeded MAX_DOWNLOAD_BYTES "
                            f"({MAX_DOWNLOAD_BYTES}) during download; aborting."
                        )
                    out_file.write(chunk)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

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
    telemetry = TelemetrySummary()

    try:
        # 1. Download video
        video_path = _download_video(video_url)

        # 2. Validate duration against the Track 2 30s-2min compliance window
        is_valid, duration_msg = validate_duration(video_path)
        if duration_msg:
            log_fn = logger.info if is_valid else logger.warning
            log_fn("[%s] %s", task_id, duration_msg)

        # 3. Transcribe with the full 4-tier fallback (Fireworks -> local Whisper ->
        #    vision -> static baseline); truncates processing to the first 120s.
        logger.info("[%s] Transcribing (with vision fallback)...", task_id)
        transcript, backend = transcribe_with_vision_fallback(video_path, api_key)
        logger.info(
            "[%s] Transcription via '%s': %d chars.", task_id, backend, len(transcript)
        )

        # 4. Visual grounding — always sample a few frames so captions reflect
        #    what's actually on screen, not just the transcript (best-effort;
        #    never blocks the pipeline on failure).
        logger.info("[%s] Sampling frames for visual grounding...", task_id)
        visual_context = get_visual_context(video_path, api_key)
        logger.info(
            "[%s] Visual context: %s", task_id, "obtained" if visual_context else "unavailable"
        )

        # 5. Context generation
        logger.info("[%s] Generating context summary...", task_id)
        client = build_client(api_key)
        context, context_telemetry = generate_raw_context(
            transcript, client=client, visual_context=visual_context
        )
        telemetry += context_telemetry
        logger.info("[%s] Context summary (%d chars) ready.", task_id, len(context))

        # 6. Self-correcting critic loop
        logger.info("[%s] Running caption critic loop...", task_id)
        evaluation, critic_telemetry = run_caption_critic_loop(transcript, context, client=client)
        telemetry += critic_telemetry
        logger.info(
            "[%s] Critic loop finished. Approved=%s, scores=%s, total_tokens=%d",
            task_id,
            evaluation.approved,
            evaluation.tonal_scores,
            telemetry.total_tokens,
        )

        # 7. Build output dict
        captions = evaluation.captions
        return {
            "formal": captions.formal,
            "sarcastic": captions.sarcastic,
            "humorous_tech": captions.humorous_tech,
            "humorous_non_tech": captions.humorous_non_tech,
        }

    finally:
        # 8. Cleanup temporary files
        logger.info("[%s] Cleaning up temporary files.", task_id)
        cleanup_temp_files(video_path)


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
    skipped = 0
    start_time = time.monotonic()

    for index, task in enumerate(tasks, start=1):
        elapsed = time.monotonic() - start_time
        if elapsed >= BATCH_DEADLINE_SECONDS:
            remaining = total - index + 1
            logger.warning(
                "Batch deadline (%.0fs) reached after %d/%d task(s); "
                "skipping remaining %d task(s) so the process can exit cleanly "
                "with results collected so far.",
                BATCH_DEADLINE_SECONDS,
                index - 1,
                total,
                remaining,
            )
            skipped += remaining
            break

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
        "=== Batch complete: %d succeeded, %d failed, %d skipped out of %d total. ===",
        succeeded,
        failed,
        skipped,
        total,
    )
    # Always exit 0 once we've reached this point: results.json already holds
    # every successfully completed task, written atomically as it went. A
    # per-clip failure (or a deadline-triggered skip) must not zero out an
    # otherwise-successful batch by signaling total process failure.


if __name__ == "__main__":
    main()
