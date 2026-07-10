"""Video audio extraction and frame-extraction utilities using MoviePy."""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


AUDIO_SUFFIX = ".mp3"


@contextmanager
def temp_audio_path(suffix: str = AUDIO_SUFFIX) -> Generator[str, None, None]:
    """Yield a temporary audio file path and remove it on exit."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Failed to remove temp audio file %s: %s", path, exc)


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    max_duration: float | None = None,
) -> str:
    """
    Extract the audio track from an MP4 (or other video) file.

    If *max_duration* is set and the clip is longer, only the first
    *max_duration* seconds are extracted (Track 2's 2-minute processing cap).

    Returns the path to the extracted audio file. Caller is responsible for
    cleanup when *output_path* is not provided (a temp file is created).
  """
    from moviepy.editor import VideoFileClip

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    owns_output = output_path is None
    if owns_output:
        fd, output_path = tempfile.mkstemp(suffix=AUDIO_SUFFIX)
        os.close(fd)
    else:
        output_path = str(output_path)

    clip = None
    try:
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            raise ValueError("Video has no audio track.")

        audio = clip.audio
        if max_duration is not None and clip.duration and clip.duration > max_duration:
            audio = clip.subclip(0, max_duration).audio

        audio.write_audiofile(
            output_path,
            fps=44100,
            codec="libmp3lame",
            verbose=False,
            logger=None,
        )
        return output_path
    except Exception:
        if owns_output and output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        raise
    finally:
        if clip is not None:
            clip.close()


def safe_extract_audio(
    video_bytes: bytes,
    suffix: str = ".mp4",
    max_duration: float | None = None,
) -> tuple[str | None, str | None, float | None]:
    """
    Write uploaded bytes to a temp video file, extract audio, and clean up the video.

    Returns (audio_path, error_message, original_duration_seconds). On success
    error_message is None. The caller must delete audio_path when finished.
    """
    video_fd, video_path = tempfile.mkstemp(suffix=suffix)
    os.close(video_fd)
    audio_path: str | None = None
    duration: float | None = None

    try:
        with open(video_path, "wb") as video_file:
            video_file.write(video_bytes)

        duration, duration_err = _clip_duration(video_path)
        if duration_err:
            logger.warning("Could not read video duration: %s", duration_err)

        audio_path = extract_audio_from_video(video_path, max_duration=max_duration)
        return audio_path, None, duration
    except ValueError as exc:
        return None, str(exc), duration
    except Exception as exc:
        logger.exception("Audio extraction failed")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        return None, f"Audio extraction failed: {exc}", duration
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


def _clip_duration(video_path: str | Path) -> tuple[float | None, str | None]:
    """Return (duration_seconds, error_message) for a video file."""
    from moviepy.editor import VideoFileClip

    clip = None
    try:
        clip = VideoFileClip(str(video_path))
        return clip.duration, None
    except Exception as exc:
        return None, str(exc)
    finally:
        if clip is not None:
            clip.close()


def duration_verdict(
    duration: float, min_sec: int = 30, max_sec: int = 120
) -> tuple[bool, str | None]:
    """
    Judge a clip duration against Track 2's 30s-2min compliance window.

    Returns (is_valid, message). Too-short clips are flagged invalid (callers
    may still choose to process them with a warning). Too-long clips are
    still considered valid but come with a truncation notice; callers should
    only process the first *max_sec* seconds.
    """
    if duration < min_sec:
        return False, (
            f"Video is {duration:.1f}s, shorter than the {min_sec}s Track 2 minimum. "
            "Processing will continue, but this clip does not meet the compliance window."
        )
    if duration > max_sec:
        return True, (
            f"Video is {duration:.1f}s, exceeding the {max_sec}s Track 2 cap. "
            f"Only the first {max_sec}s will be processed."
        )
    return True, None


def validate_duration(
    video_path: str | Path, min_sec: int = 30, max_sec: int = 120
) -> tuple[bool, str | None]:
    """Validate a video file's duration against the Track 2 compliance window."""
    video_path = Path(video_path)
    if not video_path.exists():
        return False, f"Video file not found: {video_path}"

    duration, error = _clip_duration(video_path)
    if error or duration is None:
        return False, f"Could not determine video duration: {error}"

    return duration_verdict(duration, min_sec=min_sec, max_sec=max_sec)


def cleanup_audio_file(audio_path: str | None) -> None:
    """Remove a single extracted audio file if it exists."""
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError as exc:
            logger.warning("Failed to remove audio file %s: %s", audio_path, exc)


def cleanup_temp_files(*paths: str | None) -> None:
    """Remove one or more temporary files (e.g. extracted MP3) after processing."""
    for path in paths:
        cleanup_audio_file(path)


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Client-type strategies to try in order. YouTube's default (web) client
# frequently fails with "The page needs to be reloaded" (SABR streaming
# rollout breaking extraction); android/ios/tv_embedded clients still
# return playable formats without a PO token. Trying several client
# combinations improves the odds of success on hosts YouTube treats with
# more suspicion (cloud/datacenter IPs).
_YT_CLIENT_STRATEGIES = [
    ["android", "ios"],
    ["tv_embedded", "web"],
    ["web_safari"],
]


def download_video(url: str, max_bytes: int = 500 * 1024 * 1024) -> str:
    """
    Download a video from a YouTube link or a direct video file URL to a temp
    MP4 file using yt-dlp (which also handles plain file URLs via its
    generic extractor).

    For YouTube URLs, tries several player-client strategies in sequence
    since YouTube's extraction behavior varies by source IP and changes
    frequently; if every strategy fails the final, most-informative error
    is raised (frequently caused by YouTube blocking the calling IP —
    common for cloud-hosted deployments — rather than a genuinely
    unavailable video).

    Returns the path to the downloaded MP4. Caller is responsible for
    cleanup via cleanup_audio_file/cleanup_temp_files.
    """
    import yt_dlp

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    os.remove(path)  # yt-dlp writes to this exact path via outtmpl

    is_youtube = "youtube.com" in url or "youtu.be" in url
    strategies = _YT_CLIENT_STRATEGIES if is_youtube else [None]

    # Optional proxy for hosts YouTube blocklists at the IP level (e.g.
    # Streamlit Community Cloud). Must be a residential/mobile proxy —
    # datacenter proxies are blocked by YouTube the same way cloud-host IPs
    # are, so they won't help here. Read from env first, falling back to
    # Streamlit secrets so it can be set without touching source or .env.
    proxy_url = os.getenv("YTDLP_PROXY_URL", "").strip()
    if not proxy_url:
        try:
            import streamlit as st

            proxy_url = str(st.secrets.get("YTDLP_PROXY_URL", "")).strip()
        except Exception:
            proxy_url = ""

    last_exc: Exception | None = None
    for player_clients in strategies:
        ydl_opts: dict = {
            "format": "mp4/best[ext=mp4]/best",
            "outtmpl": path,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": max_bytes,
            "http_headers": {"User-Agent": _BROWSER_USER_AGENT},
        }
        if player_clients:
            ydl_opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
        if proxy_url:
            ydl_opts["proxy"] = proxy_url

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
            last_exc = RuntimeError("Download produced no output file.")
        except Exception as exc:
            last_exc = exc
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            continue

    hint = (
        " This is commonly caused by YouTube blocking cloud-hosted server IPs "
        "rather than the video actually being unavailable — try a direct video "
        "file URL instead if this persists."
        if is_youtube
        else ""
    )
    raise RuntimeError(f"Failed to download video from {url}: {last_exc}.{hint}") from last_exc


def extract_midpoint_frame(
    video_path: str | Path,
    output_image_path: str | Path,
) -> tuple[str | None, str | None]:
    """
    Extract the frame at the video's midpoint and save it as a JPEG.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_image_path:
        Destination path for the output JPEG.

    Returns
    -------
    (output_path, None) on success; (None, error_message) on failure.
    """
    try:
        from moviepy.editor import VideoFileClip

        video_path = Path(video_path)
        if not video_path.exists():
            return None, f"Video file not found: {video_path}"

        clip = None
        try:
            clip = VideoFileClip(str(video_path))
            if clip.duration is None or clip.duration <= 0:
                return None, "Video has zero or unknown duration; cannot extract midpoint frame."
            return _save_frame_at(clip, clip.duration / 2.0, output_image_path)
        finally:
            if clip is not None:
                clip.close()

    except Exception as exc:
        logger.warning("extract_midpoint_frame failed for %s: %s", video_path, exc)
        return None, f"Frame extraction failed: {exc}"


def _save_frame_at(clip, timestamp: float, output_image_path: str | Path) -> tuple[str | None, str | None]:
    """Grab the frame at *timestamp* from an already-open MoviePy clip and save it as JPEG."""
    from PIL import Image
    import numpy as np

    frame: np.ndarray = clip.get_frame(timestamp)  # shape (H, W, 3), dtype uint8
    image = Image.fromarray(frame, mode="RGB")
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_image_path), format="JPEG", quality=85)
    logger.info("Frame (t=%.2fs) saved to %s", timestamp, output_image_path)
    return str(output_image_path), None


def extract_sample_frames(
    video_path: str | Path,
    num_frames: int = 3,
) -> list[str]:
    """
    Sample up to *num_frames* evenly-spaced frames across the video (skipping
    the very first/last instant) and save each as a temp JPEG.

    Used for always-on visual grounding: even when speech transcription
    succeeds, a few sampled frames let the vision model verify/ground the
    caption content in what's actually on screen, not just what's said.

    Returns a list of temp file paths (possibly shorter than num_frames, or
    empty on failure). Caller is responsible for deleting the returned files
    via cleanup_audio_file/cleanup_temp_files.
    """
    from moviepy.editor import VideoFileClip

    video_path = Path(video_path)
    if not video_path.exists() or num_frames <= 0:
        return []

    clip = None
    saved: list[str] = []
    try:
        clip = VideoFileClip(str(video_path))
        if clip.duration is None or clip.duration <= 0:
            return []

        # Evenly spaced timestamps strictly inside (0, duration), avoiding
        # black frames at the very start/end of the clip.
        fraction_step = 1.0 / (num_frames + 1)
        timestamps = [clip.duration * fraction_step * (i + 1) for i in range(num_frames)]

        for ts in timestamps:
            fd, frame_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            path, err = _save_frame_at(clip, ts, frame_path)
            if path:
                saved.append(path)
            else:
                logger.warning("Sample frame at t=%.2fs failed: %s", ts, err)
                cleanup_audio_file(frame_path)
        return saved
    except Exception as exc:
        logger.warning("extract_sample_frames failed for %s: %s", video_path, exc)
        for path in saved:
            cleanup_audio_file(path)
        return []
    finally:
        if clip is not None:
            clip.close()
