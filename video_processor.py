"""Video audio extraction utilities using MoviePy."""

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


def extract_audio_from_video(video_path: str | Path, output_path: str | Path | None = None) -> str:
    """
    Extract the audio track from an MP4 (or other video) file.

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

        clip.audio.write_audiofile(
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


def safe_extract_audio(video_bytes: bytes, suffix: str = ".mp4") -> tuple[str | None, str | None]:
    """
    Write uploaded bytes to a temp video file, extract audio, and clean up the video.

    Returns (audio_path, error_message). On success error_message is None.
    The caller must delete audio_path when finished.
    """
    video_fd, video_path = tempfile.mkstemp(suffix=suffix)
    os.close(video_fd)
    audio_path: str | None = None

    try:
        with open(video_path, "wb") as video_file:
            video_file.write(video_bytes)

        audio_path = extract_audio_from_video(video_path)
        return audio_path, None
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        logger.exception("Audio extraction failed")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass
        return None, f"Audio extraction failed: {exc}"
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


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
