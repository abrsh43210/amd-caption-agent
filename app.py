"""
AMD Developer Hackathon 2026 — Track 2: Video Captioning via Fireworks AI.

Streamlit dashboard for MP4 upload, audio extraction, and multi-agent caption generation.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

from audio_transcriber import get_visual_context, transcribe_with_vision_fallback
from pipeline import (
    SERVERLESS_MODELS,
    build_client,
    generate_raw_context,
    mock_transcript,
    resolve_fireworks_api_key,
    run_caption_critic_loop,
)
from schemas import CriticEvaluation, TelemetrySummary
from video_processor import cleanup_temp_files, download_video, validate_duration

_BACKEND_LABELS = {
    "fireworks-whisper-v3": "Fireworks Whisper-v3",
    "local-whisper-base": "local Whisper (CPU)",
    "vision-fallback": "vision fallback (LLaMA Vision on midpoint frame)",
    "static-baseline": "static silent-video baseline",
}

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AMD Caption Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ED1C24;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .caption-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ED1C24;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .caption-card h3 {
        color: #f9fafb;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .caption-card p {
        color: #d1d5db;
        margin: 0;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .caption-card.formal { border-left-color: #3b82f6; }
    .caption-card.sarcastic { border-left-color: #a855f7; }
    .caption-card.humorous-tech { border-left-color: #10b981; }
    .caption-card.humorous-non-tech { border-left-color: #f59e0b; }
    .score-bar {
        background: #374151;
        border-radius: 6px;
        height: 8px;
        margin-top: 4px;
        overflow: hidden;
    }
    .score-fill {
        background: linear-gradient(90deg, #ED1C24, #f97316);
        height: 100%;
        border-radius: 6px;
    }
    .critic-panel {
        background: #111827;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #374151;
    }
    .badge-approved {
        background: #065f46;
        color: #6ee7b7;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-rejected {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .transcript-scroll {
        max-height: 220px;
        overflow-y: auto;
        border-radius: 8px;
        background: #111827;
        padding: 0.75rem 1rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #d1d5db;
        border: 1px solid #374151;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .transcript-scroll::-webkit-scrollbar {
        width: 6px;
    }
    .transcript-scroll::-webkit-scrollbar-thumb {
        background: #4b5563;
        border-radius: 3px;
    }
    .metric-inline {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.82rem;
        color: #6ee7b7;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AVAILABLE_MODELS = SERVERLESS_MODELS


def resolve_api_key(sidebar_key: str) -> str | None:
    return resolve_fireworks_api_key(sidebar_key)


def validate_api_key_ui(sidebar_key: str, resolved_key: str | None) -> None:
    """Surface helpful warnings when the sidebar key looks misconfigured."""
    sidebar = sidebar_key.strip()
    if sidebar and not sidebar.startswith("fw_"):
        st.sidebar.warning(
            "Sidebar key does not start with `fw_` — ignoring it and using `.env` / secrets instead."
        )
    elif sidebar and sidebar.startswith("fw_") and len(sidebar) <= 10:
        st.sidebar.warning("Sidebar key looks incomplete. Using `.env` / secrets if available.")
    if resolved_key and not resolved_key.startswith("fw_"):
        st.sidebar.error("Loaded Fireworks API key is invalid — it must start with `fw_`.")


def render_caption_card(title: str, text: str, css_class: str, score: float | None = None) -> None:
    score_html = ""
    if score is not None:
        pct = int(min(max(score, 0.0), 1.0) * 100)
        score_html = f"""
        <div style="margin-top:0.75rem;">
            <span style="color:#9ca3af;font-size:0.8rem;">Tonal score: {score:.2f}</span>
            <div class="score-bar"><div class="score-fill" style="width:{pct}%;"></div></div>
        </div>
        """
    st.markdown(
        f"""
        <div class="caption-card {css_class}">
            <h3>{title}</h3>
            <p>{text}</p>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(evaluation: CriticEvaluation) -> None:
    scores = evaluation.tonal_scores
    captions = evaluation.captions

    col1, col2 = st.columns(2)
    with col1:
        render_caption_card(
            "Formal",
            captions.formal,
            "formal",
            scores.get("formal"),
        )
        render_caption_card(
            "Humorous — Tech",
            captions.humorous_tech,
            "humorous-tech",
            scores.get("humorous_tech"),
        )
    with col2:
        render_caption_card(
            "Sarcastic",
            captions.sarcastic,
            "sarcastic",
            scores.get("sarcastic"),
        )
        render_caption_card(
            "Humorous — Non-Tech",
            captions.humorous_non_tech,
            "humorous-non-tech",
            scores.get("humorous_non_tech"),
        )

    badge = (
        '<span class="badge-approved">✓ Critic Approved</span>'
        if evaluation.approved
        else '<span class="badge-rejected">⚠ Not Fully Approved</span>'
    )
    st.markdown(
        f"""
        <div class="critic-panel">
            <h3 style="color:#f9fafb;margin-top:0;">Critic Evaluation {badge}</h3>
            <p style="color:#9ca3af;margin:0.75rem 0 0 0;line-height:1.6;">{evaluation.critique_notes}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def process_pipeline(
    *,
    video_path: str | None,
    video_name: str,
    transcript_override: str,
    use_mock: bool,
    api_key: str,
    model: str,
    max_retries: int,
) -> dict:
    """
    Run transcription (+ always-on visual grounding), context analysis, and
    the critic loop for one video or transcript. Used by the batch-links
    section so each link gets an independent, isolated pipeline run.

    Raises on unrecoverable failure (empty transcript with no mock/manual
    fallback, API errors, etc.) — callers should catch per-item.
    """
    telemetry = TelemetrySummary()
    stage_times: dict[str, float] = {}
    transcript = transcript_override.strip()
    stt_backend: str | None = None
    visual_context: str | None = None

    _t0 = time.perf_counter()
    if video_path and not transcript:
        transcript, stt_backend = transcribe_with_vision_fallback(video_path, api_key)
        if not transcript:
            if use_mock:
                transcript = mock_transcript(video_name)
            else:
                raise RuntimeError("Transcription returned an empty result.")
    elif not transcript and use_mock:
        transcript = mock_transcript(video_name)
    stage_times["transcribe"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    if video_path:
        visual_context = get_visual_context(video_path, api_key)
    client = build_client(api_key)
    context, context_telemetry = generate_raw_context(
        transcript, client=client, model=model, visual_context=visual_context
    )
    telemetry += context_telemetry
    stage_times["context"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    evaluation, critic_telemetry = run_caption_critic_loop(
        transcript, context, client=client, model=model, max_retries=max_retries
    )
    telemetry += critic_telemetry
    stage_times["generate_critic"] = time.perf_counter() - _t0

    return {
        "video_name": video_name,
        "transcript": transcript,
        "stt_backend": stt_backend,
        "visual_context": visual_context,
        "context": context,
        "evaluation": evaluation,
        "telemetry": telemetry,
        "stage_times": stage_times,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://www.amd.com/content/dam/code/images/header/amd-header-logo.svg", width=120)
    st.markdown("### Configuration")

    api_key_input = st.text_input(
        "Fireworks API Key",
        type="password",
        help="Leave blank to use `.env` or `st.secrets`. Enter a key starting with `fw_` to override.",
        placeholder="Leave blank to load from .env",
    )
    api_key = resolve_api_key(api_key_input)
    validate_api_key_ui(api_key_input, api_key)

    if api_key:
        st.session_state["fireworks_api_key"] = api_key

    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=0,
        format_func=lambda m: {
            "accounts/fireworks/models/gpt-oss-20b": "GPT-OSS 20B (fast, default)",
            "accounts/fireworks/models/gpt-oss-120b": "GPT-OSS 120B (high quality)",
        }.get(m, m),
    )
    max_retries = st.slider("Critic max retries", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown("**Transcript fallback**")
    use_mock = st.checkbox("Use mock transcript (UI testing)", value=False)
    manual_transcript = st.text_area(
        "Manual transcript override",
        height=120,
        placeholder="Paste transcript here if audio is unclear or missing…",
    )

    st.divider()
    st.caption("AMD Developer Hackathon 2026 · Track 2")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<p class="main-header">🎬 AMD Caption Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-agent video captioning powered by Fireworks AI</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload an MP4 video", type=["mp4"])

if st.button("Generate Captions", type="primary", disabled=not (uploaded or use_mock or manual_transcript.strip())):
    api_key = st.session_state.get("fireworks_api_key") or resolve_api_key(api_key_input)
    if not api_key:
        st.error("Please provide a Fireworks API key in `.env`, Streamlit secrets, or the sidebar.")
        st.stop()
    if not api_key.startswith("fw_"):
        st.error("Invalid Fireworks API key — it must start with `fw_`. Check your `.env` file.")
        st.stop()

    transcript = manual_transcript.strip()
    video_path: str | None = None
    video_name = "demo.mp4"
    context: str | None = None
    evaluation: CriticEvaluation | None = None
    telemetry = TelemetrySummary()
    stage_times: dict[str, float] = {}

    try:
        with st.status("Pipeline progress", expanded=True) as status:
            # Step 1 — Prepare video & validate duration
            _t0 = time.perf_counter()
            try:
                st.write("**[1/5] Preparing video…**")
                if uploaded is not None and not manual_transcript.strip():
                    video_name = uploaded.name
                    video_bytes = uploaded.read()
                    fd, video_path = tempfile.mkstemp(suffix=".mp4")
                    with os.fdopen(fd, "wb") as video_file:
                        video_file.write(video_bytes)

                    is_valid, duration_msg = validate_duration(video_path)
                    if duration_msg:
                        st.warning(duration_msg)
                    st.success(f"Video ready ({len(video_bytes) // 1024} KB).")
                elif uploaded is not None and manual_transcript.strip():
                    video_name = uploaded.name
                    st.info("Skipping video processing — using manual transcript override.")
                elif use_mock or transcript:
                    if not transcript:
                        transcript = mock_transcript(video_name)
                    st.info("Skipping video upload — using text/mock input.")
                else:
                    st.error("Upload a video or enable mock transcript.")
                    status.update(label="Pipeline failed", state="error")
                    st.stop()
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 1", state="error")
                st.stop()
            stage_times["extract"] = time.perf_counter() - _t0

            # Step 2 — Transcribe (Whisper with vision/static-baseline fallback)
            _t0 = time.perf_counter()
            try:
                if video_path and not transcript:
                    st.write("**[2/5] 🎵 Transcribing (Whisper + vision fallback)…**")
                    try:
                        transcript, stt_backend = transcribe_with_vision_fallback(video_path, api_key)
                        if not transcript:
                            raise ValueError("Transcription returned an empty result.")
                        st.success(
                            f"Transcription complete ({len(transcript.split())} words) "
                            f"via **{_BACKEND_LABELS.get(stt_backend, stt_backend)}**."
                        )
                    except Exception as exc:
                        st.warning(f"Transcription failed: {exc}")
                        if use_mock:
                            transcript = mock_transcript(video_name)
                            st.info("Falling back to mock transcript.")
                        else:
                            st.error(
                                "Could not transcribe audio. Paste a manual transcript in the sidebar "
                                "or enable mock transcript for testing."
                            )
                            status.update(label="Pipeline failed", state="error")
                            st.stop()
                elif transcript:
                    st.write("**[2/5] 🎵 Transcribing (Whisper + vision fallback)…**")
                    st.info("Skipped — using manual transcript override.")
                else:
                    st.write("**[2/5] 🎵 Transcribing (Whisper + vision fallback)…**")
                    st.info("Skipped — no video to transcribe.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 2", state="error")
                st.stop()
            stage_times["transcribe"] = time.perf_counter() - _t0

            # Step 3 — Analyze context (transcript + always-on visual grounding)
            _t0 = time.perf_counter()
            visual_context: str | None = None
            try:
                st.write("**[3/5] Analyzing context (audio + visual grounding)…**")
                if video_path:
                    visual_context = get_visual_context(video_path, api_key)
                    if visual_context:
                        st.caption("🖼️ Visual grounding: sampled frames analyzed and fused into context.")
                    else:
                        st.caption("🖼️ Visual grounding unavailable for this video; using transcript only.")
                client = build_client(api_key)
                context, context_telemetry = generate_raw_context(
                    transcript, client=client, model=model, visual_context=visual_context
                )
                telemetry += context_telemetry
                st.success("Context analysis complete.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 3", state="error")
                st.stop()
            stage_times["context"] = time.perf_counter() - _t0

            # Step 4 — Style agents + critic loop
            _t0 = time.perf_counter()
            try:
                st.write("**[4/5] Orchestrating style agents…**")
                evaluation, critic_telemetry = run_caption_critic_loop(
                    transcript,
                    context,
                    client=client,
                    model=model,
                    max_retries=max_retries,
                )
                telemetry += critic_telemetry
                st.success("Four copywriter personas drafted captions.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 4", state="error")
                st.stop()
            stage_times["generate_critic"] = time.perf_counter() - _t0

            # Step 5 — Critic verification
            try:
                st.write("**[5/5] Running critic verification…**")
                if evaluation.approved:
                    st.success("Critic approved all caption styles.")
                else:
                    st.warning("Critic completed with reservations — see notes below.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 5", state="error")
                st.stop()

            status.update(label="Pipeline complete", state="complete")

        with st.expander("Transcript", expanded=False):
            # Stats row + full-width code block (Streamlit's st.code includes a native copy button)
            st.caption(f"{len(transcript.split())} words · {len(transcript)} characters")
            st.code(transcript, language=None)

        if visual_context:
            with st.expander("Visual grounding (sampled frames)", expanded=False):
                st.markdown(visual_context)

        with st.expander("Context analysis", expanded=False):
            st.markdown(context)

        st.markdown("### Generated Captions")
        render_results(evaluation)

        st.markdown("#### Telemetry")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total tokens", f"{telemetry.total_tokens:,}")
        metric_cols[1].metric("Prompt tokens", f"{telemetry.prompt_tokens:,}")
        metric_cols[2].metric("Completion tokens", f"{telemetry.completion_tokens:,}")
        metric_cols[3].metric("API calls", telemetry.calls)
        st.caption(
            " · ".join(f"{stage}: {elapsed:.1f}s" for stage, elapsed in stage_times.items())
        )

    except Exception as exc:
        logger.exception("Pipeline error")
        st.error(f"Error details: {str(exc)}")
        st.code(traceback.format_exc())
    finally:
        cleanup_temp_files(video_path)

st.markdown("---")
st.markdown("### 🔗 Batch: YouTube / Video Links")
st.caption(
    "Paste one or more YouTube links or direct video URLs (one per line). "
    "Each is downloaded and run through the full pipeline independently."
)
links_input = st.text_area(
    "Video links",
    height=100,
    placeholder="https://www.youtube.com/watch?v=...\nhttps://example.com/clip.mp4",
    key="batch_links",
    label_visibility="collapsed",
)

MAX_BATCH_LINKS = 5

if st.button("Process Links", disabled=not links_input.strip()):
    batch_api_key = st.session_state.get("fireworks_api_key") or resolve_api_key(api_key_input)
    if not batch_api_key or not batch_api_key.startswith("fw_"):
        st.error("Please provide a valid Fireworks API key (starts with `fw_`) before processing links.")
        st.stop()

    urls = list(dict.fromkeys(u.strip() for u in links_input.splitlines() if u.strip()))
    if len(urls) > MAX_BATCH_LINKS:
        st.warning(f"Only the first {MAX_BATCH_LINKS} links will be processed in this batch.")
        urls = urls[:MAX_BATCH_LINKS]

    for idx, url in enumerate(urls, start=1):
        dl_path: str | None = None
        result: dict | None = None
        with st.status(f"[{idx}/{len(urls)}] {url}", expanded=True) as link_status:
            try:
                st.write("Downloading video…")
                dl_path = download_video(url)
                is_valid, duration_msg = validate_duration(dl_path)
                if duration_msg:
                    st.warning(duration_msg)

                st.write("Transcribing, grounding in visual frames, analyzing context, orchestrating agents…")
                result = process_pipeline(
                    video_path=dl_path,
                    video_name=url,
                    transcript_override="",
                    use_mock=use_mock,
                    api_key=batch_api_key,
                    model=model,
                    max_retries=max_retries,
                )
                link_status.update(label=f"✓ {url}", state="complete")
            except Exception as exc:
                logger.exception("Batch link failed: %s", url)
                link_status.update(label=f"✗ Failed — {url}", state="error")
                st.error(f"Error details: {str(exc)}")
            finally:
                cleanup_temp_files(dl_path)

        if result is not None:
            with st.expander(f"Results — {url}", expanded=True):
                backend_note = (
                    f" · via **{_BACKEND_LABELS.get(result['stt_backend'], result['stt_backend'])}**"
                    if result["stt_backend"]
                    else ""
                )
                st.caption(f"{len(result['transcript'].split())} words{backend_note}")
                if result["visual_context"]:
                    with st.expander("Visual grounding (sampled frames)", expanded=False):
                        st.markdown(result["visual_context"])
                render_results(result["evaluation"])
                t = result["telemetry"]
                st.caption(
                    f"Tokens: {t.total_tokens:,} total ({t.calls} API calls) · "
                    + " · ".join(f"{stage}: {elapsed:.1f}s" for stage, elapsed in result["stage_times"].items())
                )

if not (uploaded or use_mock or manual_transcript.strip() or links_input.strip()):
    st.info("Upload an MP4, paste video links above, or click **Generate Captions** to begin.")

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        """
        1. **Prepare video** — validate duration against Track 2's 30s-2min window (over-long clips are auto-truncated to the first 2 minutes)
        2. **Transcribe & Analyze Vision**: Transcribe speech via Whisper (Fireworks/local fallback). If the video is silent, we automatically extract a midpoint keyframe and use Fireworks LLaMA Vision to describe the scene, falling back to a static baseline caption if that fails too.
        3. **Analyze context (audio + vision fusion)** — themes, mood, and technical jargon from the transcript, always fused with a Vision-LLM description of frames sampled across the *whole* video — so captions are grounded in what's actually shown, not just what's said, even when speech transcription succeeds
        4. **Orchestrate agents** — Formal, Sarcastic, Humorous-Tech, Humorous-Non-Tech
        5. **Critic loop** — automated quality gate (LLM critic + embedding-based distinctness check) with up to 3 self-correction passes
        """
    )
