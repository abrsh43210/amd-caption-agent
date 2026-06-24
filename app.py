"""
AMD Developer Hackathon 2026 — Track 2: Video Captioning via Fireworks AI.

Streamlit dashboard for MP4 upload, audio extraction, and multi-agent caption generation.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from audio_transcriber import transcribe_audio
from pipeline import (
    SERVERLESS_MODELS,
    build_client,
    generate_raw_context,
    mock_transcript,
    resolve_fireworks_api_key,
    run_caption_critic_loop,
)
from schemas import CriticEvaluation
from video_processor import cleanup_temp_files, safe_extract_audio

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
    audio_path: str | None = None
    video_name = "demo.mp4"
    context: str | None = None
    evaluation: CriticEvaluation | None = None

    try:
        with st.status("Pipeline progress", expanded=True) as status:
            # Step 1 — Extract audio
            try:
                st.write("**[1/5] Extracting audio…**")
                if uploaded is not None and not manual_transcript.strip():
                    video_name = uploaded.name
                    video_bytes = uploaded.read()
                    audio_path, audio_error = safe_extract_audio(video_bytes)
                    if audio_error:
                        st.warning(f"Audio issue: {audio_error}")
                        if not transcript:
                            if use_mock:
                                transcript = mock_transcript(video_name)
                                st.info("Using mock transcript for testing.")
                            else:
                                st.error(
                                    "No usable audio and no manual transcript. "
                                    "Enable mock transcript or paste text in the sidebar."
                                )
                                status.update(label="Pipeline failed", state="error")
                                st.stop()
                    elif audio_path:
                        st.success(f"Audio extracted ({Path(audio_path).stat().st_size // 1024} KB MP3).")
                elif uploaded is not None and manual_transcript.strip():
                    video_name = uploaded.name
                    st.info("Skipping audio extraction — using manual transcript override.")
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

            # Step 2 — Transcribe audio via Whisper-v3
            try:
                if audio_path and not transcript:
                    st.write("**[2/5] 🎵 Transcribing audio via Whisper…**")
                    try:
                        transcript, stt_backend = transcribe_audio(audio_path, api_key=api_key)
                        if transcript:
                            st.success(
                                f"Transcription complete ({len(transcript.split())} words) "
                                f"via **{stt_backend}**."
                            )
                            if stt_backend == "local-whisper-base":
                                st.info(
                                    "Fireworks serverless audio was unavailable (deprecated June 2026). "
                                    "Used local Whisper on CPU instead."
                                )
                        else:
                            raise ValueError("Whisper returned an empty transcript.")
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
                    finally:
                        cleanup_temp_files(audio_path)
                        audio_path = None
                elif transcript:
                    st.write("**[2/5] 🎵 Transcribing audio via Whisper…**")
                    st.info("Skipped — using manual transcript override.")
                else:
                    st.write("**[2/5] 🎵 Transcribing audio via Whisper…**")
                    st.info("Skipped — no audio file to transcribe.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 2", state="error")
                st.stop()

            # Step 3 — Analyze context
            try:
                st.write("**[3/5] Analyzing context…**")
                client = build_client(api_key)
                context = generate_raw_context(transcript, client=client, model=model)
                st.success("Context analysis complete.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 3", state="error")
                st.stop()

            # Step 4 — Style agents + critic loop
            try:
                st.write("**[4/5] Orchestrating style agents…**")
                evaluation = run_caption_critic_loop(
                    transcript,
                    context,
                    client=client,
                    model=model,
                    max_retries=max_retries,
                )
                st.success("Four copywriter personas drafted captions.")
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                st.code(traceback.format_exc())
                status.update(label="Pipeline failed at step 4", state="error")
                st.stop()

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
            st.markdown(transcript)

        with st.expander("Context analysis", expanded=False):
            st.markdown(context)

        st.markdown("### Generated Captions")
        render_results(evaluation)

    except Exception as exc:
        logger.exception("Pipeline error")
        st.error(f"Error details: {str(exc)}")
        st.code(traceback.format_exc())
    finally:
        cleanup_temp_files(audio_path)

else:
    st.info("Upload an MP4 and click **Generate Captions** to begin.")

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        """
        1. **Extract audio** from your MP4 with MoviePy (MP3)
        2. **Transcribe** speech via Whisper (Fireworks when available, local CPU fallback)
        3. **Analyze context** — themes, mood, and technical jargon
        4. **Orchestrate agents** — Formal, Sarcastic, Humorous-Tech, Humorous-Non-Tech
        5. **Critic loop** — automated quality gate with up to 3 self-correction passes
        """
    )
