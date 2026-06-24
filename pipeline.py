"""Fireworks AI pipeline for context analysis and caption generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from schemas import CriticEvaluation

logger = logging.getLogger(__name__)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-20b"
SERVERLESS_MODELS = [
    "accounts/fireworks/models/gpt-oss-20b",    # Standard / fast (default)
    "accounts/fireworks/models/gpt-oss-120b",   # High-tier upgrade
]

COPYWRITER_SYSTEM_PROMPT = """You are a multi-agent caption orchestrator for AMD Developer Hackathon video content.

You manage FOUR distinct copywriter personas. Each must produce a UNIQUE caption for the same video transcript.
Captions should be concise (1-3 sentences), engaging, and suitable as social/video descriptions.

## Persona 1: Formal
- Tone: Professional, authoritative, polished.
- Audience: Press, enterprise developers, official channels.
- Style: Clear value proposition, no slang, no emojis.

## Persona 2: Sarcastic
- Tone: Dry wit, subtle irony, deadpan confidence.
- Audience: Developers who appreciate sharp humor without being mean.
- Style: Understated punchlines; never offensive or cruel.

## Persona 3: Humorous-Tech
- Tone: Playful insider humor.
- Audience: Engineers, hackathon participants, tech Twitter.
- Style: Light memes, dev jargon, references to GPUs, APIs, debugging — keep it fun not cringe.

## Persona 4: Humorous-Non-Tech
- Tone: Warm, accessible, universally funny.
- Audience: General public, casual viewers.
- Style: Plain language, relatable analogies, no technical jargon.

## Critic Role (same response)
After drafting all four captions, act as a strict critic:
- Score each style's tonal alignment from 0.0 to 1.0 in `tonal_scores` (keys: formal, sarcastic, humorous_tech, humorous_non_tech).
- Write actionable `critique_notes` if any caption misses its persona or is too similar to another.
- Set `approved` to true ONLY when all four captions are distinct, on-brand, and score >= 0.75 each.
"""

MOCK_TRANSCRIPT = (
    "Welcome to the AMD Developer Hackathon 2026. Today we're showcasing a video captioning "
    "pipeline powered by Fireworks AI and Llama models running on high-performance hardware. "
    "Developers upload an MP4, we extract context from the audio, and multi-agent workflows "
    "generate formal, sarcastic, and humorous captions validated by an automated critic loop."
)


def resolve_fireworks_api_key(sidebar_override: str | None = None) -> str | None:
    """
    Resolve Fireworks API key with priority:
    valid sidebar override > FIREWORKS_API_KEY env > st.secrets.
    """
    override = (sidebar_override or "").strip()
    if override.startswith("fw_") and len(override) > 10:
        return override

    key = os.getenv("FIREWORKS_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        if "FIREWORKS_API_KEY" in st.secrets:
            return st.secrets["FIREWORKS_API_KEY"]
    except Exception:
        pass
    return None


def get_api_key() -> str | None:
    """Resolve Fireworks API key from environment or Streamlit secrets."""
    return resolve_fireworks_api_key()


def build_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL)


def mock_transcript(video_name: str = "demo.mp4") -> str:
    """Fallback transcript for UI testing when audio is missing or unclear."""
    return f"[Mock transcript for {video_name}]\n\n{MOCK_TRANSCRIPT}"


def _critic_response_format() -> dict[str, Any]:
    schema = CriticEvaluation.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "critic_evaluation",
            "schema": schema,
            "strict": True,
        },
    }


def generate_raw_context(
    transcript: str,
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize themes, mood, and technical jargon from a transcript."""
    if not transcript.strip():
        return "No transcript provided. Unable to analyze context."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze video transcripts for a captioning pipeline. "
                    "Summarize in 3-5 bullet points: main themes, emotional mood, "
                    "key technical jargon, and target audience signals. Be concise."
                ),
            },
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ],
        temperature=0.4,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


def _request_critic_evaluation(
    *,
    client: OpenAI,
    model: str,
    transcript: str,
    context: str,
    prior_feedback: str | None = None,
) -> CriticEvaluation:
    user_content = (
        f"## Transcript\n{transcript}\n\n"
        f"## Context Analysis\n{context}\n\n"
        "Generate all four caption styles and perform critic evaluation. "
        "Respond with valid JSON matching the schema."
    )
    if prior_feedback:
        user_content += f"\n\n## Prior Critic Feedback (address these issues)\n{prior_feedback}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COPYWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_critic_response_format(),
        temperature=0.7,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content or "{}"
    try:
        return CriticEvaluation.model_validate_json(raw)
    except ValidationError:
        payload = json.loads(raw)
        return CriticEvaluation.model_validate(payload)


def run_caption_critic_loop(
    transcript: str,
    context: str,
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> CriticEvaluation:
    """
    Run copywriter + critic agents with self-correction up to max_retries.

    If the critic rejects captions, feedback is fed back into the next attempt.
    """
    prior_feedback: str | None = None
    last_evaluation: CriticEvaluation | None = None

    for attempt in range(1, max_retries + 1):
        logger.info("Critic loop attempt %s/%s", attempt, max_retries)
        evaluation = _request_critic_evaluation(
            client=client,
            model=model,
            transcript=transcript,
            context=context,
            prior_feedback=prior_feedback,
        )
        last_evaluation = evaluation

        if evaluation.approved:
            return evaluation

        prior_feedback = evaluation.critique_notes
        logger.info("Critic rejected attempt %s: %s", attempt, prior_feedback)

    if last_evaluation is not None:
        return last_evaluation

    raise RuntimeError("Critic loop completed without producing an evaluation.")


def run_full_pipeline(
    transcript: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> tuple[str, CriticEvaluation]:
    """Execute context analysis and caption critic loop end-to-end."""
    client = build_client(api_key)
    context = generate_raw_context(transcript, client=client, model=model)
    evaluation = run_caption_critic_loop(
        transcript,
        context,
        client=client,
        model=model,
        max_retries=max_retries,
    )
    return context, evaluation
