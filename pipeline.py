"""Fireworks AI pipeline for context analysis and caption generation."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from schemas import CaptionSet, CriticEvaluation, TelemetrySummary

logger = logging.getLogger(__name__)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-20b"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
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
- MUST NOT: use puns, use exclamation points, or land the punchline on a piece of tech jargon. The
  humor comes from tone and understatement, not from a technical punchline — that's Persona 3's job.

## Persona 3: Humorous-Tech
- Tone: Playful insider humor.
- Audience: Engineers, hackathon participants, tech Twitter.
- Style: Light memes, dev jargon, references to GPUs, APIs, debugging — keep it fun not cringe.
- MUST NOT: be deadpan or understated — that's Persona 2's job. The caption MUST land its punchline
  on a concrete, specific technical detail (a model name, an API, a metric, a piece of hardware) —
  a generic joke with no technical anchor is a Sarcastic caption wearing a costume.

## Persona 4: Humorous-Non-Tech
- Tone: Warm, accessible, universally funny.
- Audience: General public, casual viewers.
- Style: Plain language, relatable analogies, no technical jargon.

## Distinctness rule
Before finalizing, read the Sarcastic and Humorous-Tech captions side by side. If you could swap
them between the two personas and neither reader would notice anything off, BOTH are wrong — revise
until the Sarcastic caption is dry and jargon-free, and the Humorous-Tech caption is upbeat and
anchored in a specific technical detail.

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
    return OpenAI(api_key=api_key, base_url=FIREWORKS_BASE_URL, timeout=30.0, max_retries=2)


def mock_transcript(video_name: str = "demo.mp4") -> str:
    """Fallback transcript for UI testing when audio is missing or unclear."""
    return f"[Mock transcript for {video_name}]\n\n{MOCK_TRANSCRIPT}"


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    wait=wait_exponential(min=2, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _chat_completion_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    """Call client.chat.completions.create with exponential-backoff retry on transient errors."""
    return client.chat.completions.create(**kwargs)


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
    visual_context: str | None = None,
) -> tuple[str, TelemetrySummary]:
    """Summarize themes, mood, and technical jargon from a transcript.

    When *visual_context* is provided (a vision-model description of sampled
    video frames), it's fused with the transcript so captions can reference
    what's actually on screen, not just what's said — see get_visual_context.
    """
    telemetry = TelemetrySummary()
    if not transcript.strip() and not (visual_context or "").strip():
        return "No transcript or visual content available. Unable to analyze context.", telemetry

    user_sections = [f"Transcript:\n\n{transcript}" if transcript.strip() else "Transcript: (none — silent or unclear audio)"]
    if visual_context and visual_context.strip():
        user_sections.append(f"Visual content (sampled video frames):\n\n{visual_context}")

    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze video transcripts and, when available, a description of "
                    "the video's visual content for a captioning pipeline. Summarize in "
                    "3-6 bullet points: main themes, emotional mood, key technical jargon, "
                    "target audience signals, and any notable visual details (setting, "
                    "on-screen action, text/UI) that a caption should reflect. Be concise."
                ),
            },
            {"role": "user", "content": "\n\n".join(user_sections)},
        ],
        temperature=0.4,
        max_tokens=512,
    )
    telemetry.add_usage(getattr(response, "usage", None))
    return (response.choices[0].message.content or "").strip(), telemetry


def _request_critic_evaluation(
    *,
    client: OpenAI,
    model: str,
    transcript: str,
    context: str,
    prior_feedback: str | None = None,
    prior_captions: CaptionSet | None = None,
) -> tuple[CriticEvaluation, TelemetrySummary]:
    telemetry = TelemetrySummary()

    if prior_feedback:
        # Retry attempt: skip the full transcript to save tokens. Anchor the
        # revision to the previous captions so the model only touches the
        # flagged style(s) instead of re-drafting everything from scratch.
        prior_captions_json = prior_captions.model_dump_json(indent=2) if prior_captions else "(none)"
        user_content = (
            f"## Context Analysis\n{context}\n\n"
            f"## Previous Captions\n{prior_captions_json}\n\n"
            f"## Prior Critic Feedback (address these issues)\n{prior_feedback}\n\n"
            "Revise ONLY the caption style(s) flagged above; keep every other caption exactly as "
            "written in Previous Captions. Perform critic evaluation on the full, updated set. "
            "Respond with valid JSON matching the schema."
        )
    else:
        user_content = (
            f"## Transcript\n{transcript}\n\n"
            f"## Context Analysis\n{context}\n\n"
            "Generate all four caption styles and perform critic evaluation. "
            "Respond with valid JSON matching the schema."
        )

    messages = [
        {"role": "system", "content": COPYWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_tokens = 2048
    response = _chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        response_format=_critic_response_format(),
        temperature=0.7,
        max_tokens=max_tokens,
    )
    telemetry.add_usage(getattr(response, "usage", None))

    if response.choices[0].finish_reason == "length":
        logger.warning("Critic response truncated (finish_reason=length); retrying with larger max_tokens.")
        max_tokens = 3072
        response = _chat_completion_with_retry(
            client,
            model=model,
            messages=messages,
            response_format=_critic_response_format(),
            temperature=0.7,
            max_tokens=max_tokens,
        )
        telemetry.add_usage(getattr(response, "usage", None))

    raw = response.choices[0].message.content or "{}"
    try:
        try:
            evaluation = CriticEvaluation.model_validate_json(raw)
        except ValidationError:
            payload = json.loads(raw)
            evaluation = CriticEvaluation.model_validate(payload)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.error("Critic returned malformed JSON: %s | raw (truncated): %s", exc, raw[:500])
        raise RuntimeError(f"Critic returned malformed JSON: {exc}") from exc

    return evaluation, telemetry


def _caption_pair_similarities(captions: CaptionSet, client: OpenAI) -> dict[tuple[str, str], float]:
    """Compute pairwise cosine similarity between the four caption styles via Fireworks embeddings."""
    style_texts = {
        "formal": captions.formal,
        "sarcastic": captions.sarcastic,
        "humorous_tech": captions.humorous_tech,
        "humorous_non_tech": captions.humorous_non_tech,
    }
    names = list(style_texts)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[style_texts[n] for n in names])
    vectors = [np.array(item.embedding) for item in response.data]

    similarities: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            denom = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            cos_sim = float(np.dot(vectors[i], vectors[j]) / denom) if denom else 0.0
            similarities[(names[i], names[j])] = cos_sim
    return similarities


def captions_are_distinct(captions: CaptionSet, client: OpenAI, threshold: float = 0.85) -> bool:
    """Return False if any pair of the four captions is too semantically similar."""
    similarities = _caption_pair_similarities(captions, client)
    return all(sim <= threshold for sim in similarities.values())


def run_caption_critic_loop(
    transcript: str,
    context: str,
    *,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    distinctness_threshold: float = 0.85,
) -> tuple[CriticEvaluation, TelemetrySummary]:
    """
    Run copywriter + critic agents with self-correction up to max_retries.

    A caption set is only accepted if the critic approves it AND an independent
    embedding-based distinctness check confirms no two captions are near-duplicates.
    If the critic rejects captions, or the distinctness check fails, feedback is fed
    back into the next attempt. A malformed critic response (RuntimeError) is logged
    and retried rather than aborting the whole loop.
    """
    telemetry = TelemetrySummary()
    prior_feedback: str | None = None
    prior_captions: CaptionSet | None = None
    last_evaluation: CriticEvaluation | None = None
    last_fully_passed = False

    for attempt in range(1, max_retries + 1):
        logger.info("Critic loop attempt %s/%s", attempt, max_retries)
        try:
            evaluation, call_telemetry = _request_critic_evaluation(
                client=client,
                model=model,
                transcript=transcript,
                context=context,
                prior_feedback=prior_feedback,
                prior_captions=prior_captions,
            )
        except RuntimeError as exc:
            logger.error("Critic evaluation attempt %s failed: %s", attempt, exc)
            prior_feedback = prior_feedback or (
                "Your previous response was not valid JSON. Return only JSON matching the schema."
            )
            continue

        telemetry += call_telemetry
        last_evaluation = evaluation
        prior_captions = evaluation.captions

        distinct_ok = True
        similarities: dict[tuple[str, str], float] = {}
        try:
            similarities = _caption_pair_similarities(evaluation.captions, client)
            distinct_ok = all(sim <= distinctness_threshold for sim in similarities.values())
        except Exception as exc:
            logger.warning(
                "Distinctness check failed (embeddings unavailable); skipping gate for this attempt: %s",
                exc,
            )
            distinct_ok = True

        fully_passed = evaluation.approved and distinct_ok
        last_fully_passed = fully_passed
        if fully_passed:
            return evaluation, telemetry

        if not distinct_ok:
            worst_pair, worst_score = max(similarities.items(), key=lambda kv: kv[1])
            prior_feedback = (
                f"Captions '{worst_pair[0]}' and '{worst_pair[1]}' are too similar "
                f"(similarity={worst_score:.2f}); revise for stronger tonal contrast."
            )
            logger.info("Distinctness check rejected attempt %s: %s", attempt, prior_feedback)
        else:
            prior_feedback = evaluation.critique_notes
            logger.info("Critic rejected attempt %s: %s", attempt, prior_feedback)

    if last_evaluation is not None:
        if not last_fully_passed and last_evaluation.approved:
            last_evaluation = last_evaluation.model_copy(update={"approved": False})
        return last_evaluation, telemetry

    raise RuntimeError("Critic loop completed without producing a usable evaluation.")


def run_full_pipeline(
    transcript: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> tuple[str, CriticEvaluation, TelemetrySummary]:
    """Execute context analysis and caption critic loop end-to-end."""
    client = build_client(api_key)
    context, context_telemetry = generate_raw_context(transcript, client=client, model=model)
    evaluation, critic_telemetry = run_caption_critic_loop(
        transcript,
        context,
        client=client,
        model=model,
        max_retries=max_retries,
    )
    context_telemetry += critic_telemetry
    return context, evaluation, context_telemetry
