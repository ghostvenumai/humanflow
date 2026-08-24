#!/usr/bin/env python3
"""Controlled real-provider TTS A/B with no subjective score fabrication."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from pathlib import Path
from time import monotonic_ns
from uuid import uuid4

from humanflow.runtime.elevenlabs_provider import (
    ElevenLabsStreamingTTSProvider,
    SynthesisBudget,
)
from humanflow.runtime.providers import GaplessSegmentTTSProvider, SpeechSynthesisRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "tts-ab-benchmark.json"
BASELINE_MODEL = "eleven_flash_v2_5"
CANDIDATE_MODEL = "eleven_v3_conversational"
PHRASES = (
    "Ja, klar. Donnerstag passt. Welche Uhrzeit wäre dir am liebsten?",
    "Ein ERP-System verbindet zentrale Geschäftsprozesse und hält die gemeinsamen Daten an einem Ort.",
    "Das klingt gerade wirklich anstrengend. Wir gehen am besten einen Schritt nach dem anderen.",
    "Der Termin ist am Donnerstag, dem zehnten September, um fünfzehn Uhr.",
)
CANCEL_PHRASE = (
    "Ich erkläre das etwas ausführlicher, damit die Unterbrechung während einer laufenden "
    "Sprachausgabe unter identischen Bedingungen gemessen werden kann."
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 3)


async def _run_model(api_key: str, voice_id: str, model: str) -> dict[str, object]:
    budget = SynthesisBudget(maximum_audio_seconds=180.0, maximum_input_characters=1_200)
    raw = ElevenLabsStreamingTTSProvider(
        api_key=api_key,
        voice_id=voice_id,
        model=model,
        budget=budget,
    )
    provider = GaplessSegmentTTSProvider(raw)
    first_playable_ms: list[float] = []
    wall_ms: list[float] = []
    audio_seconds: list[float] = []
    source_chunks: list[int] = []
    failures: list[str] = []
    for index, phrase in enumerate(PHRASES):
        request = SpeechSynthesisRequest(
            text=phrase,
            response_id=f"ab-{model}-{index}-{uuid4()}",
            sequence_start=0,
            language_code="de",
            speaking_rate=1.0,
            stability=0.48,
            similarity_boost=0.78,
            style=0.08,
            use_speaker_boost=False,
            intent="information",
        )
        started_ns = monotonic_ns()
        try:
            chunks = [
                chunk
                async for chunk in provider.stream_speech(
                    request, cancel_event=asyncio.Event()
                )
            ]
        except Exception as error:
            failures.append(type(error).__name__)
            continue
        completed_ns = monotonic_ns()
        if len(chunks) != 1:
            failures.append("non_single_gapless_playback_unit")
            continue
        chunk = chunks[0]
        first_playable_ms.append((completed_ns - started_ns) / 1_000_000.0)
        wall_ms.append((completed_ns - started_ns) / 1_000_000.0)
        audio_seconds.append(chunk.frame.duration_ms / 1_000.0)
        source_chunks.append(int(chunk.provider.get("source_chunk_count", "0")))

    cancel_event = asyncio.Event()
    cancel_request = SpeechSynthesisRequest(
        text=CANCEL_PHRASE,
        response_id=f"ab-cancel-{model}-{uuid4()}",
        sequence_start=0,
        language_code="de",
    )

    async def collect_cancelled() -> int:
        emitted = 0
        async for _ in provider.stream_speech(
            cancel_request, cancel_event=cancel_event
        ):
            emitted += 1
        return emitted

    cancellation_task = asyncio.create_task(collect_cancelled())
    await asyncio.sleep(0.25)
    cancellation_requested_ns = monotonic_ns()
    cancel_event.set()
    try:
        chunks_after_cancel = await asyncio.wait_for(cancellation_task, timeout=3.0)
        cancellation_completed_ms: float | None = (
            monotonic_ns() - cancellation_requested_ns
        ) / 1_000_000.0
        cancellation_failure = None
    except Exception as error:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)
        cancellation_completed_ms = None
        cancellation_failure = type(error).__name__
        chunks_after_cancel = -1

    metrics = budget.to_dict()
    successful = len(first_playable_ms)
    return {
        "model": model,
        "voice": "same_configured_german_professional_voice",
        "requests": len(PHRASES),
        "successful_requests": successful,
        "integration_reliability": successful / len(PHRASES),
        "first_playable_audio_latency_ms": {
            "median": (
                round(statistics.median(first_playable_ms), 3)
                if first_playable_ms
                else None
            ),
            "p95": _percentile(first_playable_ms, 0.95),
            "samples": len(first_playable_ms),
        },
        "request_wall_ms": {
            "median": round(statistics.median(wall_ms), 3) if wall_ms else None,
            "p95": _percentile(wall_ms, 0.95),
        },
        "generated_audio_seconds": round(sum(audio_seconds), 3),
        "gapless_playback_units": successful,
        "provider_source_chunks": source_chunks,
        "sequence_and_playback_stability": (
            "PASS" if successful == len(PHRASES) and not failures else "FAIL"
        ),
        "cancellation": {
            "request_after_ms": 250.0,
            "completion_ms": (
                None
                if cancellation_completed_ms is None
                else round(cancellation_completed_ms, 3)
            ),
            "playback_units_emitted_after_cancel": chunks_after_cancel,
            "result": (
                "PASS"
                if cancellation_failure is None and chunks_after_cancel == 0
                else "FAIL"
            ),
            "failure": cancellation_failure,
        },
        "cost_evidence": {
            "submitted_characters": metrics["submitted_characters"],
            "provider_reported_billable_character_credits": metrics[
                "reported_billable_characters"
            ],
            "usd_cost": None,
            "usd_cost_reason": "not_fabricated_provider_did_not_return_dollar_charge",
        },
        "failures": failures,
    }


async def main() -> None:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
    if not api_key.strip() or not voice_id.strip():
        raise SystemExit("ElevenLabs runtime configuration missing")
    baseline = await _run_model(api_key, voice_id, BASELINE_MODEL)
    candidate = await _run_model(api_key, voice_id, CANDIDATE_MODEL)
    objective_pass = all(
        item["sequence_and_playback_stability"] == "PASS"
        and item["cancellation"]["result"] == "PASS"
        for item in (baseline, candidate)
    )
    payload = {
        "benchmark": "controlled_elevenlabs_tts_ab",
        "same_german_phrases": list(PHRASES),
        "same_voice": True,
        "same_prosody_settings": True,
        "same_gapless_browser_playback_pipeline": True,
        "baseline": baseline,
        "candidate": candidate,
        "objective_gate": "PASS" if objective_pass else "FAIL",
        "subjective_human_ratings": {
            "sample_count": 0,
            "naturalness": None,
            "prosody": None,
            "pacing": None,
            "warmth": None,
            "mechanical_impression": None,
            "overall_realism": None,
        },
        "keep_or_revert": "PENDING_HUMAN_AB",
        "manual_validation": "REQUIRED_NOT_ATTESTED",
        "measurement_scope": (
            "real ElevenLabs provider to the same gapless semantic playback unit; "
            "browser/device perception is excluded and must be rated by a human"
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not objective_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
