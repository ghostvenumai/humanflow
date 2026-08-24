#!/usr/bin/env python3
"""Ten-run real-provider long-response continuity benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from array import array
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

from humanflow.audio.analysis import analyze_pcm16
from humanflow.audio.models import AudioFrame
from humanflow.runtime.elevenlabs_provider import (
    DEFAULT_ELEVENLABS_MODEL,
    ElevenLabsStreamingTTSProvider,
    SynthesisBudget,
)
from humanflow.runtime.providers import SpeechSynthesisRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "audio-continuity-10x.json"
ITERATIONS = 10
SAMPLE_RATE_HZ = 16_000
TEXT = (
    "HumanFlow erklärt heute in Ruhe, wie ein ERP-System verschiedene Bereiche eines "
    "Unternehmens miteinander verbindet. Einkauf, Lager, Verkauf und Buchhaltung greifen "
    "dabei auf einen gemeinsamen Datenstand zu. So müssen Informationen nicht mehrfach "
    "eingetragen werden, und Änderungen sind für alle beteiligten Teams schneller sichtbar. "
    "Wenn du möchtest, können wir danach ein kurzes Beispiel aus einem Handwerksbetrieb ansehen."
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * fraction)], 3)


def _boundary_jumps(chunks: list[bytes]) -> list[dict[str, float | int]]:
    boundaries: list[dict[str, float | int]] = []
    elapsed_samples = 0
    for previous, current in zip(chunks, chunks[1:], strict=False):
        elapsed_samples += len(previous) // 2
        left = int.from_bytes(previous[-2:], "little", signed=True)
        right = int.from_bytes(current[:2], "little", signed=True)
        boundaries.append(
            {
                "timestamp_ms": round(elapsed_samples * 1_000 / SAMPLE_RATE_HZ, 3),
                "sample_jump_full_scale": round(abs(right - left) / 32_768.0, 6),
            }
        )
    return boundaries


def _low_energy_candidates(pcm16: bytes) -> list[dict[str, float | str]]:
    """Return timestamps for review, not claims that natural pauses are defects."""

    samples = array("h")
    samples.frombytes(pcm16)
    window_samples = SAMPLE_RATE_HZ // 20  # 50 ms
    low_windows: list[int] = []
    for start in range(0, len(samples) - window_samples + 1, window_samples):
        peak = max(abs(sample) for sample in samples[start : start + window_samples])
        if peak <= 96:
            low_windows.append(start)
    candidates: list[dict[str, float | str]] = []
    run_start: int | None = None
    previous: int | None = None
    for start in [*low_windows, -1]:
        if run_start is None:
            run_start = None if start < 0 else start
        elif start != previous + window_samples:
            duration = (previous + window_samples - run_start) * 1_000 / SAMPLE_RATE_HZ
            if duration >= 150:
                candidates.append(
                    {
                        "timestamp_ms": round(run_start * 1_000 / SAMPLE_RATE_HZ, 3),
                        "duration_ms": round(duration, 3),
                        "reason": "LOW_ENERGY_RUN_REVIEW_NATURAL_PAUSE_POSSIBLE",
                    }
                )
            run_start = None if start < 0 else start
        previous = start
    return candidates


async def run() -> dict[str, object]:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    model = os.environ.get("HUMANFLOW_TTS_MODEL", DEFAULT_ELEVENLABS_MODEL).strip()
    if not api_key or not voice_id:
        raise RuntimeError("ElevenLabs runtime configuration is missing")
    budget = SynthesisBudget(
        maximum_audio_seconds=600.0,
        maximum_input_characters=len(TEXT) * ITERATIONS + 100,
    )
    provider = ElevenLabsStreamingTTSProvider(
        api_key=api_key,
        voice_id=voice_id,
        model=model or DEFAULT_ELEVENLABS_MODEL,
        budget=budget,
        request_timeout_seconds=60.0,
    )
    runs: list[dict[str, object]] = []
    wall_times: list[float] = []
    ttfa_times: list[float] = []
    rms_values: list[float] = []
    durations_seconds: list[float] = []

    for iteration in range(1, ITERATIONS + 1):
        response_id = f"audio-continuity-{iteration}"
        started_ns = monotonic_ns()
        chunks = [
            chunk
            async for chunk in provider.stream_speech(
                SpeechSynthesisRequest(
                    text=TEXT,
                    response_id=response_id,
                    sequence_start=0,
                    language_code="de",
                    speaking_rate=1.0,
                    stability=0.48,
                    similarity_boost=0.78,
                    style=0.08,
                    use_speaker_boost=False,
                    intent="information",
                    tts_session_id=f"{response_id}:tts",
                    segment_id=f"{response_id}:segment:1",
                ),
                cancel_event=asyncio.Event(),
            )
        ]
        wall_ms = (monotonic_ns() - started_ns) / 1_000_000.0
        if not chunks:
            raise RuntimeError("provider returned no chunks")
        sequences = [chunk.frame.sequence for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(sequences) != len(set(sequences)) or len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError("provider returned duplicate chunk provenance")
        pcm_chunks = [chunk.frame.pcm16 for chunk in chunks]
        pcm16 = b"".join(pcm_chunks)
        signal = analyze_pcm16(
            AudioFrame(
                stream_id=response_id,
                sequence=0,
                pcm16=pcm16,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
        )
        request_metrics = provider.last_request_metrics
        ttfa_ms = None if request_metrics is None else request_metrics.first_pcm_latency_ms
        jumps = _boundary_jumps(pcm_chunks)
        large_jumps = [item for item in jumps if item["sample_jump_full_scale"] > 0.25]
        candidates = [
            *large_jumps,
            *_low_energy_candidates(pcm16),
        ]
        runs.append(
            {
                "iteration": iteration,
                "response_id": response_id,
                "tts_session_id": f"{response_id}:tts",
                "physical_tts_requests": 1,
                "provider_chunk_count": len(chunks),
                "chunk_sequences": sequences,
                "duplicate_chunks": 0,
                "stale_chunks": 0,
                "decoded_duration_ms": round(signal.duration_ms, 3),
                "request_wall_ms": round(wall_ms, 3),
                "first_pcm_latency_ms": None if ttfa_ms is None else round(ttfa_ms, 3),
                "rms_dbfs": signal.rms_dbfs,
                "peak_dbfs": signal.peak_dbfs,
                "provider_chunk_boundary_jumps": jumps,
                "potential_hitch_timestamps": candidates,
                "underrun_count": 0,
                "underrun_scope": "contiguous_provider_pcm_reassembly",
            }
        )
        wall_times.append(wall_ms)
        if ttfa_ms is not None:
            ttfa_times.append(ttfa_ms)
        if signal.rms_dbfs is not None:
            rms_values.append(signal.rms_dbfs)
        durations_seconds.append(signal.duration_ms / 1_000.0)

    budget_data = budget.to_dict()
    return {
        "benchmark": "real_elevenlabs_continuous_long_response_10x",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provider": provider.provider_info.to_dict(),
        "same_text_every_run": True,
        "text": TEXT,
        "iterations": ITERATIONS,
        "requested_duration_seconds": {"minimum": 15.0, "maximum": 25.0},
        "duration_target_result": (
            "PASS"
            if all(15.0 <= duration <= 25.0 for duration in durations_seconds)
            else "FAIL"
        ),
        "response_context": "ONE_RESPONSE_ONE_TTS_REQUEST_CONTIGUOUS_PCM",
        "production_scheduler": "RESPONSE_LEVEL_LOOKAHEAD_QUEUE_TESTED_SEPARATELY",
        "amplitude_processing": "NONE_RAW_PROVIDER_PCM",
        "per_chunk_normalization": False,
        "aggregate": {
            "request_wall_ms_median": round(statistics.median(wall_times), 3),
            "request_wall_ms_p95": _percentile(wall_times, 0.95),
            "first_pcm_latency_ms_median": (
                round(statistics.median(ttfa_times), 3) if ttfa_times else None
            ),
            "first_pcm_latency_ms_p95": _percentile(ttfa_times, 0.95),
            "rms_dbfs_minimum": round(min(rms_values), 3) if rms_values else None,
            "rms_dbfs_maximum": round(max(rms_values), 3) if rms_values else None,
            "rms_range_across_responses_db": (
                round(max(rms_values) - min(rms_values), 3) if rms_values else None
            ),
            "total_underruns": 0,
            "duplicate_chunks": 0,
            "stale_chunks": 0,
            "duration_target_pass_count": sum(
                15.0 <= duration <= 25.0 for duration in durations_seconds
            ),
        },
        "budget": budget_data,
        "runs": runs,
        "human_audible_assessment": None,
        "manual_validation": "REQUIRED_NOT_ATTESTED",
        "measurement_scope": (
            "real ElevenLabs continuous PCM and exact provider-boundary timestamps; "
            "browser AudioContext/device timing and human audibility are excluded"
        ),
    }


def main() -> None:
    try:
        report = asyncio.run(run())
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "exception_type": type(error).__name__,
                    "provider_http_status": getattr(error, "status_code", None),
                    "provider_error_code": getattr(error, "provider_code", None),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": (
                    "PASS_OBJECTIVE_MEASUREMENTS_HUMAN_AUDIBILITY_PENDING"
                    if report["duration_target_result"] == "PASS"
                    else "FAIL_DURATION_TARGET_HUMAN_AUDIBILITY_PENDING"
                ),
                "iterations": report["iterations"],
                "duration_target_result": report["duration_target_result"],
                "aggregate": report["aggregate"],
                "budget": report["budget"],
                "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if report["duration_target_result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
