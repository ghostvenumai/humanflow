#!/usr/bin/env python3
"""Measure local realtime-core timing from emitted monotonic events."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import get_clock_info, monotonic_ns

from humanflow.audio.models import AudioFrame
from humanflow.runtime.providers import (
    EchoReasoner,
    NullTranscriber,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnSignals


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "mean": round(fmean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p90": round(_percentile(values, 90), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


async def _wait_for_event(sink: InMemoryTelemetrySink, event_type: EventType) -> None:
    for _ in range(2_000):
        if any(event.event_type is event_type for event in sink.events):
            return
        await asyncio.sleep(0.0005)
    raise TimeoutError(f"event did not arrive: {event_type}")


async def _sample(index: int) -> dict[str, float | int]:
    sink = InMemoryTelemetrySink()
    session = RealtimeVoiceSession(
        conversation_id=f"benchmark-{index}",
        sink=sink,
        transcriber=NullTranscriber(),
        reasoner=EchoReasoner(first_token_delay_ms=8.0, chunk_delay_ms=0.0),
        synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=100.0),
        audio_output=TimedPcmOutput(quantum_ms=5.0),
    )
    await session.start()
    update = TranscriptUpdate(
        text="Bitte prüfen Sie meinen Termin",
        is_final=True,
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=1_200,
            semantic_complete=True,
            acoustic_completion=0.9,
        ),
    )
    await session.submit_transcript(update)
    await _wait_for_event(sink, EventType.AGENT_AUDIO_STARTED)

    enqueue_started_ns = monotonic_ns()
    for sequence in range(20):
        session.receive_audio(
            AudioFrame(
                stream_id=f"caller-{index}",
                sequence=sequence,
                pcm16=b"\x00\x00" * 320,
                captured_ns=monotonic_ns(),
            )
        )
    enqueue_elapsed_ms = (monotonic_ns() - enqueue_started_ns) / 1_000_000.0
    input_drain_started_ns = monotonic_ns()
    await session.wait_for_input()
    input_drain_ms = (monotonic_ns() - input_drain_started_ns) / 1_000_000.0

    await asyncio.sleep((3 + index % 8 * 2) / 1000.0)
    returned_latency_ms = await session.interrupt()
    await session.wait_for_response()
    turn_event = next(event for event in sink.events if event.event_type is EventType.TURN_CONFIRMED)
    audio_event = next(
        event for event in sink.events if event.event_type is EventType.AGENT_AUDIO_STARTED
    )
    cancel_event = next(
        event for event in sink.events if event.event_type is EventType.AGENT_AUDIO_CANCELLED
    )
    candidate_event = next(
        event for event in sink.events if event.event_type is EventType.INTERRUPTION_CANDIDATE
    )
    event_latency_ms = (
        cancel_event.monotonic_ns - candidate_event.monotonic_ns
    ) / 1_000_000.0
    assert isinstance(returned_latency_ms, float)
    assert event_latency_ms >= 0.0
    assert returned_latency_ms >= 0.0
    result: dict[str, float | int] = {
        "sample": index,
        "ttfa_ms": (audio_event.monotonic_ns - turn_event.monotonic_ns) / 1_000_000.0,
        "audible_barge_in_latency_ms": event_latency_ms,
        "audio_sink_request_to_stop_ms": returned_latency_ms,
        "pcm_enqueue_20_frames_ms": enqueue_elapsed_ms,
        "pcm_input_drain_20_frames_ms": input_drain_ms,
        "played_samples_before_stop": int(cancel_event.payload["played_samples"]),
    }
    await session.close(reason_code="benchmark_complete")
    return result


async def _run(sample_count: int) -> list[dict[str, float | int]]:
    return [await _sample(index) for index in range(sample_count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/realtime-core-benchmark.json")
    )
    args = parser.parse_args()
    if args.samples < 20:
        parser.error("--samples must be at least 20 for a latency distribution")

    raw = asyncio.run(_run(args.samples))
    ttfa = [float(sample["ttfa_ms"]) for sample in raw]
    barge = [float(sample["audible_barge_in_latency_ms"]) for sample in raw]
    enqueue = [float(sample["pcm_enqueue_20_frames_ms"]) for sample in raw]
    drain = [float(sample["pcm_input_drain_20_frames_ms"]) for sample in raw]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "measurement_scope": {
            "kind": "local_event_loop_transport_benchmark",
            "audio_input": "20 real in-memory PCM16 frames per sample",
            "audio_output": "TimedPcmOutput paced by monotonic event-loop sleeps",
            "reasoner": "EchoReasoner with configured 8 ms first-fragment delay",
            "synthesizer": "local PCM16 tone chunks; not human speech quality",
            "hardware_device": False,
            "network_provider": False,
            "claim_limit": (
                "Measures controller, queue and cancellation behavior on this process only; "
                "it is not a browser, telephony, speaker, STT, model, or production-call SLA."
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "monotonic_clock": vars(get_clock_info("monotonic")),
        },
        "metrics": {
            "ttfa_ms": _summary(ttfa),
            "audible_barge_in_latency_ms": _summary(barge),
            "pcm_enqueue_20_frames_ms": _summary(enqueue),
            "pcm_input_drain_20_frames_ms": _summary(drain),
        },
        "samples": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
