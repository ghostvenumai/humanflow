#!/usr/bin/env python3
"""Measure two-stage barge-in on paced local PCM and acknowledged playback."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import monotonic_ns
from typing import Any

from humanflow.audio.models import AudioFrame
from humanflow.runtime.providers import (
    EchoReasoner,
    NullTranscriber,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnSignals
from humanflow.web.transport import BrowserAcknowledgedAudioOutput, OutboundItem


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
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
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def _turn(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(final=True),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=650,
            utterance_duration_ms=500,
            semantic_complete=True,
            acoustic_completion=1.0,
            provider_endpointed=True,
        ),
    )


async def _wait_for_event(
    sink: InMemoryTelemetrySink, event_type: EventType
) -> None:
    for _ in range(2_000):
        if any(event.event_type is event_type for event in sink.events):
            return
        await asyncio.sleep(0.001)
    raise TimeoutError(f"event did not arrive: {event_type}")


async def _paced_frames(
    session: RealtimeVoiceSession,
    *,
    sequence_start: int,
    count: int,
    amplitude: int,
) -> int:
    sequence = sequence_start
    for _ in range(count):
        captured_ns = monotonic_ns()
        await asyncio.sleep(0.02)
        session.receive_audio(
            AudioFrame(
                stream_id="paced-authoritative-pcm",
                sequence=sequence,
                pcm16=struct.pack("<h", amplitude) * 320,
                sample_rate_hz=16_000,
                captured_ns=captured_ns,
            )
        )
        sequence += 1
    return sequence


async def _browser_ack_loop(
    outbound: asyncio.Queue[OutboundItem | None],
    output: BrowserAcknowledgedAudioOutput,
    session: RealtimeVoiceSession,
) -> None:
    active_chunk_id: str | None = None
    while True:
        item = await outbound.get()
        try:
            if item is None:
                return
            if isinstance(item, bytes):
                if active_chunk_id is not None:
                    output.acknowledge(
                        {"type": "playback_started", "chunk_id": active_chunk_id}
                    )
                continue
            message_type = item.get("type")
            if message_type == "audio_chunk":
                active_chunk_id = str(item["chunk_id"])
            elif message_type == "playback_duck":
                session.acknowledge_playback_control(
                    {
                        "type": "playback_ducked",
                        "response_id": item["response_id"],
                        "target_gain": item["target_gain"],
                    }
                )
            elif message_type == "playback_resume":
                session.acknowledge_playback_control(
                    {
                        "type": "playback_resumed",
                        "response_id": item["response_id"],
                    }
                )
            elif message_type == "invalidate_playback" and active_chunk_id is not None:
                output.acknowledge(
                    {
                        "type": "playback_stopped",
                        "chunk_id": active_chunk_id,
                        "played_samples": 0,
                        "player_stop_callback_latency_ms": 0.0,
                    }
                )
                active_chunk_id = None
        finally:
            outbound.task_done()


async def _sample(index: int) -> dict[str, float | int]:
    outbound: asyncio.Queue[OutboundItem | None] = asyncio.Queue()
    output = BrowserAcknowledgedAudioOutput(outbound)
    sink = InMemoryTelemetrySink()
    session = RealtimeVoiceSession(
        conversation_id=f"acoustic-benchmark-{index}",
        sink=sink,
        transcriber=NullTranscriber(),
        reasoner=EchoReasoner(first_token_delay_ms=1, chunk_delay_ms=0),
        synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=4_000),
        audio_output=output,
        soft_yield_recovery_delay_ms=40,
    )
    browser = asyncio.create_task(
        _browser_ack_loop(outbound, output, session),
        name=f"acoustic-browser-ack-{index}",
    )
    await session.start()
    await session.submit_transcript(_turn("Bitte sprich weiter."))
    await _wait_for_event(sink, EventType.AGENT_AUDIO_STARTED)

    sequence = await _paced_frames(
        session, sequence_start=0, count=10, amplitude=5_000
    )
    sequence = await _paced_frames(
        session, sequence_start=sequence, count=10, amplitude=0
    )
    await session.submit_transcript(_turn("mhm"))
    await _wait_for_event(sink, EventType.BACKCHANNEL_RECOVERY)

    sequence = await _paced_frames(
        session, sequence_start=sequence, count=32, amplitude=5_000
    )
    del sequence
    await session.wait_for_response()
    await _wait_for_event(sink, EventType.AUDIBLE_STOP_ACK)

    onset_events = [
        event
        for event in sink.events
        if event.event_type is EventType.USER_AUDIO_STARTED
        and "acoustic_speech_onset_latency_ms" in event.payload
    ]
    duck_events = [
        event for event in sink.events if event.event_type is EventType.PLAYBACK_DUCK_STARTED
    ]
    hard = next(
        event
        for event in sink.events
        if event.event_type is EventType.INTERRUPTION_CONFIRMED
    )
    stop = next(
        event for event in sink.events if event.event_type is EventType.AUDIBLE_STOP_ACK
    )
    recovery = next(
        event for event in sink.events if event.event_type is EventType.BACKCHANNEL_RECOVERY
    )
    false_count = sum(
        event.event_type is EventType.FALSE_INTERRUPTION_DETECTED
        for event in sink.events
    )
    result: dict[str, float | int] = {
        "sample": index,
        "acoustic_speech_onset_latency_ms": float(
            onset_events[-1].payload["acoustic_speech_onset_latency_ms"]
        ),
        "speech_onset_to_soft_duck_ms": float(
            duck_events[-1].payload["speech_onset_to_soft_duck_ms"]
        ),
        "speech_onset_to_hard_cancel_ms": float(
            hard.payload["speech_onset_to_hard_cancel_ms"]
        ),
        "speech_onset_to_audible_stop_ms": float(
            stop.payload["speech_onset_to_audible_stop_ms"]
        ),
        "backchannel_recovery_latency_ms": float(
            recovery.payload["backchannel_recovery_latency_ms"]
        ),
        "false_interruption_count": false_count,
    }
    await session.close(reason_code="acoustic_benchmark_complete")
    outbound.put_nowait(None)
    await browser
    return result


async def _run(samples: int) -> list[dict[str, float | int]]:
    return [await _sample(index) for index in range(samples)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/acoustic-barge-in-benchmark.json"),
    )
    args = parser.parse_args()
    if args.samples < 10:
        parser.error("--samples must be at least 10")
    samples = asyncio.run(_run(args.samples))
    metric_names = (
        "acoustic_speech_onset_latency_ms",
        "speech_onset_to_soft_duck_ms",
        "speech_onset_to_hard_cancel_ms",
        "speech_onset_to_audible_stop_ms",
        "backchannel_recovery_latency_ms",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "measurement_scope": {
            "kind": "paced_local_pcm_two_stage_barge_in_benchmark",
            "pcm": "real paced PCM16 frames on the authoritative session input",
            "playback": "BrowserAcknowledgedAudioOutput with in-process acknowledgement",
            "stt": "not used for acoustic onset or sustained takeover",
            "browser_audio_device": False,
            "human_perception": False,
            "claim_limit": (
                "Measures PCM detector, controller, epoch invalidation and acknowledged "
                "stop locally; browser/device audible yield remains manual evidence."
            ),
        },
        "metrics": {
            name: _summary([float(sample[name]) for sample in samples])
            for name in metric_names
        },
        "false_interruption_count": sum(
            int(sample["false_interruption_count"]) for sample in samples
        ),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metrics": report["metrics"], "false_interruption_count": report["false_interruption_count"]}, indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
