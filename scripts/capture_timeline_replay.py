#!/usr/bin/env python3
"""Capture synthetic runtime events and prove deterministic strict replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from humanflow.domain.conversation import ConversationState
from humanflow.evaluation.timeline import load_jsonl_events, replay_timeline
from humanflow.runtime.providers import (
    EchoReasoner,
    NullTranscriber,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.telemetry.sinks import JsonlTelemetrySink
from humanflow.turns.models import TurnSignals


ROOT = Path(__file__).resolve().parents[1]
TIMELINE = ROOT / "reports" / "realtime-timeline.jsonl"
REPORT = ROOT / "reports" / "timeline-replay.json"


def _update(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=900,
            semantic_complete=True,
            acoustic_completion=0.9,
        ),
    )


async def _wait_speaking(session: RealtimeVoiceSession) -> None:
    for _ in range(1_000):
        if session.state is ConversationState.SPEAKING:
            return
        await asyncio.sleep(0.001)
    raise TimeoutError("session did not start playback")


async def _capture() -> None:
    TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    TIMELINE.write_text("", encoding="utf-8")
    sink = JsonlTelemetrySink(TIMELINE)
    for index, interrupted in enumerate((False, True), start=1):
        session = RealtimeVoiceSession(
            conversation_id=f"replay-call-{index}",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=EchoReasoner(first_token_delay_ms=3, chunk_delay_ms=0),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=50),
            audio_output=TimedPcmOutput(quantum_ms=3),
        )
        await session.start()
        await session.submit_transcript(_update("Bitte bestätigen Sie meinen Termin"))
        if interrupted:
            await _wait_speaking(session)
            await session.submit_transcript(
                TranscriptUpdate(
                    text="Moment, stopp",
                    is_final=True,
                    signals=TurnSignals(
                        speech_active=True,
                        silence_duration_ms=0,
                        utterance_duration_ms=250,
                        interruption_probability=0.98,
                    ),
                )
            )
        await session.wait_for_response()
        await session.close(reason_code="replay_fixture_complete")


def main() -> None:
    asyncio.run(_capture())
    first = replay_timeline(load_jsonl_events(TIMELINE))
    second = replay_timeline(load_jsonl_events(TIMELINE))
    if first != second:
        raise RuntimeError("timeline replay is not deterministic")
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "timeline": {
            "path": str(TIMELINE.relative_to(ROOT)),
            "sha256": hashlib.sha256(TIMELINE.read_bytes()).hexdigest(),
            "synthetic": True,
            "contains_real_runtime_monotonic_timestamps": True,
        },
        "replay": first.to_dict(),
        "deterministic_second_pass_equal": True,
        "claim_limit": "Synthetic local sessions; not production-call quality evidence.",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["replay"], indent=2, sort_keys=True))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
