#!/usr/bin/env python3
"""Run the fixed runtime-depth evaluation used by quality-loop iterations."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from humanflow.domain.conversation import ConversationState
from humanflow.runtime.providers import (
    EchoReasoner,
    NullTranscriber,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnSignals


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "quality" / "runtime_scenarios.json"
OUTPUT = ROOT / "reports" / "runtime-quality-eval.json"


async def _run_scenario(scenario: dict[str, object]) -> dict[str, object]:
    sink = InMemoryTelemetrySink()
    session = RealtimeVoiceSession(
        conversation_id=str(scenario["id"]),
        sink=sink,
        transcriber=NullTranscriber(),
        reasoner=EchoReasoner(first_token_delay_ms=0, chunk_delay_ms=0),
        synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=25),
        audio_output=TimedPcmOutput(quantum_ms=2),
    )
    await session.start()
    await session.submit_transcript(
        TranscriptUpdate(
            text="Bitte nennen Sie mir einen Termin",
            is_final=True,
            provenance=TranscriptProvenance.user_fixture(
                final=True, source="evaluation_fixture"
            ),
            signals=TurnSignals(
                speech_active=False,
                silence_duration_ms=350,
                utterance_duration_ms=900,
                semantic_complete=True,
                acoustic_completion=0.9,
            ),
        )
    )
    for _ in range(1_000):
        if session.state is ConversationState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    else:
        raise TimeoutError("agent did not start speaking")
    raw_input = scenario["input"]
    assert isinstance(raw_input, dict)
    decision = await session.submit_transcript(
        TranscriptUpdate(
            text=str(raw_input["text"]),
            is_final=True,
            provenance=TranscriptProvenance.user_fixture(
                final=True, source="evaluation_fixture"
            ),
            signals=TurnSignals(
                speech_active=True,
                silence_duration_ms=0,
                utterance_duration_ms=500,
                background_speech_probability=float(raw_input["background_speech_probability"]),
                interruption_probability=float(raw_input["interruption_probability"]),
            ),
        )
    )
    await session.wait_for_response()
    transitions = [
        event.payload["to_state"]
        for event in sink.events
        if event.event_type is EventType.STATE_TRANSITIONED
    ]
    cancellations = sum(
        event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
    )
    expected = scenario["expected"]
    assert isinstance(expected, dict)
    expected_states = list(expected["ordered_states"])
    cursor = 0
    for state in transitions:
        if cursor < len(expected_states) and state == expected_states[cursor]:
            cursor += 1
    passed = (
        decision.decision.value == expected["decision"]
        and cursor == len(expected_states)
        and cancellations == expected["agent_audio_cancelled_events"]
    )
    await session.close(reason_code="runtime_quality_eval_complete")
    return {
        "id": scenario["id"],
        "passed": passed,
        "decision": decision.decision.value,
        "observed_transitions": transitions,
        "agent_audio_cancelled_events": cancellations,
    }


async def _run(scenarios: list[dict[str, object]]) -> list[dict[str, object]]:
    return [await _run_scenario(scenario) for scenario in scenarios]


def main() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    results = asyncio.run(_run(fixture["scenarios"]))
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "schema_version": 1,
        "fixture": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "score": passed / len(results),
        },
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
