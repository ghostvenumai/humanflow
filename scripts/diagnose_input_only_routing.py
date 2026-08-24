#!/usr/bin/env python3
"""Exercise browser-STT event routing with reasoning and TTS disabled."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from humanflow.telemetry.events import EventType
from humanflow.web.app import _handle_json
from humanflow.web.transport import BrowserAcknowledgedAudioOutput


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "input-routing-isolation-B.json"
TEXTS = (
    "Das ist der erste echte Mikrofonsatz.",
    "Hier folgt der zweite echte Mikrofonsatz.",
    "Nur meine Sprache darf als Nutzertext erscheinen.",
)


class RecordingStateMachine:
    def __init__(self) -> None:
        self.events: list[tuple[EventType, dict[str, Any]]] = []

    def record(self, event_type: EventType, **kwargs: Any) -> None:
        self.events.append((event_type, kwargs))


class ForbiddenControllerSession:
    def __init__(self) -> None:
        self.state_machine = RecordingStateMachine()
        self.reasoning_or_tts_submissions = 0

    async def accept_user_transcript(self, update: object) -> None:
        del update
        self.reasoning_or_tts_submissions += 1
        raise AssertionError("input-only mode called the conversation controller")


def browser_final(text: str, result_id: str) -> str:
    return json.dumps(
        {
            "type": "transcript",
            "source": "browser_stt",
            "recognition_result_id": result_id,
            "text": text,
            "final": True,
            "provenance": {
                "transcript_id": f"{result_id}:final",
                "event_kind": "USER_TRANSCRIPT_FINAL",
                "source": "browser_stt",
                "origin": "BROWSER_SPEECH_RECOGNITION",
                "stream_id": "browser-recognition:input-only",
                "browser_recognition_session_id": "input-only",
                "audio_capture_id": "input-only-capture",
                "recognition_input_binding": (
                    "UNVERIFIED_INDEPENDENT_BROWSER_CAPTURE"
                ),
            },
            "signals": {
                "speech_active": False,
                "silence_duration_ms": 350,
                "utterance_duration_ms": 900,
                "semantic_complete": True,
                "provider_endpointed": True,
            },
        }
    )


async def run() -> dict[str, object]:
    session = ForbiddenControllerSession()
    outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
    output = BrowserAcknowledgedAudioOutput(outbound)
    seen: set[str] = set()
    payloads = [
        browser_final(text, f"input-only-result-{index}")
        for index, text in enumerate(TEXTS)
    ]
    for payload in payloads:
        await _handle_json(  # type: ignore[arg-type]
            payload,
            session,
            output,
            outbound,
            seen_final_transcript_ids=seen,
            input_only=True,
        )
    await _handle_json(  # type: ignore[arg-type]
        payloads[1],
        session,
        output,
        outbound,
        seen_final_transcript_ids=seen,
        input_only=True,
    )
    await _handle_json(  # type: ignore[arg-type]
        json.dumps(
            {
                "type": "transcript",
                "source": "assistant_reasoning",
                "text": "Dieser Assistententext darf nie Nutzertext werden.",
                "final": True,
                "provenance": {
                    "transcript_id": "forbidden-assistant-text",
                    "event_kind": "ASSISTANT_TEXT",
                    "source": "assistant_reasoning",
                    "origin": "ASSISTANT_REASONING",
                    "stream_id": "assistant-response-stream",
                    "response_id": "assistant-response-id",
                },
                "signals": {},
            }
        ),
        session,
        output,
        outbound,
        seen_final_transcript_ids=seen,
        input_only=True,
    )

    final_events = [
        kwargs
        for event_type, kwargs in session.state_machine.events
        if event_type is EventType.FINAL_TRANSCRIPT
    ]
    duplicate_events = [
        kwargs
        for event_type, kwargs in session.state_machine.events
        if event_type is EventType.DUPLICATE_TRANSCRIPT_REJECTED
    ]
    rejected_assistant_events = [
        kwargs
        for event_type, kwargs in session.state_machine.events
        if event_type is EventType.TRANSCRIPT_REJECTED
        and kwargs.get("reason_code")
        == "assistant_origin_event_forbidden_from_user_history"
    ]
    accepted_texts = [event["payload"]["raw_text"] for event in final_events]
    if accepted_texts != list(TEXTS):
        raise RuntimeError("unexpected_text_entered_user_transcript_route")
    if session.reasoning_or_tts_submissions:
        raise RuntimeError("conversation_controller_called_in_input_only_mode")
    if len(duplicate_events) != 1:
        raise RuntimeError("duplicate_final_was_not_rejected_exactly_once")
    if len(rejected_assistant_events) != 1:
        raise RuntimeError("assistant_origin_event_was_not_hard_rejected")
    return {
        "status": "PASS_AUTOMATED_INPUT_ROUTING_ISOLATION",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": (
            "server browser-STT event routing with input-only mode; physical "
            "microphone/browser recognition remains manual validation"
        ),
        "browser_stt_final_payloads_sent": len(payloads) + 1,
        "unique_final_transcripts_accepted": len(final_events),
        "duplicate_final_transcripts_rejected": len(duplicate_events),
        "assistant_routing_payloads_attempted": 1,
        "assistant_origin_events_hard_rejected": len(rejected_assistant_events),
        "assistant_payloads_accepted_as_user_transcript": 0,
        "conversation_controller_calls": session.reasoning_or_tts_submissions,
        "reasoning_calls": 0,
        "tts_calls": 0,
        "accepted_texts": accepted_texts,
        "manual_browser_microphone_validation": "PENDING",
        "agent_attestation": False,
    }


def main() -> None:
    report = asyncio.run(run())
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
