#!/usr/bin/env python3
"""Live regression for independent orthopedist and hairdresser objects."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import websocket


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "live-multi-appointment-smoke.json"
URL = "ws://127.0.0.1:8765/ws?tts=baseline"
TURNS = (
    "Ich brauche einen Orthopädentermin.",
    "Am besten nächste Woche Freitag um 14 Uhr.",
    "Mmm, warte mal, dann machen wir vielleicht 16 Uhr nächste Woche Donnerstag.",
    "Ich brauch noch 'n Friseurtermin, nächste Woche Mittwoch um 14 Uhr.",
    "Der Friseurtermin, was mit dem?",
    "Nee, ich will ihn absagen.",
    "Nicht Orthopäde. Der soll bleiben. Den Friseurtermin brauche ich nicht mehr.",
)


def _receive(connection: websocket.WebSocket) -> dict[str, Any] | bytes:
    message = connection.recv()
    if isinstance(message, bytes):
        return message
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid websocket payload")
    if payload.get("type") == "error":
        raise RuntimeError(f"live pipeline error: {payload.get('code')}")
    return payload


def _send_turn(connection: websocket.WebSocket, text: str) -> None:
    connection.send(
        json.dumps(
            {
                "type": "transcript",
                "source": "diagnostic_smoke",
                "text": text,
                "final": True,
                "provenance": {
                    "transcript_id": str(uuid4()),
                    "event_kind": "USER_TRANSCRIPT_FINAL",
                    "source": "diagnostic_smoke",
                    "origin": "DIAGNOSTIC_TEXT_INPUT",
                    "stream_id": "live-multi-appointment-smoke",
                },
                "signals": {
                    "speech_active": False,
                    "silence_duration_ms": 500,
                    "utterance_duration_ms": 900,
                    "semantic_complete": True,
                    "provider_endpointed": True,
                    "acoustic_completion": 1.0,
                },
            },
            ensure_ascii=False,
        )
    )


def _run_turn(connection: websocket.WebSocket, text: str) -> dict[str, Any]:
    _send_turn(connection, text)
    pending_audio: dict[str, Any] | None = None
    assistant_chunks: list[str] = []
    state_event: dict[str, Any] | None = None
    while True:
        payload = _receive(connection)
        if isinstance(payload, bytes):
            if pending_audio is None:
                raise RuntimeError("audio bytes without metadata")
            chunk_id = pending_audio["chunk_id"]
            connection.send(json.dumps({"type": "playback_started", "chunk_id": chunk_id}))
            connection.send(json.dumps({"type": "playback_completed", "chunk_id": chunk_id}))
            pending_audio = None
            continue
        if payload.get("type") == "audio_chunk":
            pending_audio = payload
            boundary = str(payload.get("text_boundary", "")).strip()
            if boundary:
                assistant_chunks.append(boundary)
            continue
        if payload.get("type") != "telemetry":
            continue
        event = payload["event"]
        if event["event_type"] == "APPOINTMENT_STATE_UPDATED":
            state_event = dict(event["payload"])
        if event["event_type"] == "AGENT_AUDIO_COMPLETED":
            return {
                "user": text,
                "assistant": " ".join(assistant_chunks),
                "state_event": state_event,
            }


def _value(appointment: dict[str, Any], slot: str) -> str | None:
    value = appointment.get(slot)
    return str(value["value"]) if isinstance(value, dict) else None


def main() -> None:
    connection = websocket.create_connection(URL, timeout=60)
    try:
        ready: dict[str, Any] | None = None
        while ready is None:
            payload = _receive(connection)
            if isinstance(payload, dict) and payload.get("type") == "ready":
                ready = payload
        turns = [_run_turn(connection, text) for text in TURNS]
    finally:
        connection.close()

    final_event = turns[-1]["state_event"] or {}
    appointments = final_event.get("appointments", {})
    orthopedist = appointments.get("appointment_1", {})
    hairdresser = appointments.get("appointment_2", {})
    checks = {
        "seven_live_turns_completed": len(turns) == 7,
        "stable_two_appointment_ids": tuple(appointments) == (
            "appointment_1",
            "appointment_2",
        ),
        "orthopedist_isolated": (
            _value(orthopedist, "purpose") == "Orthopädie"
            and _value(orthopedist, "date") == "2026-09-03"
            and _value(orthopedist, "time") == "16:00"
            and _value(orthopedist, "status") == "READY_TO_BOOK"
        ),
        "hairdresser_isolated_and_cancelled": (
            _value(hairdresser, "purpose") == "Friseur"
            and _value(hairdresser, "date") == "2026-09-02"
            and _value(hairdresser, "time") == "14:00"
            and _value(hairdresser, "status") == "CANCELLED"
        ),
        "no_false_booking_claim_in_state": all(
            _value(appointment, "status") != "BOOKED"
            for appointment in appointments.values()
        ),
        "assistant_never_claims_booked": all(
            "gebucht" not in turn["assistant"].casefold() for turn in turns
        ),
        "assistant_never_claims_unexecuted_external_action": all(
            not any(
                marker in turn["assistant"].casefold()
                for marker in (
                    "gebucht",
                    "eingetragen",
                    "storniert",
                    "abgesagt",
                    "gelöscht",
                )
            )
            for turn in turns
        ),
        "local_cancellation_uses_terminwunsch": (
            "terminwunsch" in turns[5]["assistant"].casefold()
            and "terminwunsch" in turns[6]["assistant"].casefold()
        ),
        "spoken_dates_match_isolated_objects": (
            "3. september" in turns[2]["assistant"].casefold()
            and "2. september" in turns[3]["assistant"].casefold()
        ),
    }
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "real Anthropic reasoning and ElevenLabs TTS websocket path",
        "providers": ready["providers"],
        "checks": checks,
        "final_appointments": appointments,
        "turns": turns,
        "manual_validation": "REQUIRED_NOT_ATTESTED",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("live multi-appointment checks failed: " + ", ".join(failed))
    print(json.dumps({"checks": checks}, ensure_ascii=False, sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
