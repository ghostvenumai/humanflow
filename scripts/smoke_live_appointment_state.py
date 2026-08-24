#!/usr/bin/env python3
"""Verify the live demo's authoritative Friday-to-Thursday appointment deltas."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import websocket


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "live-appointment-state-smoke.json"
URL = "ws://127.0.0.1:8765/ws?tts=baseline"
TURNS = (
    "Ich möchte den Termin nächsten Freitag.",
    "Äh, machen wir übernächste Woche Donnerstag, das wäre besser.",
    "Am besten gegen 14 Uhr.",
    "Vielleicht 15 Uhr.",
)


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
                    "stream_id": "live-appointment-state-smoke",
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


def _acknowledge_audio(
    connection: websocket.WebSocket,
    metadata: dict[str, Any],
) -> None:
    chunk_id = metadata["chunk_id"]
    connection.send(json.dumps({"type": "playback_started", "chunk_id": chunk_id}))
    connection.send(
        json.dumps({"type": "playback_completed", "chunk_id": chunk_id})
    )


def _run_turn(connection: websocket.WebSocket, text: str) -> dict[str, Any]:
    _send_turn(connection, text)
    pending_audio: dict[str, Any] | None = None
    assistant_chunks: list[str] = []
    latest_state: dict[str, Any] | None = None
    updated_slots: list[str] = []
    while True:
        payload = _receive(connection)
        if isinstance(payload, bytes):
            if pending_audio is None:
                raise RuntimeError("audio bytes without metadata")
            _acknowledge_audio(connection, pending_audio)
            pending_audio = None
            continue
        if payload.get("type") == "audio_chunk":
            pending_audio = payload
            boundary = str(payload.get("text_boundary", "")).strip()
            if boundary:
                assistant_chunks.append(boundary)
        if payload.get("type") != "telemetry":
            continue
        event = payload["event"]
        if event["event_type"] == "APPOINTMENT_STATE_UPDATED":
            latest_state = dict(event["payload"]["state"])
            updated_slots = list(event["payload"]["updated_slots"])
        if event["event_type"] == "AGENT_AUDIO_COMPLETED":
            return {
                "user": text,
                "assistant": " ".join(assistant_chunks),
                "updated_slots": updated_slots,
                "state": latest_state,
            }


def _slot_value(state: dict[str, Any], name: str) -> str | None:
    slot = state.get(name)
    return str(slot["value"]) if isinstance(slot, dict) else None


def main() -> None:
    connection = websocket.create_connection(URL, timeout=45)
    try:
        ready: dict[str, Any] | None = None
        while ready is None:
            payload = _receive(connection)
            if isinstance(payload, dict) and payload.get("type") == "ready":
                ready = payload
        results = [_run_turn(connection, text) for text in TURNS]
    finally:
        connection.close()

    final_state = results[-1]["state"] or {}
    assistant_after_thursday = " ".join(
        result["assistant"].casefold() for result in results[1:]
    )
    checks = {
        "four_turns_completed": len(results) == 4,
        "final_date_is_thursday": _slot_value(final_state, "date") == "2026-09-10",
        "final_time_is_15": _slot_value(final_state, "time") == "15:00",
        "date_provenance_is_second_turn": (
            isinstance(final_state.get("date"), dict)
            and isinstance(results[1]["state"], dict)
            and isinstance(results[1]["state"].get("date"), dict)
            and final_state["date"]["source_turn"]
            == results[1]["state"]["date"]["source_turn"]
        ),
        "time_only_deltas_preserve_date": results[2]["updated_slots"]
        == ["time", "status"]
        and results[3]["updated_slots"] == ["time"],
        "old_friday_not_resurrected_in_later_assistant_text": (
            "freitag" not in assistant_after_thursday
        ),
        "spoken_date_matches_authoritative_state": (
            "10. september" in results[1]["assistant"].casefold()
        ),
        "assistant_never_claims_external_action": all(
            not any(
                marker in result["assistant"].casefold()
                for marker in (
                    "gebucht",
                    "eingetragen",
                    "storniert",
                    "abgesagt",
                    "gelöscht",
                )
            )
            for result in results
        ),
    }
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "real Anthropic reasoning and ElevenLabs TTS websocket path",
        "tts_ab_selection": ready["tts_ab_selection"],
        "checks": checks,
        "final_state": final_state,
        "turns": results,
        "manual_validation": "REQUIRED_NOT_ATTESTED",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("live appointment checks failed: " + ", ".join(failed))
    print(json.dumps({"checks": checks, "final_state": final_state}, sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
