#!/usr/bin/env python3
"""Verify live-provider follow-up after websocket cancellation acknowledgement."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import websocket


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "live-barge-in-smoke.json"
URL = "ws://127.0.0.1:8765/ws"


def send_transcript(connection: websocket.WebSocket, text: str) -> None:
    connection.send(
        json.dumps(
            {
                "type": "transcript",
                "source": "diagnostic_smoke",
                "text": text,
                "final": True,
                "signals": {
                    "speech_active": False,
                    "silence_duration_ms": 0,
                    "utterance_duration_ms": 900,
                    "semantic_complete": True,
                    "provider_endpointed": True,
                    "acoustic_completion": 1.0,
                    "interruption_probability": (
                        1.0 if text.casefold().startswith("moment") else 0.0
                    ),
                },
            }
        )
    )


def receive_payload(connection: websocket.WebSocket) -> dict[str, Any] | bytes:
    message = connection.recv()
    if isinstance(message, bytes):
        return message
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise RuntimeError("websocket JSON payload is not an object")
    if payload.get("type") == "error":
        raise RuntimeError(f"live pipeline error: {payload}")
    return payload


def contains_56(text: str) -> bool:
    normalized = "".join(character for character in text.casefold() if character.isalnum())
    return "56" in normalized or "sechsundfünfzig" in normalized


def main() -> None:
    connection = websocket.create_connection(URL, timeout=30)
    try:
        while True:
            ready = receive_payload(connection)
            if isinstance(ready, dict) and ready.get("type") == "ready":
                break
        connection.send(
            json.dumps(
                {
                    "type": "provider_capabilities",
                    "stt_available": True,
                    "tts_available": True,
                }
            )
        )
        send_transcript(
            connection,
            "Erkläre ausführlich, wie ein modernes Warenlager organisiert wird.",
        )

        initial_chunk: dict[str, Any] | None = None
        while initial_chunk is None:
            payload = receive_payload(connection)
            if isinstance(payload, dict) and payload.get("type") == "audio_chunk":
                initial_chunk = payload
        audio = receive_payload(connection)
        if not isinstance(audio, bytes):
            raise RuntimeError("initial audio envelope missing")
        initial_chunk_id = str(initial_chunk["chunk_id"])
        connection.send(
            json.dumps({"type": "playback_started", "chunk_id": initial_chunk_id})
        )
        send_transcript(connection, "Moment, stopp. Was ist sieben mal acht?")

        cancel_seen = False
        cancellation_event: dict[str, Any] | None = None
        followup_confirmed = False
        followup_chunks: list[str] = []
        pending_followup: dict[str, Any] | None = None
        first_followup_output_ms: float | None = None
        completed = False
        while not completed:
            payload = receive_payload(connection)
            if isinstance(payload, bytes):
                if pending_followup is None:
                    raise RuntimeError("follow-up audio bytes arrived without metadata")
                chunk_id = str(pending_followup["chunk_id"])
                connection.send(
                    json.dumps({"type": "playback_started", "chunk_id": chunk_id})
                )
                connection.send(
                    json.dumps({"type": "playback_completed", "chunk_id": chunk_id})
                )
                pending_followup = None
                continue
            message_type = payload.get("type")
            if message_type == "cancel_audio":
                if str(payload["chunk_id"]) != initial_chunk_id:
                    raise RuntimeError("unexpected cancellation chunk")
                cancel_seen = True
                connection.send(
                    json.dumps(
                        {
                            "type": "playback_stopped",
                            "chunk_id": initial_chunk_id,
                            "played_samples": 0,
                        }
                    )
                )
                continue
            if message_type == "audio_chunk":
                pending_followup = payload
                followup_chunks.append(str(payload["text_boundary"]))
                continue
            if message_type != "telemetry":
                continue
            event = payload["event"]
            if event["event_type"] == "RECOVERY_STARTED":
                raise RuntimeError(f"unexpected recovery: {event['payload']}")
            if event["event_type"] == "AGENT_AUDIO_CANCELLED":
                cancellation_event = event
            elif (
                event["event_type"] == "TURN_CONFIRMED"
                and event["reason_code"] == "barge_in_followup_complete"
            ):
                followup_confirmed = True
            elif (
                event["event_type"] == "FIRST_MODEL_OUTPUT"
                and followup_confirmed
            ):
                first_followup_output_ms = float(
                    event["payload"]["provider_latency_ms"]
                )
            elif event["event_type"] == "AGENT_AUDIO_COMPLETED" and followup_chunks:
                completed = True
    finally:
        connection.close()

    response = " ".join(followup_chunks)
    checks = {
        "cancel_message_seen": cancel_seen,
        "server_cancellation_event_seen": cancellation_event is not None,
        "followup_confirmed_after_cancel": followup_confirmed,
        "followup_answer_contains_56": contains_56(response),
        "no_canned_acknowledgement": "ich habe sie verstanden" not in response.casefold(),
    }
    assert cancellation_event is not None
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "real reasoning plus websocket cancellation; browser/audio device excluded",
        "manual_validation": "REQUIRED_NOT_ATTESTED",
        "checks": checks,
        "metrics": {
            "controller_stop_ack_latency_ms": cancellation_event["payload"][
                "audible_barge_in_latency_ms"
            ],
            "followup_first_model_output_ms": first_followup_output_ms,
            "reported_played_samples": cancellation_event["payload"]["played_samples"],
        },
        "followup": {
            "transcript": "Was ist sieben mal acht?",
            "response": response,
            "speech_chunks": len(followup_chunks),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("live barge-in checks failed: " + ", ".join(failed))
    print(json.dumps({"checks": checks, "metrics": report["metrics"]}, sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
