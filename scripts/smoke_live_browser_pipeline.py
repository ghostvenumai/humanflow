#!/usr/bin/env python3
"""Exercise the live websocket reasoning path without claiming browser audio quality."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import websocket


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "live-provider-smoke.json"
URL = "ws://127.0.0.1:8765/ws"
PROMPTS = (
    ("capabilities", "Was kannst du?"),
    ("weather_limit", "Wie ist das Wetter?"),
    ("arithmetic", "Was ist 25 mal 17?"),
    ("erp", "Kannst du mir erklären, was ein ERP-System ist?"),
    ("context_followup", "Wie lautete noch einmal das Ergebnis meiner Rechenaufgabe?"),
    ("appointment", "Ich möchte einen Termin am Freitag."),
)


def receive_json(connection: websocket.WebSocket) -> dict[str, Any]:
    message = connection.recv()
    if isinstance(message, bytes):
        raise RuntimeError("unexpected binary frame without audio metadata")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise RuntimeError("websocket JSON payload is not an object")
    return payload


def wait_until_ready(connection: websocket.WebSocket) -> dict[str, Any]:
    while True:
        payload = receive_json(connection)
        if payload.get("type") == "ready":
            return payload


def run_turn(connection: websocket.WebSocket, prompt: str) -> dict[str, Any]:
    print(f"live_turn_started={prompt}", flush=True)
    connection.send(
        json.dumps(
            {
                "type": "transcript",
                "source": "diagnostic_smoke",
                "text": prompt,
                "final": True,
                "provenance": {
                    "transcript_id": str(uuid4()),
                    "event_kind": "USER_TRANSCRIPT_FINAL",
                    "source": "diagnostic_smoke",
                    "origin": "DIAGNOSTIC_TEXT_INPUT",
                    "stream_id": "live-provider-smoke-text-input",
                },
                "signals": {
                    "speech_active": False,
                    "silence_duration_ms": 350,
                    "utterance_duration_ms": 900,
                    "semantic_complete": True,
                    "acoustic_completion": 0.9,
                },
            }
        )
    )
    chunks: list[str] = []
    first_output_ms: float | None = None
    generation_ms: float | None = None
    usage: dict[str, int] | None = None
    pending_chunk: dict[str, Any] | None = None
    messages_seen = 0
    last_milestone = "transcript_sent"
    turn_started = time.monotonic()
    event_counts: dict[str, int] = {}
    connection.settimeout(1.0)
    while True:
        try:
            message = connection.recv()
        except websocket.WebSocketTimeoutException:
            if time.monotonic() - turn_started > 90:
                raise TimeoutError(
                    "live turn exceeded 90 seconds "
                    f"after {last_milestone}; event_counts={event_counts}"
                ) from None
            continue
        messages_seen += 1
        if time.monotonic() - turn_started > 90:
            raise TimeoutError(
                "live turn exceeded 90 seconds "
                f"after {last_milestone}; event_counts={event_counts}"
            )
        if messages_seen > 2_000:
            raise RuntimeError(
                f"live turn message limit exceeded after {last_milestone}"
            )
        if isinstance(message, bytes):
            if pending_chunk is None:
                raise RuntimeError("audio bytes arrived without metadata")
            chunk_id = str(pending_chunk["chunk_id"])
            connection.send(json.dumps({"type": "playback_started", "chunk_id": chunk_id}))
            decoded_duration_ms = float(pending_chunk["decoded_duration_ms"])
            time.sleep(decoded_duration_ms / 1_000.0)
            connection.send(json.dumps({"type": "playback_completed", "chunk_id": chunk_id}))
            pending_chunk = None
            continue
        payload = json.loads(message)
        if payload.get("type") == "error":
            raise RuntimeError(f"live pipeline error: {payload}")
        if payload.get("type") == "audio_chunk":
            pending_chunk = payload
            chunks.append(str(payload["text_boundary"]))
            last_milestone = "audio_chunk"
            continue
        if payload.get("type") != "telemetry":
            continue
        event = payload["event"]
        event_type = event["event_type"]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        if event_type == "FIRST_MODEL_OUTPUT":
            first_output_ms = float(event["payload"]["provider_latency_ms"])
            last_milestone = "first_model_output"
        elif event_type == "TTS_REQUEST_STARTED":
            last_milestone = "tts_request_started"
        elif event_type == "FIRST_AUDIO_CHUNK":
            last_milestone = "first_audio_chunk"
        elif event_type == "AGENT_GENERATION_COMPLETED":
            generation_ms = float(event["payload"]["duration_ms"])
            last_milestone = "generation_completed"
            raw_usage = event["payload"].get("usage")
            if isinstance(raw_usage, dict):
                usage = {key: int(value) for key, value in raw_usage.items()}
        elif event_type == "RECOVERY_STARTED":
            raise RuntimeError(
                "provider recovery entered "
                f"after {last_milestone}; event_counts={event_counts}; "
                f"payload={event['payload']}"
            )
        elif event_type == "AGENT_AUDIO_COMPLETED":
            break
    print(
        f"live_turn_completed={prompt} messages={messages_seen} chunks={len(chunks)}",
        flush=True,
    )
    return {
        "prompt": prompt,
        "response": " ".join(chunks),
        "speech_chunks": len(chunks),
        "first_model_output_ms": first_output_ms,
        "generation_ms": generation_ms,
        "usage": usage,
    }


def semantic_checks(results: dict[str, dict[str, Any]]) -> dict[str, bool]:
    canned = "ich habe sie verstanden"
    responses = {name: result["response"].casefold() for name, result in results.items()}
    return {
        "all_responses_nonempty": all(len(text.strip()) >= 10 for text in responses.values()),
        "no_canned_acknowledgement": all(canned not in text for text in responses.values()),
        "plain_speech_has_no_markdown": all(
            not any(marker in text for marker in ("*", "#", "`"))
            for text in responses.values()
        ),
        "arithmetic_contains_425": _contains_425(responses["arithmetic"]),
        "erp_is_semantically_relevant": all(
            term in responses["erp"] for term in ("erp", "unternehmen")
        ),
        "weather_states_live_data_limit": any(
            term in responses["weather_limit"]
            for term in ("keinen zugriff", "nicht auf", "keine live", "standort")
        ),
        "followup_uses_prior_math_context": _contains_425(
            responses["context_followup"]
        ),
        "appointment_is_semantically_relevant": all(
            term in responses["appointment"] for term in ("termin", "freitag")
        ),
    }


def _contains_425(text: str) -> bool:
    normalized = "".join(character for character in text.casefold() if character.isalnum())
    return "425" in normalized or "vierhundertfünfundzwanzig" in normalized


def main() -> None:
    connection = websocket.create_connection(URL, timeout=30)
    try:
        ready = wait_until_ready(connection)
        connection.send(
            json.dumps(
                {
                    "type": "provider_capabilities",
                    "stt_available": True,
                    "tts_available": True,
                }
            )
        )
        results = {name: run_turn(connection, prompt) for name, prompt in PROMPTS}
    finally:
        connection.close()

    checks = semantic_checks(results)
    first_output = [
        result["first_model_output_ms"]
        for result in results.values()
        if result["first_model_output_ms"] is not None
    ]
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "real reasoning websocket path; browser microphone/STT/TTS excluded",
        "manual_validation": "REQUIRED_NOT_ATTESTED",
        "providers": ready["providers"],
        "checks": checks,
        "summary": {
            "turns": len(results),
            "first_model_output_ms_median": round(statistics.median(first_output), 3),
            "first_model_output_ms_max": round(max(first_output), 3),
            "input_tokens": sum(
                (result["usage"] or {}).get("input_tokens", 0)
                for result in results.values()
            ),
            "output_tokens": sum(
                (result["usage"] or {}).get("output_tokens", 0)
                for result in results.values()
            ),
        },
        "turns": results,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("semantic live checks failed: " + ", ".join(failed))
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"report={REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
