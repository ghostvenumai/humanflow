from __future__ import annotations

import json
from datetime import UTC, datetime

from humanflow.domain.conversation import ConversationState
from humanflow.telemetry.events import EventType, TelemetryEvent
from humanflow.telemetry.sinks import JsonlTelemetrySink


def event(**payload: object) -> TelemetryEvent:
    return TelemetryEvent.create(
        event_type=EventType.TOOL_STARTED,
        conversation_id="call-1",
        correlation_id="tool-1",
        sequence=1,
        timestamp_utc=datetime(2026, 8, 23, 15, 0, tzinfo=UTC),
        monotonic_ns=123,
        state=ConversationState.TOOL_WAIT,
        reason_code="appointment_lookup",
        payload=payload,
    )


def test_sensitive_payload_keys_are_redacted_recursively() -> None:
    item = event(
        tool="lookup_customer",
        api_key="do-not-store",
        nested={"Authorization": "Bearer secret", "safe": "kept"},
    )

    assert item.payload["tool"] == "lookup_customer"
    assert item.payload["api_key"] == "[REDACTED]"
    assert item.payload["nested"]["Authorization"] == "[REDACTED]"
    assert item.payload["nested"]["safe"] == "kept"


def test_jsonl_sink_writes_replayable_event(tmp_path) -> None:
    path = tmp_path / "timeline.jsonl"
    sink = JsonlTelemetrySink(path)
    item = event(tool="lookup_customer")

    sink.emit(item)

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["event_id"] == item.event_id
    assert parsed["event_type"] == "TOOL_STARTED"
    assert parsed["timestamp_utc"] == "2026-08-23T15:00:00Z"
    assert parsed["payload"] == {"tool": "lookup_customer"}

