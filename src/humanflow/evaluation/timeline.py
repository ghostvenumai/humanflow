"""Strict replay validation and metrics derived from raw telemetry events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from humanflow.controller.state_machine import ALLOWED_TRANSITIONS
from humanflow.domain.conversation import ConversationState
from humanflow.telemetry.events import EventType


@dataclass(frozen=True, slots=True)
class TimelineReplayResult:
    events: int
    conversations: int
    ttfa_ms: tuple[float, ...]
    audible_barge_in_latency_ms: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "conversations": self.conversations,
            "ttfa_ms": list(self.ttfa_ms),
            "audible_barge_in_latency_ms": list(self.audible_barge_in_latency_ms),
        }


def load_jsonl_events(path: Path) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"event must be an object at {path}:{line_number}")
        events.append(payload)
    if not events:
        raise ValueError("timeline must not be empty")
    return tuple(events)


def replay_timeline(events: Iterable[Mapping[str, Any]]) -> TimelineReplayResult:
    states: dict[str, ConversationState] = {}
    sequences: dict[str, int] = {}
    clocks: dict[str, int] = {}
    turn_started: dict[tuple[str, str], int] = {}
    interruption_started: dict[tuple[str, str], int] = {}
    ttfa_ms: list[float] = []
    barge_ms: list[float] = []
    event_count = 0

    for event in events:
        event_count += 1
        try:
            conversation_id = str(event["conversation_id"])
            correlation_id = str(event["correlation_id"])
            sequence = int(event["sequence"])
            timestamp_ns = int(event["monotonic_ns"])
            event_type = EventType(str(event["event_type"]))
            recorded_state = ConversationState(str(event["state"]))
            payload = event.get("payload", {})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid telemetry envelope at event {event_count}") from error
        if not conversation_id or not correlation_id or not isinstance(payload, Mapping):
            raise ValueError(f"invalid telemetry identity/payload at event {event_count}")
        expected_sequence = sequences.get(conversation_id, 0) + 1
        if sequence != expected_sequence:
            raise ValueError(
                f"sequence gap for {conversation_id}: expected {expected_sequence}, got {sequence}"
            )
        if timestamp_ns < clocks.get(conversation_id, 0):
            raise ValueError(f"monotonic time moved backwards for {conversation_id}")
        sequences[conversation_id] = sequence
        clocks[conversation_id] = timestamp_ns
        current_state = states.get(conversation_id, ConversationState.IDLE)

        if event_type is EventType.STATE_TRANSITIONED:
            try:
                source = ConversationState(str(payload["from_state"]))
                target = ConversationState(str(payload["to_state"]))
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid transition payload at event {event_count}") from error
            if source is not current_state:
                raise ValueError(
                    f"transition source mismatch for {conversation_id}: {source} != {current_state}"
                )
            if target not in ALLOWED_TRANSITIONS[source]:
                raise ValueError(f"illegal replay transition: {source} -> {target}")
            if recorded_state is not target:
                raise ValueError("transition event state does not match target")
            states[conversation_id] = target
        elif recorded_state is not current_state:
            raise ValueError(
                f"event state mismatch for {conversation_id}: {recorded_state} != {current_state}"
            )

        key = (conversation_id, correlation_id)
        if event_type is EventType.TURN_CONFIRMED:
            turn_started[key] = timestamp_ns
        elif event_type is EventType.AGENT_AUDIO_STARTED and key in turn_started:
            ttfa_ms.append((timestamp_ns - turn_started.pop(key)) / 1_000_000.0)
        elif event_type is EventType.INTERRUPTION_CANDIDATE:
            interruption_started[key] = timestamp_ns
        elif event_type is EventType.AGENT_AUDIO_CANCELLED:
            if key not in interruption_started:
                raise ValueError("audio cancellation has no correlated interruption candidate")
            barge_ms.append((timestamp_ns - interruption_started.pop(key)) / 1_000_000.0)

    return TimelineReplayResult(
        events=event_count,
        conversations=len(states),
        ttfa_ms=tuple(ttfa_ms),
        audible_barge_in_latency_ms=tuple(barge_ms),
    )
