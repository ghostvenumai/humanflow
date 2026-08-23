from __future__ import annotations

import pytest

from humanflow.evaluation.timeline import replay_timeline


def _event(
    sequence: int,
    event_type: str,
    state: str,
    monotonic_ns: int,
    *,
    correlation_id: str = "turn-1",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "conversation_id": "call-1",
        "correlation_id": correlation_id,
        "sequence": sequence,
        "monotonic_ns": monotonic_ns,
        "event_type": event_type,
        "state": state,
        "payload": payload or {},
    }


def test_replay_validates_state_and_derives_protected_event_metrics() -> None:
    events = [
        _event(
            1,
            "STATE_TRANSITIONED",
            "LISTENING",
            100,
            payload={"from_state": "IDLE", "to_state": "LISTENING"},
        ),
        _event(2, "TURN_CONFIRMED", "LISTENING", 200),
        _event(
            3,
            "STATE_TRANSITIONED",
            "THINKING",
            250,
            payload={"from_state": "LISTENING", "to_state": "THINKING"},
        ),
        _event(
            4,
            "STATE_TRANSITIONED",
            "SPEAKING",
            300,
            payload={"from_state": "THINKING", "to_state": "SPEAKING"},
        ),
        _event(5, "AGENT_AUDIO_STARTED", "SPEAKING", 700),
        _event(6, "INTERRUPTION_CANDIDATE", "SPEAKING", 800, correlation_id="barge-1"),
        _event(7, "AGENT_AUDIO_CANCELLED", "SPEAKING", 1_100, correlation_id="barge-1"),
    ]
    replay = replay_timeline(events)
    assert replay.ttfa_ms == (0.0005,)
    assert replay.audible_barge_in_latency_ms == (0.0003,)


def test_replay_rejects_sequence_gaps() -> None:
    with pytest.raises(ValueError, match="sequence gap"):
        replay_timeline(
            [
                _event(
                    2,
                    "STATE_TRANSITIONED",
                    "LISTENING",
                    100,
                    payload={"from_state": "IDLE", "to_state": "LISTENING"},
                )
            ]
        )
