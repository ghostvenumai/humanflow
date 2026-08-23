from __future__ import annotations

import pytest

from humanflow.controller.state_machine import ALLOWED_TRANSITIONS, ConversationStateMachine
from humanflow.domain.conversation import ConversationState, InvalidTransition
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink


def machine() -> tuple[ConversationStateMachine, InMemoryTelemetrySink]:
    values = iter(range(100, 10_000))
    sink = InMemoryTelemetrySink()
    return (
        ConversationStateMachine(
            conversation_id="call-1",
            sink=sink,
            clock_ns=lambda: next(values),
        ),
        sink,
    )


def test_transition_graph_covers_every_state() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(ConversationState)
    assert all(targets for targets in ALLOWED_TRANSITIONS.values())


def test_legal_path_emits_ordered_reason_coded_events() -> None:
    controller, sink = machine()
    controller.transition(
        ConversationState.LISTENING,
        reason_code="call_connected",
        correlation_id="turn-1",
    )
    event = controller.transition(
        ConversationState.POSSIBLE_END_OF_TURN,
        reason_code="silence_candidate",
        correlation_id="turn-1",
        payload={"silence_ms": 320},
    )

    assert controller.state is ConversationState.POSSIBLE_END_OF_TURN
    assert controller.revision == 2
    assert [item.sequence for item in sink.events] == [1, 2]
    assert event.event_type is EventType.STATE_TRANSITIONED
    assert event.payload["from_state"] == "LISTENING"
    assert event.payload["to_state"] == "POSSIBLE_END_OF_TURN"
    assert event.payload["silence_ms"] == 320


def test_illegal_and_self_transitions_fail_closed_without_telemetry() -> None:
    controller, sink = machine()

    with pytest.raises(InvalidTransition):
        controller.transition(
            ConversationState.SPEAKING,
            reason_code="skip_lifecycle",
            correlation_id="turn-1",
        )
    with pytest.raises(InvalidTransition):
        controller.transition(
            ConversationState.IDLE,
            reason_code="same_state",
            correlation_id="turn-1",
        )

    assert controller.state is ConversationState.IDLE
    assert controller.revision == 0
    assert sink.events == ()


def test_operation_epoch_rejects_late_async_result() -> None:
    controller, sink = machine()
    token = controller.issue_operation(kind="model_generation")

    assert controller.accept_result(token, correlation_id="turn-1") is True
    controller.invalidate_operations(
        reason_code="intentional_interruption",
        correlation_id="turn-1",
    )
    assert controller.accept_result(token, correlation_id="turn-1") is False

    assert [event.event_type for event in sink.events] == [
        EventType.OPERATION_INVALIDATED,
        EventType.STALE_RESULT_REJECTED,
    ]
    assert sink.events[-1].payload["token_epoch"] == 0
    assert sink.events[-1].payload["current_epoch"] == 1


def test_token_from_another_conversation_is_rejected() -> None:
    first, _ = machine()
    second_sink = InMemoryTelemetrySink()
    second = ConversationStateMachine(conversation_id="call-2", sink=second_sink)
    token = first.issue_operation(kind="tool")

    assert second.accept_result(token, correlation_id="tool-2") is False
    assert second_sink.events[0].event_type is EventType.STALE_RESULT_REJECTED

