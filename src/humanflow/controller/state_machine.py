"""Explicit, observable conversation state machine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from humanflow.domain.conversation import ConversationState, InvalidTransition, OperationToken
from humanflow.telemetry.events import EventType, TelemetryEvent
from humanflow.telemetry.sinks import TelemetrySink


ALLOWED_TRANSITIONS: Mapping[ConversationState, frozenset[ConversationState]] = {
    ConversationState.IDLE: frozenset(
        {ConversationState.LISTENING, ConversationState.DISCONNECTING, ConversationState.FAILED}
    ),
    ConversationState.LISTENING: frozenset(
        {
            ConversationState.POSSIBLE_END_OF_TURN,
            ConversationState.THINKING,
            ConversationState.OVERLAP,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.POSSIBLE_END_OF_TURN: frozenset(
        {
            ConversationState.LISTENING,
            ConversationState.THINKING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.THINKING: frozenset(
        {
            ConversationState.SPEAKING,
            ConversationState.TOOL_WAIT,
            ConversationState.LISTENING,
            ConversationState.INTERRUPTED,
            ConversationState.RECOVERING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.SPEAKING: frozenset(
        {
            ConversationState.POSSIBLE_INTERRUPTION,
            ConversationState.INTERRUPTED,
            ConversationState.OVERLAP,
            ConversationState.LISTENING,
            ConversationState.RECOVERING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.POSSIBLE_INTERRUPTION: frozenset(
        {
            ConversationState.SPEAKING,
            ConversationState.INTERRUPTED,
            ConversationState.OVERLAP,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.INTERRUPTED: frozenset(
        {
            ConversationState.LISTENING,
            ConversationState.THINKING,
            ConversationState.OVERLAP,
            ConversationState.RECOVERING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.OVERLAP: frozenset(
        {
            ConversationState.LISTENING,
            ConversationState.SPEAKING,
            ConversationState.INTERRUPTED,
            ConversationState.RECOVERING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.TOOL_WAIT: frozenset(
        {
            ConversationState.SPEAKING,
            ConversationState.THINKING,
            ConversationState.LISTENING,
            ConversationState.RECOVERING,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.RECOVERING: frozenset(
        {
            ConversationState.LISTENING,
            ConversationState.THINKING,
            ConversationState.SPEAKING,
            ConversationState.HANDOFF,
            ConversationState.DISCONNECTING,
            ConversationState.FAILED,
        }
    ),
    ConversationState.HANDOFF: frozenset(
        {ConversationState.LISTENING, ConversationState.DISCONNECTING, ConversationState.FAILED}
    ),
    ConversationState.DISCONNECTING: frozenset({ConversationState.IDLE, ConversationState.FAILED}),
    ConversationState.FAILED: frozenset(
        {ConversationState.RECOVERING, ConversationState.HANDOFF, ConversationState.DISCONNECTING}
    ),
}


class ConversationStateMachine:
    def __init__(
        self,
        *,
        conversation_id: str,
        sink: TelemetrySink,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        self.conversation_id = conversation_id
        self.state = ConversationState.IDLE
        self.revision = 0
        self._sequence = 0
        self._operation_epoch = 0
        self._sink = sink
        self._clock_ns = clock_ns

    @property
    def operation_epoch(self) -> int:
        return self._operation_epoch

    def transition(
        self,
        target: ConversationState,
        *,
        reason_code: str,
        correlation_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> TelemetryEvent:
        source = self.state
        if target == source or target not in ALLOWED_TRANSITIONS[source]:
            raise InvalidTransition(f"{source.value} -> {target.value} is not allowed")
        if not reason_code.strip():
            raise ValueError("reason_code must not be empty")
        self.state = target
        self.revision += 1
        details = dict(payload or {})
        details.update(
            {
                "from_state": source.value,
                "to_state": target.value,
                "revision": self.revision,
                "operation_epoch": self._operation_epoch,
            }
        )
        return self._emit(
            EventType.STATE_TRANSITIONED,
            correlation_id=correlation_id,
            reason_code=reason_code,
            payload=details,
        )

    def issue_operation(self, *, kind: str) -> OperationToken:
        return OperationToken(
            conversation_id=self.conversation_id,
            epoch=self._operation_epoch,
            operation_id=str(uuid4()),
            kind=kind,
        )

    def invalidate_operations(self, *, reason_code: str, correlation_id: str) -> TelemetryEvent:
        previous_epoch = self._operation_epoch
        self._operation_epoch += 1
        return self._emit(
            EventType.OPERATION_INVALIDATED,
            correlation_id=correlation_id,
            reason_code=reason_code,
            payload={"previous_epoch": previous_epoch, "new_epoch": self._operation_epoch},
        )

    def accept_result(self, token: OperationToken, *, correlation_id: str) -> bool:
        current = token.conversation_id == self.conversation_id and token.epoch == self._operation_epoch
        if not current:
            self._emit(
                EventType.STALE_RESULT_REJECTED,
                correlation_id=correlation_id,
                reason_code="operation_epoch_mismatch",
                payload={
                    "operation_id": token.operation_id,
                    "operation_kind": token.kind,
                    "token_epoch": token.epoch,
                    "current_epoch": self._operation_epoch,
                },
            )
        return current

    def _emit(
        self,
        event_type: EventType,
        *,
        correlation_id: str,
        reason_code: str,
        payload: Mapping[str, Any],
    ) -> TelemetryEvent:
        self._sequence += 1
        event = TelemetryEvent.create(
            event_type=event_type,
            conversation_id=self.conversation_id,
            correlation_id=correlation_id,
            sequence=self._sequence,
            monotonic_ns=self._clock_ns(),
            state=self.state,
            reason_code=reason_code,
            payload=payload,
        )
        self._sink.emit(event)
        return event

