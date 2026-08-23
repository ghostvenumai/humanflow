"""Evidence-first runtime event envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from humanflow.domain.conversation import ConversationState


class EventType(StrEnum):
    USER_AUDIO_STARTED = "USER_AUDIO_STARTED"
    USER_AUDIO_STOPPED = "USER_AUDIO_STOPPED"
    PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
    FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT"
    TURN_CANDIDATE = "TURN_CANDIDATE"
    TURN_CONFIRMED = "TURN_CONFIRMED"
    BACKCHANNEL_DETECTED = "BACKCHANNEL_DETECTED"
    INTERRUPTION_CANDIDATE = "INTERRUPTION_CANDIDATE"
    INTERRUPTION_CONFIRMED = "INTERRUPTION_CONFIRMED"
    AGENT_GENERATION_STARTED = "AGENT_GENERATION_STARTED"
    FIRST_MODEL_OUTPUT = "FIRST_MODEL_OUTPUT"
    FIRST_AUDIO_CHUNK = "FIRST_AUDIO_CHUNK"
    AGENT_AUDIO_STARTED = "AGENT_AUDIO_STARTED"
    AGENT_AUDIO_CANCELLED = "AGENT_AUDIO_CANCELLED"
    AGENT_AUDIO_COMPLETED = "AGENT_AUDIO_COMPLETED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    STATE_TRANSITIONED = "STATE_TRANSITIONED"
    OPERATION_INVALIDATED = "OPERATION_INVALIDATED"
    STALE_RESULT_REJECTED = "STALE_RESULT_REJECTED"
    CALL_ENDED = "CALL_ENDED"


SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
SENSITIVE_KEY_SUFFIXES = (
    "_api_key",
    "_auth_token",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_access_token",
    "_refresh_token",
)


def _sanitize(value: Any, *, key: str = "") -> Any:
    normalized_key = key.lower().replace("-", "_")
    if normalized_key in SENSITIVE_KEY_NAMES or normalized_key.endswith(SENSITIVE_KEY_SUFFIXES):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(child_key): _sanitize(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """A serializable event with both wall-clock and monotonic time."""

    event_type: EventType
    conversation_id: str
    correlation_id: str
    sequence: int
    timestamp_utc: datetime
    monotonic_ns: int
    state: ConversationState
    reason_code: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if not self.correlation_id.strip():
            raise ValueError("correlation_id must not be empty")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        if not self.reason_code.strip():
            raise ValueError("reason_code must not be empty")
        sanitized = _sanitize(dict(self.payload))
        object.__setattr__(self, "payload", MappingProxyType(sanitized))

    @classmethod
    def create(
        cls,
        *,
        event_type: EventType,
        conversation_id: str,
        correlation_id: str,
        sequence: int,
        monotonic_ns: int,
        state: ConversationState,
        reason_code: str,
        payload: Mapping[str, Any] | None = None,
        timestamp_utc: datetime | None = None,
    ) -> "TelemetryEvent":
        return cls(
            event_type=event_type,
            conversation_id=conversation_id,
            correlation_id=correlation_id,
            sequence=sequence,
            timestamp_utc=timestamp_utc or datetime.now(UTC),
            monotonic_ns=monotonic_ns,
            state=state,
            reason_code=reason_code,
            payload=payload or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "conversation_id": self.conversation_id,
            "correlation_id": self.correlation_id,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "monotonic_ns": self.monotonic_ns,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "payload": dict(self.payload),
        }
