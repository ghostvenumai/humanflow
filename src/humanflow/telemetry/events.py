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
    POSSIBLE_INTERRUPTION = "POSSIBLE_INTERRUPTION"
    PLAYBACK_DUCK_REQUESTED = "PLAYBACK_DUCK_REQUESTED"
    PLAYBACK_DUCK_STARTED = "PLAYBACK_DUCK_STARTED"
    PLAYBACK_RESUME_REQUESTED = "PLAYBACK_RESUME_REQUESTED"
    PLAYBACK_RESUMED = "PLAYBACK_RESUMED"
    BACKCHANNEL_RECOVERY = "BACKCHANNEL_RECOVERY"
    FALSE_INTERRUPTION_DETECTED = "FALSE_INTERRUPTION_DETECTED"
    PROVIDER_STATUS = "PROVIDER_STATUS"
    STT_PROVIDER_FAILED = "STT_PROVIDER_FAILED"
    PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
    FINAL_TRANSCRIPT = "FINAL_TRANSCRIPT"
    TURN_CANDIDATE = "TURN_CANDIDATE"
    TURN_CONFIRMED = "TURN_CONFIRMED"
    APPOINTMENT_STATE_UPDATED = "APPOINTMENT_STATE_UPDATED"
    BACKCHANNEL_DETECTED = "BACKCHANNEL_DETECTED"
    INTERRUPTION_CANDIDATE = "INTERRUPTION_CANDIDATE"
    TAKEOVER_EVIDENCE = "TAKEOVER_EVIDENCE"
    INTERRUPTION_CONFIRMED = "INTERRUPTION_CONFIRMED"
    AUDIO_CANCEL_SIGNAL = "AUDIO_CANCEL_SIGNAL"
    AUDIBLE_STOP_ACK = "AUDIBLE_STOP_ACK"
    AGENT_GENERATION_STARTED = "AGENT_GENERATION_STARTED"
    AGENT_GENERATION_COMPLETED = "AGENT_GENERATION_COMPLETED"
    FIRST_MODEL_OUTPUT = "FIRST_MODEL_OUTPUT"
    SEMANTIC_CHUNK_READY = "SEMANTIC_CHUNK_READY"
    TTS_REQUEST_STARTED = "TTS_REQUEST_STARTED"
    TTS_PROVIDER_FALLBACK = "TTS_PROVIDER_FALLBACK"
    TTS_PROVIDER_ACTIVATED = "TTS_PROVIDER_ACTIVATED"
    TTS_PROVIDER_DEACTIVATED = "TTS_PROVIDER_DEACTIVATED"
    FIRST_AUDIO_CHUNK = "FIRST_AUDIO_CHUNK"
    PLAYBACK_OWNER_CREATED = "PLAYBACK_OWNER_CREATED"
    PLAYBACK_OWNER_DESTROYED = "PLAYBACK_OWNER_DESTROYED"
    AUDIO_CHUNK_RECEIVED = "AUDIO_CHUNK_RECEIVED"
    AUDIO_CHUNK_SCHEDULED = "AUDIO_CHUNK_SCHEDULED"
    AUDIO_CHUNK_PLAYED = "AUDIO_CHUNK_PLAYED"
    AUDIO_SEGMENT_METRICS = "AUDIO_SEGMENT_METRICS"
    PLAYBACK_UNDERRUN = "PLAYBACK_UNDERRUN"
    DUPLICATE_CHUNK_REJECTED = "DUPLICATE_CHUNK_REJECTED"
    STALE_CHUNK_REJECTED = "STALE_CHUNK_REJECTED"
    DUPLICATE_TRANSCRIPT_REJECTED = "DUPLICATE_TRANSCRIPT_REJECTED"
    TRANSCRIPT_PROVENANCE_RECORDED = "TRANSCRIPT_PROVENANCE_RECORDED"
    TRANSCRIPT_REJECTED = "TRANSCRIPT_REJECTED"
    SELF_SPEECH_CANDIDATE = "SELF_SPEECH_CANDIDATE"
    SELF_SPEECH_SUPPRESSED = "SELF_SPEECH_SUPPRESSED"
    SELF_SPEECH_ACCEPTED_AS_REAL_USER = "SELF_SPEECH_ACCEPTED_AS_REAL_USER"
    AGENT_AUDIO_STARTED = "AGENT_AUDIO_STARTED"
    PLAYBACK_SINK_METRICS = "PLAYBACK_SINK_METRICS"
    AGENT_AUDIO_CANCELLED = "AGENT_AUDIO_CANCELLED"
    AGENT_AUDIO_STOP_UNCONFIRMED = "AGENT_AUDIO_STOP_UNCONFIRMED"
    AGENT_AUDIO_COMPLETED = "AGENT_AUDIO_COMPLETED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    AVAILABILITY_QUERIED = "AVAILABILITY_QUERIED"
    APPOINTMENT_CREATED = "APPOINTMENT_CREATED"
    APPOINTMENT_RESCHEDULED = "APPOINTMENT_RESCHEDULED"
    APPOINTMENT_CANCELLED = "APPOINTMENT_CANCELLED"
    APPOINTMENTS_LISTED = "APPOINTMENTS_LISTED"
    BOOKING_CONFLICT_DETECTED = "BOOKING_CONFLICT_DETECTED"
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
