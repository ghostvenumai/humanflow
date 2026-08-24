"""Immutable transcript provenance and role boundaries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic_ns
from uuid import uuid4


class ConversationEventKind(StrEnum):
    USER_AUDIO = "USER_AUDIO"
    USER_TRANSCRIPT_PARTIAL = "USER_TRANSCRIPT_PARTIAL"
    USER_TRANSCRIPT_FINAL = "USER_TRANSCRIPT_FINAL"
    ASSISTANT_TEXT = "ASSISTANT_TEXT"
    ASSISTANT_TTS_AUDIO = "ASSISTANT_TTS_AUDIO"
    ASSISTANT_PLAYBACK = "ASSISTANT_PLAYBACK"
    SYSTEM_EVENT = "SYSTEM_EVENT"


class TranscriptOrigin(StrEnum):
    BROWSER_SPEECH_RECOGNITION = "BROWSER_SPEECH_RECOGNITION"
    STREAMING_STT_PROVIDER = "STREAMING_STT_PROVIDER"
    DIAGNOSTIC_TEXT_INPUT = "DIAGNOSTIC_TEXT_INPUT"
    ASSISTANT_REASONING = "ASSISTANT_REASONING"
    ASSISTANT_TTS = "ASSISTANT_TTS"
    SYSTEM = "SYSTEM"


USER_TRANSCRIPT_KINDS = frozenset(
    {
        ConversationEventKind.USER_TRANSCRIPT_PARTIAL,
        ConversationEventKind.USER_TRANSCRIPT_FINAL,
    }
)
ALLOWED_USER_SOURCE_BINDINGS = {
    TranscriptOrigin.STREAMING_STT_PROVIDER: frozenset({"streaming_stt"}),
    TranscriptOrigin.DIAGNOSTIC_TEXT_INPUT: frozenset(
        {
            "manual_diagnostic",
            "diagnostic_smoke",
            "test_fixture",
            "evaluation_fixture",
            "replay_fixture",
        }
    ),
}


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\wäöüß]+", normalized, flags=re.UNICODE))


@dataclass(frozen=True, slots=True)
class TranscriptProvenance:
    transcript_id: str
    event_kind: ConversationEventKind
    source: str
    origin: TranscriptOrigin
    stream_id: str
    timestamp_ns: int
    browser_recognition_session_id: str | None = None
    audio_capture_id: str | None = None
    response_id: str | None = None
    stt_session_id: str | None = None
    browser_timestamp_ms: float | None = None
    recognition_input_binding: str | None = None
    audio_frame_sequence: int | None = None
    provider_event_type: str | None = None

    def __post_init__(self) -> None:
        for name in ("transcript_id", "source", "stream_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.browser_timestamp_ms is not None and self.browser_timestamp_ms < 0:
            raise ValueError("browser_timestamp_ms must be non-negative")
        if self.origin is TranscriptOrigin.BROWSER_SPEECH_RECOGNITION:
            if not self.browser_recognition_session_id:
                raise ValueError("browser recognition session id required")
            if not self.audio_capture_id:
                raise ValueError("browser audio capture id required")
        if self.origin is TranscriptOrigin.STREAMING_STT_PROVIDER:
            if not self.stt_session_id:
                raise ValueError("streaming STT session id required")
            if not self.audio_capture_id:
                raise ValueError("streaming STT audio capture id required")

    @property
    def is_allowlisted_user_input(self) -> bool:
        return (
            self.event_kind in USER_TRANSCRIPT_KINDS
            and self.source in ALLOWED_USER_SOURCE_BINDINGS.get(self.origin, frozenset())
        )

    @property
    def is_assistant_origin(self) -> bool:
        return self.event_kind in {
            ConversationEventKind.ASSISTANT_TEXT,
            ConversationEventKind.ASSISTANT_TTS_AUDIO,
            ConversationEventKind.ASSISTANT_PLAYBACK,
        } or self.origin in {
            TranscriptOrigin.ASSISTANT_REASONING,
            TranscriptOrigin.ASSISTANT_TTS,
        }

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "transcript_id": self.transcript_id,
            "event_kind": self.event_kind.value,
            "source": self.source,
            "origin": self.origin.value,
            "stream_id": self.stream_id,
            "browser_recognition_session_id": self.browser_recognition_session_id,
            "audio_capture_id": self.audio_capture_id,
            "response_id": self.response_id,
            "stt_session_id": self.stt_session_id,
            "timestamp_ns": self.timestamp_ns,
            "browser_timestamp_ms": self.browser_timestamp_ms,
            "recognition_input_binding": self.recognition_input_binding,
            "audio_frame_sequence": self.audio_frame_sequence,
            "provider_event_type": self.provider_event_type,
        }

    @classmethod
    def user_fixture(
        cls,
        *,
        final: bool,
        source: str = "test_fixture",
        stream_id: str = "fixture-text-input",
    ) -> "TranscriptProvenance":
        return cls(
            transcript_id=str(uuid4()),
            event_kind=(
                ConversationEventKind.USER_TRANSCRIPT_FINAL
                if final
                else ConversationEventKind.USER_TRANSCRIPT_PARTIAL
            ),
            source=source,
            origin=TranscriptOrigin.DIAGNOSTIC_TEXT_INPUT,
            stream_id=stream_id,
            timestamp_ns=monotonic_ns(),
        )


class TranscriptRejected(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
