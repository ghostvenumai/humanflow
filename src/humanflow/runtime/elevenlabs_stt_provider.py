"""Source-bound realtime PCM transcription with ElevenLabs Scribe v2."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections import deque
from collections.abc import Awaitable, Callable
from time import monotonic_ns
from typing import Any
from urllib.parse import urlencode

from humanflow.audio.models import AudioFrame
from humanflow.turns.models import TurnSignals

from .providers import (
    ProviderInfo,
    ProviderMode,
    STTProviderCapabilities,
    TranscriptUpdate,
)
from .transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
    TranscriptProvenance,
)


DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2_realtime"
DEFAULT_ELEVENLABS_STT_ENDPOINT = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
)
_INTERRUPTION_PREFIX = re.compile(
    r"^\s*(?:(?:nein\s+)?stopp|moment|warte(?:\s+mal)?|halt)\b",
    re.IGNORECASE,
)
_FILLER_ENDING = re.compile(r"(?:\b(?:äh|ähm|also|und))\s*[,.!?-]*$", re.IGNORECASE)
_ERROR_EVENTS = frozenset(
    {
        "auth_error",
        "quota_exceeded",
        "transcriber_error",
        "input_error",
        "invalid_request",
        "error",
        "commit_throttled",
        "unaccepted_terms",
        "rate_limited",
        "queue_overflow",
        "resource_exhausted",
        "session_time_limit_exceeded",
        "chunk_size_exceeded",
        "insufficient_audio_activity",
    }
)


class ElevenLabsRealtimeSTTError(RuntimeError):
    """Credential-safe provider failure exposing only a bounded event code."""

    def __init__(self, provider_code: str) -> None:
        safe_code = (
            provider_code
            if re.fullmatch(r"[a-z0-9_.-]{1,80}", provider_code)
            else "unknown_provider_error"
        )
        self.provider_code = safe_code
        super().__init__(f"elevenlabs_stt_{safe_code}")


ConnectFactory = Callable[..., Awaitable[Any]]


class ElevenLabsRealtimeSTTProvider:
    """Send the authoritative browser PCM stream to Scribe over one WebSocket."""

    capabilities = STTProviderCapabilities(
        streaming_audio=True,
        source_bound=True,
        partial_transcripts=True,
        final_transcripts=True,
        provider_endpointing=True,
        cancellation=True,
    )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ELEVENLABS_STT_MODEL,
        language_code: str = "de",
        endpoint: str = DEFAULT_ELEVENLABS_STT_ENDPOINT,
        vad_threshold: float = 0.4,
        vad_silence_threshold_seconds: float = 0.65,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 100,
        connect_timeout_seconds: float = 8.0,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ElevenLabs API key is required for realtime STT")
        if not model.strip() or not language_code.strip() or not endpoint.strip():
            raise ValueError("STT model, language, and endpoint are required")
        if not 0.0 < vad_threshold < 1.0:
            raise ValueError("VAD threshold must be between zero and one")
        if vad_silence_threshold_seconds <= 0:
            raise ValueError("VAD silence threshold must be positive")
        if min_speech_duration_ms < 1 or min_silence_duration_ms < 1:
            raise ValueError("minimum speech and silence durations must be positive")
        if connect_timeout_seconds <= 0:
            raise ValueError("connect timeout must be positive")
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._language_code = language_code.strip()
        self._endpoint = endpoint.strip()
        self._vad_threshold = vad_threshold
        self._vad_silence_threshold_seconds = vad_silence_threshold_seconds
        self._min_speech_duration_ms = min_speech_duration_ms
        self._min_silence_duration_ms = min_silence_duration_ms
        self._connect_timeout_seconds = connect_timeout_seconds
        self._connect_factory = connect_factory
        self._connection: Any | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._failure: Exception | None = None
        self._provider_session_id: str | None = None
        self._audio_capture_id: str | None = None
        self._pcm_stream_id: str | None = None
        self._last_audio_sequence: int | None = None
        self._partial_sequence = 0
        self._committed_sequence = 0
        self._latest_partial: TranscriptUpdate | None = None
        self._committed: deque[TranscriptUpdate] = deque()
        self._utterance_started_ns: int | None = None
        self._closed = False

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            role="stt",
            provider="elevenlabs-scribe-realtime",
            model=self._model,
            mode=ProviderMode.REAL,
            runtime="server",
        )

    @property
    def provider_session_id(self) -> str | None:
        return self._provider_session_id

    @property
    def audio_source_binding(self) -> tuple[str, str] | None:
        if self._audio_capture_id is None or self._pcm_stream_id is None:
            return None
        return self._audio_capture_id, self._pcm_stream_id

    def bind_audio_source(self, *, audio_capture_id: str, stream_id: str) -> None:
        """Bind once to the exact getUserMedia stream carried by the WebSocket."""

        if not audio_capture_id.strip() or not stream_id.strip():
            raise ValueError("audio capture id and PCM stream id are required")
        binding = (audio_capture_id.strip(), stream_id.strip())
        current = (self._audio_capture_id, self._pcm_stream_id)
        if current != (None, None) and current != binding:
            raise RuntimeError("stt_audio_source_binding_is_immutable")
        self._audio_capture_id, self._pcm_stream_id = binding

    async def start(self) -> None:
        """Authenticate and establish the provider session after source binding."""

        if self._closed:
            raise RuntimeError("realtime STT provider is closed")
        if self._audio_capture_id is None or self._pcm_stream_id is None:
            raise RuntimeError("stt_audio_source_not_bound")
        await self._ensure_connected()

    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        if self._closed:
            raise RuntimeError("realtime STT provider is closed")
        self._validate_frame(frame)
        await self.start()
        self._raise_failure()
        self._last_audio_sequence = frame.sequence
        await self._connection.send(
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(frame.pcm16).decode("ascii"),
                },
                separators=(",", ":"),
            )
        )
        await asyncio.sleep(0)
        self._raise_failure()
        return self._drain_updates()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            await asyncio.gather(self._receiver_task, return_exceptions=True)
        connection = self._connection
        if connection is not None:
            try:
                await asyncio.wait_for(connection.close(), timeout=3.0)
            except TimeoutError:
                transport = getattr(connection, "transport", None)
                if transport is not None:
                    transport.abort()

    def _validate_frame(self, frame: AudioFrame) -> None:
        if self._audio_capture_id is None or self._pcm_stream_id is None:
            raise RuntimeError("stt_audio_source_not_bound")
        if frame.stream_id != self._pcm_stream_id:
            raise RuntimeError("stt_received_unbound_pcm_stream")
        if frame.sample_rate_hz != 16_000 or frame.channels != 1:
            raise ValueError("Scribe realtime requires mono pcm_16000")
        if self._last_audio_sequence is not None and frame.sequence <= self._last_audio_sequence:
            raise RuntimeError("stale_or_duplicate_pcm_frame")

    async def _ensure_connected(self) -> None:
        if self._connection is not None:
            self._raise_failure()
            return
        query = urlencode(
            {
                "model_id": self._model,
                "audio_format": "pcm_16000",
                "language_code": self._language_code,
                "commit_strategy": "vad",
                "vad_threshold": self._vad_threshold,
                "vad_silence_threshold_secs": self._vad_silence_threshold_seconds,
                "min_speech_duration_ms": self._min_speech_duration_ms,
                "min_silence_duration_ms": self._min_silence_duration_ms,
            }
        )
        connect_factory = self._connect_factory
        if connect_factory is None:
            from websockets.asyncio.client import connect

            connect_factory = connect
        self._connection = await asyncio.wait_for(
            connect_factory(
                f"{self._endpoint}?{query}",
                additional_headers={"xi-api-key": self._api_key},
                open_timeout=self._connect_timeout_seconds,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=2,
                max_size=1_048_576,
            ),
            timeout=self._connect_timeout_seconds,
        )
        self._receiver_task = asyncio.create_task(
            self._receive_events(), name="humanflow-elevenlabs-scribe-receiver"
        )
        await asyncio.wait_for(
            self._ready.wait(), timeout=self._connect_timeout_seconds
        )
        self._raise_failure()

    async def _receive_events(self) -> None:
        try:
            async for raw_message in self._connection:
                self._handle_provider_message(raw_message)
                if self._failure is not None:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error
        finally:
            if self._provider_session_id is None and self._failure is None:
                self._failure = ElevenLabsRealtimeSTTError(
                    "connection_closed_before_session_started"
                )
            self._ready.set()

    def _handle_provider_message(self, raw_message: str | bytes) -> None:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            payload = json.loads(raw_message)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._failure = ElevenLabsRealtimeSTTError("invalid_json_event")
            self._ready.set()
            raise self._failure from error
        if not isinstance(payload, dict):
            self._failure = ElevenLabsRealtimeSTTError("invalid_event_shape")
            self._ready.set()
            raise self._failure
        message_type = str(payload.get("message_type", ""))
        if message_type == "session_started":
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                self._failure = ElevenLabsRealtimeSTTError("missing_session_id")
            else:
                self._provider_session_id = session_id.strip()
            self._ready.set()
            return
        if message_type in _ERROR_EVENTS:
            self._failure = ElevenLabsRealtimeSTTError(message_type)
            self._ready.set()
            return
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        if message_type in {"partial_transcript", "final_transcript"}:
            self._latest_partial = self._build_update(
                text=text,
                final=False,
                provider_event_type=message_type,
            )
            return
        if message_type == "committed_transcript":
            self._latest_partial = None
            self._committed.append(
                self._build_update(
                    text=text,
                    final=True,
                    provider_event_type=message_type,
                )
            )
            self._utterance_started_ns = None

    def _build_update(
        self,
        *,
        text: str,
        final: bool,
        provider_event_type: str,
    ) -> TranscriptUpdate:
        observed_ns = monotonic_ns()
        if self._utterance_started_ns is None:
            self._utterance_started_ns = observed_ns
        utterance_ms = max(
            1, round((observed_ns - self._utterance_started_ns) / 1_000_000)
        )
        if final:
            self._committed_sequence += 1
            event_sequence = self._committed_sequence
            transcript_id = (
                f"{self._provider_session_id}:committed:{event_sequence}"
            )
        else:
            self._partial_sequence += 1
            event_sequence = self._partial_sequence
            transcript_id = f"{self._provider_session_id}:partial:{event_sequence}"
        clean_text = text.strip()
        return TranscriptUpdate(
            text=clean_text,
            raw_text=text,
            is_final=final,
            provenance=TranscriptProvenance(
                transcript_id=transcript_id,
                event_kind=(
                    ConversationEventKind.USER_TRANSCRIPT_FINAL
                    if final
                    else ConversationEventKind.USER_TRANSCRIPT_PARTIAL
                ),
                source="streaming_stt",
                origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
                stream_id=self._pcm_stream_id or "unbound",
                stt_session_id=self._provider_session_id,
                audio_capture_id=self._audio_capture_id,
                timestamp_ns=observed_ns,
                recognition_input_binding="EXACT_GETUSERMEDIA_PCM16",
                audio_frame_sequence=self._last_audio_sequence,
                provider_event_type=provider_event_type,
            ),
            provider=self.provider_info,
            signals=TurnSignals(
                speech_active=not final,
                silence_duration_ms=(
                    round(self._vad_silence_threshold_seconds * 1_000)
                    if final
                    else 0
                ),
                utterance_duration_ms=utterance_ms,
                semantic_complete=(
                    final or provider_event_type == "final_transcript"
                ),
                filler_ending=bool(_FILLER_ENDING.search(clean_text)),
                acoustic_completion=1.0 if final else 0.0,
                interruption_probability=(
                    0.98 if _INTERRUPTION_PREFIX.search(clean_text) else 0.0
                ),
                provider_endpointed=final,
            ),
        )

    def _drain_updates(self) -> tuple[TranscriptUpdate, ...]:
        updates: list[TranscriptUpdate] = []
        if self._latest_partial is not None:
            updates.append(self._latest_partial)
            self._latest_partial = None
        while self._committed:
            updates.append(self._committed.popleft())
        return tuple(updates)

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
