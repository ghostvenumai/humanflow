"""Provider boundaries and dependency-free local realtime adapters."""

from __future__ import annotations

import asyncio
import inspect
import math
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from time import monotonic_ns
from typing import Protocol
from uuid import uuid4

from humanflow.audio.models import (
    AudioChunk,
    AudioFrame,
    AudioPlaybackMode,
    PlaybackReceipt,
)
from humanflow.domain.conversation import OperationToken
from humanflow.turns.models import TurnSignals

from .transcript_events import (
    ConversationEventKind,
    TranscriptProvenance,
    USER_TRANSCRIPT_KINDS,
    normalize_transcript,
)


class ProviderMode(StrEnum):
    """Whether a runtime provider performs real work or deterministic test work."""

    REAL = "REAL"
    MOCK = "MOCK"


@dataclass(frozen=True, slots=True)
class STTProviderCapabilities:
    streaming_audio: bool
    source_bound: bool
    partial_transcripts: bool
    final_transcripts: bool
    provider_endpointing: bool
    cancellation: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "streaming_audio": self.streaming_audio,
            "source_bound": self.source_bound,
            "partial_transcripts": self.partial_transcripts,
            "final_transcripts": self.final_transcripts,
            "provider_endpointing": self.provider_endpointing,
            "cancellation": self.cancellation,
        }


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Secret-free provider identity exposed in telemetry and the live demo."""

    role: str
    provider: str
    model: str
    mode: ProviderMode
    runtime: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode.value,
            "runtime": self.runtime,
        }


def provider_info(provider: object, *, role: str) -> ProviderInfo:
    """Return an adapter's declared identity, or an explicit unknown/mock identity."""

    info = getattr(provider, "provider_info", None)
    if isinstance(info, ProviderInfo):
        return info
    return ProviderInfo(
        role=role,
        provider=type(provider).__name__,
        model="undeclared",
        mode=ProviderMode.MOCK,
        runtime="server",
    )


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """A provider transcript plus the signals used for a turn decision."""

    text: str
    is_final: bool
    signals: TurnSignals
    provenance: TranscriptProvenance
    provider: ProviderInfo | None = None
    raw_text: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        expected_kind = (
            ConversationEventKind.USER_TRANSCRIPT_FINAL
            if self.is_final
            else ConversationEventKind.USER_TRANSCRIPT_PARTIAL
        )
        if (
            self.provenance.event_kind in USER_TRANSCRIPT_KINDS
            and self.provenance.event_kind is not expected_kind
        ):
            raise ValueError("transcript finality does not match event provenance")
        object.__setattr__(self, "raw_text", self.raw_text or self.text)

    @property
    def normalized_text(self) -> str:
        return normalize_transcript(self.text)


class StreamingTranscriber(Protocol):
    provider_info: ProviderInfo
    capabilities: STTProviderCapabilities

    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]: ...

    async def close(self) -> None: ...


StreamingSTTProvider = StreamingTranscriber


class StreamingReasoner(Protocol):
    def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]: ...


@dataclass(frozen=True, slots=True)
class SpeechSynthesisRequest:
    """Vendor-neutral request for one stable, speech-ready semantic segment."""

    text: str
    response_id: str
    sequence_start: int
    display_text: str | None = None
    language_code: str = "de"
    voice: str | None = None
    speaking_rate: float = 1.0
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None
    use_speaker_boost: bool | None = None
    pause_after_ms: int = 0
    intent: str = "information"
    previous_text: str = ""
    tts_session_id: str = ""
    segment_id: str = ""

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.response_id.strip():
            raise ValueError("text and response_id must not be empty")
        if self.display_text is not None and not self.display_text.strip():
            raise ValueError("display_text must be non-empty when present")
        if self.sequence_start < 0:
            raise ValueError("sequence_start must be non-negative")
        if not 0.5 <= self.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")
        for name in ("stability", "similarity_boost", "style"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.pause_after_ms < 0:
            raise ValueError("pause_after_ms must be non-negative")
        if self.tts_session_id and not self.tts_session_id.strip():
            raise ValueError("tts_session_id must be non-empty when present")
        if self.segment_id and not self.segment_id.strip():
            raise ValueError("segment_id must be non-empty when present")


class StreamingTTSProvider(Protocol):
    """A cancellable provider that yields playable audio before synthesis ends."""

    def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]: ...


# Compatibility name for integrations that imported the original boundary.
StreamingSpeechSynthesizer = StreamingTTSProvider


PlaybackStarted = Callable[[int], None]
AudioObserver = Callable[[AudioChunk, int, int], Awaitable[None] | None]


class AudioOutput(Protocol):
    async def play(
        self,
        chunk: AudioChunk,
        *,
        cancel_event: asyncio.Event,
        on_started: PlaybackStarted,
    ) -> PlaybackReceipt: ...


class NullTranscriber:
    """Consumes real PCM frames while transcript input arrives over another adapter."""

    provider_info = ProviderInfo(
        role="stt",
        provider="null-transcriber",
        model="none",
        mode=ProviderMode.MOCK,
        runtime="server",
    )
    capabilities = STTProviderCapabilities(
        streaming_audio=True,
        source_bound=True,
        partial_transcripts=False,
        final_transcripts=False,
        provider_endpointing=False,
        cancellation=False,
    )

    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        del frame
        await asyncio.sleep(0)
        return ()

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class EchoReasoner:
    """Local streaming reasoner for offline demos; it makes no model-quality claim."""

    first_token_delay_ms: float = 20.0
    chunk_delay_ms: float = 5.0

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            role="reasoning",
            provider="echo-reasoner",
            model="deterministic-echo",
            mode=ProviderMode.MOCK,
            runtime="server",
        )

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        await asyncio.sleep(self.first_token_delay_ms / 1000.0)
        words = ("Ich", "habe", "verstanden:", *transcript.strip().split())
        for index, word in enumerate(words):
            if index:
                await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            yield word


@dataclass(slots=True)
class ToneSpeechSynthesizer:
    """Produces correctly timed PCM16 chunks for transport and cancellation testing."""

    sample_rate_hz: int = 16_000
    chunk_duration_ms: float = 80.0
    amplitude: int = 2_400
    frequency_hz: float = 220.0

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            role="tts",
            provider="tone-synthesizer",
            model=f"sine-{self.frequency_hz:g}hz",
            mode=ProviderMode.MOCK,
            runtime="server",
        )

    async def synthesize(
        self, text: str, *, response_id: str, sequence: int
    ) -> AudioChunk:
        if self.chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        sample_count = max(1, round(self.sample_rate_hz * self.chunk_duration_ms / 1000.0))
        samples = array(
            "h",
            (
                round(
                    self.amplitude
                    * math.sin(2.0 * math.pi * self.frequency_hz * i / self.sample_rate_hz)
                )
                for i in range(sample_count)
            ),
        )
        frame = AudioFrame(
            stream_id=response_id,
            sequence=sequence,
            pcm16=samples.tobytes(),
            sample_rate_hz=self.sample_rate_hz,
            captured_ns=monotonic_ns(),
        )
        await asyncio.sleep(0)
        return AudioChunk(
            chunk_id=str(uuid4()),
            response_id=response_id,
            text=text.strip(),
            frame=frame,
            provider=self.provider_info.to_dict(),
        )

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        if cancel_event.is_set():
            return
        chunk = await self.synthesize(
            request.text,
            response_id=request.response_id,
            sequence=request.sequence_start,
        )
        yield replace(
            chunk,
            text=request.display_text or request.text,
            semantic_text=request.text,
            display_text=request.display_text or request.text,
            pause_after_ms=request.pause_after_ms,
            speaking_rate=request.speaking_rate,
            tts_session_id=request.tts_session_id or request.response_id,
            segment_id=request.segment_id or chunk.semantic_id,
        )


@dataclass(slots=True)
class BrowserSpeechSynthesisAdapter:
    """Send semantic text boundaries to real browser speech synthesis.

    The silent PCM envelope supplies a conservative duration/sample budget for
    the transport-neutral ledger. It is never rendered when the browser demo's
    mandatory Web Speech synthesis provider is available.
    """

    sample_rate_hz: int = 16_000
    words_per_second: float = 2.6
    minimum_duration_ms: float = 450.0
    maximum_duration_ms: float = 15_000.0

    provider_info = ProviderInfo(
        role="tts",
        provider="browser-web-speech-api",
        model="de-DE-system-voice",
        mode=ProviderMode.REAL,
        runtime="browser",
    )

    async def synthesize(
        self, text: str, *, response_id: str, sequence: int
    ) -> AudioChunk:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")
        if self.words_per_second <= 0:
            raise ValueError("words_per_second must be positive")
        estimated_ms = len(clean_text.split()) / self.words_per_second * 1000.0
        duration_ms = min(
            self.maximum_duration_ms,
            max(self.minimum_duration_ms, estimated_ms),
        )
        sample_count = max(1, round(self.sample_rate_hz * duration_ms / 1000.0))
        frame = AudioFrame(
            stream_id=response_id,
            sequence=sequence,
            pcm16=b"\x00\x00" * sample_count,
            sample_rate_hz=self.sample_rate_hz,
            captured_ns=monotonic_ns(),
        )
        await asyncio.sleep(0)
        return AudioChunk(
            chunk_id=str(uuid4()),
            response_id=response_id,
            text=clean_text,
            frame=frame,
            playback_mode=AudioPlaybackMode.BROWSER_SPEECH,
            provider=self.provider_info.to_dict(),
        )

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        if cancel_event.is_set():
            return
        chunk = await self.synthesize(
            request.text,
            response_id=request.response_id,
            sequence=request.sequence_start,
        )
        yield replace(
            chunk,
            text=request.display_text or request.text,
            semantic_text=request.text,
            display_text=request.display_text or request.text,
            pause_after_ms=request.pause_after_ms,
            speaking_rate=request.speaking_rate,
            tts_session_id=request.tts_session_id or request.response_id,
            segment_id=request.segment_id or chunk.semantic_id,
        )


class GaplessSegmentTTSProvider:
    """Coalesce raw PCM transport chunks into one gapless semantic playback unit.

    Provider streaming still runs concurrently and remains cancellable. The
    browser receives one source buffer per stable semantic segment, avoiding a
    stop-and-wait source-node gap at every provider packet boundary.
    """

    def __init__(self, upstream: StreamingTTSProvider) -> None:
        self._upstream = upstream

    @property
    def provider_info(self) -> ProviderInfo:
        return self._upstream.provider_info  # type: ignore[attr-defined]

    @property
    def fallback_info(self) -> ProviderInfo | None:
        return getattr(self._upstream, "fallback_info", None)

    @property
    def last_request_metrics(self) -> object | None:
        return getattr(self._upstream, "last_request_metrics", None)

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        stream = self._upstream.stream_speech(request, cancel_event=cancel_event)
        chunks: list[AudioChunk] = []
        seen_chunk_ids: set[str] = set()
        seen_sequences: set[int] = set()
        expected_sequence = request.sequence_start
        try:
            async for chunk in stream:
                if cancel_event.is_set():
                    break
                if chunk.chunk_id in seen_chunk_ids:
                    raise RuntimeError("duplicate_tts_chunk_id")
                if chunk.frame.sequence in seen_sequences:
                    raise RuntimeError("duplicate_tts_chunk_sequence")
                if chunk.response_id != request.response_id:
                    raise RuntimeError("stale_tts_response_id")
                if chunk.frame.sequence != expected_sequence:
                    raise RuntimeError("non_contiguous_tts_chunk_sequence")
                seen_chunk_ids.add(chunk.chunk_id)
                seen_sequences.add(chunk.frame.sequence)
                expected_sequence += 1
                chunks.append(chunk)
        finally:
            close_stream = getattr(stream, "aclose", None)
            if close_stream is not None:
                await close_stream()

        if cancel_event.is_set() or not chunks:
            return
        playback_modes = {chunk.playback_mode for chunk in chunks}
        if len(playback_modes) != 1:
            raise RuntimeError("multiple_tts_playback_modes_in_segment")
        if chunks[0].playback_mode is AudioPlaybackMode.BROWSER_SPEECH:
            if len(chunks) != 1:
                raise RuntimeError("browser_fallback_must_emit_one_chunk")
            yield chunks[0]
            return

        sample_rates = {chunk.frame.sample_rate_hz for chunk in chunks}
        channels = {chunk.frame.channels for chunk in chunks}
        providers = {
            (chunk.provider.get("provider"), chunk.provider.get("model"))
            for chunk in chunks
        }
        if len(sample_rates) != 1 or len(channels) != 1 or len(providers) != 1:
            raise RuntimeError("inconsistent_tts_stream_metadata")
        if not chunks[-1].semantic_boundary or chunks[-1].semantic_text != request.text:
            raise RuntimeError("tts_stream_missing_final_semantic_boundary")
        provider_metadata = dict(chunks[0].provider)
        provider_metadata.update(
            {
                "delivery": "gapless-semantic-buffer",
                "source_chunk_count": str(len(chunks)),
                "source_sequence_first": str(chunks[0].frame.sequence),
                "source_sequence_last": str(chunks[-1].frame.sequence),
            }
        )
        yield AudioChunk(
            chunk_id=str(uuid4()),
            response_id=request.response_id,
            text=request.display_text or request.text,
            semantic_id=chunks[0].semantic_id,
            semantic_text=request.text,
            tts_session_id=request.tts_session_id or request.response_id,
            segment_id=request.segment_id or chunks[0].semantic_id,
            frame=AudioFrame(
                stream_id=request.response_id,
                sequence=request.sequence_start,
                pcm16=b"".join(chunk.frame.pcm16 for chunk in chunks),
                sample_rate_hz=chunks[0].frame.sample_rate_hz,
                channels=chunks[0].frame.channels,
                captured_ns=chunks[0].frame.captured_ns,
            ),
            playback_mode=AudioPlaybackMode.PCM,
            semantic_boundary=True,
            display_text=request.display_text or request.text,
            pause_after_ms=request.pause_after_ms,
            speaking_rate=request.speaking_rate,
            provider=provider_metadata,
        )


@dataclass(slots=True)
class TimedPcmOutput:
    """A cancel-aware PCM sink that advances according to the real event-loop clock."""

    quantum_ms: float = 10.0
    observer: AudioObserver | None = None
    clock_ns: Callable[[], int] = monotonic_ns

    async def play(
        self,
        chunk: AudioChunk,
        *,
        cancel_event: asyncio.Event,
        on_started: PlaybackStarted,
    ) -> PlaybackReceipt:
        if self.quantum_ms <= 0:
            raise ValueError("quantum_ms must be positive")
        frame = chunk.frame
        requested_samples = frame.samples_per_channel
        quantum_samples = max(1, round(frame.sample_rate_hz * self.quantum_ms / 1000.0))
        played_samples = 0
        started_ns = self.clock_ns()
        on_started(started_ns)

        while played_samples < requested_samples and not cancel_event.is_set():
            block_samples = min(quantum_samples, requested_samples - played_samples)
            block_started = played_samples
            await asyncio.sleep(block_samples / frame.sample_rate_hz)
            if cancel_event.is_set():
                break
            played_samples += block_samples
            if self.observer is not None:
                observed = self.observer(chunk, block_started, played_samples)
                if inspect.isawaitable(observed):
                    await observed

        stopped_ns = self.clock_ns()
        cancelled = cancel_event.is_set() and played_samples < requested_samples
        return PlaybackReceipt(
            chunk_id=chunk.chunk_id,
            requested_samples=requested_samples,
            played_samples=played_samples,
            playback_started_ns=started_ns,
            playback_stopped_ns=stopped_ns,
            cancelled=cancelled,
        )
