"""Provider boundaries and dependency-free local realtime adapters."""

from __future__ import annotations

import asyncio
import inspect
import math
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from time import monotonic_ns
from typing import Protocol
from uuid import uuid4

from humanflow.audio.models import AudioChunk, AudioFrame, PlaybackReceipt
from humanflow.domain.conversation import OperationToken
from humanflow.turns.models import TurnSignals


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """A provider transcript plus the signals used for a turn decision."""

    text: str
    is_final: bool
    signals: TurnSignals

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")


class StreamingTranscriber(Protocol):
    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]: ...

    async def close(self) -> None: ...


class StreamingReasoner(Protocol):
    def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]: ...


class StreamingSpeechSynthesizer(Protocol):
    async def synthesize(
        self, text: str, *, response_id: str, sequence: int
    ) -> AudioChunk: ...


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
