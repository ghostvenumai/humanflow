"""Transport-neutral PCM16 audio types used by the realtime core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


PCM16_BYTES_PER_SAMPLE = 2


class AudioPlaybackMode(StrEnum):
    """How a transport must render an audio chunk."""

    PCM = "pcm"
    BROWSER_SPEECH = "browser_speech"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A sequence-ordered, interleaved little-endian PCM16 frame."""

    stream_id: str
    sequence: int
    pcm16: bytes
    sample_rate_hz: int = 16_000
    channels: int = 1
    captured_ns: int = 0

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("stream_id must not be empty")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.captured_ns < 0:
            raise ValueError("captured_ns must be non-negative")
        frame_width = PCM16_BYTES_PER_SAMPLE * self.channels
        if not self.pcm16 or len(self.pcm16) % frame_width:
            raise ValueError("pcm16 must contain complete non-empty PCM16 samples")

    @property
    def samples_per_channel(self) -> int:
        return len(self.pcm16) // (PCM16_BYTES_PER_SAMPLE * self.channels)

    @property
    def duration_ms(self) -> float:
        return self.samples_per_channel * 1000.0 / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A generated speech chunk with conservative semantic-delivery metadata.

    Streaming providers may emit several PCM chunks for one semantic segment.
    Only the final chunk carries ``text`` and therefore advances the played-text
    ledger. ``display_text`` is transport UI metadata and is never counted as
    delivered speech.
    """

    chunk_id: str
    response_id: str
    text: str
    frame: AudioFrame
    semantic_id: str = ""
    semantic_text: str = ""
    playback_mode: AudioPlaybackMode = AudioPlaybackMode.PCM
    semantic_boundary: bool = True
    display_text: str = ""
    pause_after_ms: int = 0
    speaking_rate: float = 1.0
    provider: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or not self.response_id.strip():
            raise ValueError("chunk_id and response_id must not be empty")
        if self.semantic_boundary and not self.text.strip():
            raise ValueError("semantic boundary text must not be empty")
        if not self.semantic_boundary and self.text.strip():
            raise ValueError("non-boundary chunks must not carry ledger text")
        semantic_id = self.semantic_id.strip() or self.chunk_id
        semantic_text = self.semantic_text.strip() or self.text.strip()
        if not semantic_text:
            raise ValueError("semantic_text must not be empty")
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "semantic_text", semantic_text)
        if self.pause_after_ms < 0:
            raise ValueError("pause_after_ms must be non-negative")
        if not 0.5 <= self.speaking_rate <= 2.0:
            raise ValueError("speaking_rate must be between 0.5 and 2.0")
        object.__setattr__(self, "provider", MappingProxyType(dict(self.provider)))


@dataclass(frozen=True, slots=True)
class PlaybackReceipt:
    """What an audio sink actually consumed, including the audible stop boundary."""

    chunk_id: str
    requested_samples: int
    played_samples: int
    playback_started_ns: int
    playback_stopped_ns: int
    cancelled: bool
    sink_base_latency_ms: float | None = None
    sink_output_latency_ms: float | None = None
    player_stop_callback_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        if self.requested_samples < 1:
            raise ValueError("requested_samples must be positive")
        if not 0 <= self.played_samples <= self.requested_samples:
            raise ValueError("played_samples must be between zero and requested_samples")
        if self.playback_started_ns < 0:
            raise ValueError("playback_started_ns must be non-negative")
        if self.playback_stopped_ns < self.playback_started_ns:
            raise ValueError("playback_stopped_ns must not precede playback_started_ns")
        if not self.cancelled and self.played_samples != self.requested_samples:
            raise ValueError("non-cancelled playback must consume every requested sample")
        for name in (
            "sink_base_latency_ms",
            "sink_output_latency_ms",
            "player_stop_callback_latency_ms",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
