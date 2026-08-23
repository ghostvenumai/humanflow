"""Transport-neutral PCM16 audio types used by the realtime core."""

from __future__ import annotations

from dataclasses import dataclass


PCM16_BYTES_PER_SAMPLE = 2


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
    """A generated speech chunk whose text is an atomic semantic boundary."""

    chunk_id: str
    response_id: str
    text: str
    frame: AudioFrame

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or not self.response_id.strip():
            raise ValueError("chunk_id and response_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")


@dataclass(frozen=True, slots=True)
class PlaybackReceipt:
    """What an audio sink actually consumed, including the audible stop boundary."""

    chunk_id: str
    requested_samples: int
    played_samples: int
    playback_started_ns: int
    playback_stopped_ns: int
    cancelled: bool

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
