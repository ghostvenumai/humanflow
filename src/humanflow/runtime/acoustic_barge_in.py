"""Low-latency acoustic barge-in signals from authoritative PCM16 input."""

from __future__ import annotations

import math
import sys
from array import array
from dataclasses import dataclass
from enum import StrEnum

from humanflow.audio.models import AudioFrame


class AcousticEventType(StrEnum):
    SPEECH_ONSET = "SPEECH_ONSET"
    SUSTAINED_TAKEOVER = "SUSTAINED_TAKEOVER"
    SPEECH_ENDED = "SPEECH_ENDED"


@dataclass(frozen=True, slots=True)
class AcousticBargeInEvent:
    event_type: AcousticEventType
    speech_onset_ns: int
    observed_ns: int
    rms: float
    peak: float
    threshold: float
    speech_duration_ms: float
    detection_latency_ms: float


class AcousticBargeInDetector:
    """Energy VAD that ducks quickly and confirms only sustained takeover.

    It never creates a conversational turn. The detector only emits acoustic
    control signals; STT and the turn policy retain semantic authority.
    """

    def __init__(
        self,
        *,
        onset_debounce_ms: float = 70.0,
        sustained_takeover_ms: float = 620.0,
        release_silence_ms: float = 180.0,
        minimum_rms: float = 0.010,
        minimum_peak: float = 0.030,
        noise_multiplier: float = 2.8,
        initial_noise_floor: float = 0.003,
    ) -> None:
        if not 20.0 <= onset_debounce_ms < sustained_takeover_ms:
            raise ValueError("acoustic onset debounce must precede takeover")
        if release_silence_ms <= 0:
            raise ValueError("acoustic release silence must be positive")
        if minimum_rms <= 0 or minimum_peak <= 0 or noise_multiplier <= 1:
            raise ValueError("acoustic thresholds must be positive and selective")
        self.onset_debounce_ms = onset_debounce_ms
        self.sustained_takeover_ms = sustained_takeover_ms
        self.release_silence_ms = release_silence_ms
        self.minimum_rms = minimum_rms
        self.minimum_peak = minimum_peak
        self.noise_multiplier = noise_multiplier
        self._noise_floor = initial_noise_floor
        self._candidate_onset_ns: int | None = None
        self._candidate_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._speech_active = False
        self._takeover_emitted = False

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def observe(
        self, frame: AudioFrame, *, assistant_playback_active: bool
    ) -> tuple[AcousticBargeInEvent, ...]:
        rms, peak = _pcm_levels(frame.pcm16)
        threshold = max(self.minimum_rms, self._noise_floor * self.noise_multiplier)
        speech = rms >= threshold and peak >= self.minimum_peak
        frame_ms = frame.duration_ms
        observed_ns = frame.captured_ns + round(frame_ms * 1_000_000)

        if not assistant_playback_active:
            self._reset_candidate()
            if not speech:
                self._update_noise_floor(rms)
            return ()

        if speech:
            self._silence_duration_ms = 0.0
            if self._candidate_onset_ns is None:
                self._candidate_onset_ns = frame.captured_ns
                self._candidate_duration_ms = 0.0
            self._candidate_duration_ms += frame_ms
            events: list[AcousticBargeInEvent] = []
            if (
                not self._speech_active
                and self._candidate_duration_ms >= self.onset_debounce_ms
            ):
                self._speech_active = True
                events.append(
                    self._event(
                        AcousticEventType.SPEECH_ONSET,
                        observed_ns=observed_ns,
                        rms=rms,
                        peak=peak,
                        threshold=threshold,
                    )
                )
            if (
                self._speech_active
                and not self._takeover_emitted
                and self._candidate_duration_ms >= self.sustained_takeover_ms
            ):
                self._takeover_emitted = True
                events.append(
                    self._event(
                        AcousticEventType.SUSTAINED_TAKEOVER,
                        observed_ns=observed_ns,
                        rms=rms,
                        peak=peak,
                        threshold=threshold,
                    )
                )
            return tuple(events)

        if self._speech_active:
            self._silence_duration_ms += frame_ms
            if self._silence_duration_ms >= self.release_silence_ms:
                event = self._event(
                    AcousticEventType.SPEECH_ENDED,
                    observed_ns=observed_ns,
                    rms=rms,
                    peak=peak,
                    threshold=threshold,
                )
                self._reset_candidate()
                self._update_noise_floor(rms)
                return (event,)
            return ()

        self._reset_candidate()
        self._update_noise_floor(rms)
        return ()

    def _event(
        self,
        event_type: AcousticEventType,
        *,
        observed_ns: int,
        rms: float,
        peak: float,
        threshold: float,
    ) -> AcousticBargeInEvent:
        onset_ns = self._candidate_onset_ns
        if onset_ns is None:
            raise RuntimeError("acoustic event has no onset")
        return AcousticBargeInEvent(
            event_type=event_type,
            speech_onset_ns=onset_ns,
            observed_ns=observed_ns,
            rms=round(rms, 6),
            peak=round(peak, 6),
            threshold=round(threshold, 6),
            speech_duration_ms=round(self._candidate_duration_ms, 3),
            detection_latency_ms=round(
                max(0.0, (observed_ns - onset_ns) / 1_000_000.0), 3
            ),
        )

    def _update_noise_floor(self, rms: float) -> None:
        bounded = min(rms, self.minimum_rms)
        self._noise_floor = self._noise_floor * 0.96 + bounded * 0.04

    def _reset_candidate(self) -> None:
        self._candidate_onset_ns = None
        self._candidate_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._speech_active = False
        self._takeover_emitted = False


def _pcm_levels(pcm16: bytes) -> tuple[float, float]:
    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0, 0.0
    scale = 32768.0
    square_sum = 0.0
    peak = 0
    for sample in samples:
        magnitude = abs(sample)
        peak = max(peak, magnitude)
        normalized = sample / scale
        square_sum += normalized * normalized
    return math.sqrt(square_sum / len(samples)), peak / scale
