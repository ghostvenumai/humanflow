"""Admission of streaming STT finals backed by authoritative PCM speech."""

from __future__ import annotations

import math
import sys
from array import array
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from humanflow.audio.models import AudioFrame

from .providers import TranscriptUpdate
from .self_speech import SelfSpeechAssessment


class PcmSpeechEpisodeEventType(StrEnum):
    STARTED = "STARTED"
    ENDED = "ENDED"


@dataclass(slots=True)
class PcmSpeechEpisode:
    speech_episode_id: str
    stream_id: str
    speech_start_monotonic: int
    speech_end_monotonic: int
    first_frame_sequence: int
    last_frame_sequence: int
    voiced_duration_ms: float
    max_rms: float
    max_peak: float
    threshold: float
    valid: bool = False
    started_emitted: bool = False
    claimed_by_transcript_id: str | None = None

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "speech_episode_id": self.speech_episode_id,
            "stream_id": self.stream_id,
            "speech_start_monotonic": self.speech_start_monotonic,
            "speech_end_monotonic": self.speech_end_monotonic,
            "first_frame_sequence": self.first_frame_sequence,
            "last_frame_sequence": self.last_frame_sequence,
            "voiced_duration_ms": round(self.voiced_duration_ms, 3),
            "max_rms": round(self.max_rms, 6),
            "max_peak": round(self.max_peak, 6),
            "threshold": round(self.threshold, 6),
            "valid": self.valid,
            "claimed_by_transcript_id": self.claimed_by_transcript_id,
        }


@dataclass(frozen=True, slots=True)
class PcmSpeechEpisodeEvent:
    event_type: PcmSpeechEpisodeEventType
    episode: PcmSpeechEpisode
    observed_ns: int


@dataclass(frozen=True, slots=True)
class FinalAdmissionAssessment:
    accepted: bool
    reason_code: str
    speech_episode_id: str | None
    speech_start_monotonic: int | None
    speech_end_monotonic: int | None
    final_received_monotonic: int
    alignment_ms: float | None
    voiced_duration_ms: float | None
    max_rms: float | None
    max_peak: float | None
    assistant_playback_active: bool
    self_speech_candidate: bool
    self_speech_confidence: float

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "speech_episode_id": self.speech_episode_id,
            "speech_start_monotonic": self.speech_start_monotonic,
            "speech_end_monotonic": self.speech_end_monotonic,
            "final_received_monotonic": self.final_received_monotonic,
            "alignment_ms": self.alignment_ms,
            "voiced_duration_ms": self.voiced_duration_ms,
            "max_rms": self.max_rms,
            "max_peak": self.max_peak,
            "assistant_playback_active": self.assistant_playback_active,
            "self_speech_candidate": self.self_speech_candidate,
            "self_speech_confidence": self.self_speech_confidence,
        }


class FinalTranscriptAdmissionGate:
    """Link each provider FINAL to one unconsumed microphone speech episode."""

    def __init__(
        self,
        *,
        minimum_voiced_ms: float = 120.0,
        release_silence_ms: float = 180.0,
        finalization_window_ms: float = 3_500.0,
        minimum_rms: float = 0.010,
        minimum_peak: float = 0.030,
        noise_multiplier: float = 2.8,
        initial_noise_floor: float = 0.003,
        maximum_episodes: int = 32,
    ) -> None:
        if minimum_voiced_ms <= 0 or release_silence_ms <= 0:
            raise ValueError("speech episode durations must be positive")
        if finalization_window_ms < release_silence_ms:
            raise ValueError("finalization window must include release silence")
        if minimum_rms <= 0 or minimum_peak <= 0 or noise_multiplier <= 1:
            raise ValueError("speech evidence thresholds must be selective")
        if maximum_episodes < 1:
            raise ValueError("maximum episodes must be positive")
        self.minimum_voiced_ms = minimum_voiced_ms
        self.release_silence_ms = release_silence_ms
        self._finalization_window_ns = round(finalization_window_ms * 1_000_000)
        self.minimum_rms = minimum_rms
        self.minimum_peak = minimum_peak
        self.noise_multiplier = noise_multiplier
        self._noise_floor = initial_noise_floor
        self._episodes: deque[PcmSpeechEpisode] = deque(maxlen=maximum_episodes)
        self._active: PcmSpeechEpisode | None = None
        self._silence_duration_ms = 0.0
        self._generation = 0

    @property
    def active_episode_id(self) -> str | None:
        return None if self._active is None else self._active.speech_episode_id

    def observe(self, frame: AudioFrame) -> tuple[PcmSpeechEpisodeEvent, ...]:
        rms, peak = _pcm_levels(frame.pcm16)
        threshold = max(self.minimum_rms, self._noise_floor * self.noise_multiplier)
        speech_like = rms >= threshold and peak >= self.minimum_peak
        frame_end_ns = frame.captured_ns + round(frame.duration_ms * 1_000_000)
        events: list[PcmSpeechEpisodeEvent] = []

        active = self._active
        if (
            active is not None
            and frame.captured_ns - active.speech_end_monotonic
            >= round(self.release_silence_ms * 1_000_000)
        ):
            events.extend(self._finish_active(observed_ns=frame.captured_ns))
            active = None

        if speech_like:
            self._silence_duration_ms = 0.0
            if active is None or active.stream_id != frame.stream_id:
                if active is not None:
                    events.extend(self._finish_active(observed_ns=frame.captured_ns))
                self._generation += 1
                active = PcmSpeechEpisode(
                    speech_episode_id=f"pcm-speech-{self._generation}",
                    stream_id=frame.stream_id,
                    speech_start_monotonic=frame.captured_ns,
                    speech_end_monotonic=frame_end_ns,
                    first_frame_sequence=frame.sequence,
                    last_frame_sequence=frame.sequence,
                    voiced_duration_ms=0.0,
                    max_rms=0.0,
                    max_peak=0.0,
                    threshold=threshold,
                )
                self._active = active
            active.speech_end_monotonic = frame_end_ns
            active.last_frame_sequence = frame.sequence
            active.voiced_duration_ms += frame.duration_ms
            active.max_rms = max(active.max_rms, rms)
            active.max_peak = max(active.max_peak, peak)
            active.threshold = max(active.threshold, threshold)
            if active.voiced_duration_ms >= self.minimum_voiced_ms:
                active.valid = True
                if not active.started_emitted:
                    active.started_emitted = True
                    events.append(
                        PcmSpeechEpisodeEvent(
                            PcmSpeechEpisodeEventType.STARTED,
                            active,
                            frame_end_ns,
                        )
                    )
            return tuple(events)

        if active is not None:
            self._silence_duration_ms += frame.duration_ms
            if self._silence_duration_ms >= self.release_silence_ms:
                events.extend(self._finish_active(observed_ns=frame_end_ns))
        else:
            self._update_noise_floor(rms)
        return tuple(events)

    def assess_final(
        self,
        update: TranscriptUpdate,
        *,
        assistant_playback_active: bool,
        self_speech: SelfSpeechAssessment,
    ) -> FinalAdmissionAssessment:
        if not update.is_final:
            raise ValueError("only final transcripts require admission")
        final_ns = update.provenance.timestamp_ns
        if self_speech.suppress:
            return self._assessment(
                accepted=False,
                reason_code=(
                    self_speech.rejection_reason or "probable_assistant_self_speech"
                ),
                episode=self._matching_episode(update),
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )

        episode = self._matching_episode(update)
        if episode is None:
            return self._assessment(
                accepted=False,
                reason_code="final_without_authoritative_pcm_speech_episode",
                episode=None,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )
        alignment_ns = final_ns - episode.speech_end_monotonic
        if alignment_ns < 0 or alignment_ns > self._finalization_window_ns:
            return self._assessment(
                accepted=False,
                reason_code="final_outside_pcm_speech_finalization_window",
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )
        if not episode.valid:
            return self._assessment(
                accepted=False,
                reason_code="pcm_speech_episode_below_minimum_evidence",
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )
        final_sequence = update.provenance.audio_frame_sequence
        if final_sequence is not None and final_sequence < episode.last_frame_sequence:
            return self._assessment(
                accepted=False,
                reason_code="final_precedes_pcm_speech_episode_frames",
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )
        if (
            self_speech.candidate
            and self_speech.confidence >= 0.58
            and episode.voiced_duration_ms < 240.0
        ):
            return self._assessment(
                accepted=False,
                reason_code="weak_pcm_with_recent_assistant_similarity",
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
            )
        episode.claimed_by_transcript_id = update.provenance.transcript_id
        return self._assessment(
            accepted=True,
            reason_code="authoritative_pcm_speech_episode_linked",
            episode=episode,
            final_ns=final_ns,
            assistant_playback_active=assistant_playback_active,
            self_speech=self_speech,
        )

    def _matching_episode(self, update: TranscriptUpdate) -> PcmSpeechEpisode | None:
        candidates = list(self._episodes)
        if self._active is not None:
            candidates.append(self._active)
        final_ns = update.provenance.timestamp_ns
        return next(
            (
                episode
                for episode in reversed(candidates)
                if episode.stream_id == update.provenance.stream_id
                and episode.claimed_by_transcript_id is None
                and episode.speech_start_monotonic <= final_ns
                and final_ns - episode.speech_end_monotonic
                <= self._finalization_window_ns
            ),
            None,
        )

    def _finish_active(self, *, observed_ns: int) -> list[PcmSpeechEpisodeEvent]:
        active = self._active
        if active is None:
            return []
        self._episodes.append(active)
        self._active = None
        self._silence_duration_ms = 0.0
        self._update_noise_floor(0.0)
        if not active.started_emitted:
            return []
        return [
            PcmSpeechEpisodeEvent(
                PcmSpeechEpisodeEventType.ENDED,
                active,
                observed_ns,
            )
        ]

    def _update_noise_floor(self, rms: float) -> None:
        self._noise_floor = self._noise_floor * 0.96 + min(
            rms, self.minimum_rms
        ) * 0.04

    @staticmethod
    def _assessment(
        *,
        accepted: bool,
        reason_code: str,
        episode: PcmSpeechEpisode | None,
        final_ns: int,
        assistant_playback_active: bool,
        self_speech: SelfSpeechAssessment,
    ) -> FinalAdmissionAssessment:
        alignment_ms = (
            None
            if episode is None
            else round(
                (final_ns - episode.speech_end_monotonic) / 1_000_000.0,
                3,
            )
        )
        return FinalAdmissionAssessment(
            accepted=accepted,
            reason_code=reason_code,
            speech_episode_id=(
                None if episode is None else episode.speech_episode_id
            ),
            speech_start_monotonic=(
                None if episode is None else episode.speech_start_monotonic
            ),
            speech_end_monotonic=(
                None if episode is None else episode.speech_end_monotonic
            ),
            final_received_monotonic=final_ns,
            alignment_ms=alignment_ms,
            voiced_duration_ms=(
                None if episode is None else round(episode.voiced_duration_ms, 3)
            ),
            max_rms=None if episode is None else round(episode.max_rms, 6),
            max_peak=None if episode is None else round(episode.max_peak, 6),
            assistant_playback_active=assistant_playback_active,
            self_speech_candidate=self_speech.candidate,
            self_speech_confidence=self_speech.confidence,
        )


def _pcm_levels(pcm16: bytes) -> tuple[float, float]:
    samples = array("h")
    samples.frombytes(pcm16)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0, 0.0
    square_sum = 0.0
    peak = 0
    for sample in samples:
        peak = max(peak, abs(sample))
        normalized = sample / 32768.0
        square_sum += normalized * normalized
    return math.sqrt(square_sum / len(samples)), peak / 32768.0
