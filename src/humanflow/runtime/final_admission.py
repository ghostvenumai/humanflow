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
from .transcript_events import normalize_transcript


class FinalAdmissionReason(StrEnum):
    ACCEPTED = "ACCEPTED"
    NO_PCM_EPISODE = "NO_PCM_EPISODE"
    PCM_EPISODE_TOO_SHORT = "PCM_EPISODE_TOO_SHORT"
    FINAL_TOO_EARLY = "FINAL_TOO_EARLY"
    FINAL_TOO_LATE = "FINAL_TOO_LATE"
    STREAM_ID_MISMATCH = "STREAM_ID_MISMATCH"
    GENERATION_MISMATCH = "GENERATION_MISMATCH"
    FRAME_RANGE_MISMATCH = "FRAME_RANGE_MISMATCH"
    EPISODE_ALREADY_CONSUMED = "EPISODE_ALREADY_CONSUMED"
    SELF_SPEECH_MATCH = "SELF_SPEECH_MATCH"
    INSUFFICIENT_ACOUSTIC_EVIDENCE = "INSUFFICIENT_ACOUSTIC_EVIDENCE"
    STALE_FINAL = "STALE_FINAL"
    DUPLICATE_FINAL = "DUPLICATE_FINAL"
    SESSION_MISMATCH = "SESSION_MISMATCH"
    UNKNOWN = "UNKNOWN"


class PcmSpeechEpisodeState(StrEnum):
    ACTIVE = "ACTIVE"
    FINALIZING = "FINALIZING"
    CONSUMED = "CONSUMED"
    EXPIRED = "EXPIRED"


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
    generation: int
    state: PcmSpeechEpisodeState = PcmSpeechEpisodeState.ACTIVE
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
            "generation": self.generation,
            "state": self.state.value,
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
    episode_state: str | None
    episode_generation: int | None
    first_frame_sequence: int | None
    last_frame_sequence: int | None
    final_frame_sequence: int | None
    required_voiced_ms: float | None

    @classmethod
    def rejected_without_episode(
        cls,
        *,
        reason_code: FinalAdmissionReason,
        final_received_monotonic: int,
        final_frame_sequence: int | None,
        assistant_playback_active: bool,
    ) -> "FinalAdmissionAssessment":
        return cls(
            accepted=False,
            reason_code=reason_code.value,
            speech_episode_id=None,
            speech_start_monotonic=None,
            speech_end_monotonic=None,
            final_received_monotonic=final_received_monotonic,
            alignment_ms=None,
            voiced_duration_ms=None,
            max_rms=None,
            max_peak=None,
            assistant_playback_active=assistant_playback_active,
            self_speech_candidate=False,
            self_speech_confidence=0.0,
            episode_state=None,
            episode_generation=None,
            first_frame_sequence=None,
            last_frame_sequence=None,
            final_frame_sequence=final_frame_sequence,
            required_voiced_ms=None,
        )

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
            "episode_state": self.episode_state,
            "episode_generation": self.episode_generation,
            "first_frame_sequence": self.first_frame_sequence,
            "last_frame_sequence": self.last_frame_sequence,
            "final_frame_sequence": self.final_frame_sequence,
            "required_voiced_ms": self.required_voiced_ms,
        }


class FinalTranscriptAdmissionGate:
    """Link each provider FINAL to one unconsumed microphone speech episode."""

    def __init__(
        self,
        *,
        minimum_voiced_ms: float = 120.0,
        minimum_short_utterance_voiced_ms: float = 40.0,
        release_silence_ms: float = 180.0,
        finalization_window_ms: float = 5_000.0,
        minimum_rms: float = 0.010,
        minimum_peak: float = 0.030,
        noise_multiplier: float = 2.8,
        initial_noise_floor: float = 0.003,
        maximum_episodes: int = 32,
    ) -> None:
        if (
            minimum_voiced_ms <= 0
            or minimum_short_utterance_voiced_ms <= 0
            or release_silence_ms <= 0
        ):
            raise ValueError("speech episode durations must be positive")
        if minimum_short_utterance_voiced_ms > minimum_voiced_ms:
            raise ValueError("short utterance threshold must not exceed normal threshold")
        if finalization_window_ms < release_silence_ms:
            raise ValueError("finalization window must include release silence")
        if minimum_rms <= 0 or minimum_peak <= 0 or noise_multiplier <= 1:
            raise ValueError("speech evidence thresholds must be selective")
        if maximum_episodes < 1:
            raise ValueError("maximum episodes must be positive")
        self.minimum_voiced_ms = minimum_voiced_ms
        self.minimum_short_utterance_voiced_ms = minimum_short_utterance_voiced_ms
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
                    generation=self._generation,
                )
                self._active = active
            active.speech_end_monotonic = frame_end_ns
            active.last_frame_sequence = frame.sequence
            active.voiced_duration_ms += frame.duration_ms
            active.max_rms = max(active.max_rms, rms)
            active.max_peak = max(active.max_peak, peak)
            active.threshold = max(active.threshold, threshold)
            if active.voiced_duration_ms >= self.minimum_short_utterance_voiced_ms:
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
        episode, association_reason = self._associate_episode(update)
        if self_speech.suppress and episode is None:
            association_reason = FinalAdmissionReason.SELF_SPEECH_MATCH
        if episode is None:
            return self._assessment(
                accepted=False,
                reason_code=association_reason,
                episode=None,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
                final_frame_sequence=update.provenance.audio_frame_sequence,
            )
        required_voiced_ms = self._required_voiced_ms(update.text)
        if episode.voiced_duration_ms < self.minimum_short_utterance_voiced_ms:
            return self._assessment(
                accepted=False,
                reason_code=FinalAdmissionReason.PCM_EPISODE_TOO_SHORT,
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
                final_frame_sequence=update.provenance.audio_frame_sequence,
                required_voiced_ms=required_voiced_ms,
            )
        strong_independent_user_evidence = (
            episode.voiced_duration_ms >= max(480.0, required_voiced_ms * 1.5)
            and episode.max_rms >= episode.threshold * 1.2
            and episode.max_peak >= self.minimum_peak * 1.5
        )
        if self_speech.suppress and not strong_independent_user_evidence:
            return self._assessment(
                accepted=False,
                reason_code=FinalAdmissionReason.SELF_SPEECH_MATCH,
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
                final_frame_sequence=update.provenance.audio_frame_sequence,
                required_voiced_ms=required_voiced_ms,
            )
        if not episode.valid or episode.voiced_duration_ms < required_voiced_ms:
            return self._assessment(
                accepted=False,
                reason_code=FinalAdmissionReason.INSUFFICIENT_ACOUSTIC_EVIDENCE,
                episode=episode,
                final_ns=final_ns,
                assistant_playback_active=assistant_playback_active,
                self_speech=self_speech,
                final_frame_sequence=update.provenance.audio_frame_sequence,
                required_voiced_ms=required_voiced_ms,
            )
        episode.claimed_by_transcript_id = update.provenance.transcript_id
        episode.state = PcmSpeechEpisodeState.CONSUMED
        return self._assessment(
            accepted=True,
            reason_code=FinalAdmissionReason.ACCEPTED,
            episode=episode,
            final_ns=final_ns,
            assistant_playback_active=assistant_playback_active,
            self_speech=self_speech,
            final_frame_sequence=update.provenance.audio_frame_sequence,
            required_voiced_ms=required_voiced_ms,
        )

    def _associate_episode(
        self, update: TranscriptUpdate
    ) -> tuple[PcmSpeechEpisode | None, FinalAdmissionReason]:
        candidates = list(self._episodes)
        if self._active is not None:
            candidates.append(self._active)
        if not candidates:
            return None, FinalAdmissionReason.NO_PCM_EPISODE
        stream_candidates = [
            episode
            for episode in candidates
            if episode.stream_id == update.provenance.stream_id
        ]
        if not stream_candidates:
            return None, FinalAdmissionReason.STREAM_ID_MISMATCH
        final_ns = update.provenance.timestamp_ns
        final_sequence = update.provenance.audio_frame_sequence
        frame_candidates = stream_candidates
        if final_sequence is not None:
            frame_candidates = [
                episode
                for episode in stream_candidates
                if final_sequence >= episode.first_frame_sequence
            ]
            if not frame_candidates:
                return None, FinalAdmissionReason.FRAME_RANGE_MISMATCH
        temporal_candidates = [
            episode
            for episode in frame_candidates
            if episode.speech_start_monotonic <= final_ns
            and final_ns - episode.speech_end_monotonic <= self._finalization_window_ns
        ]
        if temporal_candidates:
            unconsumed = [
                episode
                for episode in temporal_candidates
                if episode.claimed_by_transcript_id is None
            ]
            if not unconsumed:
                return None, FinalAdmissionReason.EPISODE_ALREADY_CONSUMED
            return max(
                unconsumed, key=lambda episode: episode.speech_start_monotonic
            ), FinalAdmissionReason.ACCEPTED
        if all(final_ns < episode.speech_start_monotonic for episode in frame_candidates):
            return None, FinalAdmissionReason.FINAL_TOO_EARLY
        if all(
            final_ns - episode.speech_end_monotonic > self._finalization_window_ns
            for episode in frame_candidates
        ):
            for episode in frame_candidates:
                if episode.state is PcmSpeechEpisodeState.FINALIZING:
                    episode.state = PcmSpeechEpisodeState.EXPIRED
            return None, FinalAdmissionReason.FINAL_TOO_LATE
        return None, FinalAdmissionReason.UNKNOWN

    def _required_voiced_ms(self, text: str) -> float:
        token_count = len(normalize_transcript(text).split())
        if token_count <= 3:
            return self.minimum_short_utterance_voiced_ms
        return max(self.minimum_voiced_ms, min(700.0, token_count * 35.0))

    def _finish_active(self, *, observed_ns: int) -> list[PcmSpeechEpisodeEvent]:
        active = self._active
        if active is None:
            return []
        self._episodes.append(active)
        self._active = None
        if active.state is PcmSpeechEpisodeState.ACTIVE:
            active.state = PcmSpeechEpisodeState.FINALIZING
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
        final_frame_sequence: int | None,
        required_voiced_ms: float | None = None,
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
            reason_code=str(reason_code),
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
            episode_state=None if episode is None else episode.state.value,
            episode_generation=None if episode is None else episode.generation,
            first_frame_sequence=(
                None if episode is None else episode.first_frame_sequence
            ),
            last_frame_sequence=(
                None if episode is None else episode.last_frame_sequence
            ),
            final_frame_sequence=final_frame_sequence,
            required_voiced_ms=required_voiced_ms,
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
