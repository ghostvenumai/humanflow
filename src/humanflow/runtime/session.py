"""Full-duplex voice session with explicit turn, playback, and cancellation truth."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, replace
from decimal import Decimal
from time import monotonic_ns
from typing import Callable
from uuid import uuid4

from humanflow.audio.analysis import analyze_pcm16
from humanflow.audio.ledger import PlayedAudioLedger
from humanflow.audio.models import AudioChunk, AudioFrame, PlaybackReceipt
from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.cost.runtime import RuntimeCostObserver
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import TelemetrySink
from humanflow.turns.models import TurnDecision, TurnDecisionType
from humanflow.turns.policies import HybridTurnPolicy
from humanflow.tools.appointment_coordinator import (
    AppointmentTransactionCoordinator,
    AppointmentTransactionOutcome,
)
from humanflow.tools.providers import ToolProvider

from .providers import (
    AudioOutput,
    SpeechSynthesisRequest,
    StreamingReasoner,
    StreamingTTSProvider,
    StreamingTranscriber,
    TranscriptUpdate,
    provider_info,
)
from .acoustic_barge_in import (
    AcousticBargeInDetector,
    AcousticBargeInEvent,
    AcousticEventType,
)
from .appointment_state import AppointmentState, AppointmentStateDelta, AppointmentStateTracker
from .final_admission import (
    FinalAdmissionAssessment,
    FinalAdmissionReason,
    FinalTranscriptAdmissionGate,
    PcmSpeechEpisodeEvent,
    PcmSpeechEpisodeEventType,
)
from .prosody import ProsodyPlanner
from .speech_text import GermanSpeechNormalizer
from .self_speech import SelfSpeechAssessment, SelfSpeechGuard
from .transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
    TranscriptRejected,
)


_BARGE_IN_PREFIX = re.compile(
    r"^\s*(?:(?:nein\s+)?stopp|moment(?:\s*,?\s*(?:stopp|warte))?|"
    r"warte(?:\s+mal)?|halt)\b[\s,.:;!?\-]*(?P<followup>.*)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _SoftYieldEpisode:
    response_id: str
    correlation_id: str
    speech_onset_ns: int
    possible_interruption_ns: int
    duck_requested_ns: int | None = None
    speech_ended_ns: int | None = None
    duck_ack_ns: int | None = None
    resume_requested_ns: int | None = None
    resume_ack_ns: int | None = None
    hard_confirmed_ns: int | None = None
    cancel_signal_ns: int | None = None
    first_partial_ns: int | None = None
    final_transcript_ns: int | None = None
    takeover_evidence_ns: int | None = None
    takeover_evidence_type: str | None = None
    queue_invalidated_ns: int | None = None
    classification: str = "PENDING"
    backchannel_recovery_recorded: bool = False


class RealtimeVoiceSession:
    """Owns live-call concurrency; providers never own conversational state."""

    def __init__(
        self,
        *,
        conversation_id: str,
        sink: TelemetrySink,
        transcriber: StreamingTranscriber,
        reasoner: StreamingReasoner,
        synthesizer: StreamingTTSProvider,
        audio_output: AudioOutput,
        turn_policy: HybridTurnPolicy | None = None,
        prosody_planner: ProsodyPlanner | None = None,
        speech_normalizer: GermanSpeechNormalizer | None = None,
        self_speech_guard: SelfSpeechGuard | None = None,
        final_admission_gate: FinalTranscriptAdmissionGate | None = None,
        acoustic_barge_in_detector: AcousticBargeInDetector | None = None,
        appointment_state_tracker: AppointmentStateTracker | None = None,
        appointment_tool_provider: ToolProvider | None = None,
        appointment_tool_timeout_ms: float = 4_000.0,
        cost_observer: RuntimeCostObserver | None = None,
        soft_yield_recovery_delay_ms: float = 420.0,
        input_queue_size: int = 256,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if input_queue_size < 1:
            raise ValueError("input_queue_size must be positive")
        if soft_yield_recovery_delay_ms < 0:
            raise ValueError("soft yield recovery delay must be non-negative")
        self.state_machine = ConversationStateMachine(
            conversation_id=conversation_id,
            sink=sink,
            clock_ns=clock_ns,
        )
        self.ledger = PlayedAudioLedger()
        self._transcriber = transcriber
        self._reasoner = reasoner
        self._synthesizer = synthesizer
        self._audio_output = audio_output
        self._turn_policy = turn_policy or HybridTurnPolicy()
        self._prosody_planner = prosody_planner or ProsodyPlanner()
        self._speech_normalizer = speech_normalizer or GermanSpeechNormalizer()
        self._self_speech_guard = self_speech_guard or SelfSpeechGuard()
        self._final_admission_gate = (
            final_admission_gate or FinalTranscriptAdmissionGate()
        )
        self._acoustic_barge_in = (
            acoustic_barge_in_detector or AcousticBargeInDetector()
        )
        self._appointment_state_tracker = (
            appointment_state_tracker or AppointmentStateTracker()
        )
        self._appointment_transaction_coordinator = (
            AppointmentTransactionCoordinator(
                conversation_id=conversation_id,
                state_machine=self.state_machine,
                provider=appointment_tool_provider,
                timeout_ms=appointment_tool_timeout_ms,
            )
            if appointment_tool_provider is not None
            else None
        )
        self._cost_observer = cost_observer
        self._stt_audio_seconds_since_commit = Decimal("0")
        self._stt_partials_since_commit = 0
        self._soft_yield_recovery_delay_ms = soft_yield_recovery_delay_ms
        self._input_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(input_queue_size)
        self._clock_ns = clock_ns
        self._input_task: asyncio.Task[None] | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._response_id: str | None = None
        self._response_token: OperationToken | None = None
        self._cancel_event = asyncio.Event()
        self._audio_stopped = asyncio.Event()
        self._audio_stopped.set()
        self._cancel_requested_ns: int | None = None
        self._cancel_correlation_id: str | None = None
        self._last_playback_receipt: PlaybackReceipt | None = None
        self._playback_owner_response_id: str | None = None
        self._active_tts_provider: tuple[str, str, str] | None = None
        self._seen_audio_chunk_ids: set[str] = set()
        self._seen_audio_sequences: set[tuple[str, int]] = set()
        self._audio_signal_metrics: dict[str, tuple[float | None, float | None]] = {}
        self._cancelled_response_ids: set[str] = set()
        self._cancel_acknowledged_response_ids: set[str] = set()
        self._seen_final_transcript_ids: set[str] = set()
        self._accepted_user_turn_ids: set[str] = set()
        self._user_audio_active = False
        self._soft_yield: _SoftYieldEpisode | None = None
        self._acoustic_interrupt_task: asyncio.Task[float | None] | None = None
        self._soft_resume_task: asyncio.Task[None] | None = None
        self._stt_failure_type: str | None = None
        self._closed = False

    @property
    def state(self) -> ConversationState:
        return self.state_machine.state

    @property
    def response_active(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    @property
    def queued_input_frames(self) -> int:
        return self._input_queue.qsize()

    @property
    def last_playback_receipt(self) -> PlaybackReceipt | None:
        return self._last_playback_receipt

    @property
    def conversation_history_roles(self) -> tuple[str, ...]:
        history = getattr(self._reasoner, "history", ())
        if not isinstance(history, (list, tuple)):
            return ()
        roles = tuple(
            str(message.get("role"))
            for message in history
            if isinstance(message, dict)
        )
        for index, role in enumerate(roles):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise RuntimeError("conversation_history_role_invariant_violated")
        return roles

    @property
    def appointment_state(self) -> AppointmentState:
        """Controller-owned transaction truth; models cannot mutate this object."""

        return self._appointment_state_tracker.state

    @property
    def appointment_states(self) -> dict[str, AppointmentState]:
        """Independent controller-owned appointment objects keyed by stable ID."""

        return self._appointment_state_tracker.appointments

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if self._input_task is not None:
            return
        correlation_id = str(uuid4())
        self.state_machine.transition(
            ConversationState.LISTENING,
            reason_code="session_started",
            correlation_id=correlation_id,
        )
        self._input_task = asyncio.create_task(
            self._input_loop(), name=f"humanflow-input-{self.state_machine.conversation_id}"
        )

    def receive_audio(self, frame: AudioFrame) -> None:
        """Accept PCM without waiting on agent playback or downstream providers."""
        if self._closed or self._input_task is None:
            raise RuntimeError("session is not running")
        if self._stt_failure_type is not None or self._input_task.done():
            raise RuntimeError("streaming STT input pipeline is unavailable")
        assistant_playback_active = (
            self.response_active
            and self._response_id is not None
            and self.state
            in {
                ConversationState.SPEAKING,
                ConversationState.POSSIBLE_INTERRUPTION,
                ConversationState.OVERLAP,
            }
        )
        admission_events = self._final_admission_gate.observe(frame)
        speech_episode_id = self._final_admission_gate.active_episode_id
        if admission_events:
            speech_episode_id = admission_events[-1].episode.speech_episode_id
        for admission_event in admission_events:
            if not assistant_playback_active:
                self._record_pcm_speech_episode_event(admission_event)
        for event in self._acoustic_barge_in.observe(
            frame, assistant_playback_active=assistant_playback_active
        ):
            self._handle_acoustic_barge_in(
                event,
                speech_episode_id=speech_episode_id,
            )
        try:
            self._input_queue.put_nowait(frame)
        except asyncio.QueueFull as error:
            raise BufferError("realtime input queue is full") from error

    async def submit_transcript(self, update: TranscriptUpdate) -> TurnDecision:
        """Compatibility entry point routed through the authoritative user gate."""
        return await self.accept_user_transcript(update)

    async def accept_user_transcript(self, update: TranscriptUpdate) -> TurnDecision:
        """Accept only allowlisted user provenance and suppress probable self-speech."""
        if self._closed or self._input_task is None:
            raise RuntimeError("session is not running")
        correlation_id = str(uuid4())
        provenance = update.provenance
        if not provenance.is_allowlisted_user_input:
            reason = (
                "assistant_origin_event_forbidden_from_user_history"
                if provenance.is_assistant_origin
                else "transcript_source_not_allowlisted"
            )
            self._record_transcript_provenance(
                update=update,
                correlation_id=correlation_id,
                accepted_by_user_ingestion=False,
                accepted_as_user_turn=False,
                rejection_reason=reason,
            )
            self.state_machine.record(
                EventType.TRANSCRIPT_REJECTED,
                correlation_id=correlation_id,
                reason_code=reason,
                payload={
                    "transcript_id": provenance.transcript_id,
                    "origin": provenance.origin.value,
                    "event_kind": provenance.event_kind.value,
                    "assistant_origin_event_to_user_history": "FORBIDDEN",
                },
            )
            raise TranscriptRejected(reason)

        if provenance.origin is TranscriptOrigin.STREAMING_STT_PROVIDER:
            rejection_reason: str | None = None
            if provenance.recognition_input_binding != "EXACT_GETUSERMEDIA_PCM16":
                rejection_reason = FinalAdmissionReason.SESSION_MISMATCH.value
            expected_source_binding = getattr(
                self._transcriber, "audio_source_binding", None
            )
            if (
                rejection_reason is None
                and isinstance(expected_source_binding, tuple)
                and len(expected_source_binding) == 2
                and (provenance.audio_capture_id, provenance.stream_id)
                != expected_source_binding
            ):
                rejection_reason = FinalAdmissionReason.STREAM_ID_MISMATCH.value
            if rejection_reason is not None:
                if update.is_final:
                    self._record_pre_admission_rejection(
                        update=update,
                        reason_code=rejection_reason,
                        correlation_id=correlation_id,
                    )
                self._record_transcript_provenance(
                    update=update,
                    correlation_id=correlation_id,
                    accepted_by_user_ingestion=False,
                    accepted_as_user_turn=False,
                    rejection_reason=rejection_reason,
                )
                self.state_machine.record(
                    EventType.TRANSCRIPT_REJECTED,
                    correlation_id=correlation_id,
                    reason_code=rejection_reason,
                    payload={
                        "transcript_id": provenance.transcript_id,
                        "non_authoritative_pcm_to_user_history": "FORBIDDEN",
                    },
                )
                raise TranscriptRejected(rejection_reason)

        expected_stt_session_id = getattr(
            self._transcriber, "provider_session_id", None
        )
        if (
            provenance.origin is TranscriptOrigin.STREAMING_STT_PROVIDER
            and isinstance(expected_stt_session_id, str)
            and provenance.stt_session_id != expected_stt_session_id
        ):
            reason = FinalAdmissionReason.SESSION_MISMATCH.value
            if update.is_final:
                self._record_pre_admission_rejection(
                    update=update,
                    reason_code=reason,
                    correlation_id=correlation_id,
                )
            self._record_transcript_provenance(
                update=update,
                correlation_id=correlation_id,
                accepted_by_user_ingestion=False,
                accepted_as_user_turn=False,
                rejection_reason=reason,
            )
            self.state_machine.record(
                EventType.TRANSCRIPT_REJECTED,
                correlation_id=correlation_id,
                reason_code=reason,
                payload={
                    "transcript_id": provenance.transcript_id,
                    "stale_stt_session_to_user_history": "FORBIDDEN",
                },
            )
            raise TranscriptRejected(reason)

        if update.is_final and provenance.transcript_id in self._seen_final_transcript_ids:
            reason = FinalAdmissionReason.DUPLICATE_FINAL.value
            self._record_pre_admission_rejection(
                update=update,
                reason_code=reason,
                correlation_id=correlation_id,
            )
            self._record_transcript_provenance(
                update=update,
                correlation_id=correlation_id,
                accepted_by_user_ingestion=False,
                accepted_as_user_turn=False,
                rejection_reason=reason,
            )
            self.state_machine.record(
                EventType.DUPLICATE_TRANSCRIPT_REJECTED,
                correlation_id=correlation_id,
                reason_code="final_transcript_id_already_processed",
                payload={
                    "transcript_id": provenance.transcript_id,
                    "duplicate_final_to_user_history": "FORBIDDEN",
                },
            )
            raise TranscriptRejected(reason)

        observed_ns = self._clock_ns()
        playback_active = self.state in {
            ConversationState.SPEAKING,
            ConversationState.POSSIBLE_INTERRUPTION,
            ConversationState.OVERLAP,
        }
        assessment = self._self_speech_guard.assess(
            text=update.text,
            observed_ns=observed_ns,
            origin=provenance.origin,
            playback_active=playback_active,
        )
        if assessment.candidate:
            self._record_self_speech_event(
                EventType.SELF_SPEECH_CANDIDATE,
                update=update,
                assessment=assessment,
                correlation_id=correlation_id,
                reason_code="recent_assistant_speech_content_and_timing_match",
            )
        final_admission: FinalAdmissionAssessment | None = None
        if update.is_final and provenance.origin is TranscriptOrigin.STREAMING_STT_PROVIDER:
            final_admission = self._final_admission_gate.assess_final(
                update,
                assistant_playback_active=playback_active,
                self_speech=assessment,
            )
            self._record_final_admission(
                update=update,
                assessment=final_admission,
                correlation_id=correlation_id,
            )
            if not final_admission.accepted:
                reason = final_admission.reason_code
                self._seen_final_transcript_ids.add(provenance.transcript_id)
                self._record_transcript_provenance(
                    update=update,
                    correlation_id=correlation_id,
                    accepted_by_user_ingestion=False,
                    accepted_as_user_turn=False,
                    rejection_reason=reason,
                    response_id=assessment.matched_response_id,
                    final_admission=final_admission,
                )
                if assessment.suppress:
                    self._record_self_speech_event(
                        EventType.SELF_SPEECH_SUPPRESSED,
                        update=update,
                        assessment=assessment,
                        correlation_id=correlation_id,
                        reason_code=reason,
                    )
                self.state_machine.record(
                    EventType.TRANSCRIPT_REJECTED,
                    correlation_id=correlation_id,
                    reason_code=reason,
                    payload={
                        "transcript_id": provenance.transcript_id,
                        "final_to_user_history": "FORBIDDEN",
                        "speech_episode_id": final_admission.speech_episode_id,
                    },
                )
                raise TranscriptRejected(reason)
        if assessment.suppress and not (
            final_admission is not None and final_admission.accepted
        ):
            reason = assessment.rejection_reason or "probable_assistant_self_speech"
            self._record_transcript_provenance(
                update=update,
                correlation_id=correlation_id,
                accepted_by_user_ingestion=False,
                accepted_as_user_turn=False,
                rejection_reason=reason,
                response_id=assessment.matched_response_id,
            )
            self._record_self_speech_event(
                EventType.SELF_SPEECH_SUPPRESSED,
                update=update,
                assessment=assessment,
                correlation_id=correlation_id,
                reason_code=reason,
            )
            raise TranscriptRejected(reason)

        if update.is_final:
            self._seen_final_transcript_ids.add(provenance.transcript_id)

        response_task_before = self._response_task
        decision = await self._handle_transcript(
            update,
            final_admission=final_admission,
        )
        accepted_as_user_turn = (
            update.is_final and self._response_task is not response_task_before
        )
        if not update.is_final and accepted_as_user_turn:
            raise RuntimeError("partial_transcript_to_user_history_forbidden")
        if accepted_as_user_turn:
            if provenance.transcript_id in self._accepted_user_turn_ids:
                raise RuntimeError("user_history_write_count_per_turn_exceeded")
            self._accepted_user_turn_ids.add(provenance.transcript_id)
        self._record_transcript_provenance(
            update=update,
            correlation_id=correlation_id,
            accepted_by_user_ingestion=True,
            accepted_as_user_turn=accepted_as_user_turn,
            rejection_reason=None,
            response_id=assessment.matched_response_id,
            final_admission=final_admission,
        )
        if assessment.candidate:
            self._record_self_speech_event(
                EventType.SELF_SPEECH_ACCEPTED_AS_REAL_USER,
                update=update,
                assessment=assessment,
                correlation_id=correlation_id,
                reason_code=(
                    "independent_pcm_evidence_overrode_self_speech_candidate"
                    if assessment.suppress
                    else "candidate_below_conservative_suppression_threshold"
                ),
            )
        return decision

    def _record_pcm_speech_episode_event(
        self, event: PcmSpeechEpisodeEvent
    ) -> None:
        started = event.event_type is PcmSpeechEpisodeEventType.STARTED
        self._user_audio_active = started
        self.state_machine.record(
            EventType.USER_AUDIO_STARTED if started else EventType.USER_AUDIO_STOPPED,
            correlation_id=event.episode.speech_episode_id,
            reason_code=(
                "authoritative_pcm_final_admission_speech_onset"
                if started
                else "authoritative_pcm_final_admission_speech_ended"
            ),
            payload={
                **event.episode.to_dict(),
                "observed_ns": event.observed_ns,
                "source": "AUTHORITATIVE_GETUSERMEDIA_PCM16",
                "consumer": "FINAL_TRANSCRIPT_ADMISSION",
            },
        )

    def _handle_acoustic_barge_in(
        self,
        event: AcousticBargeInEvent,
        *,
        speech_episode_id: str | None = None,
    ) -> None:
        """React to PCM VAD before STT produces language."""

        response_id = self._response_id
        if response_id is None or not self.response_active:
            return
        observed_ns = event.observed_ns or self._clock_ns()
        speech_onset_ns = event.speech_onset_ns or observed_ns
        payload = {
            "response_id": response_id,
            "speech_onset_ns": speech_onset_ns,
            "observed_ns": observed_ns,
            "rms": event.rms,
            "peak": event.peak,
            "threshold": event.threshold,
            "speech_duration_ms": event.speech_duration_ms,
            "acoustic_speech_onset_latency_ms": event.detection_latency_ms,
            "source": "AUTHORITATIVE_GETUSERMEDIA_PCM16",
            "stt_dependency": "NONE",
            "speech_episode_id": speech_episode_id,
        }
        if event.event_type is AcousticEventType.SPEECH_ONSET:
            if (
                self._soft_yield is not None
                and self._soft_yield.response_id == response_id
                and self._soft_yield.speech_ended_ns is None
            ):
                return
            correlation_id = str(uuid4())
            self._soft_yield = _SoftYieldEpisode(
                response_id=response_id,
                correlation_id=correlation_id,
                speech_onset_ns=speech_onset_ns,
                possible_interruption_ns=observed_ns,
            )
            self._user_audio_active = True
            self.state_machine.record(
                EventType.USER_AUDIO_STARTED,
                correlation_id=correlation_id,
                reason_code="authoritative_pcm_vad_speech_onset",
                payload=payload,
            )
            self.state_machine.record(
                EventType.POSSIBLE_INTERRUPTION,
                correlation_id=correlation_id,
                reason_code="stable_pcm_speech_during_assistant_playback",
                payload=payload,
            )
            duck_requested_ns = self._clock_ns()
            self._soft_yield.duck_requested_ns = duck_requested_ns
            self.state_machine.record(
                EventType.PLAYBACK_DUCK_REQUESTED,
                correlation_id=correlation_id,
                reason_code="soft_yield_before_semantic_classification",
                payload={
                    **payload,
                    "duck_requested_ns": duck_requested_ns,
                    "duck_stage": "MILD_SOFT_YIELD",
                    "target_gain": 0.55,
                    "speech_onset_to_duck_request_ms": max(
                        0.0,
                        (duck_requested_ns - speech_onset_ns) / 1_000_000.0,
                    ),
                },
            )
            duck = getattr(self._audio_output, "soft_duck", None)
            if callable(duck):
                duck(response_id=response_id, speech_onset_ns=speech_onset_ns)
            return

        episode = self._soft_yield
        if episode is None or episode.response_id != response_id:
            return
        if event.event_type is AcousticEventType.SUSTAINED_TAKEOVER:
            if episode.hard_confirmed_ns is not None or self._cancel_event.is_set():
                return
            episode.classification = "SUSTAINED_TAKEOVER"
            self._record_takeover_evidence(
                episode,
                evidence_type="ACOUSTIC_SUSTAINED_TAKEOVER",
                semantic=False,
            )
            if self._soft_resume_task is not None:
                self._soft_resume_task.cancel()
            self._acoustic_interrupt_task = asyncio.create_task(
                self.interrupt(
                    correlation_id=episode.correlation_id,
                    speech_onset_ns=episode.speech_onset_ns,
                    reason_code="acoustic_sustained_takeover",
                ),
                name=f"humanflow-acoustic-interrupt-{response_id}",
            )
            return

        if event.event_type is AcousticEventType.SPEECH_ENDED:
            episode.speech_ended_ns = observed_ns
            self._user_audio_active = False
            self.state_machine.record(
                EventType.USER_AUDIO_STOPPED,
                correlation_id=episode.correlation_id,
                reason_code="authoritative_pcm_vad_speech_ended",
                payload=payload,
            )
            if episode.hard_confirmed_ns is not None or self._cancel_event.is_set():
                return
            episode.classification = "AWAITING_STT_OR_TRANSIENT_RECOVERY"
            if self._soft_resume_task is not None:
                self._soft_resume_task.cancel()
            self._soft_resume_task = asyncio.create_task(
                self._resume_soft_yield_after_classification_window(episode),
                name=f"humanflow-soft-resume-{response_id}",
            )

    def _record_takeover_evidence(
        self,
        episode: _SoftYieldEpisode,
        *,
        evidence_type: str,
        semantic: bool,
    ) -> None:
        """Record the first evidence that can promote soft yield to hard cancel."""

        if episode.takeover_evidence_ns is not None:
            return
        evidence_ns = self._clock_ns()
        episode.takeover_evidence_ns = evidence_ns
        episode.takeover_evidence_type = evidence_type
        self.state_machine.record(
            EventType.TAKEOVER_EVIDENCE,
            correlation_id=episode.correlation_id,
            reason_code=evidence_type.casefold(),
            payload={
                "response_id": episode.response_id,
                "speech_onset_ns": episode.speech_onset_ns,
                "evidence_ns": evidence_ns,
                "evidence_type": evidence_type,
                "semantic_evidence": semantic,
                "speech_onset_to_takeover_evidence_ms": max(
                    0.0,
                    (evidence_ns - episode.speech_onset_ns) / 1_000_000.0,
                ),
            },
        )

    async def _resume_soft_yield_after_classification_window(
        self, episode: _SoftYieldEpisode
    ) -> None:
        try:
            await asyncio.sleep(self._soft_yield_recovery_delay_ms / 1_000.0)
        except asyncio.CancelledError:
            return
        if (
            self._soft_yield is not episode
            or episode.hard_confirmed_ns is not None
            or self._cancel_event.is_set()
        ):
            return
        self._request_playback_resume(
            episode, reason_code="uncertain_transient_classification_window_elapsed"
        )

    def _request_playback_resume(
        self, episode: _SoftYieldEpisode, *, reason_code: str
    ) -> None:
        if episode.resume_requested_ns is not None:
            return
        episode.resume_requested_ns = self._clock_ns()
        self.state_machine.record(
            EventType.PLAYBACK_RESUME_REQUESTED,
            correlation_id=episode.correlation_id,
            reason_code=reason_code,
            payload={
                "response_id": episode.response_id,
                "speech_onset_ns": episode.speech_onset_ns,
                "speech_ended_ns": episode.speech_ended_ns,
                "resume_requested_ns": episode.resume_requested_ns,
                "classification_hold_ms": self._soft_yield_recovery_delay_ms,
            },
        )
        resume = getattr(self._audio_output, "resume_playback", None)
        if callable(resume):
            resume(
                response_id=episode.response_id,
                speech_onset_ns=episode.speech_onset_ns,
            )

    async def interrupt(
        self,
        *,
        correlation_id: str | None = None,
        speech_onset_ns: int | None = None,
        reason_code: str = "intentional_interruption",
    ) -> float | None:
        """Stop audible output and return request-to-actual-stop latency in milliseconds."""
        if not self.response_active:
            return None
        correlation_id = correlation_id or str(uuid4())
        request_ns = self._clock_ns()
        if self._cancel_event.is_set():
            await self._audio_stopped.wait()
            receipt = self._last_playback_receipt
            if receipt is None or not receipt.cancelled:
                return None
            return max(
                0.0,
                (receipt.playback_stopped_ns - request_ns) / 1_000_000.0,
            )
        self._cancel_requested_ns = request_ns
        self._cancel_correlation_id = correlation_id
        episode = self._soft_yield
        if episode is not None and episode.response_id == self._response_id:
            speech_onset_ns = episode.speech_onset_ns
            episode.hard_confirmed_ns = request_ns
            episode.classification = "INTENTIONAL_INTERRUPTION"
            if self._soft_resume_task is not None:
                self._soft_resume_task.cancel()
        speech_onset_ns = speech_onset_ns or request_ns
        state = self.state
        if state is ConversationState.SPEAKING:
            self.state_machine.transition(
                ConversationState.POSSIBLE_INTERRUPTION,
                reason_code="intentional_speech_detected",
                correlation_id=correlation_id,
            )
            self.state_machine.record(
                EventType.INTERRUPTION_CANDIDATE,
                correlation_id=correlation_id,
                reason_code="turn_policy_interruption",
                payload={"detected_ns": request_ns},
            )
            self.state_machine.transition(
                ConversationState.INTERRUPTED,
                reason_code="intentional_interruption_confirmed",
                correlation_id=correlation_id,
            )
        elif state in {
            ConversationState.POSSIBLE_INTERRUPTION,
            ConversationState.OVERLAP,
        }:
            self.state_machine.transition(
                ConversationState.INTERRUPTED,
                reason_code="intentional_interruption_confirmed",
                correlation_id=correlation_id,
            )
        elif state is ConversationState.THINKING:
            self.state_machine.transition(
                ConversationState.INTERRUPTED,
                reason_code="interrupted_before_audio",
                correlation_id=correlation_id,
            )
        self.state_machine.invalidate_operations(
            reason_code="barge_in_cancelled_response",
            correlation_id=correlation_id,
        )
        self.state_machine.record(
            EventType.INTERRUPTION_CONFIRMED,
            correlation_id=correlation_id,
            reason_code=reason_code,
            payload={
                "speech_onset_ns": speech_onset_ns,
                "confirmed_ns": request_ns,
                "speech_onset_to_hard_cancel_ms": max(
                    0.0, (request_ns - speech_onset_ns) / 1_000_000.0
                ),
                "takeover_evidence_type": (
                    episode.takeover_evidence_type if episode is not None else None
                ),
                "takeover_evidence_to_confirmation_ms": (
                    None
                    if episode is None or episode.takeover_evidence_ns is None
                    else max(
                        0.0,
                        (request_ns - episode.takeover_evidence_ns) / 1_000_000.0,
                    )
                ),
            },
        )
        if self._response_id is not None:
            self._cancelled_response_ids.add(self._response_id)
            invalidate = getattr(self._audio_output, "invalidate_response", None)
            if callable(invalidate):
                invalidate(
                    response_id=self._response_id,
                    speech_onset_ns=speech_onset_ns,
                )
            if episode is not None and episode.response_id == self._response_id:
                episode.queue_invalidated_ns = self._clock_ns()
        cancel_signal_ns = self._clock_ns()
        if episode is not None and episode.response_id == self._response_id:
            episode.cancel_signal_ns = cancel_signal_ns
        self.state_machine.record(
            EventType.AUDIO_CANCEL_SIGNAL,
            correlation_id=correlation_id,
            reason_code="active_response_epoch_invalidated",
            payload={
                "response_id": self._response_id,
                "speech_onset_ns": speech_onset_ns,
                "cancel_signal_ns": cancel_signal_ns,
                "speech_onset_to_cancel_signal_ms": max(
                    0.0, (cancel_signal_ns - speech_onset_ns) / 1_000_000.0
                ),
                "queue_invalidated_ns": (
                    episode.queue_invalidated_ns if episode is not None else None
                ),
                "confirmation_to_queue_invalidation_ms": (
                    None
                    if episode is None or episode.queue_invalidated_ns is None
                    else max(
                        0.0,
                        (episode.queue_invalidated_ns - request_ns) / 1_000_000.0,
                    )
                ),
                "queue_invalidation_to_cancel_signal_ms": (
                    None
                    if episode is None or episode.queue_invalidated_ns is None
                    else max(
                        0.0,
                        (cancel_signal_ns - episode.queue_invalidated_ns)
                        / 1_000_000.0,
                    )
                ),
            },
        )
        self._cancel_event.set()
        await self._audio_stopped.wait()
        receipt = self._last_playback_receipt
        if receipt is None or not receipt.cancelled:
            return None
        return max(0.0, (receipt.playback_stopped_ns - request_ns) / 1_000_000.0)

    def acknowledge_playback_control(self, message: dict[str, object]) -> bool:
        """Record browser-applied duck/resume receipts on the server clock."""

        message_type = message.get("type")
        response_id = message.get("response_id")
        episode = self._soft_yield
        if (
            episode is None
            or not isinstance(response_id, str)
            or response_id != episode.response_id
        ):
            return False
        acknowledged_ns = self._clock_ns()
        if message_type == "playback_ducked" and episode.duck_ack_ns is None:
            episode.duck_ack_ns = acknowledged_ns
            self.state_machine.record(
                EventType.PLAYBACK_DUCK_STARTED,
                correlation_id=episode.correlation_id,
                reason_code="browser_gain_duck_acknowledged",
                payload={
                    "response_id": response_id,
                    "speech_onset_ns": episode.speech_onset_ns,
                    "duck_ack_ns": acknowledged_ns,
                    "speech_onset_to_soft_duck_ms": max(
                        0.0,
                        (acknowledged_ns - episode.speech_onset_ns) / 1_000_000.0,
                    ),
                    "target_gain": message.get("target_gain"),
                    "measurement_scope": "server_pcm_onset_to_browser_ack_upper_bound",
                },
            )
            return True
        if message_type == "playback_resumed" and episode.resume_ack_ns is None:
            episode.resume_ack_ns = acknowledged_ns
            self.state_machine.record(
                EventType.PLAYBACK_RESUMED,
                correlation_id=episode.correlation_id,
                reason_code="browser_gain_resume_acknowledged",
                payload={
                    "response_id": response_id,
                    "speech_onset_ns": episode.speech_onset_ns,
                    "resume_ack_ns": acknowledged_ns,
                    "speech_onset_to_playback_resume_ms": max(
                        0.0,
                        (acknowledged_ns - episode.speech_onset_ns) / 1_000_000.0,
                    ),
                    "measurement_scope": "server_pcm_onset_to_browser_ack_upper_bound",
                },
            )
            if episode.classification == "BACKCHANNEL":
                self._record_backchannel_recovery(episode, text=None)
            return True
        return False

    def _record_backchannel_recovery(
        self, episode: _SoftYieldEpisode, *, text: str | None
    ) -> None:
        if (
            episode.backchannel_recovery_recorded
            or episode.resume_ack_ns is None
        ):
            return
        episode.backchannel_recovery_recorded = True
        self.state_machine.record(
            EventType.BACKCHANNEL_RECOVERY,
            correlation_id=episode.correlation_id,
            reason_code="backchannel_classified_and_playback_resumed",
            payload={
                "response_id": episode.response_id,
                "text": text,
                "backchannel_recovery_latency_ms": max(
                    0.0,
                    (episode.resume_ack_ns - episode.speech_onset_ns)
                    / 1_000_000.0,
                ),
            },
        )

    async def wait_for_response(self) -> None:
        task = self._response_task
        if task is not None:
            await task

    async def wait_for_input(self) -> None:
        """Wait until all PCM frames accepted so far have reached the transcriber."""
        await self._input_queue.join()
        if self._stt_failure_type is not None:
            raise RuntimeError("streaming STT input pipeline is unavailable")

    async def close(self, *, reason_code: str = "normal_shutdown") -> None:
        if self._closed:
            return
        if self.response_active:
            await self.interrupt(correlation_id=str(uuid4()))
            await self.wait_for_response()
        if self._acoustic_interrupt_task is not None:
            await asyncio.gather(
                self._acoustic_interrupt_task, return_exceptions=True
            )
        if self._soft_resume_task is not None:
            self._soft_resume_task.cancel()
            await asyncio.gather(self._soft_resume_task, return_exceptions=True)
        self._closed = True
        if self._input_task is not None:
            if self._input_task.done():
                await asyncio.gather(self._input_task, return_exceptions=True)
                self._drain_input_queue()
            else:
                await self._input_queue.put(None)
                await self._input_task
        await self._transcriber.close()
        if self._cost_observer is not None:
            try:
                await self._cost_observer.close()
            except Exception as error:
                self._record_cost_failure("close", error)
        correlation_id = str(uuid4())
        if self.state is not ConversationState.LISTENING:
            if self.state in {ConversationState.INTERRUPTED, ConversationState.RECOVERING}:
                self.state_machine.transition(
                    ConversationState.LISTENING,
                    reason_code="shutdown_normalized_state",
                    correlation_id=correlation_id,
                )
        self.state_machine.transition(
            ConversationState.DISCONNECTING,
            reason_code=reason_code,
            correlation_id=correlation_id,
        )
        self.state_machine.record(
            EventType.CALL_ENDED,
            correlation_id=correlation_id,
            reason_code=reason_code,
        )
        self.state_machine.transition(
            ConversationState.IDLE,
            reason_code="resources_released",
            correlation_id=correlation_id,
        )

    def _record_stt_cost(self, update: TranscriptUpdate) -> None:
        observer = self._cost_observer
        if observer is None:
            return
        provider = update.provider or provider_info(self._transcriber, role="stt")
        self._observe_cost(
            "record_stt",
            operation_id=f"stt:{update.provenance.transcript_id}",
            turn_id=update.provenance.transcript_id,
            provider=provider.provider,
            model=provider.model,
            audio_seconds=self._stt_audio_seconds_since_commit,
            partial_count=self._stt_partials_since_commit,
            provider_session_id=update.provenance.stt_session_id,
        )
        self._stt_audio_seconds_since_commit = Decimal("0")
        self._stt_partials_since_commit = 0

    def _record_tool_cost(
        self,
        *,
        outcome: AppointmentTransactionOutcome,
        token: OperationToken,
        turn_id: str,
    ) -> None:
        observer = self._cost_observer
        if observer is None:
            return
        appointment_id = outcome.value.get("appointment_id")
        result_status = outcome.value.get("result_status") or outcome.value.get("status")
        self._observe_cost(
            "record_tool",
            operation_id=f"{token.operation_id}:tool:{outcome.tool_name}",
            turn_id=turn_id,
            tool_name=outcome.tool_name,
            duration_ms=outcome.elapsed_ms,
            success=outcome.success,
            retry=False,
            appointment_id=(str(appointment_id) if appointment_id is not None else None),
            transaction_result=(str(result_status) if result_status is not None else None),
            failure_class=outcome.failure_reason,
        )

    def _record_tts_cost(
        self,
        *,
        token: OperationToken,
        turn_id: str,
        response_id: str,
        segment_id: str,
        submitted_characters: int,
        generated_audio_seconds: Decimal,
        actual_provider: dict[str, object],
        fallback: bool,
        cancelled: bool,
    ) -> None:
        observer = self._cost_observer
        if observer is None:
            return
        primary = provider_info(self._synthesizer, role="tts")
        if fallback:
            failure_class = getattr(
                self._synthesizer, "last_fallback_reason", None
            ) or "primary_failed_before_audio"
            self._observe_cost(
                "record_primary_tts_failure",
                operation_id=f"{token.operation_id}:tts:{segment_id}:primary-failure",
                turn_id=turn_id,
                response_id=response_id,
                provider=primary.provider,
                model=primary.model,
                failure_class=str(failure_class),
            )
        metrics = getattr(self._synthesizer, "last_request_metrics", None)
        reported_characters = getattr(metrics, "reported_billable_characters", None)
        first_audio_latency_ms = getattr(metrics, "first_pcm_latency_ms", None)
        self._observe_cost(
            "record_tts",
            operation_id=f"{token.operation_id}:tts:{segment_id}",
            turn_id=turn_id,
            response_id=response_id,
            provider=str(actual_provider.get("provider") or primary.provider),
            model=str(actual_provider.get("model") or primary.model),
            characters=submitted_characters,
            audio_seconds=generated_audio_seconds,
            reported_billable_characters=(
                int(reported_characters)
                if isinstance(reported_characters, int)
                else None
            ),
            fallback=fallback,
            cancelled=cancelled,
            latency_ms=(
                float(first_audio_latency_ms)
                if isinstance(first_audio_latency_ms, (int, float))
                else None
            ),
        )

    def _record_llm_cost(
        self,
        *,
        usage_payload: object,
        token: OperationToken,
        turn_id: str,
        response_id: str,
        provider: dict[str, str],
        generation_started_ns: int,
    ) -> None:
        observer = self._cost_observer
        if observer is None or not isinstance(usage_payload, dict):
            return
        self._observe_cost(
            "record_llm",
            operation_id=f"{token.operation_id}:llm",
            turn_id=turn_id,
            response_id=response_id,
            provider=provider["provider"],
            model=provider["model"],
            input_tokens=int(usage_payload.get("input_tokens", 0)),
            output_tokens=int(usage_payload.get("output_tokens", 0)),
            latency_ms=max(
                0.0,
                (self._clock_ns() - generation_started_ns) / 1_000_000.0,
            ),
            success=True,
        )

    def _observe_cost(self, method_name: str, **arguments: object) -> object | None:
        observer = self._cost_observer
        if observer is None:
            return None
        try:
            result = getattr(observer, method_name)(**arguments)
            if result is True:
                self.state_machine.record(
                    EventType.COST_EVENT_RECORDED,
                    correlation_id=str(uuid4()),
                    reason_code="queued_for_nonblocking_cost_persistence",
                    payload={
                        "cost_operation": method_name,
                        "operation_id": arguments.get("operation_id"),
                        "provider": arguments.get("provider"),
                        "model": arguments.get("model"),
                        "persistence_status": "QUEUED_BEST_EFFORT",
                        "conversation_dependency": "NONE",
                    },
                )
            return result
        except Exception as error:
            self._record_cost_failure(method_name, error)
            return None

    def _record_cost_failure(self, operation: str, error: Exception) -> None:
        self.state_machine.record(
            EventType.COST_LEDGER_WRITE_FAILED,
            correlation_id=str(uuid4()),
            reason_code="cost_observability_isolated_from_conversation",
            payload={
                "operation": operation,
                "exception_type": type(error).__name__,
                "conversation_continues": True,
            },
        )

    async def _input_loop(self) -> None:
        while True:
            frame = await self._input_queue.get()
            try:
                if frame is None:
                    return
                updates = await self._transcriber.ingest(frame)
                self._stt_audio_seconds_since_commit += Decimal(
                    frame.samples_per_channel
                ) / Decimal(frame.sample_rate_hz)
                for update in updates:
                    if update.is_final:
                        self._record_stt_cost(update)
                    else:
                        self._stt_partials_since_commit += 1
                    try:
                        await self.accept_user_transcript(update)
                    except TranscriptRejected:
                        continue
            except Exception as error:
                self._stt_failure_type = type(error).__name__
                self.state_machine.record(
                    EventType.STT_PROVIDER_FAILED,
                    correlation_id=str(uuid4()),
                    reason_code="streaming_stt_input_pipeline_failed",
                    payload={
                        "exception_type": type(error).__name__,
                        "provider": provider_info(
                            self._transcriber, role="stt"
                        ).to_dict(),
                    },
                )
                self._drain_input_queue()
                return
            finally:
                self._input_queue.task_done()

    def _drain_input_queue(self) -> None:
        while True:
            try:
                self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._input_queue.task_done()

    async def _handle_transcript(
        self,
        update: TranscriptUpdate,
        *,
        final_admission: FinalAdmissionAssessment | None = None,
    ) -> TurnDecision:
        correlation_id = str(uuid4())
        transcript_provider = update.provider or provider_info(
            self._transcriber, role="stt"
        )
        transcript_ns = update.provenance.timestamp_ns or self._clock_ns()
        episode = self._soft_yield
        stt_timing: dict[str, float] = {}
        if episode is not None and transcript_ns >= episode.speech_onset_ns:
            latency_ms = max(
                0.0, (transcript_ns - episode.speech_onset_ns) / 1_000_000.0
            )
            if update.is_final:
                episode.final_transcript_ns = transcript_ns
                stt_timing["final_stt_ms"] = latency_ms
            elif episode.first_partial_ns is None:
                episode.first_partial_ns = transcript_ns
                stt_timing["first_stt_partial_ms"] = latency_ms
        speech_active = update.signals.speech_active
        acoustic_episode_active = (
            episode is not None
            and transcript_ns >= episode.speech_onset_ns
            and transcript_ns - episode.speech_onset_ns <= 5_000_000_000
        )
        if speech_active and not self._user_audio_active and not acoustic_episode_active:
            self._user_audio_active = True
            self.state_machine.record(
                EventType.USER_AUDIO_STARTED,
                correlation_id=correlation_id,
                reason_code="transcriber_speech_started",
            )
        elif not speech_active and self._user_audio_active:
            self._user_audio_active = False
            self.state_machine.record(
                EventType.USER_AUDIO_STOPPED,
                correlation_id=correlation_id,
                reason_code="transcriber_speech_stopped",
            )

        event_type = EventType.FINAL_TRANSCRIPT if update.is_final else EventType.PARTIAL_TRANSCRIPT
        self.state_machine.record(
            event_type,
            correlation_id=correlation_id,
            reason_code="provider_transcript",
            payload={
                "text": update.text,
                "event_kind": update.provenance.event_kind.value,
                "transcript_id": update.provenance.transcript_id,
                "provider": transcript_provider.to_dict(),
                "final_admission": (
                    final_admission.to_dict()
                    if final_admission is not None
                    else None
                ),
                "partial_transcript_to_user_history": (
                    "FORBIDDEN" if not update.is_final else None
                ),
                **stt_timing,
                "signals": {
                    "speech_active": update.signals.speech_active,
                    "silence_duration_ms": update.signals.silence_duration_ms,
                    "utterance_duration_ms": update.signals.utterance_duration_ms,
                    "semantic_complete": update.signals.semantic_complete,
                    "provider_endpointed": update.signals.provider_endpointed,
                    "interruption_probability": (
                        update.signals.interruption_probability
                    ),
                },
            },
        )
        agent_speaking = self.state in {
            ConversationState.SPEAKING,
            ConversationState.POSSIBLE_INTERRUPTION,
            ConversationState.OVERLAP,
        } or acoustic_episode_active
        signals = replace(
            update.signals,
            partial_transcript="" if update.is_final else update.text,
            final_transcript=update.text if update.is_final else "",
            agent_speaking=agent_speaking,
        )
        decision = self._turn_policy.decide(signals)
        semantic_acoustic_takeover = (
            update.is_final
            and acoustic_episode_active
            and decision.decision is not TurnDecisionType.BACKCHANNEL
            and not update.signals.filler_ending
        )
        if semantic_acoustic_takeover and decision.decision is not TurnDecisionType.INTERRUPTION:
            decision = TurnDecision(
                TurnDecisionType.INTERRUPTION,
                0.92,
                ("acoustic_overlap_with_semantic_takeover",),
                ("authoritative_pcm_vad", "final_transcript", "agent_playback_overlap"),
            )
        self.state_machine.record(
            EventType.TURN_CANDIDATE,
            correlation_id=correlation_id,
            reason_code=decision.reason_codes[0],
            payload={
                "decision": decision.decision.value,
                "confidence": decision.confidence,
                "reason_codes": list(decision.reason_codes),
                "signals_used": list(decision.signals_used),
            },
        )

        if decision.decision is TurnDecisionType.BACKCHANNEL:
            if episode is not None and acoustic_episode_active:
                episode.classification = "BACKCHANNEL"
                if episode.hard_confirmed_ns is not None:
                    self.state_machine.record(
                        EventType.FALSE_INTERRUPTION_DETECTED,
                        correlation_id=episode.correlation_id,
                        reason_code="backchannel_was_hard_cancelled",
                        payload={
                            "response_id": episode.response_id,
                            "text": update.text,
                            "false_interruption_count_increment": 1,
                        },
                    )
                if episode.resume_ack_ns is not None:
                    self._record_backchannel_recovery(
                        episode, text=update.text
                    )
                elif self.response_active:
                    if self._soft_resume_task is not None:
                        self._soft_resume_task.cancel()
                    self._request_playback_resume(
                        episode, reason_code="semantic_backchannel_classified"
                    )
            self.state_machine.record(
                EventType.BACKCHANNEL_DETECTED,
                correlation_id=correlation_id,
                reason_code="non_interrupting_acknowledgement",
                payload={"text": update.text},
            )
        elif (
            decision.decision is TurnDecisionType.UNCERTAIN
            and agent_speaking
            and signals.background_speech_probability >= 0.75
        ):
            self.state_machine.transition(
                ConversationState.OVERLAP,
                reason_code="probable_background_speech_overlap",
                correlation_id=correlation_id,
                payload={
                    "background_speech_probability": signals.background_speech_probability
                },
            )
            self.state_machine.transition(
                ConversationState.SPEAKING,
                reason_code="non_interrupting_overlap_resolved",
                correlation_id=correlation_id,
            )
        elif decision.decision is TurnDecisionType.INTERRUPTION:
            if episode is not None and acoustic_episode_active:
                self._record_takeover_evidence(
                    episode,
                    evidence_type=(
                        "SEMANTIC_FINAL_TAKEOVER"
                        if update.is_final
                        else "SEMANTIC_PARTIAL_TAKEOVER"
                    ),
                    semantic=True,
                )
            await self.interrupt(
                correlation_id=(
                    episode.correlation_id
                    if episode is not None and acoustic_episode_active
                    else correlation_id
                ),
                speech_onset_ns=(
                    episode.speech_onset_ns
                    if episode is not None and acoustic_episode_active
                    else None
                ),
                reason_code=(
                    "semantic_takeover_after_soft_yield"
                    if semantic_acoustic_takeover
                    else "intentional_interruption"
                ),
            )
            followup = ""
            if update.is_final:
                followup = _barge_in_followup(update.text)
                if semantic_acoustic_takeover and not followup:
                    followup = update.text.strip()
            if followup:
                await self.wait_for_response()
                self.state_machine.record(
                    EventType.TURN_CONFIRMED,
                    correlation_id=correlation_id,
                    reason_code="barge_in_followup_complete",
                    payload={
                        "text": followup,
                        "original_text": update.text,
                        "confidence": decision.confidence,
                        "stt_provider": transcript_provider.to_dict(),
                    },
                )
                await self._begin_response(
                    followup,
                    correlation_id=correlation_id,
                    user_transcript_id=update.provenance.transcript_id,
                )
        elif decision.decision is TurnDecisionType.COMPLETE and update.is_final:
            self.state_machine.record(
                EventType.TURN_CONFIRMED,
                correlation_id=correlation_id,
                reason_code="hybrid_policy_complete",
                payload={
                    "text": update.text,
                    "confidence": decision.confidence,
                    "stt_provider": transcript_provider.to_dict(),
                },
            )
            await self._begin_response(
                update.text,
                correlation_id=correlation_id,
                user_transcript_id=update.provenance.transcript_id,
            )
        return decision

    async def _begin_response(
        self,
        transcript: str,
        *,
        correlation_id: str,
        user_transcript_id: str,
    ) -> None:
        if self.response_active:
            return
        if self.state is not ConversationState.LISTENING:
            return
        appointment_delta = self._appointment_state_tracker.apply_user_turn(
            transcript,
            source_turn=user_transcript_id,
        )
        if self._appointment_transaction_coordinator is None:
            appointment_context = self._appointment_state_tracker.reasoning_context(
                appointment_delta
            )
            set_context = getattr(
                self._reasoner, "set_authoritative_transaction_context", None
            )
            if callable(set_context):
                set_context(
                    appointment_context,
                    state=appointment_delta.appointments,
                )
        if appointment_delta.changed:
            self.state_machine.record(
                EventType.APPOINTMENT_STATE_UPDATED,
                correlation_id=correlation_id,
                reason_code="user_turn_delta_applied_to_authoritative_state",
                payload={
                    **appointment_delta.to_dict(),
                    "assistant_history_used_as_state_source": False,
                    "unchanged_slots_preserved": True,
                },
            )
        self.state_machine.transition(
            ConversationState.THINKING,
            reason_code="user_turn_complete",
            correlation_id=correlation_id,
        )
        self._cancel_event = asyncio.Event()
        self._audio_stopped = asyncio.Event()
        self._cancel_requested_ns = None
        self._cancel_correlation_id = None
        self._last_playback_receipt = None
        if self._soft_resume_task is not None:
            self._soft_resume_task.cancel()
        self._soft_yield = None
        self._response_id = str(uuid4())
        self._response_token = self.state_machine.issue_operation(kind="reasoning_and_speech")
        history_user_count_before = (
            self.conversation_history_roles.count("user")
            if hasattr(self._reasoner, "history")
            else None
        )
        self._response_task = asyncio.create_task(
            self._run_response(
                transcript=transcript,
                response_id=self._response_id,
                token=self._response_token,
                correlation_id=correlation_id,
                user_transcript_id=user_transcript_id,
                history_user_count_before=history_user_count_before,
                appointment_delta=appointment_delta,
            ),
            name=f"humanflow-response-{self._response_id}",
        )

    async def _run_response(
        self,
        *,
        transcript: str,
        response_id: str,
        token: OperationToken,
        correlation_id: str,
        user_transcript_id: str,
        history_user_count_before: int | None,
        appointment_delta: AppointmentStateDelta,
    ) -> None:
        first_model_output = True
        first_audio = True
        audio_chunk_sequence = 0
        semantic_chunk_sequence = 0
        speech_segment_count = 0
        output_characters = 0
        previous_spoken_text = ""
        playback_tasks: list[asyncio.Task[PlaybackReceipt]] = []
        generation_started_ns = self._clock_ns()
        reasoning_provider = provider_info(self._reasoner, role="reasoning")
        speech_provider = provider_info(self._synthesizer, role="tts")
        try:
            coordinator = self._appointment_transaction_coordinator
            tool_outcome = None
            if coordinator is not None:
                tool_delta, tool_outcome = await coordinator.execute(
                    transcript=transcript,
                    delta=appointment_delta,
                    tracker=self._appointment_state_tracker,
                    correlation_id=correlation_id,
                    source_turn=user_transcript_id,
                    parent_token=token,
                )
                if tool_outcome is not None:
                    self._record_tool_cost(
                        outcome=tool_outcome,
                        token=token,
                        turn_id=user_transcript_id,
                    )
                appointment_context = self._appointment_state_tracker.reasoning_context(
                    tool_delta
                )
                appointment_context = coordinator.enrich_reasoning_context(
                    appointment_context, tool_outcome
                )
                set_context = getattr(
                    self._reasoner, "set_authoritative_transaction_context", None
                )
                if callable(set_context):
                    set_context(
                        appointment_context,
                        state=tool_delta.appointments,
                    )
                if tool_outcome is not None and tool_delta.updated_slots:
                    self.state_machine.record(
                        EventType.APPOINTMENT_STATE_UPDATED,
                        correlation_id=correlation_id,
                        reason_code="database_tool_result_applied_to_action_state",
                        payload={
                            **tool_delta.to_dict(),
                            "tool_name": tool_outcome.tool_name,
                            "tool_success": tool_outcome.success,
                        },
                    )
                if self._cancel_event.is_set() or not self.state_machine.accept_result(
                    token, correlation_id=correlation_id
                ):
                    return
            self.state_machine.record(
                EventType.AGENT_GENERATION_STARTED,
                correlation_id=correlation_id,
                reason_code="reasoner_stream_started",
                payload={
                    "response_id": response_id,
                    "provider": reasoning_provider.to_dict(),
                },
            )
            async for text in self._reasoner.stream_response(transcript, token):
                self._raise_completed_playback_failure(playback_tasks)
                if self._cancel_event.is_set() or not self.state_machine.accept_result(
                    token, correlation_id=correlation_id
                ):
                    break
                semantic_ready_ns = self._clock_ns()
                if first_model_output:
                    first_model_output = False
                    self.state_machine.record(
                        EventType.FIRST_MODEL_OUTPUT,
                        correlation_id=correlation_id,
                        reason_code="first_semantic_chunk_ready",
                        payload={
                            "response_id": response_id,
                            "provider": reasoning_provider.to_dict(),
                            "provider_latency_ms": max(
                                0.0,
                                (semantic_ready_ns - generation_started_ns) / 1_000_000.0,
                            ),
                        },
                    )
                output_characters += len(text)
                self.state_machine.record(
                    EventType.SEMANTIC_CHUNK_READY,
                    correlation_id=correlation_id,
                    reason_code="stable_reasoner_boundary",
                    payload={
                        "response_id": response_id,
                        "semantic_chunk_sequence": semantic_chunk_sequence,
                        "characters": len(text),
                        "provider": reasoning_provider.to_dict(),
                        "event_kind": ConversationEventKind.ASSISTANT_TEXT.value,
                        "origin": "ASSISTANT_REASONING",
                    },
                )
                semantic_chunk_sequence += 1

                for segment in self._prosody_planner.plan(text):
                    self._raise_completed_playback_failure(playback_tasks)
                    if self._cancel_event.is_set():
                        break
                    speech_segment_count += 1
                    tts_session_id = f"{response_id}:tts"
                    segment_id = f"{response_id}:segment:{speech_segment_count}"
                    spoken_text = self._speech_normalizer.normalize(segment.text)
                    request = SpeechSynthesisRequest(
                        text=spoken_text,
                        display_text=segment.text,
                        response_id=response_id,
                        sequence_start=audio_chunk_sequence,
                        language_code="de",
                        speaking_rate=segment.speaking_rate,
                        stability=segment.stability,
                        similarity_boost=segment.similarity_boost,
                        style=segment.style,
                        use_speaker_boost=segment.use_speaker_boost,
                        pause_after_ms=segment.pause_after_ms,
                        intent=segment.intent.value,
                        previous_text=previous_spoken_text,
                        tts_session_id=tts_session_id,
                        segment_id=segment_id,
                    )
                    tts_request_started_ns = self._clock_ns()
                    self.state_machine.record(
                        EventType.TTS_REQUEST_STARTED,
                        correlation_id=correlation_id,
                        reason_code="prosody_segment_submitted",
                        payload={
                            "response_id": response_id,
                            "tts_session_id": tts_session_id,
                            "segment_id": segment_id,
                            "physical_request_sequence": speech_segment_count - 1,
                            "provider": speech_provider.to_dict(),
                            "intent": segment.intent.value,
                            "characters": len(segment.text),
                            "spoken_characters": len(spoken_text),
                            "speech_normalized": spoken_text != segment.text,
                            "display_text_boundary": segment.text,
                            "spoken_text_boundary": spoken_text,
                            "intentional_linguistic_pause_ms": segment.pause_after_ms,
                            "boundary_pause_class": (
                                "INTENTIONAL_LINGUISTIC"
                                if segment.pause_after_ms > 0
                                else "CONTINUOUS_GAPLESS"
                            ),
                            "speaking_rate": segment.speaking_rate,
                            "pause_after_ms": segment.pause_after_ms,
                            "event_kind": (
                                ConversationEventKind.ASSISTANT_TTS_AUDIO.value
                            ),
                            "origin": "ASSISTANT_TTS",
                        },
                    )
                    stream = self._synthesizer.stream_speech(
                        request, cancel_event=self._cancel_event
                    )
                    segment_audio_emitted = False
                    segment_audio_seconds = Decimal("0")
                    fallback_recorded = False
                    actual_speech_provider = speech_provider.to_dict()
                    try:
                        async for chunk in stream:
                            self._raise_completed_playback_failure(playback_tasks)
                            if self._cancel_event.is_set() or not self.state_machine.accept_result(
                                token, correlation_id=correlation_id
                            ):
                                self._record_stale_chunk(
                                    chunk=chunk,
                                    response_id=response_id,
                                    correlation_id=correlation_id,
                                )
                                break
                            await self._wait_for_playback_capacity(
                                playback_tasks, maximum_pending=2
                            )
                            if not self._accept_audio_chunk(
                                chunk=chunk,
                                response_id=response_id,
                                correlation_id=correlation_id,
                            ):
                                continue
                            segment_audio_emitted = True
                            actual_speech_provider = (
                                dict(chunk.provider)
                                if chunk.provider
                                else speech_provider.to_dict()
                            )
                            if (
                                not fallback_recorded
                                and actual_speech_provider.get("provider")
                                != speech_provider.provider
                            ):
                                fallback_recorded = True
                                self.state_machine.record(
                                    EventType.TTS_PROVIDER_FALLBACK,
                                    correlation_id=correlation_id,
                                    reason_code="primary_failed_before_audio",
                                    payload={
                                        "response_id": response_id,
                                        "primary": speech_provider.to_dict(),
                                        "active": actual_speech_provider,
                                    },
                                )
                            audio_chunk_sequence = max(
                                audio_chunk_sequence, chunk.frame.sequence + 1
                            )
                            segment_audio_seconds += Decimal(
                                chunk.frame.samples_per_channel
                            ) / Decimal(chunk.frame.sample_rate_hz)
                            generated_ns = self._clock_ns()
                            self.ledger.register_generated(chunk, generated_ns=generated_ns)
                            self.ledger.mark_queued(
                                chunk.chunk_id, queued_ns=self._clock_ns()
                            )
                            self._self_speech_guard.register_pending(
                                chunk_id=chunk.chunk_id,
                                response_id=response_id,
                                text=chunk.semantic_text,
                            )
                            if self._cancel_event.is_set() or not self.state_machine.accept_result(
                                token, correlation_id=correlation_id
                            ):
                                self.ledger.cancel_unplayed(
                                    response_id=response_id,
                                    cancelled_ns=self._clock_ns(),
                                )
                                break
                            if first_audio:
                                first_audio = False
                                self.state_machine.record(
                                    EventType.FIRST_AUDIO_CHUNK,
                                    correlation_id=correlation_id,
                                    reason_code="first_streaming_tts_pcm_ready",
                                    payload={
                                        "response_id": response_id,
                                        "chunk_id": chunk.chunk_id,
                                        "provider": actual_speech_provider,
                                        "tts_request_to_first_audio_ms": max(
                                            0.0,
                                            (
                                                generated_ns
                                                - tts_request_started_ns
                                            )
                                            / 1_000_000.0,
                                        ),
                                    },
                                )
                                if self.state is ConversationState.THINKING:
                                    self.state_machine.transition(
                                        ConversationState.SPEAKING,
                                        reason_code="playback_ready",
                                        correlation_id=correlation_id,
                                    )

                            self.state_machine.record(
                                EventType.AUDIO_CHUNK_SCHEDULED,
                                correlation_id=correlation_id,
                                reason_code="single_owner_playback_schedule",
                                payload=self._audio_chunk_payload(chunk),
                            )
                            playback_tasks.append(
                                asyncio.create_task(
                                    self._play_scheduled_chunk(
                                        chunk=chunk,
                                        generated_ns=generated_ns,
                                        semantic_ready_ns=semantic_ready_ns,
                                        tts_request_started_ns=tts_request_started_ns,
                                        response_id=response_id,
                                        correlation_id=correlation_id,
                                    ),
                                    name=f"humanflow-playback-{chunk.chunk_id}",
                                )
                            )
                    finally:
                        close_stream = getattr(stream, "aclose", None)
                        if close_stream is not None:
                            await close_stream()
                    self._record_tts_cost(
                        token=token,
                        turn_id=user_transcript_id,
                        response_id=response_id,
                        segment_id=segment_id,
                        submitted_characters=len(spoken_text),
                        generated_audio_seconds=segment_audio_seconds,
                        actual_provider=actual_speech_provider,
                        fallback=fallback_recorded,
                        cancelled=self._cancel_event.is_set(),
                    )
                    if not segment_audio_emitted and not self._cancel_event.is_set():
                        raise RuntimeError("tts_provider_returned_no_chunks")
                    if self._cancel_event.is_set():
                        break
                    previous_spoken_text = (
                        f"{previous_spoken_text} {spoken_text}".strip()[-1_000:]
                    )

                if self._cancel_event.is_set():
                    break

            if playback_tasks:
                await asyncio.gather(*playback_tasks)

            if not self._cancel_event.is_set():
                usage = getattr(self._reasoner, "last_usage", None)
                usage_payload = usage.to_dict() if hasattr(usage, "to_dict") else None
                self._record_llm_cost(
                    usage_payload=usage_payload,
                    token=token,
                    turn_id=user_transcript_id,
                    response_id=response_id,
                    provider=reasoning_provider.to_dict(),
                    generation_started_ns=generation_started_ns,
                )
                history_roles = self.conversation_history_roles
                user_history_writes = None
                if history_user_count_before is not None:
                    user_history_writes = (
                        history_roles.count("user") - history_user_count_before
                    )
                    if user_history_writes != 1:
                        raise RuntimeError(
                            "user_history_write_count_per_turn_invariant_violated"
                        )
                self.state_machine.record(
                    EventType.AGENT_GENERATION_COMPLETED,
                    correlation_id=correlation_id,
                    reason_code="reasoner_stream_completed",
                    payload={
                        "response_id": response_id,
                        "provider": reasoning_provider.to_dict(),
                        "duration_ms": max(
                            0.0,
                            (self._clock_ns() - generation_started_ns) / 1_000_000.0,
                        ),
                        "output_characters": output_characters,
                        "semantic_chunks": semantic_chunk_sequence,
                        "speech_segments": speech_segment_count,
                        "audio_chunks": audio_chunk_sequence,
                        "usage": usage_payload,
                        "conversation_history_roles": list(history_roles),
                        "user_transcript_id": user_transcript_id,
                        "user_history_writes_for_turn": user_history_writes,
                        "number_of_user_history_writes_per_turn_maximum": 1,
                    },
                )
                self.state_machine.record(
                    EventType.AGENT_AUDIO_COMPLETED,
                    correlation_id=correlation_id,
                    reason_code="response_stream_played",
                    payload={
                        "response_id": response_id,
                        "tts_session_id": f"{response_id}:tts",
                        "logical_tts_sessions": 1,
                        "physical_tts_requests": speech_segment_count,
                        "playback_scheduling": "response_level_lookahead_queue",
                        "playback_lookahead_limit": 2,
                        "delivered_text": self.ledger.delivered_text(response_id=response_id),
                    },
                )
                if self.state is ConversationState.SPEAKING:
                    self.state_machine.transition(
                        ConversationState.LISTENING,
                        reason_code="agent_response_complete",
                        correlation_id=correlation_id,
                    )
        except Exception as error:
            failure_ns = self._clock_ns()
            self._cancel_event.set()
            self.state_machine.invalidate_operations(
                reason_code="response_pipeline_failed",
                correlation_id=correlation_id,
            )
            self.state_machine.record(
                EventType.RECOVERY_STARTED,
                correlation_id=correlation_id,
                reason_code="response_pipeline_failure",
                payload={
                    "response_id": response_id,
                    "exception_type": type(error).__name__,
                    "reasoning_provider": reasoning_provider.to_dict(),
                    "tts_provider": speech_provider.to_dict(),
                },
            )
            if self.state in {ConversationState.THINKING, ConversationState.SPEAKING}:
                self.state_machine.transition(
                    ConversationState.RECOVERING,
                    reason_code="response_pipeline_failure",
                    correlation_id=correlation_id,
                )
            playback_unconfirmed = self.ledger.mark_playback_unconfirmed(
                response_id=response_id
            )
            self.ledger.cancel_unplayed(response_id=response_id, cancelled_ns=failure_ns)
            if playback_unconfirmed:
                self.state_machine.record(
                    EventType.AGENT_AUDIO_STOP_UNCONFIRMED,
                    correlation_id=correlation_id,
                    reason_code="audio_sink_ack_missing",
                    payload={"response_id": response_id},
                )
                if self.state is ConversationState.RECOVERING:
                    self.state_machine.transition(
                        ConversationState.HANDOFF,
                        reason_code="audible_state_unknown",
                        correlation_id=correlation_id,
                    )
                recovery_reason = "safe_handoff_after_unconfirmed_audio"
            else:
                if self.state is ConversationState.RECOVERING:
                    self.state_machine.transition(
                        ConversationState.LISTENING,
                        reason_code="response_failure_before_playback",
                        correlation_id=correlation_id,
                    )
                recovery_reason = "continued_listening_after_provider_failure"
            self.state_machine.record(
                EventType.RECOVERY_COMPLETED,
                correlation_id=correlation_id,
                reason_code=recovery_reason,
                payload={"response_id": response_id},
            )
        finally:
            if playback_tasks:
                await asyncio.gather(*playback_tasks, return_exceptions=True)
            if self._cancel_event.is_set():
                stopped_ns = self._clock_ns()
                self.ledger.cancel_unplayed(response_id=response_id, cancelled_ns=stopped_ns)
                if self.state in {
                    ConversationState.INTERRUPTED,
                    ConversationState.THINKING,
                }:
                    self.state_machine.transition(
                        ConversationState.LISTENING,
                        reason_code="barge_in_output_stopped",
                        correlation_id=correlation_id,
                    )
            self.ledger.assert_invariants()
            if self._cost_observer is not None:
                self._observe_cost(
                    "record_played_audio",
                    operation_id=f"{token.operation_id}:played-audio",
                    turn_id=user_transcript_id,
                    response_id=response_id,
                    entries=self.ledger.entries,
                )
            self._release_playback_owner(
                response_id=response_id,
                correlation_id=correlation_id,
            )
            self._audio_stopped.set()

    async def _play_scheduled_chunk(
        self,
        *,
        chunk: AudioChunk,
        generated_ns: int,
        semantic_ready_ns: int,
        tts_request_started_ns: int,
        response_id: str,
        correlation_id: str,
    ) -> PlaybackReceipt:
        receipt = await self._audio_output.play(
            chunk,
            cancel_event=self._cancel_event,
            on_started=lambda started_ns: self._playback_started(
                chunk=chunk,
                started_ns=started_ns,
                audio_ready_ns=generated_ns,
                semantic_ready_ns=semantic_ready_ns,
                tts_request_started_ns=tts_request_started_ns,
                response_id=response_id,
                correlation_id=correlation_id,
            ),
        )
        self._last_playback_receipt = receipt
        self.ledger.record_playback(receipt)
        self._self_speech_guard.mark_stopped(
            chunk_id=chunk.chunk_id,
            stopped_ns=receipt.playback_stopped_ns,
        )
        if not receipt.cancelled:
            signal = analyze_pcm16(chunk.frame)
            self._audio_signal_metrics[chunk.chunk_id] = (
                signal.rms_dbfs,
                signal.peak_dbfs,
            )
        playback_payload = {
            **self._audio_chunk_payload(chunk),
            "played_samples": receipt.played_samples,
            "requested_samples": receipt.requested_samples,
            "cancelled": receipt.cancelled,
            "playback_started_ns": receipt.playback_started_ns,
            "playback_stopped_ns": receipt.playback_stopped_ns,
            "source_node_id": receipt.source_node_id,
            "browser_scheduled_start_ms": receipt.browser_scheduled_start_ms,
            "browser_actual_playback_start_ms": (
                receipt.browser_actual_playback_start_ms
            ),
            "browser_actual_playback_end_ms": receipt.browser_actual_playback_end_ms,
            "previous_segment_end_ms": receipt.previous_segment_end_ms,
            "inter_segment_gap_ms": receipt.inter_segment_gap_ms,
            "intentional_linguistic_pause_ms": (
                receipt.intentional_linguistic_pause_ms
            ),
            "scheduler_generated_gap_ms": receipt.scheduler_generated_gap_ms,
            "queue_depth_ms": receipt.queue_depth_ms,
            "underrun_count": receipt.underrun_count,
        }
        self.state_machine.record(
            EventType.AUDIO_CHUNK_PLAYED,
            correlation_id=correlation_id,
            reason_code=(
                "browser_playback_cancelled"
                if receipt.cancelled
                else "browser_playback_completed"
            ),
            payload=playback_payload,
        )
        self.state_machine.record(
            EventType.AUDIO_SEGMENT_METRICS,
            correlation_id=correlation_id,
            reason_code="raw_provider_signal_and_browser_timeline_measured",
            payload={
                **playback_payload,
                "amplitude_processing": "NONE_STABLE_RESPONSE_GAIN",
                "per_chunk_normalization": False,
            },
        )
        if receipt.underrun_count > 0:
            self.state_machine.record(
                EventType.PLAYBACK_UNDERRUN,
                correlation_id=correlation_id,
                reason_code="browser_timeline_unplanned_gap_over_18ms",
                payload=playback_payload,
            )
        if (
            receipt.sink_base_latency_ms is not None
            or receipt.sink_output_latency_ms is not None
            or receipt.player_stop_callback_latency_ms is not None
        ):
            self.state_machine.record(
                EventType.PLAYBACK_SINK_METRICS,
                correlation_id=correlation_id,
                reason_code="browser_reported_audio_context",
                payload={
                    "response_id": response_id,
                    "chunk_id": chunk.chunk_id,
                    "sink_base_latency_ms": receipt.sink_base_latency_ms,
                    "sink_output_latency_ms": receipt.sink_output_latency_ms,
                    "player_stop_callback_latency_ms": (
                        receipt.player_stop_callback_latency_ms
                    ),
                    "source_node_id": receipt.source_node_id,
                    "browser_scheduled_start_ms": receipt.browser_scheduled_start_ms,
                    "browser_actual_playback_start_ms": (
                        receipt.browser_actual_playback_start_ms
                    ),
                    "browser_actual_playback_end_ms": (
                        receipt.browser_actual_playback_end_ms
                    ),
                    "previous_segment_end_ms": receipt.previous_segment_end_ms,
                    "inter_segment_gap_ms": receipt.inter_segment_gap_ms,
                    "intentional_linguistic_pause_ms": (
                        receipt.intentional_linguistic_pause_ms
                    ),
                    "scheduler_generated_gap_ms": (
                        receipt.scheduler_generated_gap_ms
                    ),
                    "queue_depth_ms": receipt.queue_depth_ms,
                    "underrun_count": receipt.underrun_count,
                },
            )
        if receipt.cancelled:
            self.ledger.cancel_unplayed(
                response_id=response_id,
                cancelled_ns=receipt.playback_stopped_ns,
            )
            if response_id not in self._cancel_acknowledged_response_ids:
                self._cancel_acknowledged_response_ids.add(response_id)
                self._record_cancelled(
                    receipt=receipt,
                    response_id=response_id,
                    correlation_id=self._cancel_correlation_id or correlation_id,
                )
        return receipt

    @staticmethod
    def _raise_completed_playback_failure(
        playback_tasks: list[asyncio.Task[PlaybackReceipt]],
    ) -> None:
        for task in playback_tasks:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is not None:
                    raise error

    @staticmethod
    async def _wait_for_playback_capacity(
        playback_tasks: list[asyncio.Task[PlaybackReceipt]],
        *,
        maximum_pending: int,
    ) -> None:
        pending = [task for task in playback_tasks if not task.done()]
        if len(pending) < maximum_pending:
            return
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()

    def _audio_chunk_payload(self, chunk: AudioChunk) -> dict[str, object]:
        signal_values = self._audio_signal_metrics.get(chunk.chunk_id, (None, None))
        return {
            "response_id": chunk.response_id,
            "tts_session_id": chunk.tts_session_id,
            "segment_id": chunk.segment_id,
            "stream_id": chunk.frame.stream_id,
            "chunk_id": chunk.chunk_id,
            "sequence": chunk.frame.sequence,
            "chunk_byte_length": len(chunk.frame.pcm16),
            "codec": "pcm_s16le",
            "sample_rate_hz": chunk.frame.sample_rate_hz,
            "decoded_duration_ms": chunk.frame.duration_ms,
            "rms_dbfs": signal_values[0],
            "peak_dbfs": signal_values[1],
            "provider": dict(chunk.provider),
        }

    def _record_stale_chunk(
        self,
        *,
        chunk: AudioChunk,
        response_id: str,
        correlation_id: str,
    ) -> None:
        self.state_machine.record(
            EventType.STALE_CHUNK_REJECTED,
            correlation_id=correlation_id,
            reason_code="cancelled_or_invalid_operation_epoch",
            payload={
                **self._audio_chunk_payload(chunk),
                "expected_response_id": response_id,
            },
        )

    def _accept_audio_chunk(
        self,
        *,
        chunk: AudioChunk,
        response_id: str,
        correlation_id: str,
    ) -> bool:
        if chunk.response_id != response_id or response_id in self._cancelled_response_ids:
            self._record_stale_chunk(
                chunk=chunk,
                response_id=response_id,
                correlation_id=correlation_id,
            )
            return False
        sequence_key = (chunk.frame.stream_id, chunk.frame.sequence)
        if (
            chunk.chunk_id in self._seen_audio_chunk_ids
            or sequence_key in self._seen_audio_sequences
        ):
            self.state_machine.record(
                EventType.DUPLICATE_CHUNK_REJECTED,
                correlation_id=correlation_id,
                reason_code="chunk_id_or_stream_sequence_already_seen",
                payload=self._audio_chunk_payload(chunk),
            )
            return False
        if self._playback_owner_response_id is None:
            self._playback_owner_response_id = response_id
            self.state_machine.record(
                EventType.PLAYBACK_OWNER_CREATED,
                correlation_id=correlation_id,
                reason_code="response_acquired_exclusive_playback",
                payload={
                    "response_id": response_id,
                    "active_playback_owners": 1,
                },
            )
        elif self._playback_owner_response_id != response_id:
            self._record_stale_chunk(
                chunk=chunk,
                response_id=response_id,
                correlation_id=correlation_id,
            )
            return False

        provider_key = (
            str(chunk.provider.get("provider", "unknown")),
            str(chunk.provider.get("model", "unknown")),
            chunk.playback_mode.value,
        )
        if self._active_tts_provider is None:
            self._active_tts_provider = provider_key
            self.state_machine.record(
                EventType.TTS_PROVIDER_ACTIVATED,
                correlation_id=correlation_id,
                reason_code="response_provider_pinned",
                payload={
                    "response_id": response_id,
                    "provider": dict(chunk.provider),
                    "playback_mode": chunk.playback_mode.value,
                    "active_tts_playback_providers": 1,
                },
            )
        elif self._active_tts_provider != provider_key:
            raise RuntimeError("active_tts_playback_providers_exceeded")

        self._seen_audio_chunk_ids.add(chunk.chunk_id)
        self._seen_audio_sequences.add(sequence_key)
        self.state_machine.record(
            EventType.AUDIO_CHUNK_RECEIVED,
            correlation_id=correlation_id,
            reason_code="unique_chunk_accepted",
            payload=self._audio_chunk_payload(chunk),
        )
        return True

    def _release_playback_owner(
        self,
        *,
        response_id: str,
        correlation_id: str,
    ) -> None:
        if self._playback_owner_response_id != response_id:
            return
        if self._active_tts_provider is not None:
            provider, model, playback_mode = self._active_tts_provider
            self.state_machine.record(
                EventType.TTS_PROVIDER_DEACTIVATED,
                correlation_id=correlation_id,
                reason_code="response_playback_released",
                payload={
                    "response_id": response_id,
                    "provider": provider,
                    "model": model,
                    "playback_mode": playback_mode,
                    "active_tts_playback_providers": 0,
                },
            )
        self.state_machine.record(
            EventType.PLAYBACK_OWNER_DESTROYED,
            correlation_id=correlation_id,
            reason_code="response_playback_released",
            payload={
                "response_id": response_id,
                "active_playback_owners": 0,
            },
        )
        self._active_tts_provider = None
        self._playback_owner_response_id = None
        self._audio_signal_metrics.clear()

    def _playback_started(
        self,
        *,
        chunk: AudioChunk,
        started_ns: int,
        audio_ready_ns: int,
        semantic_ready_ns: int,
        tts_request_started_ns: int,
        response_id: str,
        correlation_id: str,
    ) -> None:
        self._self_speech_guard.mark_started(
            chunk_id=chunk.chunk_id,
            started_ns=started_ns,
        )
        self.ledger.mark_playback_started(chunk.chunk_id, started_ns=started_ns)
        self.state_machine.record(
            EventType.AGENT_AUDIO_STARTED,
            correlation_id=correlation_id,
            reason_code="audio_sink_started",
            payload={
                "response_id": response_id,
                "chunk_id": chunk.chunk_id,
                "playback_started_ns": started_ns,
                "provider": dict(chunk.provider),
                "event_kind": ConversationEventKind.ASSISTANT_PLAYBACK.value,
                "origin": "ASSISTANT_TTS",
                "semantic_to_playback_start_ms": max(
                    0.0, (started_ns - semantic_ready_ns) / 1_000_000.0
                ),
                "tts_request_to_playback_start_ms": max(
                    0.0, (started_ns - tts_request_started_ns) / 1_000_000.0
                ),
                "audio_ready_to_playback_start_ms": max(
                    0.0, (started_ns - audio_ready_ns) / 1_000_000.0
                ),
            },
        )

    def _record_transcript_provenance(
        self,
        *,
        update: TranscriptUpdate,
        correlation_id: str,
        accepted_by_user_ingestion: bool,
        accepted_as_user_turn: bool,
        rejection_reason: str | None,
        response_id: str | None = None,
        final_admission: FinalAdmissionAssessment | None = None,
    ) -> None:
        provenance = update.provenance
        self.state_machine.record(
            EventType.TRANSCRIPT_PROVENANCE_RECORDED,
            correlation_id=correlation_id,
            reason_code=(
                "user_transcript_accepted"
                if accepted_by_user_ingestion
                else rejection_reason or "user_transcript_rejected"
            ),
            payload={
                **provenance.to_dict(),
                "response_id": response_id or provenance.response_id,
                "is_partial": not update.is_final,
                "is_final": update.is_final,
                "raw_text": update.raw_text,
                "normalized_text": update.normalized_text,
                "accepted_by_user_ingestion": accepted_by_user_ingestion,
                "accepted_as_user_turn": accepted_as_user_turn,
                "rejection_reason": rejection_reason,
                "final_admission": (
                    final_admission.to_dict()
                    if final_admission is not None
                    else None
                ),
                "provider": (
                    update.provider.to_dict()
                    if update.provider is not None
                    else provider_info(self._transcriber, role="stt").to_dict()
                ),
                "partial_transcript_to_user_history": (
                    "FORBIDDEN" if not update.is_final else None
                ),
            },
        )

    def _record_final_admission(
        self,
        *,
        update: TranscriptUpdate,
        assessment: FinalAdmissionAssessment,
        correlation_id: str,
    ) -> None:
        self.state_machine.record(
            (
                EventType.FINAL_ADMISSION_ACCEPTED
                if assessment.accepted
                else EventType.FINAL_ADMISSION_REJECTED
            ),
            correlation_id=correlation_id,
            reason_code=assessment.reason_code,
            payload={
                "transcript_id": update.provenance.transcript_id,
                "stream_id": update.provenance.stream_id,
                "audio_capture_id": update.provenance.audio_capture_id,
                "audio_frame_sequence": update.provenance.audio_frame_sequence,
                **assessment.to_dict(),
            },
        )

    def _record_pre_admission_rejection(
        self,
        *,
        update: TranscriptUpdate,
        reason_code: str,
        correlation_id: str,
    ) -> None:
        try:
            reason = FinalAdmissionReason(reason_code)
        except ValueError:
            reason = FinalAdmissionReason.UNKNOWN
        assessment = FinalAdmissionAssessment.rejected_without_episode(
            reason_code=reason,
            final_received_monotonic=update.provenance.timestamp_ns,
            final_frame_sequence=update.provenance.audio_frame_sequence,
            assistant_playback_active=self.state
            in {
                ConversationState.SPEAKING,
                ConversationState.POSSIBLE_INTERRUPTION,
                ConversationState.OVERLAP,
            },
        )
        self._record_final_admission(
            update=update,
            assessment=assessment,
            correlation_id=correlation_id,
        )

    def _record_self_speech_event(
        self,
        event_type: EventType,
        *,
        update: TranscriptUpdate,
        assessment: SelfSpeechAssessment,
        correlation_id: str,
        reason_code: str,
    ) -> None:
        self.state_machine.record(
            event_type,
            correlation_id=correlation_id,
            reason_code=reason_code,
            payload={
                "transcript_id": update.provenance.transcript_id,
                "raw_text": update.raw_text,
                "normalized_text": update.normalized_text,
                "confidence": assessment.confidence,
                "matched_response_id": assessment.matched_response_id,
                "matched_chunk_id": assessment.matched_chunk_id,
                "signals": assessment.signals,
            },
        )

    def _record_cancelled(
        self,
        *,
        receipt: PlaybackReceipt,
        response_id: str,
        correlation_id: str,
    ) -> None:
        requested_ns = self._cancel_requested_ns
        latency_ms = (
            None
            if requested_ns is None
            else max(0.0, (receipt.playback_stopped_ns - requested_ns) / 1_000_000.0)
        )
        episode = self._soft_yield
        speech_onset_ns = (
            episode.speech_onset_ns
            if episode is not None and episode.response_id == response_id
            else requested_ns
        )
        speech_onset_to_audible_stop_ms = (
            None
            if speech_onset_ns is None
            else max(
                0.0,
                (receipt.playback_stopped_ns - speech_onset_ns) / 1_000_000.0,
            )
        )
        self.state_machine.record(
            EventType.AGENT_AUDIO_CANCELLED,
            correlation_id=correlation_id,
            reason_code="audio_sink_confirmed_stop",
            payload={
                "response_id": response_id,
                "chunk_id": receipt.chunk_id,
                "played_samples": receipt.played_samples,
                "requested_samples": receipt.requested_samples,
                "playback_stopped_ns": receipt.playback_stopped_ns,
                "cancel_requested_ns": requested_ns,
                "audible_barge_in_latency_ms": latency_ms,
                "speech_onset_ns": speech_onset_ns,
                "speech_onset_to_hard_cancel_ms": (
                    None
                    if episode is None or episode.hard_confirmed_ns is None
                    else max(
                        0.0,
                        (
                            episode.hard_confirmed_ns
                            - episode.speech_onset_ns
                        )
                        / 1_000_000.0,
                    )
                ),
                "speech_onset_to_audible_stop_ms": (
                    speech_onset_to_audible_stop_ms
                ),
                "player_stop_callback_latency_ms": (
                    receipt.player_stop_callback_latency_ms
                ),
                "sink_output_latency_ms": receipt.sink_output_latency_ms,
                "delivered_text": self.ledger.delivered_text(response_id=response_id),
                "unheard_text": self.ledger.unheard_text(response_id=response_id),
            },
        )
        self.state_machine.record(
            EventType.AUDIBLE_STOP_ACK,
            correlation_id=correlation_id,
            reason_code="browser_confirmed_zero_future_old_response_audio",
            payload={
                "response_id": response_id,
                "chunk_id": receipt.chunk_id,
                "speech_onset_ns": speech_onset_ns,
                "playback_stopped_ns": receipt.playback_stopped_ns,
                "speech_onset_to_audible_stop_ms": (
                    speech_onset_to_audible_stop_ms
                ),
                "future_audible_audio_from_old_response": 0,
                "cancelled_epoch_future_playback": "FORBIDDEN",
                "latency_decomposition_ms": (
                    _barge_in_latency_decomposition(
                        episode,
                        audible_stop_ns=receipt.playback_stopped_ns,
                    )
                    if episode is not None and speech_onset_ns is not None
                    else None
                ),
            },
        )


def _barge_in_followup(text: str) -> str:
    """Return substantive text following an explicit German stop phrase."""

    match = _BARGE_IN_PREFIX.match(text)
    if match is None:
        return ""
    followup = match.group("followup").strip()
    words = re.findall(r"[a-zäöüß0-9]+", followup.casefold())
    return followup if len(words) >= 2 else ""


def _barge_in_latency_decomposition(
    episode: _SoftYieldEpisode,
    *,
    audible_stop_ns: int,
) -> dict[str, float | str | None]:
    def elapsed(start_ns: int | None, end_ns: int | None) -> float | None:
        if start_ns is None or end_ns is None:
            return None
        return max(0.0, (end_ns - start_ns) / 1_000_000.0)

    return {
        "speech_onset_to_possible_interruption": elapsed(
            episode.speech_onset_ns, episode.possible_interruption_ns
        ),
        "possible_interruption_to_duck_request": elapsed(
            episode.possible_interruption_ns, episode.duck_requested_ns
        ),
        "duck_request_to_duck_ack": elapsed(
            episode.duck_requested_ns, episode.duck_ack_ns
        ),
        "duck_ack_to_takeover_evidence": elapsed(
            episode.duck_ack_ns, episode.takeover_evidence_ns
        ),
        "takeover_evidence_type": episode.takeover_evidence_type,
        "takeover_evidence_to_confirmation": elapsed(
            episode.takeover_evidence_ns, episode.hard_confirmed_ns
        ),
        "confirmation_to_queue_invalidation": elapsed(
            episode.hard_confirmed_ns, episode.queue_invalidated_ns
        ),
        "queue_invalidation_to_cancel_signal": elapsed(
            episode.queue_invalidated_ns, episode.cancel_signal_ns
        ),
        "cancel_signal_to_audible_stop": elapsed(
            episode.cancel_signal_ns, audible_stop_ns
        ),
        "speech_onset_to_audible_stop": elapsed(
            episode.speech_onset_ns, audible_stop_ns
        ),
    }
