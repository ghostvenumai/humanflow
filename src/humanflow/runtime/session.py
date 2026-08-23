"""Full-duplex voice session with explicit turn, playback, and cancellation truth."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import monotonic_ns
from typing import Callable
from uuid import uuid4

from humanflow.audio.ledger import PlayedAudioLedger
from humanflow.audio.models import AudioFrame, PlaybackReceipt
from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import TelemetrySink
from humanflow.turns.models import TurnDecision, TurnDecisionType
from humanflow.turns.policies import HybridTurnPolicy

from .providers import (
    AudioOutput,
    StreamingReasoner,
    StreamingSpeechSynthesizer,
    StreamingTranscriber,
    TranscriptUpdate,
    provider_info,
)


class RealtimeVoiceSession:
    """Owns live-call concurrency; providers never own conversational state."""

    def __init__(
        self,
        *,
        conversation_id: str,
        sink: TelemetrySink,
        transcriber: StreamingTranscriber,
        reasoner: StreamingReasoner,
        synthesizer: StreamingSpeechSynthesizer,
        audio_output: AudioOutput,
        turn_policy: HybridTurnPolicy | None = None,
        input_queue_size: int = 256,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if input_queue_size < 1:
            raise ValueError("input_queue_size must be positive")
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
        self._user_audio_active = False
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
        try:
            self._input_queue.put_nowait(frame)
        except asyncio.QueueFull as error:
            raise BufferError("realtime input queue is full") from error

    async def submit_transcript(self, update: TranscriptUpdate) -> TurnDecision:
        """Accept transcript/signals from browser, telephony, or an STT adapter."""
        if self._closed or self._input_task is None:
            raise RuntimeError("session is not running")
        return await self._handle_transcript(update)

    async def interrupt(self, *, correlation_id: str | None = None) -> float | None:
        """Stop audible output and return request-to-actual-stop latency in milliseconds."""
        if not self.response_active:
            return None
        correlation_id = correlation_id or str(uuid4())
        request_ns = self._clock_ns()
        self._cancel_requested_ns = request_ns
        self._cancel_correlation_id = correlation_id
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
            reason_code="intentional_interruption",
            payload={"detected_ns": request_ns},
        )
        self._cancel_event.set()
        await self._audio_stopped.wait()
        receipt = self._last_playback_receipt
        if receipt is None or not receipt.cancelled:
            return None
        return max(0.0, (receipt.playback_stopped_ns - request_ns) / 1_000_000.0)

    async def wait_for_response(self) -> None:
        task = self._response_task
        if task is not None:
            await task

    async def wait_for_input(self) -> None:
        """Wait until all PCM frames accepted so far have reached the transcriber."""
        await self._input_queue.join()

    async def close(self, *, reason_code: str = "normal_shutdown") -> None:
        if self._closed:
            return
        if self.response_active:
            await self.interrupt(correlation_id=str(uuid4()))
            await self.wait_for_response()
        self._closed = True
        if self._input_task is not None:
            await self._input_queue.put(None)
            await self._input_task
        await self._transcriber.close()
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

    async def _input_loop(self) -> None:
        while True:
            frame = await self._input_queue.get()
            try:
                if frame is None:
                    return
                updates = await self._transcriber.ingest(frame)
                for update in updates:
                    await self._handle_transcript(update)
            finally:
                self._input_queue.task_done()

    async def _handle_transcript(self, update: TranscriptUpdate) -> TurnDecision:
        correlation_id = str(uuid4())
        transcript_provider = update.provider or provider_info(
            self._transcriber, role="stt"
        )
        speech_active = update.signals.speech_active
        if speech_active and not self._user_audio_active:
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
            payload={"text": update.text, "provider": transcript_provider.to_dict()},
        )
        agent_speaking = self.state in {
            ConversationState.SPEAKING,
            ConversationState.POSSIBLE_INTERRUPTION,
            ConversationState.OVERLAP,
        }
        signals = replace(
            update.signals,
            partial_transcript="" if update.is_final else update.text,
            final_transcript=update.text if update.is_final else "",
            agent_speaking=agent_speaking,
        )
        decision = self._turn_policy.decide(signals)
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
            await self.interrupt(correlation_id=correlation_id)
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
            await self._begin_response(update.text, correlation_id=correlation_id)
        return decision

    async def _begin_response(self, transcript: str, *, correlation_id: str) -> None:
        if self.response_active:
            return
        if self.state is not ConversationState.LISTENING:
            return
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
        self._response_id = str(uuid4())
        self._response_token = self.state_machine.issue_operation(kind="reasoning_and_speech")
        self._response_task = asyncio.create_task(
            self._run_response(
                transcript=transcript,
                response_id=self._response_id,
                token=self._response_token,
                correlation_id=correlation_id,
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
    ) -> None:
        first_model_output = True
        first_audio = True
        chunk_sequence = 0
        output_characters = 0
        generation_started_ns = self._clock_ns()
        reasoning_provider = provider_info(self._reasoner, role="reasoning")
        speech_provider = provider_info(self._synthesizer, role="tts")
        self.state_machine.record(
            EventType.AGENT_GENERATION_STARTED,
            correlation_id=correlation_id,
            reason_code="reasoner_stream_started",
            payload={
                "response_id": response_id,
                "provider": reasoning_provider.to_dict(),
            },
        )
        try:
            async for text in self._reasoner.stream_response(transcript, token):
                if self._cancel_event.is_set() or not self.state_machine.accept_result(
                    token, correlation_id=correlation_id
                ):
                    break
                if first_model_output:
                    first_model_output = False
                    first_output_ns = self._clock_ns()
                    self.state_machine.record(
                        EventType.FIRST_MODEL_OUTPUT,
                        correlation_id=correlation_id,
                        reason_code="first_stream_fragment",
                        payload={
                            "response_id": response_id,
                            "provider": reasoning_provider.to_dict(),
                            "provider_latency_ms": max(
                                0.0,
                                (first_output_ns - generation_started_ns) / 1_000_000.0,
                            ),
                        },
                    )
                output_characters += len(text)
                chunk = await self._synthesizer.synthesize(
                    text, response_id=response_id, sequence=chunk_sequence
                )
                chunk_sequence += 1
                generated_ns = self._clock_ns()
                self.ledger.register_generated(chunk, generated_ns=generated_ns)
                self.ledger.mark_queued(chunk.chunk_id, queued_ns=self._clock_ns())
                if self._cancel_event.is_set() or not self.state_machine.accept_result(
                    token, correlation_id=correlation_id
                ):
                    self.ledger.cancel_unplayed(
                        response_id=response_id, cancelled_ns=self._clock_ns()
                    )
                    break
                if first_audio:
                    first_audio = False
                    self.state_machine.record(
                        EventType.FIRST_AUDIO_CHUNK,
                        correlation_id=correlation_id,
                        reason_code="first_tts_chunk_ready",
                        payload={
                            "response_id": response_id,
                            "chunk_id": chunk.chunk_id,
                            "provider": speech_provider.to_dict(),
                        },
                    )
                    if self.state is ConversationState.THINKING:
                        self.state_machine.transition(
                            ConversationState.SPEAKING,
                            reason_code="playback_ready",
                            correlation_id=correlation_id,
                        )

                receipt = await self._audio_output.play(
                    chunk,
                    cancel_event=self._cancel_event,
                    on_started=lambda started_ns, chunk_id=chunk.chunk_id: self._playback_started(
                        chunk_id=chunk_id,
                        started_ns=started_ns,
                        response_id=response_id,
                        correlation_id=correlation_id,
                    ),
                )
                self._last_playback_receipt = receipt
                self.ledger.record_playback(receipt)
                if receipt.cancelled:
                    self.ledger.cancel_unplayed(
                        response_id=response_id, cancelled_ns=receipt.playback_stopped_ns
                    )
                    self._record_cancelled(
                        receipt=receipt,
                        response_id=response_id,
                        correlation_id=self._cancel_correlation_id or correlation_id,
                    )
                    break

            if not self._cancel_event.is_set():
                usage = getattr(self._reasoner, "last_usage", None)
                usage_payload = usage.to_dict() if hasattr(usage, "to_dict") else None
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
                        "speech_chunks": chunk_sequence,
                        "usage": usage_payload,
                    },
                )
                self.state_machine.record(
                    EventType.AGENT_AUDIO_COMPLETED,
                    correlation_id=correlation_id,
                    reason_code="response_stream_played",
                    payload={
                        "response_id": response_id,
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
                    "provider": reasoning_provider.to_dict(),
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
            if self._cancel_event.is_set():
                stopped_ns = self._clock_ns()
                self.ledger.cancel_unplayed(response_id=response_id, cancelled_ns=stopped_ns)
                if self.state is ConversationState.INTERRUPTED:
                    self.state_machine.transition(
                        ConversationState.LISTENING,
                        reason_code="barge_in_output_stopped",
                        correlation_id=correlation_id,
                    )
            self.ledger.assert_invariants()
            self._audio_stopped.set()

    def _playback_started(
        self,
        *,
        chunk_id: str,
        started_ns: int,
        response_id: str,
        correlation_id: str,
    ) -> None:
        self.ledger.mark_playback_started(chunk_id, started_ns=started_ns)
        self.state_machine.record(
            EventType.AGENT_AUDIO_STARTED,
            correlation_id=correlation_id,
            reason_code="audio_sink_started",
            payload={
                "response_id": response_id,
                "chunk_id": chunk_id,
                "playback_started_ns": started_ns,
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
                "delivered_text": self.ledger.delivered_text(response_id=response_id),
                "unheard_text": self.ledger.unheard_text(response_id=response_id),
            },
        )
