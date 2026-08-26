from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic_ns

import pytest

from humanflow.audio.ledger import LedgerState
from humanflow.audio.models import AudioChunk, AudioFrame, PlaybackReceipt
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.runtime.providers import (
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnDecisionType, TurnSignals


class CountingTranscriber:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.closed = False

    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        await asyncio.sleep(0.001)
        self.frames.append(frame.sequence)
        return ()

    async def close(self) -> None:
        self.closed = True


class FailingTranscriber(CountingTranscriber):
    async def ingest(self, frame: AudioFrame) -> tuple[TranscriptUpdate, ...]:
        del frame
        raise RuntimeError("sanitized_provider_failure")


class WordReasoner:
    def __init__(self, *, initial_delay_ms: float = 0.0) -> None:
        self.initial_delay_ms = initial_delay_ms
        self.transcripts: list[str] = []

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        self.transcripts.append(transcript)
        if self.initial_delay_ms:
            await asyncio.sleep(self.initial_delay_ms / 1000.0)
        for word in ("Ihr", "Termin", "ist", "Donnerstag", "um", "15", "Uhr"):
            yield word


class LookaheadAudioOutput:
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []
        self.two_scheduled = asyncio.Event()
        self.release = asyncio.Event()

    async def play(
        self, chunk: AudioChunk, *, cancel_event: asyncio.Event, on_started
    ) -> PlaybackReceipt:  # type: ignore[no-untyped-def]
        del cancel_event
        started_ns = monotonic_ns()
        on_started(started_ns)
        self.chunks.append(chunk)
        if len(self.chunks) >= 2:
            self.two_scheduled.set()
        await self.release.wait()
        return PlaybackReceipt(
            chunk_id=chunk.chunk_id,
            requested_samples=chunk.frame.samples_per_channel,
            played_samples=chunk.frame.samples_per_channel,
            playback_started_ns=started_ns,
            playback_stopped_ns=max(started_ns, monotonic_ns()),
            cancelled=False,
            browser_scheduled_start_ms=float(chunk.frame.sequence * 100),
            browser_actual_playback_start_ms=float(chunk.frame.sequence * 100),
            browser_actual_playback_end_ms=float(chunk.frame.sequence * 100 + 80),
            previous_segment_end_ms=(
                None if chunk.frame.sequence == 0 else float(chunk.frame.sequence * 100 - 20)
            ),
            inter_segment_gap_ms=None if chunk.frame.sequence == 0 else 20.0,
            queue_depth_ms=100.0,
            underrun_count=0,
        )


def _complete(text: str = "Ich brauche einen Termin") -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(final=True),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=1_500,
            semantic_complete=True,
            acoustic_completion=0.9,
        ),
    )


async def _wait_for_state(session: RealtimeVoiceSession, state: ConversationState) -> None:
    for _ in range(200):
        if session.state is state:
            return
        await asyncio.sleep(0.002)
    raise AssertionError(f"session did not reach {state}; current={session.state}")


def test_full_duplex_backchannel_and_audible_barge_in() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        transcriber = CountingTranscriber()
        session = RealtimeVoiceSession(
            conversation_id="call-full-duplex",
            sink=sink,
            transcriber=transcriber,
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=120),
            audio_output=TimedPcmOutput(quantum_ms=5),
        )
        await session.start()
        decision = await session.submit_transcript(_complete())
        assert decision.decision is TurnDecisionType.COMPLETE
        await _wait_for_state(session, ConversationState.SPEAKING)

        for sequence in range(12):
            session.receive_audio(
                AudioFrame(
                    stream_id="caller",
                    sequence=sequence,
                    pcm16=b"\x00\x00" * 320,
                )
            )
        await session.wait_for_input()
        assert transcriber.frames == list(range(12))
        assert session.response_active

        backchannel = await session.submit_transcript(
            TranscriptUpdate(
                text="mhm",
                is_final=True,
                provenance=TranscriptProvenance.user_fixture(final=True),
                signals=TurnSignals(
                    speech_active=True,
                    silence_duration_ms=0,
                    utterance_duration_ms=250,
                ),
            )
        )
        assert backchannel.decision is TurnDecisionType.BACKCHANNEL
        assert session.response_active

        interruption = await session.submit_transcript(
            TranscriptUpdate(
                text="Moment, stopp",
                is_final=True,
                provenance=TranscriptProvenance.user_fixture(final=True),
                signals=TurnSignals(
                    speech_active=True,
                    silence_duration_ms=0,
                    utterance_duration_ms=300,
                    interruption_probability=0.98,
                ),
            )
        )
        assert interruption.decision is TurnDecisionType.INTERRUPTION
        await session.wait_for_response()
        assert session.state is ConversationState.LISTENING
        assert session.last_playback_receipt is not None
        assert session.last_playback_receipt.cancelled
        assert session.ledger.unheard_text()
        assert any(event.event_type is EventType.BACKCHANNEL_DETECTED for event in sink.events)
        cancellation = next(
            event for event in sink.events if event.event_type is EventType.AGENT_AUDIO_CANCELLED
        )
        latency = cancellation.payload["audible_barge_in_latency_ms"]
        assert isinstance(latency, float)
        assert latency >= 0.0
        assert cancellation.payload["playback_stopped_ns"] >= cancellation.payload[
            "cancel_requested_ns"
        ]
        await session.close()
        assert transcriber.closed
        assert session.state is ConversationState.IDLE

    asyncio.run(scenario())


def test_stt_failure_drains_pcm_queue_and_session_closes_without_hanging() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        transcriber = FailingTranscriber()
        session = RealtimeVoiceSession(
            conversation_id="stt-failure-recovery",
            sink=sink,
            transcriber=transcriber,
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        for sequence in range(4):
            session.receive_audio(
                AudioFrame(
                    stream_id="caller",
                    sequence=sequence,
                    pcm16=b"\x00\x00" * 320,
                )
            )

        with pytest.raises(RuntimeError, match="streaming STT input pipeline"):
            await session.wait_for_input()
        with pytest.raises(RuntimeError, match="streaming STT input pipeline"):
            session.receive_audio(
                AudioFrame(
                    stream_id="caller",
                    sequence=5,
                    pcm16=b"\x00\x00" * 320,
                )
            )
        assert any(
            event.event_type is EventType.STT_PROVIDER_FAILED for event in sink.events
        )
        await asyncio.wait_for(session.close(), timeout=0.2)
        assert transcriber.closed

    asyncio.run(scenario())


def test_interruption_invalidates_thinking_work_and_is_idempotent() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="call-stale-work",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(initial_delay_ms=20),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=20),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.submit_transcript(_complete())
        assert session.state is ConversationState.THINKING
        first_latency = await session.interrupt()
        await session.wait_for_response()
        second_latency = await session.interrupt()

        assert first_latency is None
        assert second_latency is None
        assert session.state is ConversationState.LISTENING
        assert not session.ledger.entries
        assert sum(
            event.event_type is EventType.OPERATION_INVALIDATED for event in sink.events
        ) == 1
        await session.close()

    asyncio.run(scenario())


def test_final_barge_in_followup_stops_audio_then_starts_contextual_turn() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = WordReasoner()
        session = RealtimeVoiceSession(
            conversation_id="call-barge-followup",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=30),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.submit_transcript(_complete("Erkläre mir ein ERP-System"))
        await _wait_for_state(session, ConversationState.SPEAKING)

        decision = await session.submit_transcript(
            TranscriptUpdate(
                text="Moment, stopp. Was ist 25 mal 17?",
                is_final=True,
                provenance=TranscriptProvenance.user_fixture(final=True),
                signals=TurnSignals(
                    speech_active=False,
                    silence_duration_ms=0,
                    utterance_duration_ms=1_100,
                    semantic_complete=True,
                    acoustic_completion=1.0,
                    interruption_probability=1.0,
                    provider_endpointed=True,
                ),
            )
        )
        await session.wait_for_response()

        assert decision.decision is TurnDecisionType.INTERRUPTION
        assert reasoner.transcripts == [
            "Erkläre mir ein ERP-System",
            "Was ist 25 mal 17?",
        ]
        followup = next(
            event
            for event in sink.events
            if event.event_type is EventType.TURN_CONFIRMED
            and event.reason_code == "barge_in_followup_complete"
        )
        assert followup.payload["text"] == "Was ist 25 mal 17?"
        cancelled_index = next(
            index
            for index, event in enumerate(sink.events)
            if event.event_type is EventType.AGENT_AUDIO_CANCELLED
        )
        followup_index = sink.events.index(followup)
        assert cancelled_index < followup_index
        assert session.state is ConversationState.LISTENING
        await session.close()

    asyncio.run(scenario())


def test_missing_audio_stop_ack_enters_safe_handoff_without_fake_metric() -> None:
    class BrokenOutput:
        async def play(self, chunk, *, cancel_event, on_started):  # type: ignore[no-untyped-def]
            del cancel_event
            on_started(monotonic_ns())
            raise TimeoutError("sink_ack_missing")

    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="call-audio-unknown",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=20),
            audio_output=BrokenOutput(),
        )
        await session.start()
        await session.submit_transcript(_complete())
        await session.wait_for_response()

        assert session.state is ConversationState.HANDOFF
        assert session.ledger.entries[0].state is LedgerState.PLAYBACK_UNCONFIRMED
        assert any(
            event.event_type is EventType.AGENT_AUDIO_STOP_UNCONFIRMED
            for event in sink.events
        )
        assert not any(
            event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
        )
        await session.close(reason_code="handoff_complete")

    asyncio.run(scenario())


def test_background_speech_uses_explicit_non_interrupting_overlap_lifecycle() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="call-background-overlap",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=40),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.submit_transcript(_complete())
        await _wait_for_state(session, ConversationState.SPEAKING)
        decision = await session.submit_transcript(
            TranscriptUpdate(
                text="Nachrichten im Hintergrund",
                is_final=True,
                provenance=TranscriptProvenance.user_fixture(final=True),
                signals=TurnSignals(
                    speech_active=True,
                    silence_duration_ms=0,
                    utterance_duration_ms=500,
                    background_speech_probability=0.95,
                    interruption_probability=0.05,
                ),
            )
        )
        await session.wait_for_response()

        transitions = [
            event.payload["to_state"]
            for event in sink.events
            if event.event_type is EventType.STATE_TRANSITIONED
        ]
        overlap_index = transitions.index("OVERLAP")
        assert decision.decision is TurnDecisionType.UNCERTAIN
        assert transitions[overlap_index : overlap_index + 2] == ["OVERLAP", "SPEAKING"]
        assert not any(
            event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
        )
        assert session.state is ConversationState.LISTENING
        await session.close()

    asyncio.run(scenario())


def test_playback_owner_provider_and_chunk_lifecycle_are_unique_and_ordered() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="call-playback-invariants",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=2),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        await session.submit_transcript(_complete())
        await session.wait_for_response()

        event_types = [event.event_type for event in sink.events]
        assert event_types.count(EventType.PLAYBACK_OWNER_CREATED) == 1
        assert event_types.count(EventType.PLAYBACK_OWNER_DESTROYED) == 1
        assert event_types.count(EventType.TTS_PROVIDER_ACTIVATED) == 1
        assert event_types.count(EventType.TTS_PROVIDER_DEACTIVATED) == 1
        received = [
            event for event in sink.events
            if event.event_type is EventType.AUDIO_CHUNK_RECEIVED
        ]
        scheduled = [
            event for event in sink.events
            if event.event_type is EventType.AUDIO_CHUNK_SCHEDULED
        ]
        played = [
            event for event in sink.events
            if event.event_type is EventType.AUDIO_CHUNK_PLAYED
        ]
        assert len(received) == len(scheduled) == len(played) > 0
        assert len({event.payload["chunk_id"] for event in received}) == len(received)
        assert len(
            {
                (event.payload["stream_id"], event.payload["sequence"])
                for event in scheduled
            }
        ) == len(scheduled)
        assert not any(
            event.event_type in {
                EventType.DUPLICATE_CHUNK_REJECTED,
                EventType.STALE_CHUNK_REJECTED,
            }
            for event in sink.events
        )
        await session.close()

    asyncio.run(scenario())


def test_response_schedules_lookahead_audio_before_prior_segment_finishes() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = LookaheadAudioOutput()
        session = RealtimeVoiceSession(
            conversation_id="response-level-lookahead",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=80),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_complete())
        await asyncio.wait_for(output.two_scheduled.wait(), timeout=0.2)

        assert session.response_active
        await asyncio.sleep(0)
        assert len(output.chunks) == 2
        assert len({chunk.tts_session_id for chunk in output.chunks}) == 1
        assert len({chunk.segment_id for chunk in output.chunks}) == len(output.chunks)
        output.release.set()
        await session.wait_for_response()

        completed = next(
            event
            for event in sink.events
            if event.event_type is EventType.AGENT_AUDIO_COMPLETED
        )
        assert completed.payload["logical_tts_sessions"] == 1
        assert completed.payload["physical_tts_requests"] == 7
        assert completed.payload["playback_scheduling"] == "response_level_lookahead_queue"
        assert completed.payload["playback_lookahead_limit"] == 2
        assert sum(
            event.event_type is EventType.AUDIO_SEGMENT_METRICS
            for event in sink.events
        ) == 7
        await session.close()

    asyncio.run(scenario())


class _StallThenSpeakSynthesizer:
    """First response stalls before first audio; later responses speak normally.

    Reproduces the real failure where a synthesizer records TTS_REQUEST_STARTED but
    never yields a first audio chunk in the pre-playback window. Barge-in still
    exits cleanly because the stall respects ``cancel_event``.
    """

    def __init__(self) -> None:
        self._delegate = ToneSpeechSynthesizer(chunk_duration_ms=10)
        self._stalled_response_id: str | None = None

    @property
    def provider_info(self):
        return self._delegate.provider_info

    async def stream_speech(self, request, *, cancel_event):
        # Stall every attempt of the first response (so the pre-playback speech
        # retry also stalls and the response deterministically recovers); later
        # responses speak normally.
        if self._stalled_response_id is None:
            self._stalled_response_id = request.response_id
        if request.response_id == self._stalled_response_id:
            if not cancel_event.is_set():
                await cancel_event.wait()  # stall: never yields first audio
            return
            yield  # pragma: no cover
        async for chunk in self._delegate.stream_speech(
            request, cancel_event=cancel_event
        ):
            yield chunk


def test_stalled_tts_first_audio_recovers_and_serves_next_turn() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="tts-first-audio-stall",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=_StallThenSpeakSynthesizer(),
            audio_output=TimedPcmOutput(quantum_ms=1),
            tts_first_audio_timeout_ms=120.0,
        )
        await session.start()
        await session.submit_transcript(_complete())
        assert session.state is ConversationState.THINKING

        # The stalled pre-playback window must not strand the session in THINKING:
        # it recovers to LISTENING within a bounded time (fails before the fix).
        await asyncio.wait_for(session.wait_for_response(), timeout=2.0)
        assert session.state is ConversationState.LISTENING
        assert not session.response_active
        names = [event.event_type.name for event in sink.events]
        assert "TTS_REQUEST_STARTED" in names
        assert "FIRST_AUDIO_CHUNK" not in names
        assert "RECOVERY_COMPLETED" in names

        # A subsequent legitimate user turn is accepted and produces audio, proving
        # the response owner was released rather than orphaned.
        await session.submit_transcript(_complete("Und am Donnerstag?"))
        await _wait_for_state(session, ConversationState.SPEAKING)
        await session.wait_for_response()
        assert session.state is ConversationState.LISTENING
        await session.close()

    asyncio.run(scenario())


def test_new_turn_during_pre_playback_stall_is_served_not_lost() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = WordReasoner()
        session = RealtimeVoiceSession(
            conversation_id="pre-playback-takeover",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=reasoner,
            synthesizer=_StallThenSpeakSynthesizer(),
            audio_output=TimedPcmOutput(quantum_ms=1),
            # Default-scale timeout on purpose: the takeover, not the timeout, must
            # serve Turn B. The test never waits for this to elapse.
            tts_first_audio_timeout_ms=6_000.0,
        )
        await session.start()
        await session.submit_transcript(_complete("Termin fuer einen Orthopaeden"))
        # Turn A is stalled pre-playback: TTS requested, no first audio, THINKING.
        for _ in range(200):
            if any(e.event_type is EventType.TTS_REQUEST_STARTED for e in sink.events):
                break
            await asyncio.sleep(0.002)
        assert session.state is ConversationState.THINKING
        assert session.response_active

        # A legitimate new user turn arrives DURING the stall, before any timeout.
        decision = await session.submit_transcript(_complete("Und am Donnerstag frueher?"))
        assert decision.decision is TurnDecisionType.COMPLETE

        # Turn B must take over and be served now (not lost, not waiting 6 s).
        await _wait_for_state(session, ConversationState.SPEAKING)
        await session.wait_for_response()
        assert session.state is ConversationState.LISTENING

        # Turn B actually reached the reasoner (was not silently discarded).
        assert "Und am Donnerstag frueher?" in reasoner.transcripts

        # Turn A was cancelled and never became audible: exactly one first-audio
        # event exists (Turn B's), and the takeover cancelled the stalled epoch.
        first_audio_events = [
            e for e in sink.events if e.event_type is EventType.FIRST_AUDIO_CHUNK
        ]
        assert len(first_audio_events) == 1
        assert any(
            e.event_type is EventType.AUDIO_CANCEL_SIGNAL for e in sink.events
        )
        # No stranded generation; the owner was handed to Turn B cleanly.
        assert not session.response_active
        await session.close()

    asyncio.run(scenario())


class _FailFirstThenSpeakSynthesizer:
    """First TTS attempt raises before any audio; the retry speaks normally."""

    def __init__(self) -> None:
        self._delegate = ToneSpeechSynthesizer(chunk_duration_ms=10)
        self.calls = 0
        self._failed = False

    @property
    def provider_info(self):
        return self._delegate.provider_info

    async def stream_speech(self, request, *, cancel_event):
        self.calls += 1
        if not self._failed:
            self._failed = True
            raise RuntimeError("elevenlabs_stream_disconnected")
            yield  # pragma: no cover
        async for chunk in self._delegate.stream_speech(
            request, cancel_event=cancel_event
        ):
            yield chunk


def test_pre_playback_tts_failure_retries_same_response_without_rerunning_reasoner() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = WordReasoner()
        synth = _FailFirstThenSpeakSynthesizer()
        session = RealtimeVoiceSession(
            conversation_id="pre-playback-tts-retry",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=reasoner,
            synthesizer=synth,
            audio_output=TimedPcmOutput(quantum_ms=1),
            tts_first_audio_timeout_ms=6_000.0,
        )
        await session.start()
        await session.submit_transcript(_complete("Welche Termine hast du frei?"))
        await _wait_for_state(session, ConversationState.SPEAKING)
        await session.wait_for_response()
        assert session.state is ConversationState.LISTENING

        names = [event.event_type.name for event in sink.events]
        # The transient failure was retried exactly once and the turn was answered.
        assert names.count("TTS_SPEECH_RETRY") == 1
        assert "FIRST_AUDIO_CHUNK" in names
        assert "AGENT_AUDIO_COMPLETED" in names
        # No provider-failure recovery: the retry succeeded.
        assert "RECOVERY_COMPLETED" not in names
        # The reasoner (and therefore the appointment tool) ran exactly once: a speech
        # retry must never re-run the LLM or duplicate tool side effects.
        assert len(reasoner.transcripts) == 1
        assert synth.calls >= 2
        await session.close()

    asyncio.run(scenario())


def test_internal_non_tts_exception_before_first_audio_is_not_retried() -> None:
    # Adversarial: an internal (non speech-stream) exception raised during chunk
    # processing before first audible audio must propagate to the normal response
    # recovery and must NEVER be reclassified as a transient TTS failure.
    from humanflow.runtime.self_speech import SelfSpeechGuard

    class _RaisingSelfSpeechGuard(SelfSpeechGuard):
        def register_pending(self, *, chunk_id: str, response_id: str, text: str) -> None:
            raise RuntimeError("internal_non_tts_bug")

    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="internal-error-no-retry",
            sink=sink,
            transcriber=CountingTranscriber(),
            reasoner=WordReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=10),
            audio_output=TimedPcmOutput(quantum_ms=1),
            self_speech_guard=_RaisingSelfSpeechGuard(),
        )
        await session.start()
        await session.submit_transcript(_complete())
        await asyncio.wait_for(session.wait_for_response(), timeout=2.0)
        assert session.state is ConversationState.LISTENING

        names = [event.event_type.name for event in sink.events]
        # The internal error went to normal recovery, not the speech retry path.
        assert names.count("TTS_SPEECH_RETRY") == 0
        assert "RECOVERY_COMPLETED" in names
        assert "FIRST_AUDIO_CHUNK" not in names
        await session.close()

    asyncio.run(scenario())
