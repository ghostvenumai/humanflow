from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic_ns

from humanflow.audio.ledger import LedgerState
from humanflow.audio.models import AudioFrame
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.runtime.providers import (
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
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


def _complete(text: str = "Ich brauche einen Termin") -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
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
