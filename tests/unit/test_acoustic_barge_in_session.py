from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from time import monotonic_ns

from humanflow.audio.models import AudioFrame
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.runtime.providers import (
    NullTranscriber,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnDecisionType, TurnSignals


class RecordingReasoner:
    def __init__(self) -> None:
        self.transcripts: list[str] = []

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        self.transcripts.append(transcript)
        yield "Ich erkläre das kurz und bleibe dabei vollständig unterbrechbar."


class SoftControlOutput(TimedPcmOutput):
    def __init__(self) -> None:
        super().__init__(quantum_ms=5)
        self.ducks: list[tuple[str, int]] = []
        self.resumes: list[tuple[str, int]] = []
        self.invalidations: list[tuple[str, int]] = []

    def soft_duck(self, *, response_id: str, speech_onset_ns: int) -> bool:
        self.ducks.append((response_id, speech_onset_ns))
        return True

    def resume_playback(self, *, response_id: str, speech_onset_ns: int) -> bool:
        self.resumes.append((response_id, speech_onset_ns))
        return True

    def invalidate_response(self, *, response_id: str, speech_onset_ns: int) -> int:
        self.invalidations.append((response_id, speech_onset_ns))
        return len(self.invalidations)


def _turn(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(final=True),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=650,
            utterance_duration_ms=900,
            semantic_complete=True,
            acoustic_completion=1.0,
            provider_endpointed=True,
        ),
    )


def _partial_interruption(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=False,
        provenance=TranscriptProvenance.user_fixture(final=False),
        signals=TurnSignals(
            speech_active=True,
            silence_duration_ms=0,
            utterance_duration_ms=250,
            semantic_complete=False,
            interruption_probability=0.98,
        ),
    )


def _pcm_frame(sequence: int, amplitude: int, base_ns: int) -> AudioFrame:
    return AudioFrame(
        stream_id="authoritative-mic",
        sequence=sequence,
        pcm16=struct.pack("<h", amplitude) * 320,
        sample_rate_hz=16_000,
        captured_ns=base_ns + sequence * 20_000_000,
    )


async def _wait_for_speaking(session: RealtimeVoiceSession) -> None:
    for _ in range(300):
        if session.state is ConversationState.SPEAKING:
            return
        await asyncio.sleep(0.002)
    raise AssertionError(f"session did not start speaking: {session.state}")


def test_mhm_soft_yields_and_resumes_without_permanent_cancellation() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="soft-backchannel",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=RecordingReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_turn("Erkläre mir ein ERP-System."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(10):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        for sequence in range(10, 20):
            session.receive_audio(_pcm_frame(sequence, 0, base_ns))

        decision = await session.submit_transcript(_turn("mhm"))

        assert decision.decision is TurnDecisionType.BACKCHANNEL
        assert len(output.ducks) == 1
        assert len(output.resumes) >= 1
        assert not output.invalidations
        assert not any(
            event.event_type is EventType.INTERRUPTION_CONFIRMED
            for event in sink.events
        )
        assert any(
            event.event_type is EventType.BACKCHANNEL_DETECTED
            for event in sink.events
        )
        await session.close()

    asyncio.run(scenario())


def test_sustained_pcm_takeover_hard_cancels_before_any_stt_event() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="sustained-takeover",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=RecordingReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=2_000),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_turn("Sprich bitte etwas länger."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(32):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        await asyncio.sleep(0)
        await session.wait_for_response()

        event_types = [event.event_type for event in sink.events]
        assert output.ducks
        assert len(output.invalidations) == 1
        assert event_types.index(EventType.PLAYBACK_DUCK_REQUESTED) < event_types.index(
            EventType.INTERRUPTION_CONFIRMED
        )
        soft_index = event_types.index(EventType.PLAYBACK_DUCK_REQUESTED)
        hard_index = event_types.index(EventType.INTERRUPTION_CONFIRMED)
        evidence_index = event_types.index(EventType.TAKEOVER_EVIDENCE)
        assert soft_index < evidence_index < hard_index
        assert not any(
            event_type in {EventType.PARTIAL_TRANSCRIPT, EventType.FINAL_TRANSCRIPT}
            for event_type in event_types[soft_index:hard_index]
        )
        assert EventType.AUDIO_CANCEL_SIGNAL in event_types
        assert EventType.AUDIBLE_STOP_ACK in event_types
        stop = next(
            event for event in sink.events if event.event_type is EventType.AUDIBLE_STOP_ACK
        )
        decomposition = stop.payload["latency_decomposition_ms"]
        assert decomposition["takeover_evidence_type"] == "ACOUSTIC_SUSTAINED_TAKEOVER"
        assert decomposition["speech_onset_to_audible_stop"] >= 0
        assert decomposition["cancel_signal_to_audible_stop"] >= 0
        assert session.state is ConversationState.LISTENING
        await session.close()

    asyncio.run(scenario())


def test_moment_stopp_partial_confirms_after_soft_yield_before_sustained_timer() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="semantic-moment-stopp",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=RecordingReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_turn("Erkläre mir ein ERP-System."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(5):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        assert output.ducks and not output.invalidations

        decision = await session.submit_transcript(
            _partial_interruption("Moment, stopp")
        )
        await session.wait_for_response()

        assert decision.decision is TurnDecisionType.INTERRUPTION
        assert len(output.invalidations) == 1
        evidence = next(
            event for event in sink.events if event.event_type is EventType.TAKEOVER_EVIDENCE
        )
        assert evidence.payload["evidence_type"] == "SEMANTIC_PARTIAL_TAKEOVER"
        assert evidence.payload["semantic_evidence"] is True
        assert any(
            event.event_type is EventType.AUDIBLE_STOP_ACK for event in sink.events
        )
        await session.close()

    asyncio.run(scenario())


def test_short_hesitation_recovers_after_hold_without_destroying_response() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="hesitation-recovery",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=RecordingReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=output,
            soft_yield_recovery_delay_ms=40,
        )
        await session.start()
        await session.submit_transcript(_turn("Erkläre mir das kurz."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(10):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        for sequence in range(10, 20):
            session.receive_audio(_pcm_frame(sequence, 0, base_ns))
        await asyncio.sleep(0.06)

        hesitation = TranscriptUpdate(
            text="äh",
            is_final=True,
            provenance=TranscriptProvenance.user_fixture(final=True),
            signals=TurnSignals(
                speech_active=False,
                silence_duration_ms=650,
                utterance_duration_ms=200,
                filler_ending=True,
                semantic_complete=False,
                acoustic_completion=0.4,
                provider_endpointed=True,
            ),
        )
        decision = await session.submit_transcript(hesitation)

        assert decision.decision is TurnDecisionType.CONTINUE_LISTENING
        assert output.resumes
        assert not output.invalidations
        assert session.response_active
        await session.close()

    asyncio.run(scenario())


def test_cough_like_transient_uses_mild_duck_and_never_hard_cancels() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="cough-transient",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=RecordingReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=output,
            soft_yield_recovery_delay_ms=40,
        )
        await session.start()
        await session.submit_transcript(_turn("Erkläre mir das kurz."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(8):
            session.receive_audio(_pcm_frame(sequence, 6_000, base_ns))
        for sequence in range(8, 18):
            session.receive_audio(_pcm_frame(sequence, 0, base_ns))
        await asyncio.sleep(0.06)

        event_types = [event.event_type for event in sink.events]
        duck = next(
            event
            for event in sink.events
            if event.event_type is EventType.PLAYBACK_DUCK_REQUESTED
        )
        assert duck.payload["duck_stage"] == "MILD_SOFT_YIELD"
        assert duck.payload["target_gain"] == 0.55
        assert output.ducks and output.resumes
        assert not output.invalidations
        assert EventType.INTERRUPTION_CONFIRMED not in event_types
        assert EventType.FALSE_INTERRUPTION_DETECTED not in event_types
        assert session.response_active
        await session.close()

    asyncio.run(scenario())


def test_correction_after_acoustic_cancel_becomes_exactly_one_new_user_turn() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="acoustic-correction",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=800),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_turn("Nenne mir einen Termin."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(32):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        await asyncio.sleep(0)
        await session.wait_for_response()

        decision = await session.submit_transcript(_turn("Nein, mach lieber Montag."))
        await session.wait_for_response()

        assert decision.decision is TurnDecisionType.INTERRUPTION
        assert reasoner.transcripts == [
            "Nenne mir einen Termin.",
            "Nein, mach lieber Montag.",
        ]
        confirmed = [
            event
            for event in sink.events
            if event.event_type is EventType.TURN_CONFIRMED
            and event.reason_code == "barge_in_followup_complete"
        ]
        assert len(confirmed) == 1
        await session.close()

    asyncio.run(scenario())


def test_short_semantic_takeover_cancels_after_final_but_not_on_onset() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        output = SoftControlOutput()
        session = RealtimeVoiceSession(
            conversation_id="short-semantic-takeover",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=output,
        )
        await session.start()
        await session.submit_transcript(_turn("Erkläre mir das ausführlich."))
        await _wait_for_speaking(session)
        base_ns = monotonic_ns() - 1_000_000_000
        for sequence in range(10):
            session.receive_audio(_pcm_frame(sequence, 5_000, base_ns))
        for sequence in range(10, 20):
            session.receive_audio(_pcm_frame(sequence, 0, base_ns))
        assert not output.invalidations

        decision = await session.submit_transcript(_turn("Warum?"))
        await session.wait_for_response()

        assert decision.decision is TurnDecisionType.INTERRUPTION
        assert len(output.invalidations) == 1
        assert reasoner.transcripts == ["Erkläre mir das ausführlich.", "Warum?"]
        await session.close()

    asyncio.run(scenario())
