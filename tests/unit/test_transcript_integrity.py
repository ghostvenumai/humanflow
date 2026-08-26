from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from time import monotonic_ns

import pytest

from humanflow.audio.models import AudioFrame
from humanflow.domain.conversation import OperationToken
from humanflow.runtime.providers import (
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.self_speech import SelfSpeechGuard
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
    TranscriptProvenance,
    TranscriptRejected,
)
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnSignals


ASSISTANT_SENTENCE = "Ich bin HumanFlow, ein KI-Assistent und helfe dir gern."
HUMAN_FAILURE_CASES = (
    (
        "Ich bin HumanFlow, ein KI-Assistent. Gerne helfe ich dir bei der Terminplanung.",
        "ich bin human flow ein ki assistent gerne helfe ich dir bei der terminplanung",
    ),
    (
        "Wofür brauchst du einen Termin, beispielsweise für einen Arzt, einen "
        "Geschäftstermin oder etwas anderes, und hast du schon eine Vorstellung, "
        "wann es dir passen würde?",
        "wofür brauchst du einen termin beispielsweise für einen arzt einen "
        "geschäftstermin oder etwas anderes und hast du schon eine vorstellung "
        "wann es dir passen würde",
    ),
)


class NullInputTranscriber:
    async def ingest(self, frame: object) -> tuple[()]:
        del frame
        return ()

    async def close(self) -> None:
        return None


class RecordingReasoner:
    def __init__(self, response: str = ASSISTANT_SENTENCE) -> None:
        self.transcripts: list[str] = []
        self.response = response

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        self.transcripts.append(transcript)
        yield self.response


def _signals(*, speech_active: bool = False) -> TurnSignals:
    return TurnSignals(
        speech_active=speech_active,
        silence_duration_ms=350,
        utterance_duration_ms=900,
        semantic_complete=True,
        acoustic_completion=1.0,
        provider_endpointed=True,
    )


def _browser_update(text: str, *, transcript_id: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        signals=_signals(),
        provenance=TranscriptProvenance(
            transcript_id=transcript_id,
            event_kind=ConversationEventKind.USER_TRANSCRIPT_FINAL,
            source="browser_stt",
            origin=TranscriptOrigin.BROWSER_SPEECH_RECOGNITION,
            stream_id="browser-recognition:test",
            browser_recognition_session_id="recognition-test",
            audio_capture_id="capture-test",
            timestamp_ns=monotonic_ns(),
            recognition_input_binding="UNVERIFIED_INDEPENDENT_BROWSER_CAPTURE",
        ),
    )


def _assistant_origin_update(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        signals=_signals(),
        provenance=TranscriptProvenance(
            transcript_id="assistant-origin-injection",
            event_kind=ConversationEventKind.ASSISTANT_TEXT,
            source="assistant_reasoning",
            origin=TranscriptOrigin.ASSISTANT_REASONING,
            stream_id="assistant-response-stream",
            response_id="response-forbidden",
            timestamp_ns=monotonic_ns(),
        ),
    )


def _streaming_update(
    text: str,
    *,
    transcript_id: str,
    final: bool = True,
    speech_active: bool = False,
    stt_session_id: str = "scribe-test-session",
    input_binding: str = "EXACT_GETUSERMEDIA_PCM16",
    timestamp_ns: int | None = None,
    audio_frame_sequence: int | None = None,
) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=final,
        signals=_signals(speech_active=speech_active),
        provenance=TranscriptProvenance(
            transcript_id=transcript_id,
            event_kind=(
                ConversationEventKind.USER_TRANSCRIPT_FINAL
                if final
                else ConversationEventKind.USER_TRANSCRIPT_PARTIAL
            ),
            source="streaming_stt",
            origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
            stream_id="browser-pcm-track",
            stt_session_id=stt_session_id,
            audio_capture_id="capture-test",
            timestamp_ns=timestamp_ns or monotonic_ns(),
            recognition_input_binding=input_binding,
            audio_frame_sequence=audio_frame_sequence,
        ),
    )


def _feed_pcm_episode(
    session: RealtimeVoiceSession,
    *,
    voiced_frames: int = 10,
    silence_frames: int = 10,
    amplitude: int = 5_000,
    start_sequence: int = 0,
) -> tuple[int, int]:
    total_frames = voiced_frames + silence_frames
    base_ns = monotonic_ns() - total_frames * 20_000_000
    for offset in range(voiced_frames):
        sequence = start_sequence + offset
        session.receive_audio(
            AudioFrame(
                stream_id="browser-pcm-track",
                sequence=sequence,
                pcm16=struct.pack("<h", amplitude) * 320,
                sample_rate_hz=16_000,
                captured_ns=base_ns + offset * 20_000_000,
            )
        )
    for offset in range(voiced_frames, total_frames):
        sequence = start_sequence + offset
        session.receive_audio(
            AudioFrame(
                stream_id="browser-pcm-track",
                sequence=sequence,
                pcm16=struct.pack("<h", 0) * 320,
                sample_rate_hz=16_000,
                captured_ns=base_ns + offset * 20_000_000,
            )
        )
    return monotonic_ns(), start_sequence + total_frames - 1


def _feed_fragmented_pcm_episode(
    session: RealtimeVoiceSession,
    *,
    voiced_frames: int = 8,
    gap_frames: int = 3,
    silence_frames: int = 10,
    amplitude: int = 5_000,
) -> tuple[int, int]:
    split = voiced_frames // 2
    pattern = (
        [amplitude] * split
        + [0] * gap_frames
        + [amplitude] * (voiced_frames - split)
        + [0] * silence_frames
    )
    base_ns = monotonic_ns() - len(pattern) * 20_000_000
    for sequence, sample in enumerate(pattern):
        session.receive_audio(
            AudioFrame(
                stream_id="browser-pcm-track",
                sequence=sequence,
                pcm16=struct.pack("<h", sample) * 320,
                sample_rate_hz=16_000,
                captured_ns=base_ns + sequence * 20_000_000,
            )
        )
    return monotonic_ns(), len(pattern) - 1


async def _wait_speaking(session: RealtimeVoiceSession) -> None:
    for _ in range(300):
        if session.state.value == "SPEAKING":
            return
        await asyncio.sleep(0.002)
    raise AssertionError("session did not begin playback")


def test_assistant_origin_event_is_hard_rejected_before_reasoner_history() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="assistant-origin-forbidden",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=10),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()

        with pytest.raises(
            TranscriptRejected,
            match="assistant_origin_event_forbidden_from_user_history",
        ):
            await session.accept_user_transcript(
                _assistant_origin_update(ASSISTANT_SENTENCE)
            )

        assert reasoner.transcripts == []
        assert not session.response_active
        rejection = next(
            event for event in sink.events
            if event.event_type is EventType.TRANSCRIPT_REJECTED
        )
        assert rejection.payload["assistant_origin_event_to_user_history"] == "FORBIDDEN"
        provenance = next(
            event for event in sink.events
            if event.event_type is EventType.TRANSCRIPT_PROVENANCE_RECORDED
        )
        assert provenance.payload["transcript_id"] == "assistant-origin-injection"
        assert provenance.payload["origin"] == "ASSISTANT_REASONING"
        assert provenance.payload["raw_text"] == ASSISTANT_SENTENCE
        assert provenance.payload["normalized_text"]
        assert provenance.payload["accepted_as_user_turn"] is False
        assert (
            provenance.payload["rejection_reason"]
            == "assistant_origin_event_forbidden_from_user_history"
        )
        await session.close()

    asyncio.run(scenario())


def test_streaming_partial_is_ephemeral_and_never_calls_reasoner() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="partial-history-forbidden",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()

        await session.accept_user_transcript(
            _streaming_update(
                "Was ist fünfundzwanzig",
                transcript_id="scribe-partial-only",
                final=False,
                speech_active=True,
            )
        )

        assert reasoner.transcripts == []
        event = next(
            event
            for event in sink.events
            if event.event_type is EventType.TRANSCRIPT_PROVENANCE_RECORDED
        )
        assert event.payload["accepted_by_user_ingestion"] is True
        assert event.payload["accepted_as_user_turn"] is False
        assert event.payload["partial_transcript_to_user_history"] == "FORBIDDEN"
        await session.close()

    asyncio.run(scenario())


def test_streaming_transcript_without_authoritative_pcm_binding_is_rejected() -> None:
    async def scenario() -> None:
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="non-authoritative-streaming-source",
            sink=InMemoryTelemetrySink(),
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()

        with pytest.raises(
            TranscriptRejected,
            match="SESSION_MISMATCH",
        ):
            await session.accept_user_transcript(
                _streaming_update(
                    "Falsch gebundener Text.",
                    transcript_id="non-authoritative-source-final",
                    input_binding="UNVERIFIED_INDEPENDENT_CAPTURE",
                )
            )

        assert reasoner.transcripts == []
        await session.close()

    asyncio.run(scenario())


def test_duplicate_streaming_final_writes_user_history_at_most_once() -> None:
    async def scenario() -> None:
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="duplicate-streaming-final",
            sink=InMemoryTelemetrySink(),
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        final_ns, final_sequence = _feed_pcm_episode(session)
        update = _streaming_update(
            "Was ist fünfundzwanzig mal siebzehn?",
            transcript_id="scribe-committed-once",
            timestamp_ns=final_ns,
            audio_frame_sequence=final_sequence,
        )

        await session.accept_user_transcript(update)
        await session.wait_for_response()
        with pytest.raises(TranscriptRejected, match="DUPLICATE_FINAL"):
            await session.accept_user_transcript(update)

        assert reasoner.transcripts == ["Was ist fünfundzwanzig mal siebzehn?"]
        await session.close()

    asyncio.run(scenario())


def test_streaming_final_requires_and_records_authoritative_pcm_episode() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="final-admission-valid-pcm",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        final_ns, final_sequence = _feed_pcm_episode(session)
        await session.accept_user_transcript(
            _streaming_update(
                "Was ist fünfundzwanzig mal siebzehn?",
                transcript_id="valid-pcm-final",
                timestamp_ns=final_ns,
                audio_frame_sequence=final_sequence,
            )
        )
        await session.wait_for_response()

        admission = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_ACCEPTED
        )
        assert admission.payload["speech_episode_id"] == "pcm-speech-1"
        assert admission.payload["voiced_duration_ms"] == 200.0
        assert admission.payload["alignment_ms"] is not None
        assert reasoner.transcripts == ["Was ist fünfundzwanzig mal siebzehn?"]
        await session.close()

    asyncio.run(scenario())


def test_exact_real_mic_orthopaedie_final_recovers_and_invokes_reasoner_once() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner("Welche Woche passt dir?")
        session = RealtimeVoiceSession(
            conversation_id="real-mic-borderline-orthopaedie",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        final_ns, final_sequence = _feed_fragmented_pcm_episode(session)

        await session.accept_user_transcript(
            _streaming_update(
                "Ich brauch 'n Termin für 'n Orthopäden.",
                transcript_id="human-real-mic-orthopaedie",
                timestamp_ns=final_ns,
                audio_frame_sequence=final_sequence,
            )
        )
        await session.wait_for_response()

        assert reasoner.transcripts == ["Ich brauch 'n Termin für 'n Orthopäden."]
        admission = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_ACCEPTED
        )
        assert admission.reason_code == "ACCEPTED_RECOVERED_ACOUSTIC"
        assert admission.payload["evidence_class"] == "RECOVERED_ACOUSTIC"
        assert admission.payload["pcm_speech_detected"] is True
        assert admission.payload["self_speech_risk"] == "LOW"
        assert admission.payload["decision"] == "ACCEPT"
        assert not any(
            event.event_type is EventType.FINAL_ADMISSION_REJECTED
            for event in sink.events
        )
        await session.close()

    asyncio.run(scenario())


def test_uncertain_final_reconciles_with_late_pcm_evidence_once() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner("Welche Woche passt dir?")
        session = RealtimeVoiceSession(
            conversation_id="late-pcm-final-reconciliation",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
            final_admission_reconciliation_ms=60,
        )
        await session.start()
        base_ns = monotonic_ns() - 300_000_000
        for sequence in range(4):
            session.receive_audio(
                AudioFrame(
                    stream_id="browser-pcm-track",
                    sequence=sequence,
                    pcm16=struct.pack("<h", 5_000) * 320,
                    captured_ns=base_ns + sequence * 20_000_000,
                )
            )
        update = _streaming_update(
            "Ich brauch 'n Termin für 'n Orthopäden.",
            transcript_id="final-before-local-evidence-complete",
            timestamp_ns=base_ns + 80_000_000,
            audio_frame_sequence=3,
        )
        admission_task = asyncio.create_task(session.accept_user_transcript(update))
        await asyncio.sleep(0.01)
        for sequence in range(4, 10):
            session.receive_audio(
                AudioFrame(
                    stream_id="browser-pcm-track",
                    sequence=sequence,
                    pcm16=struct.pack("<h", 5_000) * 320,
                    captured_ns=base_ns + sequence * 20_000_000,
                )
            )

        await admission_task
        await session.wait_for_response()

        assert reasoner.transcripts == ["Ich brauch 'n Termin für 'n Orthopäden."]
        accepted = [
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_ACCEPTED
        ]
        assert len(accepted) == 1
        assert accepted[0].reason_code == "ACCEPTED_RECOVERED_ACOUSTIC"
        assert accepted[0].payload["initial_reason_code"] == (
            "INSUFFICIENT_ACOUSTIC_EVIDENCE"
        )
        assert accepted[0].payload["reconciliation_delay_ms"] >= 50
        await session.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ("Ja.", "Nein.", "Okay.", "Stopp.", "mhm"))
def test_valid_short_streaming_finals_reach_user_history_path(text: str) -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id=f"short-final-{text}",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        final_ns, final_sequence = _feed_pcm_episode(
            session, voiced_frames=3, silence_frames=10
        )

        await session.accept_user_transcript(
            _streaming_update(
                text,
                transcript_id=f"valid-short-{text}",
                timestamp_ns=final_ns,
                audio_frame_sequence=final_sequence,
            )
        )
        await session.wait_for_response()

        assert reasoner.transcripts == [text]
        accepted = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_ACCEPTED
        )
        assert accepted.reason_code == "ACCEPTED"
        assert accepted.payload["voiced_duration_ms"] == 60.0
        await session.close()

    asyncio.run(scenario())


def test_phantom_final_without_pcm_is_suppressed_and_session_keeps_listening() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="phantom-farewell-no-pcm",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()

        with pytest.raises(
            TranscriptRejected,
            match="NO_PCM_EPISODE",
        ):
            await session.accept_user_transcript(
                _streaming_update(
                    "So, tschüss, meine Mademoiselle.",
                    transcript_id="phantom-farewell-final",
                )
            )

        assert reasoner.transcripts == []
        assert session.appointment_states == {}
        assert session.state.value == "LISTENING"
        assert not any(
            event.event_type is EventType.CALL_ENDED for event in sink.events
        )
        rejected = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_REJECTED
        )
        assert rejected.reason_code == "NO_PCM_EPISODE"
        assert rejected.payload["speech_episode_id"] is None

        final_ns, final_sequence = _feed_pcm_episode(session)
        await session.accept_user_transcript(
            _streaming_update(
                "Ich brauche einen Termin.",
                transcript_id="real-turn-after-phantom",
                timestamp_ns=final_ns,
                audio_frame_sequence=final_sequence,
            )
        )
        await session.wait_for_response()
        assert reasoner.transcripts == ["Ich brauche einen Termin."]
        await session.close()

    asyncio.run(scenario())


def test_short_noise_episode_cannot_authorize_unrelated_final() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="short-noise-phantom-final",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        final_ns, final_sequence = _feed_pcm_episode(
            session,
            voiced_frames=4,
            silence_frames=10,
            amplitude=7_000,
        )

        with pytest.raises(
            TranscriptRejected,
            match="INSUFFICIENT_ACOUSTIC_EVIDENCE",
        ):
            await session.accept_user_transcript(
                _streaming_update(
                    "Ja, okay, dann viel Erfolg.",
                    transcript_id="short-noise-unrelated-final",
                    timestamp_ns=final_ns,
                    audio_frame_sequence=final_sequence,
                )
            )

        assert reasoner.transcripts == []
        rejected = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_REJECTED
        )
        assert rejected.payload["voiced_duration_ms"] == 80.0
        assert session.appointment_states == {}
        await session.close()

    asyncio.run(scenario())


def test_stale_streaming_stt_session_cannot_write_user_history() -> None:
    class SessionBoundTranscriber(NullInputTranscriber):
        provider_session_id = "scribe-current-session"

    async def scenario() -> None:
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="stale-streaming-session",
            sink=InMemoryTelemetrySink(),
            transcriber=SessionBoundTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()

        with pytest.raises(TranscriptRejected, match="SESSION_MISMATCH"):
            await session.accept_user_transcript(
                _streaming_update(
                    "Ein verspätetes Ergebnis.",
                    transcript_id="scribe-stale-final",
                    stt_session_id="scribe-old-session",
                )
            )

        assert reasoner.transcripts == []
        await session.close()

    asyncio.run(scenario())


def test_repeated_assistant_fragment_during_playback_cannot_poison_history() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="self-speech-protected",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=400),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.accept_user_transcript(
            TranscriptUpdate(
                text="Was kannst du?",
                is_final=True,
                signals=_signals(),
                provenance=TranscriptProvenance.user_fixture(final=True),
            )
        )
        await _wait_speaking(session)

        with pytest.raises(TranscriptRejected, match="SELF_SPEECH_MATCH"):
            await session.accept_user_transcript(
                _streaming_update(
                    "ich bin human flow ein ki assistent und helfe dir gern",
                    transcript_id="browser-self-speech-final",
                )
            )

        assert reasoner.transcripts == ["Was kannst du?"]
        assert any(
            event.event_type is EventType.SELF_SPEECH_SUPPRESSED
            for event in sink.events
        )
        await session.wait_for_response()
        await session.close()

    asyncio.run(scenario())


def test_weak_pcm_plus_active_assistant_similarity_rejects_final() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="weak-pcm-active-assistant-final",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=900),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.accept_user_transcript(
            TranscriptUpdate(
                text="Was kannst du?",
                is_final=True,
                signals=_signals(),
                provenance=TranscriptProvenance.user_fixture(final=True),
            )
        )
        await _wait_speaking(session)
        final_ns, final_sequence = _feed_pcm_episode(
            session,
            voiced_frames=8,
            silence_frames=10,
        )

        with pytest.raises(TranscriptRejected, match="SELF_SPEECH_MATCH"):
            await session.accept_user_transcript(
                _streaming_update(
                    "Ich bin HumanFlow, ein KI-Assistent und helfe dir gern.",
                    transcript_id="weak-active-echo-final",
                    timestamp_ns=final_ns,
                    audio_frame_sequence=final_sequence,
                )
            )

        assert reasoner.transcripts == ["Was kannst du?"]
        rejected = next(
            event
            for event in sink.events
            if event.event_type is EventType.FINAL_ADMISSION_REJECTED
        )
        assert rejected.payload["assistant_playback_active"] is True
        assert rejected.payload["self_speech_candidate"] is True
        assert rejected.payload["voiced_duration_ms"] == 160.0
        await session.wait_for_response()
        await session.close()

    asyncio.run(scenario())


def test_browser_speech_recognition_is_rejected_from_production_history() -> None:
    async def scenario() -> None:
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="clean-browser-user",
            sink=InMemoryTelemetrySink(),
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
        )
        await session.start()
        with pytest.raises(TranscriptRejected, match="transcript_source_not_allowlisted"):
            await session.accept_user_transcript(
                _browser_update(
                    "Was ist 25 mal 17?",
                    transcript_id="browser-production-forbidden",
                )
            )

        assert reasoner.transcripts == []
        await session.close()

    asyncio.run(scenario())


def test_legitimate_human_repetition_after_guard_window_remains_accepted() -> None:
    guard = SelfSpeechGuard(recent_window_ms=5)
    now = monotonic_ns()
    guard.register_pending(
        chunk_id="spoken-chunk",
        response_id="spoken-response",
        text=ASSISTANT_SENTENCE,
    )
    guard.mark_started(chunk_id="spoken-chunk", started_ns=now - 20_000_000)
    guard.mark_stopped(chunk_id="spoken-chunk", stopped_ns=now - 10_000_000)

    assessment = guard.assess(
        text=ASSISTANT_SENTENCE,
        observed_ns=now,
        origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
        playback_active=False,
    )

    assert not assessment.suppress
    assert not assessment.candidate


def test_legitimate_repetition_after_guard_window_reaches_history_path() -> None:
    async def scenario() -> None:
        guard = SelfSpeechGuard(recent_window_ms=5)
        now = monotonic_ns()
        guard.register_pending(
            chunk_id="old-spoken-chunk",
            response_id="old-spoken-response",
            text=ASSISTANT_SENTENCE,
        )
        guard.mark_started(
            chunk_id="old-spoken-chunk", started_ns=now - 20_000_000
        )
        guard.mark_stopped(
            chunk_id="old-spoken-chunk", stopped_ns=now - 10_000_000
        )
        reasoner = RecordingReasoner()
        session = RealtimeVoiceSession(
            conversation_id="legitimate-human-repetition",
            sink=InMemoryTelemetrySink(),
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=5),
            audio_output=TimedPcmOutput(quantum_ms=1),
            self_speech_guard=guard,
        )
        await session.start()
        final_ns, final_sequence = _feed_pcm_episode(session, voiced_frames=30)
        await session.accept_user_transcript(
            _streaming_update(
                ASSISTANT_SENTENCE,
                transcript_id="legitimate-repeat-final",
                timestamp_ns=final_ns,
                audio_frame_sequence=final_sequence,
            )
        )
        await session.wait_for_response()

        assert reasoner.transcripts == [ASSISTANT_SENTENCE]
        await session.close()

    asyncio.run(scenario())


def test_short_ordered_assistant_fragment_is_suppressed_only_during_playback() -> None:
    guard = SelfSpeechGuard()
    now = monotonic_ns()
    guard.register_pending(
        chunk_id="long-active-chunk",
        response_id="long-active-response",
        text=(
            "Ich bin HumanFlow, ein KI-Assistent. Ich kann Fragen beantworten "
            "und dir Dinge verständlich erklären."
        ),
    )
    guard.mark_started(chunk_id="long-active-chunk", started_ns=now)

    assessment = guard.assess(
        text="ich bin human flow ein ki assistent",
        observed_ns=now + 20_000_000,
        origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
        playback_active=True,
    )

    assert assessment.candidate
    assert assessment.suppress
    assert assessment.signals["compact_fragment"] is True


@pytest.mark.parametrize("text", ["mhm", "Moment, stopp"])
def test_short_backchannel_and_explicit_barge_in_are_never_self_speech_suppressed(
    text: str,
) -> None:
    guard = SelfSpeechGuard()
    now = monotonic_ns()
    guard.register_pending(
        chunk_id="active-chunk",
        response_id="active-response",
        text=f"{ASSISTANT_SENTENCE} {text}",
    )
    guard.mark_started(chunk_id="active-chunk", started_ns=now)

    assessment = guard.assess(
        text=text,
        observed_ns=now + 1_000_000,
        origin=TranscriptOrigin.STREAMING_STT_PROVIDER,
        playback_active=True,
    )

    assert not assessment.suppress
    assert not assessment.candidate


@pytest.mark.parametrize(("assistant_text", "self_transcript"), HUMAN_FAILURE_CASES)
def test_exact_human_self_transcription_failures_never_write_user_history(
    assistant_text: str,
    self_transcript: str,
) -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner(assistant_text)
        session = RealtimeVoiceSession(
            conversation_id="exact-human-self-transcription-regression",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=600),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.accept_user_transcript(
            TranscriptUpdate(
                text="Ich brauche Hilfe bei einem Termin.",
                is_final=True,
                signals=_signals(),
                provenance=TranscriptProvenance.user_fixture(final=True),
            )
        )
        await _wait_speaking(session)
        await session.wait_for_response()

        with pytest.raises(TranscriptRejected, match="SELF_SPEECH_MATCH"):
            await session.accept_user_transcript(
                _streaming_update(
                    self_transcript,
                    transcript_id=f"self-echo-{len(self_transcript)}",
                )
            )

        assert reasoner.transcripts == ["Ich brauche Hilfe bei einem Termin."]
        rejected = [
            event
            for event in sink.events
            if event.event_type is EventType.TRANSCRIPT_PROVENANCE_RECORDED
            and event.payload["transcript_id"] == f"self-echo-{len(self_transcript)}"
        ]
        assert len(rejected) == 1
        assert rejected[0].payload["accepted_as_user_turn"] is False
        await session.close()

    asyncio.run(scenario())


def test_actual_distinct_user_partial_during_playback_remains_ephemeral_and_accepted() -> None:
    async def scenario() -> None:
        sink = InMemoryTelemetrySink()
        reasoner = RecordingReasoner(HUMAN_FAILURE_CASES[0][0])
        session = RealtimeVoiceSession(
            conversation_id="real-user-during-playback",
            sink=sink,
            transcriber=NullInputTranscriber(),  # type: ignore[arg-type]
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=600),
            audio_output=TimedPcmOutput(quantum_ms=2),
        )
        await session.start()
        await session.accept_user_transcript(
            TranscriptUpdate(
                text="Erzähl mir etwas über ERP.",
                is_final=True,
                signals=_signals(),
                provenance=TranscriptProvenance.user_fixture(final=True),
            )
        )
        await _wait_speaking(session)

        decision = await session.accept_user_transcript(
            _streaming_update(
                "Nein, ich möchte etwas über Warenwirtschaft wissen",
                transcript_id="distinct-real-user-partial",
                final=False,
                speech_active=True,
            )
        )

        assert decision.decision.value == "CONTINUE_LISTENING"
        assert reasoner.transcripts == ["Erzähl mir etwas über ERP."]
        provenance = next(
            event
            for event in sink.events
            if event.event_type is EventType.TRANSCRIPT_PROVENANCE_RECORDED
            and event.payload["transcript_id"] == "distinct-real-user-partial"
        )
        assert provenance.payload["accepted_by_user_ingestion"] is True
        assert provenance.payload["accepted_as_user_turn"] is False
        assert provenance.payload["partial_transcript_to_user_history"] == "FORBIDDEN"
        await session.wait_for_response()
        await session.close()

    asyncio.run(scenario())
