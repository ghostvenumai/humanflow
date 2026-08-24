from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import monotonic_ns

import pytest

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


class NullInputTranscriber:
    async def ingest(self, frame: object) -> tuple[()]:
        del frame
        return ()

    async def close(self) -> None:
        return None


class RecordingReasoner:
    def __init__(self) -> None:
        self.transcripts: list[str] = []

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        self.transcripts.append(transcript)
        yield ASSISTANT_SENTENCE


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

        with pytest.raises(TranscriptRejected, match="probable_assistant_self_speech"):
            await session.accept_user_transcript(
                _browser_update(
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


def test_clean_browser_user_speech_reaches_reasoner() -> None:
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
        await session.accept_user_transcript(
            _browser_update(
                "Was ist 25 mal 17?",
                transcript_id="clean-browser-final",
            )
        )
        await session.wait_for_response()

        assert reasoner.transcripts == ["Was ist 25 mal 17?"]
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
        origin=TranscriptOrigin.BROWSER_SPEECH_RECOGNITION,
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
        await session.accept_user_transcript(
            _browser_update(
                ASSISTANT_SENTENCE,
                transcript_id="legitimate-repeat-final",
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
        origin=TranscriptOrigin.BROWSER_SPEECH_RECOGNITION,
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
        origin=TranscriptOrigin.BROWSER_SPEECH_RECOGNITION,
        playback_active=True,
    )

    assert not assessment.suppress
    assert not assessment.candidate
