from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, date, datetime

from humanflow.domain.conversation import OperationToken
from humanflow.runtime.appointment_state import AppointmentStateTracker
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
from humanflow.turns.models import TurnSignals


FIXED_TODAY = date(2026, 8, 24)
FIXED_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _tracker() -> AppointmentStateTracker:
    return AppointmentStateTracker(today=lambda: FIXED_TODAY, now=lambda: FIXED_NOW)


def test_appointment_updates_are_delta_based_and_old_friday_never_resurfaces() -> None:
    tracker = _tracker()

    first = tracker.apply_user_turn(
        "Ich möchte den Termin nächsten Freitag.", source_turn="turn-1"
    )
    changed_date = tracker.apply_user_turn(
        "Äh, machen wir übernächste Woche Donnerstag, das wäre besser.",
        source_turn="turn-2",
    )
    fourteen = tracker.apply_user_turn(
        "Am besten gegen 14 Uhr.", source_turn="turn-3"
    )
    fifteen = tracker.apply_user_turn(
        "Vielleicht 15 Uhr.", source_turn="turn-4"
    )

    assert first.updated_slots == ("date", "status")
    assert changed_date.updated_slots == ("date",)
    assert fourteen.updated_slots == ("time",)
    assert fifteen.updated_slots == ("time",)
    assert tracker.state.date is not None
    assert tracker.state.date.value == "2026-09-10"
    assert tracker.state.date.source_turn == "turn-2"
    assert tracker.state.time is not None
    assert tracker.state.time.value == "15:00"
    assert tracker.state.time.source_turn == "turn-4"
    assert tracker.state.date.updated_at == "2026-08-24T12:00:00Z"
    assert tracker.state.time.confidence == 0.99


def test_correction_changes_only_explicit_slot_and_hesitation_is_not_a_value() -> None:
    tracker = _tracker()
    tracker.apply_user_turn(
        "Ich brauche einen Termin am Freitag um 14 Uhr.", source_turn="turn-1"
    )
    correction = tracker.apply_user_turn(
        "Äh, nein, nicht Freitag. Ich meinte Montag.", source_turn="turn-2"
    )
    hesitation = tracker.apply_user_turn("Äh ...", source_turn="turn-3")

    assert correction.updated_slots == ("date",)
    assert tracker.state.date is not None
    assert tracker.state.date.value == "2026-08-31"
    assert tracker.state.time is not None
    assert tracker.state.time.value == "14:00"
    assert hesitation.updated_slots == ()


class ContextRecordingReasoner:
    def __init__(self) -> None:
        self.contexts: list[str | None] = []
        self.states: list[dict[str, object] | None] = []
        self.transcripts: list[str] = []

    def set_authoritative_transaction_context(
        self, context: str | None, *, state: dict[str, object] | None = None
    ) -> None:
        self.contexts.append(context)
        self.states.append(state)

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        self.transcripts.append(transcript)
        yield "Kurz bestätigt."


def _turn(text: str, turn_id: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=replace(
            TranscriptProvenance.user_fixture(final=True),
            transcript_id=turn_id,
        ),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=500,
            utterance_duration_ms=900,
            semantic_complete=True,
            provider_endpointed=True,
            acoustic_completion=0.95,
        ),
    )


def test_session_supplies_authoritative_state_without_rewriting_user_transcript() -> None:
    async def scenario() -> None:
        reasoner = ContextRecordingReasoner()
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="appointment-state-session",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1),
            audio_output=TimedPcmOutput(quantum_ms=1),
            appointment_state_tracker=_tracker(),
        )
        await session.start()
        turns = (
            ("Termin nächsten Freitag.", "turn-1"),
            ("Lieber übernächste Woche Donnerstag.", "turn-2"),
            ("Gegen 14 Uhr.", "turn-3"),
            ("Vielleicht 15 Uhr.", "turn-4"),
        )
        for text, turn_id in turns:
            await session.submit_transcript(_turn(text, turn_id))
            await session.wait_for_response()

        assert reasoner.transcripts == [text for text, _ in turns]
        assert reasoner.contexts[-1] is not None
        assert '"date": "2026-09-10"' in reasoner.contexts[-1]
        assert '"time": "15:00"' in reasoner.contexts[-1]
        assert '"updated_slots_this_user_turn": ["time"]' in reasoner.contexts[-1]
        state_events = [
            event
            for event in sink.events
            if event.event_type is EventType.APPOINTMENT_STATE_UPDATED
        ]
        assert len(state_events) == 4
        assert state_events[-1].payload["updated_slots"] == ["time"]
        assert state_events[-1].payload["unchanged_slots_preserved"] is True
        assert state_events[-1].payload["assistant_history_used_as_state_source"] is False
        await session.close()

    asyncio.run(scenario())
