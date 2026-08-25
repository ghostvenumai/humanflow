from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState
from humanflow.domain.conversation import OperationToken
from humanflow.runtime.anthropic_provider import (
    _authoritative_database_reply,
    _guard_transaction_fragment,
)
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
from humanflow.tools.appointment_coordinator import AppointmentTransactionCoordinator
from humanflow.tools.sqlite_appointments import SQLiteAppointmentToolProvider
from humanflow.turns.models import TurnSignals


FIXED_TODAY = date(2026, 8, 24)
FIXED_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _tracker() -> AppointmentStateTracker:
    return AppointmentStateTracker(today=lambda: FIXED_TODAY, now=lambda: FIXED_NOW)


def _machine() -> tuple[ConversationStateMachine, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    machine = ConversationStateMachine(conversation_id="sqlite-demo", sink=sink)
    machine.transition(
        ConversationState.LISTENING,
        reason_code="test_started",
        correlation_id="setup",
    )
    machine.transition(
        ConversationState.THINKING,
        reason_code="turn_complete",
        correlation_id="setup",
    )
    return machine, sink


async def _call(
    provider: SQLiteAppointmentToolProvider, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    response = await provider.call(name, arguments)
    return dict(response.value)


def _create_arguments(appointment_id: str, start: str) -> dict[str, Any]:
    return {
        "appointment_id": appointment_id,
        "appointment_type": "Orthopädie",
        "start_datetime": start,
    }


class _ContextReasoner:
    def __init__(self) -> None:
        self.contexts: list[str | None] = []

    def set_authoritative_transaction_context(
        self, context: str | None, *, state: dict[str, object] | None = None
    ) -> None:
        del state
        self.contexts.append(context)

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del transcript, token
        yield "Kurze Terminantwort."


def _turn(text: str, turn_id: str) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=replace(
            TranscriptProvenance.user_fixture(final=True), transcript_id=turn_id
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


def test_sqlite_initialization_and_deterministic_demo_seed(tmp_path: Path) -> None:
    database = tmp_path / "appointments.sqlite3"
    SQLiteAppointmentToolProvider(database)
    SQLiteAppointmentToolProvider(database)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        providers = connection.execute("SELECT COUNT(*) FROM providers").fetchone()[0]
        availability = connection.execute("SELECT COUNT(*) FROM availability").fetchone()[0]
        locations = {
            row[0] for row in connection.execute("SELECT DISTINCT location FROM providers")
        }

    assert {"providers", "availability", "appointments"} <= tables
    assert providers == 2
    assert availability == 23
    assert locations == {"Ingolstadt"}


def test_ingolstadt_seed_has_morning_midday_and_afternoon_slots(
    tmp_path: Path,
) -> None:
    provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
    for appointment_type in ("Orthopädie", "Friseur"):
        result = provider.search_availability(
            {
                "appointment_type": appointment_type,
                "location": "Ingolstadt",
                "start_date": "2026-08-25",
                "end_date": "2026-09-15",
            }
        )
        hours = [int(slot["start_datetime"][11:13]) for slot in result["slots"]]
        assert result["result_status"] == "AVAILABLE"
        assert sum(hour < 12 for hour in hours) >= 3
        assert sum(12 <= hour < 15 for hour in hours) >= 3
        assert sum(hour >= 15 for hour in hours) >= 3


@pytest.mark.parametrize(
    "transcript",
    (
        "Welche Termine sind frei?",
        "Was ist frei?",
        "Wann habt ihr etwas frei?",
        "Welche Zeiten wären verfügbar?",
    ),
)
def test_availability_intents_route_only_to_search(
    tmp_path: Path, transcript: str
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(transcript, source_turn="availability-turn")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="availability-routing",
            state_machine=machine,
            provider=provider,
            today=lambda: date(2026, 8, 25),
        )
        _, outcome = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="availability-turn",
            source_turn="availability-turn",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert outcome is not None
        assert outcome.tool_name == "search_availability"
        assert outcome.value["result_status"] == "AVAILABLE"
        assert provider.call_counts == {"search_availability": 1}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "transcript",
    (
        "Welche Termine habe ich?",
        "Was habe ich gebucht?",
        "Zeig mir meine Termine.",
    ),
)
def test_booked_appointment_intents_route_only_to_list(
    tmp_path: Path, transcript: str
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(transcript, source_turn="list-turn")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="list-routing",
            state_machine=machine,
            provider=provider,
        )
        _, outcome = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="list-turn",
            source_turn="list-turn",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert outcome is not None
        assert outcome.tool_name == "list_appointments"
        assert provider.call_counts == {"list_appointments": 1}

    asyncio.run(scenario())


def test_unavailable_requested_time_is_not_booked_and_offers_nearest_slots(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        transcript = "Orthopädentermin Mittwoch in zwei Wochen um 12:30 Uhr."
        tracker = _tracker()
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="unavailable-slot",
            state_machine=machine,
            provider=provider,
        )
        tool_delta, outcome = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert outcome is not None and outcome.tool_name == "search_availability"
        assert outcome.value["result_status"] == "UNAVAILABLE"
        assert outcome.value["slots"] == []
        assert [
            slot["start_datetime"][11:16]
            for slot in outcome.value["alternative_slots"][:2]
        ] == ["11:30", "14:00"]
        assert provider.call_counts == {"search_availability": 1}
        assert provider.list_appointments({})["appointments"] == []
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(tool_delta), outcome
        )
        response = _authoritative_database_reply(json.loads(context or "{}"))
        assert response is not None
        assert "12:30 Uhr ist leider nicht frei" in response
        assert "11:30 Uhr" in response and "14 Uhr" in response
        assert "technisch" not in response.casefold()

    asyncio.run(scenario())


def test_booking_conflict_is_business_result_not_technical_failure() -> None:
    response = _authoritative_database_reply(
        {
            "database_tool_result": {
                "tool_name": "create_appointment",
                "success": False,
                "result": {
                    "success": False,
                    "result_status": "BOOKING_CONFLICT",
                    "error_code": "BOOKING_CONFLICT",
                },
            }
        }
    )

    assert response == "Der gewünschte Termin ist leider nicht frei."
    assert "technisch" not in response.casefold()


def test_availability_create_conflict_reschedule_cancel_and_list(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        available = await _call(
            provider,
            "search_availability",
            {"appointment_type": "Orthopädie", "date": "2026-09-03"},
        )
        assert [slot["start_datetime"][11:16] for slot in available["slots"]] == [
            "10:30",
            "14:00",
        ]

        first = await _call(
            provider,
            "create_appointment",
            _create_arguments("appointment-a", "2026-09-03T10:30:00+02:00"),
        )
        conflict = await _call(
            provider,
            "create_appointment",
            _create_arguments("appointment-b", "2026-09-03T10:30:00+02:00"),
        )
        moved = await _call(
            provider,
            "reschedule_appointment",
            {
                "appointment_id": "appointment-a",
                "start_datetime": "2026-09-03T14:00:00+02:00",
            },
        )
        listed = await _call(provider, "list_appointments", {})
        cancelled = await _call(
            provider, "cancel_appointment", {"appointment_id": "appointment-a"}
        )
        active_after_cancel = await _call(provider, "list_appointments", {})

        assert first["success"] is True and first["status"] == "BOOKED"
        assert conflict == {
            "success": False,
            "tool": "create_appointment",
            "result_status": "BOOKING_CONFLICT",
            "error_code": "BOOKING_CONFLICT",
            "appointment_id": "appointment-b",
        }
        assert moved["appointment_id"] == "appointment-a"
        assert moved["old_slot"]["start_datetime"].endswith("10:30:00+02:00")
        assert moved["new_slot"]["start_datetime"].endswith("14:00:00+02:00")
        assert [item["appointment_id"] for item in listed["appointments"]] == [
            "appointment-a"
        ]
        assert cancelled["status"] == "CANCELLED"
        assert active_after_cancel["appointments"] == []

    asyncio.run(scenario())


def test_two_independent_appointments_and_reschedule_preserves_identity(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        orthopedics = await _call(
            provider,
            "create_appointment",
            _create_arguments("orthopedics-id", "2026-09-03T10:30:00+02:00"),
        )
        hairdresser = await _call(
            provider,
            "create_appointment",
            {
                "appointment_id": "hairdresser-id",
                "appointment_type": "Friseur",
                "start_datetime": "2026-08-31T11:00:00+02:00",
            },
        )
        moved = await _call(
            provider,
            "reschedule_appointment",
            {
                "appointment_id": "orthopedics-id",
                "start_datetime": "2026-09-03T14:00:00+02:00",
            },
        )
        listed = await _call(provider, "list_appointments", {})

        assert orthopedics["appointment_id"] == moved["appointment_id"]
        assert hairdresser["appointment_id"] == "hairdresser-id"
        assert {item["appointment_id"] for item in listed["appointments"]} == {
            "orthopedics-id",
            "hairdresser-id",
        }
        hair = next(
            item for item in listed["appointments"] if item["appointment_id"] == "hairdresser-id"
        )
        assert hair["start_datetime"] == "2026-08-31T11:00:00+02:00"

    asyncio.run(scenario())


def test_ambiguous_reference_never_calls_or_mutates_database(tmp_path: Path) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        tracker.apply_user_turn(
            "Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.", source_turn="turn-1"
        )
        tracker.apply_user_turn(
            "Noch ein Friseurtermin nächste Woche Mittwoch um 14 Uhr.", source_turn="turn-2"
        )
        tracker.apply_user_turn("Welche Termine habe ich?", source_turn="turn-3")
        ambiguous = tracker.apply_user_turn("Den will ich absagen.", source_turn="turn-4")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="ambiguous",
            state_machine=machine,
            provider=provider,
        )
        token = machine.issue_operation(kind="response")
        unchanged, outcome = await coordinator.execute(
            transcript="Den will ich absagen.",
            delta=ambiguous,
            tracker=tracker,
            correlation_id="turn-4",
            source_turn="turn-4",
            parent_token=token,
        )

        assert unchanged.clarification_required is True
        assert outcome is None
        assert provider.call_counts == {}
        assert provider.list_appointments({})["appointments"] == []

    asyncio.run(scenario())


def test_conversation_desire_is_separate_until_database_tool_success(tmp_path: Path) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(
            "Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            source_turn="turn-1",
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        assert provider.list_appointments({})["appointments"] == []
        assert tracker.state.status is not None
        assert tracker.state.status.value == "READY_TO_BOOK"

        machine, sink = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="separation",
            state_machine=machine,
            provider=provider,
        )
        token = machine.issue_operation(kind="response")
        searched_delta, searched = await coordinator.execute(
            transcript="Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=token,
        )
        searched_context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(searched_delta), searched
        )

        assert searched is not None and searched.tool_name == "search_availability"
        assert searched.value["result_status"] == "AVAILABLE"
        assert provider.list_appointments({})["appointments"] == []
        assert tracker.state.status is not None
        assert tracker.state.status.value == "AVAILABLE"
        assert searched_context is not None
        assert json.loads(searched_context)["external_action_performed"] is False

        confirmation = tracker.apply_user_turn("Ja, bitte buchen.", source_turn="turn-2")
        booked_delta, booked = await coordinator.execute(
            transcript="Ja, bitte buchen.",
            delta=confirmation,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(booked_delta), booked
        )

        assert booked is not None and booked.tool_name == "create_appointment"
        assert booked.success is True and booked.value["result_status"] == "BOOKED"
        assert tracker.state.status is not None and tracker.state.status.value == "BOOKED"
        assert len(provider.list_appointments({})["appointments"]) == 1
        assert context is not None and json.loads(context)["external_action_performed"] is True
        assert any(event.event_type is EventType.APPOINTMENT_CREATED for event in sink.events)

    asyncio.run(scenario())


def test_tool_failure_leaves_database_unchanged_and_forbids_false_success(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(
            "Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            source_turn="turn-1",
        )
        provider = SQLiteAppointmentToolProvider(
            tmp_path / "appointments.sqlite3",
            failure_tool="create_appointment",
        )
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="failure",
            state_machine=machine,
            provider=provider,
        )
        searched_delta, searched = await coordinator.execute(
            transcript="Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert searched is not None and searched.tool_name == "search_availability"
        confirmation = tracker.apply_user_turn("Ja, bitte buchen.", source_turn="turn-2")
        tool_delta, outcome = await coordinator.execute(
            transcript="Ja, bitte buchen.",
            delta=confirmation,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(tool_delta), outcome
        )

        assert outcome is not None and outcome.success is False
        assert outcome.value["result_status"] == "TECHNICAL_FAILURE"
        assert provider.list_appointments({})["appointments"] == []
        assert context is not None
        payload = json.loads(context)
        assert payload["external_action_performed"] is False
        repaired = _guard_transaction_fragment("Der Termin ist gebucht.", payload)
        assert "gebucht" not in repaired.casefold()

    asyncio.run(scenario())


def test_configured_slow_search_executes_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(
            "Ich brauche nächste Woche Donnerstag einen Orthopädentermin.",
            source_turn="turn-1",
        )
        provider = SQLiteAppointmentToolProvider(
            tmp_path / "appointments.sqlite3", delay_ms=25
        )
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="slow",
            state_machine=machine,
            provider=provider,
            timeout_ms=200,
        )
        outcome_delta, outcome = await coordinator.execute(
            transcript="Ich brauche nächste Woche Donnerstag einen Orthopädentermin.",
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert outcome_delta.appointment_id == "appointment_1"
        assert outcome is not None and outcome.success is True
        assert provider.call_counts["search_availability"] == 1

    asyncio.run(scenario())


def test_stale_tool_result_cannot_commit_or_change_action_state(tmp_path: Path) -> None:
    class SlowForwarder:
        def __init__(self, provider: SQLiteAppointmentToolProvider) -> None:
            self.provider = provider

        async def call(self, name: str, arguments: dict[str, Any]):
            await asyncio.sleep(0.02)
            return await self.provider.call(name, arguments)

    async def scenario() -> None:
        tracker = _tracker()
        delta = tracker.apply_user_turn(
            "Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            source_turn="turn-1",
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, sink = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="stale",
            state_machine=machine,
            provider=SlowForwarder(provider),
            timeout_ms=200,
        )
        parent_token = machine.issue_operation(kind="response")
        task = asyncio.create_task(
            coordinator.execute(
                transcript="Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
                delta=delta,
                tracker=tracker,
                correlation_id="turn-1",
                source_turn="turn-1",
                parent_token=parent_token,
            )
        )
        await asyncio.sleep(0.005)
        machine.invalidate_operations(reason_code="newer_turn", correlation_id="interrupt")
        unchanged, _ = await task

        assert provider.list_appointments({})["appointments"] == []
        assert unchanged is delta
        assert tracker.state.status is not None
        assert tracker.state.status.value == "READY_TO_BOOK"
        assert any(event.event_type is EventType.STALE_RESULT_REJECTED for event in sink.events)

    asyncio.run(scenario())


def test_realtime_session_uses_persistent_tools_for_full_demo_flow(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        reasoner = _ContextReasoner()
        session = RealtimeVoiceSession(
            conversation_id="transactional-session",
            sink=InMemoryTelemetrySink(),
            transcriber=NullTranscriber(),
            reasoner=reasoner,
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1),
            audio_output=TimedPcmOutput(quantum_ms=1),
            appointment_state_tracker=_tracker(),
            appointment_tool_provider=provider,
        )
        await session.start()
        turns = (
            ("Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.", "turn-1"),
            ("Moment, mach lieber 14 Uhr.", "turn-2"),
            ("Ja, bitte buchen.", "turn-3"),
            ("Ich brauche noch einen Friseurtermin nächsten Montag um 11 Uhr.", "turn-4"),
            ("Ja, bitte buchen.", "turn-5"),
            ("Welche Termine habe ich?", "turn-6"),
            ("Den Friseurtermin brauche ich nicht mehr.", "turn-7"),
        )
        for text, turn_id in turns:
            await session.submit_transcript(_turn(text, turn_id))
            await session.wait_for_response()

        active = provider.list_appointments({})["appointments"]
        all_rows = provider.list_appointments({"include_cancelled": True})["appointments"]
        assert len(active) == 1
        assert active[0]["appointment_type"] == "Orthopädie"
        assert active[0]["start_datetime"] == "2026-09-03T14:00:00+02:00"
        assert len(all_rows) == 2
        hairdresser = next(
            row for row in all_rows if row["appointment_type"] == "Friseur"
        )
        assert hairdresser["status"] == "CANCELLED"
        assert any(
            context is not None and '"tool_name": "list_appointments"' in context
            for context in reasoner.contexts
        )
        await session.close()

    asyncio.run(scenario())
