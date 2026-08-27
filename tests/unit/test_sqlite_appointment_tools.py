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
from humanflow.tools.appointment_coordinator import (
    AppointmentTransactionCoordinator,
    _confirms_booking,
)
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

    orthopedics = provider.search_availability(
        {
            "appointment_type": "Orthopädie",
            "location": "Ingolstadt",
            "date": "2026-09-02",
        }
    )
    assert [slot["start_datetime"][11:16] for slot in orthopedics["slots"]] == [
        "11:30",
        "14:00",
    ]


def test_relative_tomorrow_datetime_matches_seeded_sqlite_slot(tmp_path: Path) -> None:
    tracker = AppointmentStateTracker(
        today=lambda: date(2026, 8, 25),
        now=lambda: datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
    )
    delta = tracker.apply_user_turn(
        "Friseurtermin morgen gegen 12 Uhr in Ingolstadt.",
        source_turn="turn-tomorrow",
    )
    appointment = tracker.appointments[delta.appointment_id or ""]
    assert appointment.date is not None
    assert appointment.date.raw_expression == "morgen"
    assert appointment.date.value == "2026-08-26"
    assert appointment.date.timezone == "Europe/Berlin"
    assert appointment.time is not None and appointment.time.value == "12:00"

    result = SQLiteAppointmentToolProvider(
        tmp_path / "appointments.sqlite3"
    ).search_availability(
        {
            "appointment_type": "Friseur",
            "location": "Ingolstadt",
            "date": appointment.date.value,
            "preferred_time": appointment.time.value,
        }
    )

    assert result["result_status"] == "AVAILABLE"
    assert result["slots"][0]["start_datetime"] == "2026-08-26T12:00:00+02:00"


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
        transcript = "Orthopädentermin Mittwoch in einer Woche um 12:30 Uhr."
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


def test_cancelled_appointment_id_cannot_replay_as_false_booked_success(
    tmp_path: Path,
) -> None:
    provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
    request = {
        "appointment_id": "stable-cancelled-id",
        "appointment_type": "Orthopädie",
        "start_datetime": "2026-09-03T10:30:00+02:00",
    }

    booked = provider.create_appointment(request)
    cancelled = provider.cancel_appointment(
        {"appointment_id": "stable-cancelled-id"}
    )
    replay = provider.create_appointment(request)
    rows = provider.list_appointments({"include_cancelled": True})["appointments"]

    assert booked["result_status"] == "BOOKED"
    assert cancelled["result_status"] == "CANCELLED"
    assert replay["success"] is False
    assert replay["result_status"] == "BOOKING_CONFLICT"
    assert len(rows) == 1
    assert rows[0]["status"] == "CANCELLED"


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


def test_demo_calendar_responses_are_truthful_about_local_scope() -> None:
    available = _authoritative_database_reply(
        {
            "database_tool_result": {
                "tool_name": "search_availability",
                "success": True,
                "result": {
                    "success": True,
                    "result_status": "AVAILABLE",
                    "requested_time": "14:00",
                    "slots": [
                        {"start_datetime": "2026-09-02T14:00:00+02:00"}
                    ],
                },
            }
        }
    )
    booked = _authoritative_database_reply(
        {
            "database_tool_result": {
                "tool_name": "create_appointment",
                "success": True,
                "result": {
                    "success": True,
                    "result_status": "BOOKED",
                    "appointment_type": "Orthopädie",
                    "start_datetime": "2026-09-02T14:00:00+02:00",
                },
            }
        }
    )

    assert available is not None and "hinterlegten Demo-Terminen" in available
    assert booked is not None and "in HumanFlow gebucht" in booked
    assert "echten Kalender" not in available
    assert "Arztpraxis" not in booked


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


@pytest.mark.parametrize(
    "confirmation",
    (
        "Ja.",
        "Ja, okay.",
        "Ja. Ja, okay.",
        "Ja ja.",
        "Ja bitte.",
        "Okay.",
        "Okay, ja.",
        "Super.",
        "Ja, super.",
        "Perfekt.",
        "Ja, perfekt.",
        "Ja, alles ist super.",
        "Passt.",
        "Passt so.",
        "Machen wir.",
        "Nehmen wir.",
    ),
)
def test_unambiguous_offer_affirmation_books_exactly_once(
    tmp_path: Path, confirmation: str
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        transcript = "Orthopädentermin diesen Donnerstag um 15:30 Uhr."
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id=f"affirm-{confirmation}",
            state_machine=machine,
            provider=provider,
        )

        _, offered = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert offered is not None and offered.tool_name == "search_availability"
        assert offered.value["result_status"] == "AVAILABLE"
        assert provider.list_appointments({})["appointments"] == []

        confirmed = tracker.apply_user_turn(confirmation, source_turn="turn-2")
        _, booked = await coordinator.execute(
            transcript=confirmation,
            delta=confirmed,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        repeated = tracker.apply_user_turn("Ja, perfekt.", source_turn="turn-3")
        _, duplicate = await coordinator.execute(
            transcript="Ja, perfekt.",
            delta=repeated,
            tracker=tracker,
            correlation_id="turn-3",
            source_turn="turn-3",
            parent_token=machine.issue_operation(kind="response"),
        )

        rows = provider.list_appointments({})["appointments"]
        assert booked is not None and booked.tool_name == "create_appointment"
        assert booked.value["result_status"] == "BOOKED"
        assert duplicate is None
        assert len(rows) == 1
        assert rows[0]["start_datetime"] == "2026-08-27T15:30:00+02:00"
        assert provider.call_counts == {
            "search_availability": 1,
            "create_appointment": 1,
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "correction",
    (
        "Ja, aber lieber 14 Uhr.",
        "Okay, nein.",
        "Ja, Moment.",
        "Super, aber morgen wäre besser.",
        "Nein.",
        "Nee.",
        "Doch lieber Donnerstag.",
    ),
)
def test_confirmation_normalizer_rejects_corrections_and_negation(
    correction: str,
) -> None:
    assert _confirms_booking(correction) is False


def test_single_search_offer_without_requested_time_super_books_once(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        transcript = "Orthopädentermin übernächste Woche Donnerstag."
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="single-offer-super",
            state_machine=machine,
            provider=provider,
        )

        _, offered = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert offered is not None and offered.tool_name == "search_availability"
        assert [
            slot["start_datetime"] for slot in offered.value["slots"]
        ] == ["2026-09-10T15:00:00+02:00"]

        confirmation = tracker.apply_user_turn("Super.", source_turn="turn-2")
        _, booked = await coordinator.execute(
            transcript="Super.",
            delta=confirmation,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )

        rows = provider.list_appointments({})["appointments"]
        assert booked is not None and booked.tool_name == "create_appointment"
        assert booked.value["result_status"] == "BOOKED"
        assert len(rows) == 1
        assert rows[0]["start_datetime"] == "2026-09-10T15:00:00+02:00"
        assert provider.call_counts == {
            "search_availability": 1,
            "create_appointment": 1,
        }

    asyncio.run(scenario())


def test_negative_correction_after_offer_searches_new_slot_without_booking(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        first_text = "Orthopädentermin diesen Donnerstag um 15:30 Uhr."
        first = tracker.apply_user_turn(first_text, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="negative-correction",
            state_machine=machine,
            provider=provider,
        )
        await coordinator.execute(
            transcript=first_text,
            delta=first,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )

        correction_text = "Nein, lieber 12 Uhr."
        correction = tracker.apply_user_turn(correction_text, source_turn="turn-2")
        _, searched = await coordinator.execute(
            transcript=correction_text,
            delta=correction,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert searched is not None and searched.tool_name == "search_availability"
        assert searched.value["result_status"] == "AVAILABLE"
        assert provider.list_appointments({})["appointments"] == []
        assert "create_appointment" not in provider.call_counts

    asyncio.run(scenario())


def test_unavailable_slot_never_prebooks_or_moves_requested_date(tmp_path: Path) -> None:
    async def scenario() -> None:
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 25),
            now=lambda: datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
        )
        transcript = "Orthopädentermin Mittwoch in zwei Wochen um 11 Uhr."
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="weekday-no-seed-contamination",
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
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(tool_delta), outcome
        )
        reply = _authoritative_database_reply(json.loads(context or "{}"))

        appointment = tracker.appointments["appointment_1"]
        assert appointment.date is not None and appointment.date.value == "2026-09-09"
        assert outcome is not None and outcome.tool_name == "search_availability"
        assert outcome.value["result_status"] == "UNAVAILABLE"
        assert outcome.value["slots"] == []
        assert outcome.value["alternative_slots"] == []
        assert provider.list_appointments({})["appointments"] == []
        assert "create_appointment" not in provider.call_counts
        assert reply is not None and "10. September" not in reply

        guarded = _guard_transaction_fragment(
            "Okay, ich buche dir den Termin am 10. September um 11 Uhr.",
            json.loads(context or "{}"),
        )
        assert "ich buche" not in guarded.casefold()
        assert "gebucht" not in guarded.casefold()
        assert "10. september" not in guarded.casefold()

    asyncio.run(scenario())


def test_unavailable_eleventh_hour_offers_seeded_fifteenth_hour_without_prebooking(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        transcript = "Orthopädentermin übernächste Woche Donnerstag um 11 Uhr."
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="invalid-eleven",
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
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(tool_delta), outcome
        )
        reply = _authoritative_database_reply(json.loads(context or "{}"))

        assert outcome is not None and outcome.tool_name == "search_availability"
        assert outcome.value["result_status"] == "UNAVAILABLE"
        assert [
            slot["start_datetime"] for slot in outcome.value["alternative_slots"]
        ] == ["2026-09-10T15:00:00+02:00"]
        assert provider.list_appointments({})["appointments"] == []
        assert "create_appointment" not in provider.call_counts
        assert reply is not None
        assert "11 Uhr ist leider nicht frei" in reply
        assert "15 Uhr" in reply
        assert "buche" not in reply.casefold()
        assert "gebucht" not in reply.casefold()

        confirmation = tracker.apply_user_turn("Super.", source_turn="turn-2")
        _, booked = await coordinator.execute(
            transcript="Super.",
            delta=confirmation,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        rows = provider.list_appointments({})["appointments"]
        assert booked is not None and booked.tool_name == "create_appointment"
        assert booked.value["result_status"] == "BOOKED"
        assert len(rows) == 1
        assert rows[0]["start_datetime"] == "2026-09-10T15:00:00+02:00"
        assert provider.call_counts == {
            "search_availability": 1,
            "create_appointment": 1,
        }

    asyncio.run(scenario())


def test_rejected_pending_offer_does_not_book_or_reuse_original_time(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        transcript = "Orthopädentermin übernächste Woche Donnerstag um 11 Uhr."
        delta = tracker.apply_user_turn(transcript, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="reject-pending-offer",
            state_machine=machine,
            provider=provider,
        )

        _, offered = await coordinator.execute(
            transcript=transcript,
            delta=delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert offered is not None
        assert offered.value["result_status"] == "UNAVAILABLE"
        assert [
            slot["start_datetime"] for slot in offered.value["alternative_slots"]
        ] == ["2026-09-10T15:00:00+02:00"]

        rejected = tracker.apply_user_turn("Nein.", source_turn="turn-2")
        _, rejected_outcome = await coordinator.execute(
            transcript="Nein.",
            delta=rejected,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        repeated = tracker.apply_user_turn("Ja, okay.", source_turn="turn-3")
        _, repeated_outcome = await coordinator.execute(
            transcript="Ja, okay.",
            delta=repeated,
            tracker=tracker,
            correlation_id="turn-3",
            source_turn="turn-3",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert rejected_outcome is None
        assert repeated_outcome is None
        assert provider.list_appointments({})["appointments"] == []
        assert provider.call_counts == {"search_availability": 1}

    asyncio.run(scenario())


def test_second_appointment_confirmation_consumes_atomic_cross_date_offer(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 25),
            now=lambda: datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="two-orthopaedics-atomic-offers",
            state_machine=machine,
            provider=provider,
        )

        first_request = (
            "Ich brauch einen Orthopäden-Termin übernächste Woche Donnerstag "
            "um 11 Uhr."
        )
        first_delta = tracker.apply_user_turn(first_request, source_turn="turn-1")
        _, first_offer = await coordinator.execute(
            transcript=first_request,
            delta=first_delta,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert first_offer is not None
        assert [
            slot["start_datetime"]
            for slot in first_offer.value["alternative_slots"]
        ] == ["2026-09-10T15:00:00+02:00"]

        first_confirmation = tracker.apply_user_turn("Super.", source_turn="turn-2")
        _, first_booking = await coordinator.execute(
            transcript="Super.",
            delta=first_confirmation,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert first_booking is not None
        assert first_booking.value["result_status"] == "BOOKED"

        second_request = (
            "Ich bräuchte noch mal Orthopäden-Termin für Mittwoch in zwei Wochen "
            "um 11 Uhr."
        )
        second_delta = tracker.apply_user_turn(second_request, source_turn="turn-3")
        _, second_unavailable = await coordinator.execute(
            transcript=second_request,
            delta=second_delta,
            tracker=tracker,
            correlation_id="turn-3",
            source_turn="turn-3",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert second_unavailable is not None
        assert second_unavailable.value["result_status"] == "UNAVAILABLE"
        assert second_unavailable.value["query_start_date"] == "2026-09-09"
        assert second_unavailable.value["query_end_date"] == "2026-09-09"
        assert second_unavailable.value["alternative_slots"] == []

        followup = tracker.apply_user_turn("Wann hast du frei?", source_turn="turn-4")
        followup_delta, followup_offer = await coordinator.execute(
            transcript="Wann hast du frei?",
            delta=followup,
            tracker=tracker,
            correlation_id="turn-4",
            source_turn="turn-4",
            parent_token=machine.issue_operation(kind="response"),
        )
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(followup_delta), followup_offer
        )
        payload = json.loads(context or "{}")
        reply = _authoritative_database_reply(payload)
        pending = payload["pending_offer"]

        assert followup_offer is not None
        assert followup_offer.value["result_status"] == "AVAILABLE"
        assert followup_offer.value["query_start_date"] == "2026-09-09"
        assert followup_offer.value["query_end_date"] == "2026-09-30"
        assert [
            slot["start_datetime"] for slot in followup_offer.value["slots"]
        ] == ["2026-09-14T12:30:00+02:00"]
        assert tracker.active_focus_appointment_id == "appointment_2"
        assert pending == {
            "appointment_id": "appointment_2",
            "appointment_type": "Orthopädie",
            "created_at_utc": pending["created_at_utc"],
            "database_appointment_id": pending["database_appointment_id"],
            "date": "2026-09-14",
            "generation": 2,
            "location": "Ingolstadt",
            "provider_name": "Praxis am Stadtpark (Demo)",
            "resource_id": "demo-ortho-1",
            "source_query": {
                "appointment_type": "Orthopädie",
                "end_date": "2026-09-30",
                "max_results": 1,
                "start_date": "2026-09-09",
            },
            "source_result_status": "AVAILABLE",
            "status": "OFFERED",
            "time": "12:30",
            "timezone": "Europe/Berlin",
        }
        assert tracker.appointments["appointment_1"].date is not None
        assert tracker.appointments["appointment_1"].date.value == "2026-09-10"
        assert tracker.appointments["appointment_1"].time is not None
        assert tracker.appointments["appointment_1"].time.value == "15:00"
        assert tracker.appointments["appointment_2"].date is not None
        assert tracker.appointments["appointment_2"].date.value == "2026-09-09"
        assert tracker.appointments["appointment_2"].time is not None
        assert tracker.appointments["appointment_2"].time.value == "11:00"
        assert reply is not None
        assert "Mittwoch, den 9. September" in reply
        assert "Montag, den 14. September" in reply
        assert "12:30 Uhr" in reply

        second_confirmation = tracker.apply_user_turn(
            "Ja, ja, okay.", source_turn="turn-5"
        )
        _, second_booking = await coordinator.execute(
            transcript="Ja, ja, okay.",
            delta=second_confirmation,
            tracker=tracker,
            correlation_id="turn-5",
            source_turn="turn-5",
            parent_token=machine.issue_operation(kind="response"),
        )
        repeated = tracker.apply_user_turn("Ja, okay.", source_turn="turn-6")
        _, duplicate = await coordinator.execute(
            transcript="Ja, okay.",
            delta=repeated,
            tracker=tracker,
            correlation_id="turn-6",
            source_turn="turn-6",
            parent_token=machine.issue_operation(kind="response"),
        )

        rows = provider.list_appointments({})["appointments"]
        assert second_booking is not None
        assert second_booking.value["result_status"] == "BOOKED"
        assert duplicate is None
        assert len(rows) == 2
        assert len({row["appointment_id"] for row in rows}) == 2
        assert {row["start_datetime"] for row in rows} == {
            "2026-09-10T15:00:00+02:00",
            "2026-09-14T12:30:00+02:00",
        }
        assert all(
            row["start_datetime"] != "2026-09-09T11:00:00+02:00" for row in rows
        )
        assert tracker.appointments["appointment_1"].date is not None
        assert tracker.appointments["appointment_1"].date.value == "2026-09-10"
        assert tracker.appointments["appointment_1"].time is not None
        assert tracker.appointments["appointment_1"].time.value == "15:00"
        assert tracker.appointments["appointment_2"].date is not None
        assert tracker.appointments["appointment_2"].date.value == "2026-09-14"
        assert tracker.appointments["appointment_2"].time is not None
        assert tracker.appointments["appointment_2"].time.value == "12:30"
        assert provider.call_counts == {
            "search_availability": 3,
            "create_appointment": 2,
        }

    asyncio.run(scenario())


def test_affirmative_with_correction_invalidates_offer_and_never_books(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        request = "Orthopädentermin übernächste Woche Donnerstag um 11 Uhr."
        first = tracker.apply_user_turn(request, source_turn="turn-1")
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="affirmative-correction",
            state_machine=machine,
            provider=provider,
        )
        _, offered = await coordinator.execute(
            transcript=request,
            delta=first,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert offered is not None
        assert [
            slot["start_datetime"] for slot in offered.value["alternative_slots"]
        ] == ["2026-09-10T15:00:00+02:00"]

        correction_text = "Ja, aber lieber 14 Uhr."
        correction = tracker.apply_user_turn(correction_text, source_turn="turn-2")
        _, searched = await coordinator.execute(
            transcript=correction_text,
            delta=correction,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert searched is not None
        assert searched.tool_name == "search_availability"
        assert searched.value["requested_time"] == "14:00"
        assert provider.list_appointments({})["appointments"] == []
        assert provider.call_counts == {"search_availability": 2}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "list_text",
    ("Welchen Termin habe ich?", "Welche Minis hab ich?"),
)
def test_list_intent_outranks_and_invalidates_pending_offer(
    tmp_path: Path,
    list_text: str,
) -> None:
    async def scenario() -> None:
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 25),
            now=lambda: datetime(2026, 8, 25, 10, 0, tzinfo=UTC),
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id=f"list-priority-{len(list_text)}",
            state_machine=machine,
            provider=provider,
        )

        first_request = "Orthopädentermin übernächste Woche Donnerstag um 11 Uhr."
        first = tracker.apply_user_turn(first_request, source_turn="turn-1")
        await coordinator.execute(
            transcript=first_request,
            delta=first,
            tracker=tracker,
            correlation_id="turn-1",
            source_turn="turn-1",
            parent_token=machine.issue_operation(kind="response"),
        )
        confirmed = tracker.apply_user_turn("Super.", source_turn="turn-2")
        _, first_booking = await coordinator.execute(
            transcript="Super.",
            delta=confirmed,
            tracker=tracker,
            correlation_id="turn-2",
            source_turn="turn-2",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert first_booking is not None
        assert first_booking.value["result_status"] == "BOOKED"

        second_request = (
            "Ich bräuchte noch mal einen Orthopädentermin für Mittwoch in zwei "
            "Wochen um 11 Uhr."
        )
        second = tracker.apply_user_turn(second_request, source_turn="turn-3")
        _, unavailable = await coordinator.execute(
            transcript=second_request,
            delta=second,
            tracker=tracker,
            correlation_id="turn-3",
            source_turn="turn-3",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert unavailable is not None
        assert unavailable.value["result_status"] == "UNAVAILABLE"
        followup = tracker.apply_user_turn("Wann hast du frei?", source_turn="turn-4")
        _, offered = await coordinator.execute(
            transcript="Wann hast du frei?",
            delta=followup,
            tracker=tracker,
            correlation_id="turn-4",
            source_turn="turn-4",
            parent_token=machine.issue_operation(kind="response"),
        )
        assert offered is not None
        assert [slot["start_datetime"] for slot in offered.value["slots"]] == [
            "2026-09-14T12:30:00+02:00"
        ]

        list_delta = tracker.apply_user_turn(list_text, source_turn="turn-5")
        enriched_delta, listed = await coordinator.execute(
            transcript=list_text,
            delta=list_delta,
            tracker=tracker,
            correlation_id="turn-5",
            source_turn="turn-5",
            parent_token=machine.issue_operation(kind="response"),
        )
        context = coordinator.enrich_reasoning_context(
            tracker.reasoning_context(enriched_delta), listed
        )
        payload = json.loads(context or "{}")
        reply = _authoritative_database_reply(payload)
        late_confirmation = tracker.apply_user_turn("Ja.", source_turn="turn-6")
        _, late_outcome = await coordinator.execute(
            transcript="Ja.",
            delta=late_confirmation,
            tracker=tracker,
            correlation_id="turn-6",
            source_turn="turn-6",
            parent_token=machine.issue_operation(kind="response"),
        )

        assert list_delta.resolution_reason == "multi_appointment_overview"
        assert listed is not None and listed.tool_name == "list_appointments"
        assert listed.value["result_status"] == "BOOKED"
        assert [
            appointment["start_datetime"]
            for appointment in listed.value["appointments"]
        ] == ["2026-09-10T15:00:00+02:00"]
        assert payload["pending_offer"] is None
        assert reply is not None
        assert "Orthopädie-Termin am Donnerstag um 15 Uhr" in reply
        assert "11 Uhr ist leider nicht frei" not in reply
        assert late_outcome is None
        assert len(provider.list_appointments({})["appointments"]) == 1
        assert provider.call_counts == {
            "search_availability": 3,
            "create_appointment": 1,
            "list_appointments": 1,
        }

    asyncio.run(scenario())


def test_demo_reset_removes_only_demo_bookings_and_preserves_seed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "appointments.sqlite3"
    provider = SQLiteAppointmentToolProvider(database_path)
    booked = provider.create_appointment(
        _create_arguments(
            "demo-booking",
            "2026-09-10T15:00:00+02:00",
        )
    )
    assert booked["result_status"] == "BOOKED"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """INSERT INTO providers
               (resource_id, appointment_type, provider_name, location, is_demo)
               VALUES ('external-resource', 'Extern', 'Extern', 'Ingolstadt', 0)"""
        )
        connection.execute(
            """INSERT INTO appointments
               (appointment_id, resource_id, appointment_type, provider_name, location,
                start_datetime, end_datetime, status, created_at, updated_at)
               VALUES ('external-booking', 'external-resource', 'Extern', 'Extern',
                       'Ingolstadt', '2026-09-20T10:00:00+02:00',
                       '2026-09-20T10:30:00+02:00', 'BOOKED',
                       '2026-08-25T10:00:00+00:00', '2026-08-25T10:00:00+00:00')"""
        )

    result = provider.reset_demo_appointments()
    rows = provider.list_appointments({"include_cancelled": True})["appointments"]
    availability = provider.search_availability(
        {
            "appointment_type": "Orthopädie",
            "date": "2026-09-10",
            "preferred_time": "15:00",
        }
    )

    assert result == {
        "deleted_demo_appointments": 1,
        "preserved_demo_providers": 2,
        "preserved_demo_availability": 23,
    }
    assert [row["appointment_id"] for row in rows] == ["external-booking"]
    assert availability["result_status"] == "AVAILABLE"
    assert [slot["start_datetime"] for slot in availability["slots"]] == [
        "2026-09-10T15:00:00+02:00"
    ]


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


def test_unavailable_corrected_date_is_acknowledged_in_reply() -> None:
    # Regression: a corrected but unavailable date (no nearby slots) must still be
    # named, so the tool result never erases the user's correction context.
    context = {
        "resolved_appointment_id_this_turn": "appointment_1",
        "appointments": {
            "appointment_1": {
                "date": "2026-09-09",
                "time": "11:30",
                "purpose": "Orthopädie",
                "status": "READY_TO_BOOK",
            }
        },
        "database_tool_result": {
            "tool_name": "search_availability",
            "success": True,
            "result": {
                "success": True,
                "result_status": "UNAVAILABLE",
                "requested_time": "11:30",
                "slots": [],
                "alternative_slots": [],
            },
        },
    }
    reply = _authoritative_database_reply(context)
    assert reply is not None
    assert "9. September" in reply
    assert "11:30 Uhr" in reply
    assert "nicht frei" in reply


def test_booking_conflict_reply_keeps_corrected_date() -> None:
    context = {
        "resolved_appointment_id_this_turn": "appointment_1",
        "appointments": {
            "appointment_1": {
                "date": "2026-09-09",
                "purpose": "Orthopädie",
                "status": "READY_TO_BOOK",
            }
        },
        "database_tool_result": {
            "tool_name": "create_appointment",
            "success": False,
            "result": {
                "success": False,
                "result_status": "BOOKING_CONFLICT",
                "error_code": "BOOKING_CONFLICT",
            },
        },
    }
    reply = _authoritative_database_reply(context)
    assert reply is not None
    assert "9. September" in reply
    assert "nicht frei" in reply


def _sticky_followup_query(
    tmp_path: Path, followup: str
) -> tuple[str | None, str | None, str | None, list[str]]:
    """Drive Orthopädie + an unavailable 'übermorgen' (2026-08-28), then one generic
    or same-date follow-up; return (tool, query_start, query_end, slots)."""

    async def scenario() -> tuple[str | None, str | None, str | None, list[str]]:
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 26),
            now=lambda: datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id=f"sticky-{len(followup)}",
            state_machine=machine,
            provider=provider,
            today=lambda: date(2026, 8, 26),
        )

        async def say(text: str, turn: str):
            delta = tracker.apply_user_turn(text, source_turn=turn)
            return await coordinator.execute(
                transcript=text,
                delta=delta,
                tracker=tracker,
                correlation_id=turn,
                source_turn=turn,
                parent_token=machine.issue_operation(kind="response"),
            )

        await say("Ich bräuchte 'n Termin im Orthopäden.", "turn-1")
        _, unavailable = await say("übermorgen?", "turn-2")
        assert unavailable is not None
        assert unavailable.value["result_status"] == "UNAVAILABLE"
        assert unavailable.value["query_start_date"] == "2026-08-28"

        _, outcome = await say(followup, "turn-3")
        if outcome is None:
            return None, None, None, []
        return (
            outcome.tool_name,
            outcome.value.get("query_start_date"),
            outcome.value.get("query_end_date"),
            [slot["start_datetime"] for slot in outcome.value.get("slots", [])],
        )

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    "followup",
    (
        "Hast du freie Termine?",
        "Welcher Termin ist frei?",
        "Wann hast du etwas frei?",
        "Was wäre denn verfügbar?",
        "Wann könnte ich kommen?",
        "Gibt es Alternativen?",
    ),
)
def test_generic_followup_broadens_past_failed_date(
    tmp_path: Path, followup: str
) -> None:
    tool, start, end, slots = _sticky_followup_query(tmp_path, followup)
    assert tool == "search_availability"
    # Broadened beyond the single failed date, and grounded in real demo slots.
    assert (start, end) == ("2026-08-28", "2026-09-18")
    assert slots, "a grounded alternative slot must be offered"
    assert slots[0] == "2026-09-02T11:30:00+02:00"
    assert all(s[:10] != "2026-08-28" for s in slots)


@pytest.mark.parametrize(
    "followup",
    (
        "Was hast du an dem Tag frei?",
        "Welche Uhrzeiten hast du da?",
    ),
)
def test_date_qualified_followup_keeps_failed_date(
    tmp_path: Path, followup: str
) -> None:
    tool, start, end, _slots = _sticky_followup_query(tmp_path, followup)
    assert tool == "search_availability"
    # Explicit reference to the current day keeps the same-date search.
    assert (start, end) == ("2026-08-28", "2026-08-28")


@pytest.mark.parametrize(
    "utterance",
    (
        "Was geht eigentlich mit meiner Versicherung?",
        "Ist der Parkplatz verfügbar?",
        "Was geht?",
    ),
)
def test_unrelated_followup_does_not_search(tmp_path: Path, utterance: str) -> None:
    tool, _start, _end, _slots = _sticky_followup_query(tmp_path, utterance)
    assert tool is None


def _issue_a_followup(
    tmp_path: Path, followup: str
) -> tuple[str | None, list[str], set[str]]:
    """Orthopädie 2026-08-27 at 11:00 (unavailable), then one follow-up.

    Returns (tool_name, slot_start_datetimes, appointment_types_of_slots)."""

    async def scenario() -> tuple[str | None, list[str], set[str]]:
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 26),
            now=lambda: datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        )
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine, _ = _machine()
        coordinator = AppointmentTransactionCoordinator(
            conversation_id=f"issue-a-{len(followup)}",
            state_machine=machine,
            provider=provider,
            today=lambda: date(2026, 8, 26),
        )

        async def say(text: str, turn: str):
            delta = tracker.apply_user_turn(text, source_turn=turn)
            return await coordinator.execute(
                transcript=text,
                delta=delta,
                tracker=tracker,
                correlation_id=turn,
                source_turn=turn,
                parent_token=machine.issue_operation(kind="response"),
            )

        await say("Ich bräuchte einen Orthopädentermin.", "t1")
        await say("Am 27. August.", "t2")
        _, unavailable = await say("Gegen 11 Uhr.", "t3")
        assert unavailable is not None
        assert unavailable.value["result_status"] == "UNAVAILABLE"

        _, outcome = await say(followup, "t4")
        if outcome is None:
            return None, [], set()
        if outcome.tool_name == "list_appointments":
            return "list_appointments", [], set()
        slots = outcome.value.get("slots", [])
        return (
            outcome.tool_name,
            [s["start_datetime"] for s in slots],
            {s.get("appointment_type") for s in slots},
        )

    return asyncio.run(scenario())


@pytest.mark.parametrize(
    "followup",
    (
        "Welche Termine wären noch frei gewesen?",
        "Was wäre sonst noch frei?",
        "Welche anderen Uhrzeiten gibt es?",
        "Welche anderen Orthopädie-Termine gibt es?",
    ),
)
def test_availability_followup_preserves_type_and_never_lists_all(
    tmp_path: Path, followup: str
) -> None:
    tool, slots, types = _issue_a_followup(tmp_path, followup)
    # Must be a grounded, type-preserving availability search — never a list dump.
    assert tool == "search_availability"
    assert slots, "a grounded Orthopädie alternative must be offered"
    # The search horizon spans a date with Friseur availability (2026-09-02),
    # so a leak would show. It must only ever return Orthopädie.
    assert types == {"Orthopädie"}
    assert "Friseur" not in types


@pytest.mark.parametrize(
    "utterance",
    (
        "Welche Termine habe ich?",
        "Welche Friseurtermine habe ich?",
    ),
)
def test_booked_list_question_lists_and_does_not_inherit_type(
    tmp_path: Path, utterance: str
) -> None:
    tool, _slots, _types = _issue_a_followup(tmp_path, utterance)
    # A booked-appointment question stays a list; it never becomes an Orthopädie
    # availability search.
    assert tool == "list_appointments"


def test_unrelated_question_does_not_trigger_appointment_search(
    tmp_path: Path,
) -> None:
    tool, _slots, _types = _issue_a_followup(
        tmp_path, "Was geht eigentlich mit meiner Versicherung?"
    )
    assert tool is None
