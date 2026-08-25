from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState
from humanflow.runtime.appointment_state import AppointmentStateTracker
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.tools.appointment_coordinator import AppointmentTransactionCoordinator
from humanflow.tools.sqlite_appointments import SQLiteAppointmentToolProvider


FIXED_TODAY = date(2026, 8, 24)
FIXED_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _tracker() -> AppointmentStateTracker:
    return AppointmentStateTracker(today=lambda: FIXED_TODAY, now=lambda: FIXED_NOW)


def _machine(conversation_id: str) -> ConversationStateMachine:
    machine = ConversationStateMachine(
        conversation_id=conversation_id, sink=InMemoryTelemetrySink()
    )
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
    return machine


async def _turn(
    *,
    text: str,
    turn: int,
    tracker: AppointmentStateTracker,
    coordinator: AppointmentTransactionCoordinator,
    machine: ConversationStateMachine,
):
    source_turn = f"turn-{turn}"
    delta = tracker.apply_user_turn(text, source_turn=source_turn)
    return await coordinator.execute(
        transcript=text,
        delta=delta,
        tracker=tracker,
        correlation_id=source_turn,
        source_turn=source_turn,
        parent_token=machine.issue_operation(kind="response"),
    )


async def _book_orthopedics(
    *,
    tracker: AppointmentStateTracker,
    coordinator: AppointmentTransactionCoordinator,
    machine: ConversationStateMachine,
) -> str:
    _, search = await _turn(
        text="Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
        turn=1,
        tracker=tracker,
        coordinator=coordinator,
        machine=machine,
    )
    assert search is not None and search.value["result_status"] == "AVAILABLE"
    booked_delta, booked = await _turn(
        text="Ja, bitte.",
        turn=2,
        tracker=tracker,
        coordinator=coordinator,
        machine=machine,
    )
    assert booked is not None and booked.value["result_status"] == "BOOKED"
    assert booked_delta.appointment_id == "appointment_1"
    return str(booked.value["appointment_id"])


def test_failed_reschedule_restores_committed_sqlite_slot_and_booked_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine = _machine("failed-reschedule")
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="failed-reschedule",
            state_machine=machine,
            provider=provider,
        )
        database_id = await _book_orthopedics(
            tracker=tracker, coordinator=coordinator, machine=machine
        )

        delta, outcome = await _turn(
            text="Lieber um 12 Uhr.",
            turn=3,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        row = provider.list_appointments({})["appointments"][0]

        assert outcome is not None
        assert outcome.value["result_status"] == "BOOKING_CONFLICT"
        assert row["appointment_id"] == database_id
        assert row["start_datetime"] == "2026-09-03T10:30:00+02:00"
        assert delta.state["date"]["value"] == "2026-09-03"
        assert delta.state["time"]["value"] == "10:30"
        assert delta.state["status"]["value"] == "BOOKED"

    asyncio.run(scenario())


def test_failed_cancel_preserves_booked_state_and_database_row(tmp_path: Path) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine = _machine("failed-cancel")
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="failed-cancel",
            state_machine=machine,
            provider=provider,
        )
        database_id = await _book_orthopedics(
            tracker=tracker, coordinator=coordinator, machine=machine
        )
        provider.failure_tool = "cancel_appointment"

        delta, outcome = await _turn(
            text="Den Orthopädietermin bitte absagen.",
            turn=3,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        rows = provider.list_appointments({})["appointments"]

        assert outcome is not None
        assert outcome.value["result_status"] == "TECHNICAL_FAILURE"
        assert [row["appointment_id"] for row in rows] == [database_id]
        assert delta.state["status"]["value"] == "BOOKED"

    asyncio.run(scenario())


def test_correction_without_new_slot_invalidates_offer_before_later_yes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        tracker = _tracker()
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        machine = _machine("offer-correction")
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="offer-correction",
            state_machine=machine,
            provider=provider,
        )
        _, offered = await _turn(
            text="Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.",
            turn=1,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        assert offered is not None and offered.value["result_status"] == "AVAILABLE"

        _, correction = await _turn(
            text="Passt, aber später.",
            turn=2,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        _, later_yes = await _turn(
            text="Ja.",
            turn=3,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )

        assert correction is None
        assert later_yes is None
        assert provider.list_appointments({})["appointments"] == []

    asyncio.run(scenario())


def test_list_after_restart_hydrates_ids_for_same_row_reschedule(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = SQLiteAppointmentToolProvider(tmp_path / "appointments.sqlite3")
        created = provider.create_appointment(
            {
                "appointment_id": "persisted-orthopedics",
                "appointment_type": "Orthopädie",
                "start_datetime": "2026-09-03T10:30:00+02:00",
            }
        )
        assert created["result_status"] == "BOOKED"
        tracker = _tracker()
        machine = _machine("restart-hydration")
        coordinator = AppointmentTransactionCoordinator(
            conversation_id="restart-hydration",
            state_machine=machine,
            provider=provider,
        )

        _, listed = await _turn(
            text="Welche Termine habe ich?",
            turn=1,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        assert listed is not None and listed.tool_name == "list_appointments"
        assert len(tracker.appointments) == 1

        _, moved = await _turn(
            text="Den Orthopädietermin lieber um 14 Uhr.",
            turn=2,
            tracker=tracker,
            coordinator=coordinator,
            machine=machine,
        )
        rows = provider.list_appointments({})["appointments"]

        assert moved is not None and moved.value["result_status"] == "RESCHEDULED"
        assert len(rows) == 1
        assert rows[0]["appointment_id"] == "persisted-orthopedics"
        assert rows[0]["start_datetime"] == "2026-09-03T14:00:00+02:00"

    asyncio.run(scenario())


def test_multiple_explicit_appointment_actions_require_clarification() -> None:
    tracker = _tracker()
    tracker.apply_user_turn(
        "Orthopädentermin nächste Woche Donnerstag um 10:30 Uhr.", source_turn="turn-1"
    )
    tracker.apply_user_turn(
        "Noch ein Friseurtermin nächste Woche Montag um 11 Uhr.", source_turn="turn-2"
    )
    before = {
        appointment_id: state.to_dict() for appointment_id, state in tracker.appointments.items()
    }

    delta = tracker.apply_user_turn(
        "Buche den Orthopäden und verschieb den Friseur.", source_turn="turn-3"
    )

    assert delta.clarification_required is True
    assert delta.resolution_reason == "multiple_appointment_actions_require_separation"
    assert {
        appointment_id: state.to_dict() for appointment_id, state in tracker.appointments.items()
    } == before
