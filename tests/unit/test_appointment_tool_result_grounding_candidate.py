"""Grounding regression tests for HF-APPT-20260826-01.

Failure: appointment-tool-result-grounding-sticky-unavailable.

After an unavailable requested time, a natural German availability follow-up that
carries no new slot must still issue a grounded SQLite ``search_availability``
against the focused appointment's date context (with the rejected preferred time
cleared) instead of returning no tool call and letting the reasoner answer
without a ``database_tool_result``. Affirmations and corrections must not be
reclassified as availability queries, and an unavailable exact requested time
must never become a bookable pending offer.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState
from humanflow.runtime.appointment_state import AppointmentStateTracker
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.tools.appointment_coordinator import (
    _AVAILABILITY_FOLLOWUP,
    AppointmentTransactionCoordinator,
)
from humanflow.tools.sqlite_appointments import SQLiteAppointmentToolProvider

FIXED_TODAY = date(2026, 8, 26)
FIXED_NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def _machine(conversation_id: str) -> ConversationStateMachine:
    machine = ConversationStateMachine(
        conversation_id=conversation_id, sink=InMemoryTelemetrySink()
    )
    machine.transition(
        ConversationState.LISTENING, reason_code="test_started", correlation_id="setup"
    )
    machine.transition(
        ConversationState.THINKING, reason_code="turn_complete", correlation_id="setup"
    )
    return machine


class _Fixture:
    def __init__(self, tmp: str) -> None:
        self.provider = SQLiteAppointmentToolProvider(
            database_path=Path(tmp) / "demo.sqlite3"
        )
        self.conversation_id = "grounding-candidate"
        self.machine = _machine(self.conversation_id)
        self.tracker = AppointmentStateTracker(
            today=lambda: FIXED_TODAY, now=lambda: FIXED_NOW
        )
        self.coordinator = AppointmentTransactionCoordinator(
            conversation_id=self.conversation_id,
            state_machine=self.machine,
            provider=self.provider,
            today=lambda: FIXED_TODAY,
        )
        self._turn = 0

    async def say(self, text: str):
        self._turn += 1
        source_turn = f"turn-{self._turn}"
        delta = self.tracker.apply_user_turn(text, source_turn=source_turn)
        _, outcome = await self.coordinator.execute(
            transcript=text,
            delta=delta,
            tracker=self.tracker,
            correlation_id=source_turn,
            source_turn=source_turn,
            parent_token=self.machine.issue_operation(kind="response"),
        )
        self.machine.transition(
            ConversationState.LISTENING, reason_code="t", correlation_id=source_turn
        )
        self.machine.transition(
            ConversationState.THINKING, reason_code="t", correlation_id=source_turn
        )
        return outcome


def _run(coro):
    return asyncio.run(coro)


async def _unavailable_then(followups: list[str]):
    with tempfile.TemporaryDirectory() as tmp:
        fx = _Fixture(tmp)
        # "Donnerstag" from the fixed Wednesday 2026-08-26 resolves to 2026-08-27,
        # which has demo Orthopädie slots at 09:00/12:00/15:30 but not 11:30.
        first = await fx.say(
            "Ich möchte einen Orthopädietermin am Donnerstag um 11:30 Uhr."
        )
        assert first is not None and first.tool_name == "search_availability"
        assert first.value.get("result_status") == "UNAVAILABLE"
        return [await fx.say(text) for text in followups]


def test_natural_availability_followup_is_grounded_after_unavailable() -> None:
    outcomes = _run(
        _unavailable_then(
            [
                "Und was hättest du sonst so an dem Tag?",
                "Wann hast du denn frei?",
            ]
        )
    )
    for outcome in outcomes:
        assert outcome is not None, "follow-up produced no grounded tool call"
        assert outcome.tool_name == "search_availability"
        # Grounded in the SQLite result, and the rejected preferred time is gone.
        assert outcome.value.get("result_status") == "AVAILABLE"
        assert outcome.value.get("requested_time") is None
        starts = [slot["start_datetime"][11:16] for slot in outcome.value["slots"]]
        assert starts, "grounded availability returned no slots"
        # The rejected preferred time is gone; the real focused-day slots remain.
        assert "11:30" not in starts
        assert {"09:00", "12:00", "15:30"}.issubset(set(starts))
        # Every grounded slot belongs to the single focused appointment day.
        days = {slot["start_datetime"][:10] for slot in outcome.value["slots"]}
        assert days == {"2026-08-27"}


def test_affirmation_and_correction_are_not_availability_searches() -> None:
    # A bare confirmation and a bare correction must not be reclassified as an
    # availability query just because an appointment is focused.
    with tempfile.TemporaryDirectory() as tmp:
        fx = _Fixture(tmp)
        first = _run(
            fx.say("Ich möchte einen Orthopädietermin am Donnerstag um 11:30 Uhr.")
        )
        assert first.value.get("result_status") == "UNAVAILABLE"
        assert _run(fx.say("Ja, super.")) is None
        assert _run(fx.say("Nein, warte.")) is None


def test_unavailable_exact_time_is_not_a_bookable_offer() -> None:
    # The unavailable 11:30 request must not become a pending offer that a later
    # affirmation can silently book.
    with tempfile.TemporaryDirectory() as tmp:
        fx = _Fixture(tmp)
        first = _run(
            fx.say("Ich möchte einen Orthopädietermin am Donnerstag um 11:30 Uhr.")
        )
        assert first.value.get("result_status") == "UNAVAILABLE"
        # A follow-up affirmation must never create the appointment from an
        # unavailable exact time.
        booked = _run(fx.say("Ja, passt."))
        assert booked is None or booked.tool_name != "create_appointment"


def test_availability_followup_regex_positive_and_negative() -> None:
    # Direct coverage of the narrowing so unrelated utterances that merely contain
    # "frei"/"verfügbar"/"was geht" can never be reclassified as an availability
    # follow-up. Decision: bare colloquial "Was geht?" is NOT an availability query.
    positives = [
        "Und was hättest du sonst so?",
        "Und was hättest du sonst so an dem Tag?",
        "Wann hast du denn frei?",
        "Wann habt ihr frei?",
        "Hast du sonst noch etwas frei?",
        "Habt ihr noch einen Termin frei?",
        "Welche Zeiten hast du?",
        "Gibt es noch was frei am Donnerstag?",
        "Hast du andere Zeiten frei?",
    ]
    negatives = [
        "Ich bin morgen frei.",
        "Was geht eigentlich mit meiner Versicherung?",
        "Ist der Parkplatz verfügbar?",
        "Was geht?",
        "Was hast du gesagt?",
        "Was hast du gemeint?",
        "Ja, das passt gut.",
        "Nein, lieber Dienstag.",
        "Wie ist das Wetter?",
        "Ich bin da flexibel und verfügbar.",
        # Weekday names must not match via the "frei" anchor.
        "Am Freitag um 11:30 Uhr bitte.",
        "Lieber Freitag.",
    ]
    for text in positives:
        assert _AVAILABILITY_FOLLOWUP.search(text) is not None, text
    for text in negatives:
        assert _AVAILABILITY_FOLLOWUP.search(text) is None, text


def test_unrelated_slotless_utterances_do_not_trigger_search() -> None:
    # Integration guard: while an unbooked dated appointment is focused, a slotless
    # unrelated utterance containing a broad token must yield no tool call, so the
    # reasoner is never handed a spurious availability result. ("Ich bin morgen
    # frei." is intentionally excluded here: it retains the focused date/time and is
    # re-grounded by the pre-existing focus search path, not by this follow-up
    # logic, and is out of scope for HF-APPT-20260826-01.)
    for utterance in (
        "Was geht eigentlich mit meiner Versicherung?",
        "Ist der Parkplatz verfügbar?",
        "Was geht?",
    ):
        with tempfile.TemporaryDirectory() as tmp:
            fx = _Fixture(tmp)
            first = _run(
                fx.say("Ich möchte einen Orthopädietermin am Donnerstag um 11:30 Uhr.")
            )
            assert first.value.get("result_status") == "UNAVAILABLE"
            assert _run(fx.say(utterance)) is None, utterance
