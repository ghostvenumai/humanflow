from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from humanflow.runtime.appointment_state import AppointmentStateTracker
from humanflow.runtime.temporal import (
    GermanTemporalResolver,
    requested_weekday_matches_date,
)


BERLIN = ZoneInfo("Europe/Berlin")
MONDAY = datetime(2026, 8, 24, 10, 15, tzinfo=BERLIN)


@pytest.mark.parametrize(
    ("utterance", "expected", "rule_prefix"),
    (
        ("diesen Freitag", "2026-08-28", "THIS_ISO_WEEKDAY"),
        (
            "nächsten Freitag",
            "2026-08-28",
            "NEXT_WEEKDAY_OCCURRENCE_STRICTLY_AFTER_TODAY",
        ),
        ("nächste Woche Freitag", "2026-09-04", "NEXT_ISO_CALENDAR_WEEK"),
        (
            "Freitag in einer Woche",
            "2026-09-04",
            "ONE_WEEK_OFFSET_FROM_CURRENT_WEEKDAY",
        ),
        (
            "Freitag in zwei Wochen",
            "2026-09-04",
            "NAMED_WEEK_COUNT_INCLUDES_CURRENT_PARTIAL_WEEK",
        ),
        (
            "Donnerstag in zwei Wochen",
            "2026-09-03",
            "NAMED_WEEK_COUNT_INCLUDES_CURRENT_PARTIAL_WEEK",
        ),
        (
            "übernächste Woche Donnerstag",
            "2026-09-10",
            "WEEK_AFTER_NEXT_CALENDAR_WEEK",
        ),
        ("morgen", "2026-08-25", "TOMORROW_PLUS_1_DAY"),
        ("übermorgen", "2026-08-26", "DAY_AFTER_TOMORROW_PLUS_2_DAYS"),
        (
            "Freitag",
            "2026-08-28",
            "NEXT_WEEKDAY_OCCURRENCE_STRICTLY_AFTER_TODAY",
        ),
    ),
)
def test_required_berlin_relative_date_semantics(
    utterance: str, expected: str, rule_prefix: str
) -> None:
    result = GermanTemporalResolver().resolve(
        utterance,
        current_local_datetime=MONDAY,
        existing_appointment_state=None,
    )

    assert result is not None
    assert result.resolved_iso_date == expected
    assert result.timezone == "Europe/Berlin"
    assert result.resolution_rule.startswith(rule_prefix)
    assert result.raw_expression


def test_week_month_and_year_boundaries_are_calendar_based() -> None:
    resolver = GermanTemporalResolver()
    year_boundary = datetime(2025, 12, 29, 9, tzinfo=BERLIN)
    sunday_boundary = datetime(2026, 8, 30, 9, tzinfo=BERLIN)

    friday = resolver.resolve(
        "Freitag in zwei Wochen", current_local_datetime=year_boundary
    )
    monday = resolver.resolve(
        "nächste Woche Montag", current_local_datetime=sunday_boundary
    )
    after_tomorrow = resolver.resolve(
        "übermorgen", current_local_datetime=datetime(2025, 12, 31, 23, tzinfo=BERLIN)
    )

    assert friday is not None and friday.resolved_iso_date == "2026-01-09"
    assert monday is not None and monday.resolved_iso_date == "2026-08-31"
    assert after_tomorrow is not None
    assert after_tomorrow.resolved_iso_date == "2026-01-02"


def test_leap_year_and_explicit_month_name_resolution() -> None:
    resolver = GermanTemporalResolver()

    leap_day = resolver.resolve(
        "am 29. Februar", current_local_datetime=datetime(2023, 3, 1, 9, tzinfo=BERLIN)
    )
    leap_tomorrow = resolver.resolve(
        "morgen", current_local_datetime=datetime(2024, 2, 28, 9, tzinfo=BERLIN)
    )

    assert leap_day is not None and leap_day.resolved_iso_date == "2024-02-29"
    assert leap_tomorrow is not None
    assert leap_tomorrow.resolved_iso_date == "2024-02-29"


def test_relative_correction_records_existing_state_provenance() -> None:
    resolver = GermanTemporalResolver()

    result = resolver.resolve(
        "Nein, Donnerstag in zwei Wochen.",
        current_local_datetime=MONDAY,
        existing_appointment_state={"date": {"value": "2026-08-28"}},
    )

    assert result is not None
    assert result.resolved_iso_date == "2026-09-03"
    assert result.resolution_rule.endswith("CORRECTION_FROM_EXISTING_APPOINTMENT_DATE")


def test_authoritative_appointment_date_stores_full_temporal_provenance() -> None:
    tracker = AppointmentStateTracker(current_local_datetime=lambda: MONDAY)

    tracker.apply_user_turn(
        "Orthopädentermin Freitag in zwei Wochen um 14 Uhr.", source_turn="turn-1"
    )
    correction = tracker.apply_user_turn(
        "Nein, Donnerstag in zwei Wochen.", source_turn="turn-2"
    )

    date_slot = tracker.appointments["appointment_1"].date
    assert date_slot is not None
    assert date_slot.value == "2026-09-03"
    assert date_slot.resolved_iso_date == "2026-09-03"
    assert date_slot.timezone == "Europe/Berlin"
    assert date_slot.raw_expression
    assert date_slot.resolution_rule is not None
    assert "CORRECTION_FROM_EXISTING_APPOINTMENT_DATE" in date_slot.resolution_rule
    context = tracker.reasoning_context(correction)
    assert context is not None
    assert '"resolved_iso_date": "2026-09-03"' in context
    assert "never calculate relative dates in the LLM" in context


@pytest.mark.parametrize(
    ("utterance", "expected"),
    (
        ("Mittwoch in zwei Wochen", "2026-09-02"),
        ("Mittwoch in drei Wochen", "2026-09-09"),
    ),
)
def test_spoken_weekday_is_invariant_for_relative_week_resolution(
    utterance: str, expected: str
) -> None:
    reference = datetime(2026, 8, 25, 10, 0, tzinfo=BERLIN)

    result = GermanTemporalResolver().resolve(
        utterance,
        current_local_datetime=reference,
    )

    assert result is not None
    assert result.resolved_iso_date == expected
    assert datetime.fromisoformat(result.resolved_iso_date).strftime("%A") == "Wednesday"
    assert requested_weekday_matches_date(utterance, result.resolved_iso_date)


@pytest.mark.parametrize(
    "utterance",
    (
        "Mittwoch in der Woche",
        "irgendwann Mittwoch",
        "nächste Zeit Mittwoch",
        "so um Mittwoch rum",
    ),
)
def test_ambiguous_weekday_requires_clarification_without_mutating_date(
    utterance: str,
) -> None:
    reference = datetime(2026, 8, 25, 10, 0, tzinfo=BERLIN)
    resolver = GermanTemporalResolver()

    assert resolver.resolve(utterance, current_local_datetime=reference) is None
    candidate = resolver.clarification_candidate(
        utterance, current_local_datetime=reference
    )

    assert candidate is not None
    assert candidate.resolved_iso_date == "2026-08-26"
    assert candidate.confidence < 0.5
    assert candidate.resolution_rule == "AMBIGUOUS_WEEKDAY_REQUIRES_CLARIFICATION"


def test_clear_weekday_expressions_do_not_require_clarification() -> None:
    reference = datetime(2026, 8, 25, 10, 0, tzinfo=BERLIN)
    resolver = GermanTemporalResolver()

    for utterance in (
        "diesen Mittwoch",
        "nächsten Mittwoch",
        "Mittwoch in zwei Wochen",
        "übernächste Woche Donnerstag",
    ):
        assert (
            resolver.clarification_candidate(
                utterance, current_local_datetime=reference
            )
            is None
        )
        assert resolver.resolve(utterance, current_local_datetime=reference) is not None


def test_explicit_weekday_date_contradiction_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="resolved date contradicts explicitly requested weekday"
    ):
        GermanTemporalResolver().resolve(
            "Mittwoch, 10. September 2026",
            current_local_datetime=datetime(2026, 8, 25, 10, 0, tzinfo=BERLIN),
        )
