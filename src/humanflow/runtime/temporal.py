"""Deterministic German temporal resolution for transactional voice turns."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


DEFAULT_TIMEZONE = "Europe/Berlin"

_WEEKDAYS = {
    "montag": 0,
    "dienstag": 1,
    "mittwoch": 2,
    "donnerstag": 3,
    "freitag": 4,
    "samstag": 5,
    "sonntag": 6,
}
_MONTHS = {
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "maerz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}
_WEEK_COUNTS = {
    "einer": 1,
    "eine": 1,
    "einem": 1,
    "eins": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
}
_WEEKDAY_PATTERN = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.I)
_EXPLICIT_NUMERIC_DATE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])[./-](?P<month>0?[1-9]|1[0-2])"
    r"(?:[./-](?P<year>\d{2}|\d{4}))?\b"
)
_EXPLICIT_NAMED_DATE = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])\.?(?:\s+)"
    r"(?P<month>" + "|".join(_MONTHS) + r")"
    r"(?:\s+(?P<year>\d{4}))?\b",
    re.I,
)
_RELATIVE_DAY = re.compile(r"\b(übermorgen|uebermorgen|morgen)\b", re.I)
_IN_WEEKS = re.compile(
    r"\bin\s+(?P<count>\d+|einer|eine|einem|eins|zwei|drei|vier)\s+wochen?\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class TemporalResolution:
    raw_expression: str
    resolved_iso_date: str
    timezone: str
    resolution_rule: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "raw_expression": self.raw_expression,
            "resolved_iso_date": self.resolved_iso_date,
            "timezone": self.timezone,
            "resolution_rule": self.resolution_rule,
            "confidence": self.confidence,
        }


class GermanTemporalResolver:
    """Resolve German dates without delegating calendar arithmetic to an LLM.

    Product semantics are deliberately explicit:

    - ``nächsten Freitag`` means the next occurrence strictly after today.
    - ``nächste Woche Freitag`` means Friday in the next ISO calendar week.
    - ``Freitag in einer Woche`` is a one-week offset from this week's Friday.
    - In the validated phrase ``Freitag in zwei Wochen``, the current partial week
      counts as week one, so the result is also Friday in the next calendar week.
      Higher counts advance by ``count - 1`` calendar weeks.
    - ``übernächste Woche`` always advances by two calendar weeks.
    """

    def __init__(self, *, timezone: str = DEFAULT_TIMEZONE) -> None:
        self.timezone = timezone
        self._zone = ZoneInfo(timezone)

    def resolve(
        self,
        utterance: str,
        *,
        current_local_datetime: datetime,
        existing_appointment_state: Mapping[str, object] | None = None,
    ) -> TemporalResolution | None:
        if not utterance.strip():
            return None
        local_now = self._as_local(current_local_datetime)
        reference = local_now.date()
        existing_date = _existing_iso_date(existing_appointment_state)

        explicit = self._resolve_explicit(utterance, reference)
        if explicit is not None:
            resolved, raw, rule, confidence = explicit
            result = self._result(resolved, raw, rule, confidence, existing_date)
            _enforce_requested_weekday(utterance, result.resolved_iso_date)
            return result

        lowered = utterance.casefold()
        relative_days = list(_RELATIVE_DAY.finditer(lowered))
        weekdays = list(_WEEKDAY_PATTERN.finditer(lowered))
        if relative_days and (
            not weekdays or relative_days[-1].start() > weekdays[-1].start()
        ):
            match = relative_days[-1]
            days = 2 if match.group(1) in {"übermorgen", "uebermorgen"} else 1
            rule = "DAY_AFTER_TOMORROW_PLUS_2_DAYS" if days == 2 else "TOMORROW_PLUS_1_DAY"
            return self._result(
                reference + timedelta(days=days),
                utterance[match.start() : match.end()],
                rule,
                0.99,
                existing_date,
            )
        if not weekdays:
            return None

        weekday_match = weekdays[-1]
        target_weekday = _WEEKDAYS[weekday_match.group(1)]
        window_start = max(0, weekday_match.start() - 48)
        window_end = min(len(lowered), weekday_match.end() + 48)
        window = lowered[window_start:window_end]
        monday = reference - timedelta(days=reference.weekday())

        if "übernächste woche" in window or "uebernaechste woche" in window:
            resolved = monday + timedelta(days=14 + target_weekday)
            rule = "WEEK_AFTER_NEXT_CALENDAR_WEEK"
            confidence = 0.99
        elif "nächste woche" in window or "naechste woche" in window:
            resolved = monday + timedelta(days=7 + target_weekday)
            rule = "NEXT_ISO_CALENDAR_WEEK"
            confidence = 0.99
        else:
            in_weeks = _nearest_week_count(window, weekday_match.start() - window_start)
            if in_weeks is not None:
                count, raw_count = in_weeks
                week_offset = 1 if count <= 2 else count - 1
                resolved = monday + timedelta(days=7 * week_offset + target_weekday)
                rule = (
                    "ONE_WEEK_OFFSET_FROM_CURRENT_WEEKDAY"
                    if count == 1
                    else "NAMED_WEEK_COUNT_INCLUDES_CURRENT_PARTIAL_WEEK"
                )
                rule = f"{rule}:{raw_count}"
                confidence = 0.99
            elif any(
                marker in window
                for marker in ("diese woche", "diesen ", "diese ", "diesem ")
            ):
                resolved = monday + timedelta(days=target_weekday)
                if resolved < reference:
                    resolved += timedelta(days=7)
                    rule = "THIS_WEEKDAY_ROLL_FORWARD_IF_PAST"
                else:
                    rule = "THIS_ISO_WEEKDAY"
                confidence = 0.97
            else:
                days_ahead = (target_weekday - reference.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                resolved = reference + timedelta(days=days_ahead)
                rule = "NEXT_WEEKDAY_OCCURRENCE_STRICTLY_AFTER_TODAY"
                confidence = 0.96

        raw = _weekday_expression(utterance, weekday_match, window_start, window_end)
        result = self._result(resolved, raw, rule, confidence, existing_date)
        _enforce_requested_weekday(utterance, result.resolved_iso_date)
        return result

    def _as_local(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=self._zone)
        return value.astimezone(self._zone)

    def _resolve_explicit(
        self, utterance: str, reference: date
    ) -> tuple[date, str, str, float] | None:
        matches: list[tuple[int, re.Match[str], bool]] = [
            (match.start(), match, False)
            for match in _EXPLICIT_NUMERIC_DATE.finditer(utterance)
        ]
        matches.extend(
            (match.start(), match, True)
            for match in _EXPLICIT_NAMED_DATE.finditer(utterance)
        )
        if not matches:
            return None
        _, match, named = max(matches, key=lambda item: item[0])
        month = (
            _MONTHS[match.group("month").casefold()]
            if named
            else int(match.group("month"))
        )
        year_text = match.group("year")
        if year_text is not None:
            year = int(year_text)
            if year < 100:
                year += 2_000
            try:
                resolved = date(year, month, int(match.group("day")))
            except ValueError:
                return None
            rule = "EXPLICIT_CALENDAR_DATE_WITH_YEAR"
        else:
            resolved = _next_valid_annual_date(
                month=month,
                day=int(match.group("day")),
                reference=reference,
            )
            if resolved is None:
                return None
            rule = "EXPLICIT_CALENDAR_DATE_ROLL_FORWARD_YEAR_IF_PAST"
        return resolved, match.group(0), rule, 1.0

    def _result(
        self,
        resolved: date,
        raw: str,
        rule: str,
        confidence: float,
        existing_date: str | None,
    ) -> TemporalResolution:
        if existing_date is not None and existing_date != resolved.isoformat():
            rule = f"{rule}|CORRECTION_FROM_EXISTING_APPOINTMENT_DATE"
        return TemporalResolution(
            raw_expression=" ".join(raw.split()),
            resolved_iso_date=resolved.isoformat(),
            timezone=self.timezone,
            resolution_rule=rule,
            confidence=confidence,
        )


def _nearest_week_count(window: str, weekday_position: int) -> tuple[int, str] | None:
    matches = list(_IN_WEEKS.finditer(window))
    if not matches:
        return None
    match = min(matches, key=lambda item: abs(item.start() - weekday_position))
    raw = match.group("count").casefold()
    count = int(raw) if raw.isdigit() else _WEEK_COUNTS[raw]
    if count < 1:
        return None
    return count, raw


def requested_weekday_matches_date(utterance: str, resolved_iso_date: str) -> bool:
    """Check an explicitly spoken weekday against the authoritative ISO date."""

    weekdays = list(_WEEKDAY_PATTERN.finditer(utterance.casefold()))
    if not weekdays:
        return True
    try:
        resolved = date.fromisoformat(resolved_iso_date)
    except ValueError:
        return False
    return resolved.weekday() == _WEEKDAYS[weekdays[-1].group(1)]


def _enforce_requested_weekday(utterance: str, resolved_iso_date: str) -> None:
    if not requested_weekday_matches_date(utterance, resolved_iso_date):
        raise ValueError("resolved date contradicts explicitly requested weekday")


def _weekday_expression(
    utterance: str,
    weekday_match: re.Match[str],
    window_start: int,
    window_end: int,
) -> str:
    del weekday_match
    return utterance[window_start:window_end].strip(" ,.;:!?")


def _existing_iso_date(state: Mapping[str, object] | None) -> str | None:
    if state is None:
        return None
    raw = state.get("date")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        value = raw.get("value")
        return value if isinstance(value, str) else None
    return None


def _next_valid_annual_date(*, month: int, day: int, reference: date) -> date | None:
    for year in range(reference.year, reference.year + 9):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= reference:
            return candidate
    return None
