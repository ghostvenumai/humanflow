"""Authoritative delta-based state for appointment conversations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo


_WEEKDAYS = {
    "montag": 0,
    "dienstag": 1,
    "mittwoch": 2,
    "donnerstag": 3,
    "freitag": 4,
    "samstag": 5,
    "sonntag": 6,
}
_WEEKDAY_PATTERN = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_CLOCK_PATTERN = re.compile(
    r"\b(?:(?:gegen|um|ab)\s+)?(?P<hour>[01]?\d|2[0-3])"
    r"(?:(?P<separator>[:.])(?P<minute>[0-5]\d))?\s*(?P<uhr>uhr)\b",
    re.IGNORECASE,
)
_COLON_CLOCK_PATTERN = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\b"
)
_EXPLICIT_DATE_PATTERN = re.compile(
    r"\b(?P<day>0?[1-9]|[12]\d|3[01])[./-](?P<month>0?[1-9]|1[0-2])"
    r"(?:[./-](?P<year>\d{2}|\d{4}))?\b"
)


@dataclass(frozen=True, slots=True)
class SlotValue:
    """One transaction value plus immutable user-turn provenance."""

    value: str
    source_turn: str
    updated_at: str
    confidence: float
    raw_value: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "value": self.value,
            "source_turn": self.source_turn,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "raw_value": self.raw_value,
        }


@dataclass(slots=True)
class AppointmentState:
    purpose: SlotValue | None = None
    specialty: SlotValue | None = None
    date: SlotValue | None = None
    time: SlotValue | None = None
    location: SlotValue | None = None
    provider: SlotValue | None = None
    status: SlotValue | None = None

    @property
    def active(self) -> bool:
        return any(getattr(self, field.name) is not None for field in fields(self))

    def to_dict(self) -> dict[str, dict[str, str | float] | None]:
        return {
            field.name: (
                None
                if (value := getattr(self, field.name)) is None
                else value.to_dict()
            )
            for field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class AppointmentStateDelta:
    source_turn: str
    updated_slots: tuple[str, ...]
    state: dict[str, dict[str, str | float] | None]

    @property
    def changed(self) -> bool:
        return bool(self.updated_slots)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_turn": self.source_turn,
            "updated_slots": list(self.updated_slots),
            "state": self.state,
        }


class AppointmentStateTracker:
    """Apply only explicitly stated user deltas to persistent appointment slots.

    Assistant text and free-form model history are deliberately not inputs. Relative
    dates are resolved once when the user states them and then remain stable.
    """

    def __init__(
        self,
        *,
        today: Callable[[], date] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.state = AppointmentState()
        self._today = today or (
            lambda: datetime.now(ZoneInfo("Europe/Berlin")).date()
        )
        self._now = now or (lambda: datetime.now(UTC))

    def apply_user_turn(
        self,
        text: str,
        *,
        source_turn: str,
    ) -> AppointmentStateDelta:
        if not text.strip() or not source_turn.strip():
            raise ValueError("text and source_turn must not be empty")
        normalized = " ".join(text.casefold().split())
        explicitly_about_appointment = bool(
            re.search(
                r"\b(termin|arzttermin|geschäftstermin|sprechstunde|behandlung)\w*\b",
                normalized,
            )
        )
        if not self.state.active and not explicitly_about_appointment:
            return AppointmentStateDelta(source_turn, (), self.state.to_dict())

        candidates: dict[str, tuple[str, str, float]] = {}
        parsed_date = _extract_date(text, reference=self._today())
        if parsed_date is not None:
            candidates["date"] = parsed_date
        parsed_time = _extract_time(text)
        if parsed_time is not None:
            candidates["time"] = parsed_time

        purpose = _extract_purpose(normalized)
        if purpose is not None:
            candidates["purpose"] = (purpose, purpose, 0.91)
        specialty = _extract_specialty(normalized)
        if specialty is not None:
            candidates["specialty"] = (specialty, specialty, 0.91)
        location = _extract_location(text)
        if location is not None:
            candidates["location"] = location
        provider = _extract_provider(text)
        if provider is not None:
            candidates["provider"] = provider
        status = _extract_status(normalized, explicitly_about_appointment)
        if status is not None:
            candidates["status"] = (status, status, 0.96)

        timestamp = self._now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        changed: list[str] = []
        for slot_name, (value, raw_value, confidence) in candidates.items():
            current = getattr(self.state, slot_name)
            if current is not None and current.value == value:
                continue
            setattr(
                self.state,
                slot_name,
                SlotValue(
                    value=value,
                    source_turn=source_turn,
                    updated_at=timestamp,
                    confidence=confidence,
                    raw_value=raw_value,
                ),
            )
            changed.append(slot_name)
        return AppointmentStateDelta(
            source_turn=source_turn,
            updated_slots=tuple(changed),
            state=self.state.to_dict(),
        )

    def reasoning_context(self, delta: AppointmentStateDelta) -> str | None:
        if not self.state.active:
            return None
        compact_state = {
            name: slot["value"]
            for name, slot in self.state.to_dict().items()
            if slot is not None
        }
        payload = {
            "appointment_state": compact_state,
            "updated_slots_this_user_turn": list(delta.updated_slots),
            "state_policy": (
                "AUTHORITATIVE_DELTA_STATE; unchanged slots must remain unchanged; "
                "never resurrect older values from chat history"
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _extract_date(text: str, *, reference: date) -> tuple[str, str, float] | None:
    explicit_matches = list(_EXPLICIT_DATE_PATTERN.finditer(text))
    if explicit_matches:
        match = explicit_matches[-1]
        year_text = match.group("year")
        year = reference.year if year_text is None else int(year_text)
        if year < 100:
            year += 2_000
        try:
            resolved = date(year, int(match.group("month")), int(match.group("day")))
        except ValueError:
            return None
        if year_text is None and resolved < reference:
            resolved = resolved.replace(year=resolved.year + 1)
        return resolved.isoformat(), match.group(0), 0.99

    lowered = text.casefold()
    relative_matches = list(re.finditer(r"\b(übermorgen|morgen)\b", lowered))
    weekday_matches = list(_WEEKDAY_PATTERN.finditer(lowered))
    if relative_matches and (
        not weekday_matches or relative_matches[-1].start() > weekday_matches[-1].start()
    ):
        match = relative_matches[-1]
        days = 2 if match.group(1) == "übermorgen" else 1
        return (reference + timedelta(days=days)).isoformat(), match.group(0), 0.98
    if not weekday_matches:
        return None

    match = weekday_matches[-1]
    weekday = _WEEKDAYS[match.group(1)]
    prefix = lowered[max(0, match.start() - 38) : match.start()]
    if "übernächste woche" in prefix or "uebernaechste woche" in prefix:
        monday = reference - timedelta(days=reference.weekday())
        resolved = monday + timedelta(days=14 + weekday)
        confidence = 0.99
    elif "nächste woche" in prefix or "naechste woche" in prefix:
        monday = reference - timedelta(days=reference.weekday())
        resolved = monday + timedelta(days=7 + weekday)
        confidence = 0.99
    elif "diese woche" in prefix or "diesen" in prefix or "diese" in prefix:
        monday = reference - timedelta(days=reference.weekday())
        resolved = monday + timedelta(days=weekday)
        if resolved < reference:
            resolved += timedelta(days=7)
        confidence = 0.95
    else:
        days_ahead = (weekday - reference.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        resolved = reference + timedelta(days=days_ahead)
        confidence = 0.94
    return resolved.isoformat(), match.group(0), confidence


def _extract_time(text: str) -> tuple[str, str, float] | None:
    matches = list(_CLOCK_PATTERN.finditer(text))
    if not matches:
        matches = list(_COLON_CLOCK_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    return f"{hour:02d}:{minute:02d}", match.group(0), 0.99


def _extract_purpose(normalized: str) -> str | None:
    if "arzttermin" in normalized or "behandlung" in normalized:
        return "medical"
    if "geschäftstermin" in normalized:
        return "business"
    return None


def _extract_specialty(normalized: str) -> str | None:
    specialties = {
        "zahnarzt": "dentistry",
        "hausarzt": "general_medicine",
        "hautärzt": "dermatology",
        "augenarzt": "ophthalmology",
        "orthopäd": "orthopedics",
    }
    return next((value for key, value in specialties.items() if key in normalized), None)


def _extract_location(text: str) -> tuple[str, str, float] | None:
    matches = list(
        re.finditer(
            r"\b(?:in|am standort)\s+([A-ZÄÖÜ][\wÄÖÜäöüß-]*(?:\s+[A-ZÄÖÜ][\wÄÖÜäöüß-]*)?)",
            text,
        )
    )
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    return raw, raw, 0.86


def _extract_provider(text: str) -> tuple[str, str, float] | None:
    matches = list(
        re.finditer(
            r"\bbei\s+((?:Dr\.?\s+)?[A-ZÄÖÜ][\wÄÖÜäöüß-]+(?:\s+[A-ZÄÖÜ][\wÄÖÜäöüß-]+)?)",
            text,
        )
    )
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    return raw, raw, 0.86


def _extract_status(normalized: str, explicitly_about_appointment: bool) -> str | None:
    if re.search(r"\b(stornier|absag|cancel)", normalized):
        return "cancelled"
    if re.search(r"\b(gebucht|verbindlich bestätig)", normalized):
        return "booked"
    if explicitly_about_appointment:
        return "collecting"
    return None
