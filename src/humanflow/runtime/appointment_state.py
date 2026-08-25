"""Authoritative multi-object state for appointment conversations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Callable
from zoneinfo import ZoneInfo

from .temporal import DEFAULT_TIMEZONE, GermanTemporalResolver, TemporalResolution


_CLOCK_PATTERN = re.compile(
    r"\b(?:(?:gegen|um|ab)\s+)?(?P<hour>[01]?\d|2[0-3])"
    r"(?:(?P<separator>[:.])(?P<minute>[0-5]\d))?\s*(?P<uhr>uhr)\b",
    re.IGNORECASE,
)
_COLON_CLOCK_PATTERN = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\b"
)
_APPOINTMENT_WORD = re.compile(
    r"\b(\w*termin\w*|sprechstunde|behandlung|verabredung)\b", re.IGNORECASE
)
_NEW_APPOINTMENT = re.compile(
    r"\b(?:noch(?:\s+(?:einen|eine|n|nen))?|weiter(?:e|en)?|zweite[snr]?)\b",
    re.IGNORECASE,
)
_GENERIC_PRONOUN = re.compile(
    r"\b(ihn|den\s+termin|der\s+termin|dieser\s+termin|"
    r"den(?=\s+(?:will|möchte)\s+ich\s+absagen))\b",
    re.IGNORECASE,
)
_PLURAL_OVERVIEW = re.compile(r"\b(welche|meine|alle)\s+termine\b", re.IGNORECASE)
_AVAILABILITY_QUERY = re.compile(
    r"\b(?:welche\s+termine\s+sind\s+frei|was\s+ist\s+frei|"
    r"wann\s+(?:habt|hättet)\s+ihr\s+etwas\s+frei|"
    r"welche\s+zeiten\s+(?:wären|sind)\s+verfügbar|"
    r"freie\s+termine|verfügbare\s+termine)\b",
    re.IGNORECASE,
)
_CANCEL_REQUEST = re.compile(
    r"\b(absagen|stornieren|canceln|brauche\s+ich\s+nicht\s+mehr|"
    r"nicht\s+mehr\s+brauch|will\s+(?:ihn|den)\s+absagen)\b",
    re.IGNORECASE,
)
_BOOKING_CONFIRMATION = re.compile(
    r"\b(?:ja|okay|ok|passt|bestätige|buchen|buch(?:e)?|nimm)\b",
    re.IGNORECASE,
)


class AppointmentActionState(StrEnum):
    PROPOSED = "PROPOSED"
    COLLECTING_DETAILS = "COLLECTING_DETAILS"
    READY_TO_CHECK = "READY_TO_CHECK"
    CHECKING_AVAILABILITY = "CHECKING_AVAILABILITY"
    AVAILABLE = "AVAILABLE"
    READY_TO_BOOK = "READY_TO_BOOK"
    TOOL_PENDING = "TOOL_PENDING"
    BOOKED = "BOOKED"
    RESCHEDULED = "RESCHEDULED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class SlotValue:
    """One transaction value plus immutable user-turn provenance."""

    value: str
    source_turn: str
    updated_at: str
    confidence: float
    raw_value: str
    raw_expression: str | None = None
    resolved_iso_date: str | None = None
    timezone: str | None = None
    resolution_rule: str | None = None

    def to_dict(self) -> dict[str, str | float]:
        payload: dict[str, str | float] = {
            "value": self.value,
            "source_turn": self.source_turn,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "raw_value": self.raw_value,
        }
        temporal = {
            "raw_expression": self.raw_expression,
            "resolved_iso_date": self.resolved_iso_date,
            "timezone": self.timezone,
            "resolution_rule": self.resolution_rule,
        }
        payload.update({name: value for name, value in temporal.items() if value is not None})
        return payload


@dataclass(slots=True)
class AppointmentState:
    appointment_id: str
    purpose: SlotValue | None = None
    specialty: SlotValue | None = None
    date: SlotValue | None = None
    time: SlotValue | None = None
    location: SlotValue | None = None
    provider: SlotValue | None = None
    status: SlotValue | None = None
    source_turn: str = ""
    updated_at: str = ""
    confidence: float = 1.0
    external_action_confirmed: bool = False

    @property
    def active(self) -> bool:
        return self.status is None or self.status.value != AppointmentActionState.CANCELLED

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "appointment_id": self.appointment_id,
            "source_turn": self.source_turn,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "external_action_confirmed": self.external_action_confirmed,
            "active": self.active,
        }
        excluded = {
            "appointment_id",
            "source_turn",
            "updated_at",
            "confidence",
            "external_action_confirmed",
        }
        for field in fields(self):
            if field.name in excluded:
                continue
            value = getattr(self, field.name)
            payload[field.name] = None if value is None else value.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AppointmentStateDelta:
    source_turn: str
    appointment_id: str | None
    updated_slots: tuple[str, ...]
    state: dict[str, object]
    appointments: dict[str, dict[str, object]]
    active_focus_appointment_id: str | None
    created: bool = False
    clarification_required: bool = False
    clarification_options: tuple[str, ...] = ()
    resolution_reason: str = "no_appointment_reference"
    external_action_performed: bool = False

    @property
    def changed(self) -> bool:
        return bool(
            self.created
            or self.updated_slots
            or self.clarification_required
            or self.appointment_id is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_turn": self.source_turn,
            "appointment_id": self.appointment_id,
            "updated_slots": list(self.updated_slots),
            "state": self.state,
            "appointments": self.appointments,
            "active_focus_appointment_id": self.active_focus_appointment_id,
            "created": self.created,
            "clarification_required": self.clarification_required,
            "clarification_options": list(self.clarification_options),
            "resolution_reason": self.resolution_reason,
            "external_action_performed": self.external_action_performed,
        }


@dataclass(frozen=True, slots=True)
class _EntityReference:
    purpose: str
    specialty: str | None
    raw: str
    position: int


@dataclass(frozen=True, slots=True)
class _Resolution:
    appointment_id: str | None
    reason: str
    clarification_options: tuple[str, ...] = ()
    create_entity: _EntityReference | None = None


class AppointmentStateTracker:
    """Maintain independent appointment objects using user turns as the only source."""

    def __init__(
        self,
        *,
        today: Callable[[], date] | None = None,
        now: Callable[[], datetime] | None = None,
        current_local_datetime: Callable[[], datetime] | None = None,
        timezone: str = DEFAULT_TIMEZONE,
        temporal_resolver: GermanTemporalResolver | None = None,
    ) -> None:
        self._appointments: dict[str, AppointmentState] = {}
        self.active_focus_appointment_id: str | None = None
        self._next_id = 1
        self._turn_index = 0
        self._last_explicit_focus_turn: int | None = None
        self._timezone = ZoneInfo(timezone)
        if current_local_datetime is not None:
            self._current_local_datetime = current_local_datetime
        elif today is not None:
            self._current_local_datetime = lambda: datetime.combine(
                today(), time(12), tzinfo=self._timezone
            )
        else:
            self._current_local_datetime = lambda: datetime.now(self._timezone)
        self._temporal_resolver = temporal_resolver or GermanTemporalResolver(
            timezone=timezone
        )
        self._now = now or (lambda: datetime.now(UTC))

    @property
    def appointments(self) -> dict[str, AppointmentState]:
        return dict(self._appointments)

    @property
    def state(self) -> AppointmentState:
        """Focused object compatibility view for existing single-object callers."""

        focused = self.focused_appointment
        return focused or AppointmentState(appointment_id="unassigned")

    @property
    def focused_appointment(self) -> AppointmentState | None:
        if self.active_focus_appointment_id is None:
            return None
        return self._appointments.get(self.active_focus_appointment_id)

    def apply_user_turn(
        self,
        text: str,
        *,
        source_turn: str,
    ) -> AppointmentStateDelta:
        if not text.strip() or not source_turn.strip():
            raise ValueError("text and source_turn must not be empty")
        self._turn_index += 1
        normalized = " ".join(text.casefold().split())
        references = _extract_entity_references(text)
        explicitly_about_appointment = bool(_APPOINTMENT_WORD.search(normalized))
        parsed_date = self._temporal_resolver.resolve(
            text,
            current_local_datetime=self._current_local_datetime(),
            existing_appointment_state=(
                self.focused_appointment.to_dict()
                if self.focused_appointment is not None
                else None
            ),
        )
        parsed_time = _extract_time(text)
        has_slot_delta = parsed_date is not None or parsed_time is not None

        resolution = self._resolve(
            normalized=normalized,
            references=references,
            explicitly_about_appointment=explicitly_about_appointment,
            has_slot_delta=has_slot_delta,
        )
        if resolution.clarification_options:
            self.active_focus_appointment_id = None
            return self._delta(
                source_turn=source_turn,
                appointment_id=None,
                updated_slots=(),
                clarification_required=True,
                clarification_options=resolution.clarification_options,
                resolution_reason=resolution.reason,
            )

        created = False
        appointment_id = resolution.appointment_id
        timestamp = self._timestamp()
        if resolution.create_entity is not None:
            appointment_id = f"appointment_{self._next_id}"
            self._next_id += 1
            appointment = AppointmentState(
                appointment_id=appointment_id,
                source_turn=source_turn,
                updated_at=timestamp,
                confidence=0.99,
            )
            self._appointments[appointment_id] = appointment
            created = True
        elif appointment_id is None:
            return self._delta(
                source_turn=source_turn,
                appointment_id=None,
                updated_slots=(),
                resolution_reason=resolution.reason,
            )

        appointment = self._appointments[appointment_id]
        self.active_focus_appointment_id = appointment_id
        parsed_date = self._temporal_resolver.resolve(
            text,
            current_local_datetime=self._current_local_datetime(),
            existing_appointment_state=appointment.to_dict(),
        )
        if references:
            self._last_explicit_focus_turn = self._turn_index

        candidates: dict[str, tuple[str, str, float]] = {}
        if parsed_date is not None:
            candidates["date"] = (
                parsed_date.resolved_iso_date,
                parsed_date.raw_expression,
                parsed_date.confidence,
            )
        if parsed_time is not None:
            candidates["time"] = parsed_time
        selected_entity = resolution.create_entity or (references[-1] if references else None)
        if selected_entity is not None and selected_entity.purpose != "Termin":
            candidates["purpose"] = (
                selected_entity.purpose,
                selected_entity.raw,
                0.99,
            )
            if selected_entity.specialty is not None:
                candidates["specialty"] = (
                    selected_entity.specialty,
                    selected_entity.raw,
                    0.99,
                )
        generic_purpose = _extract_generic_purpose(normalized)
        if generic_purpose is not None and "purpose" not in candidates:
            candidates["purpose"] = (generic_purpose, generic_purpose, 0.91)
        location = _extract_location(text)
        if location is not None:
            candidates["location"] = location
        provider = _extract_provider(text)
        if provider is not None:
            candidates["provider"] = provider

        changed: list[str] = []
        for slot_name, (value, raw_value, confidence) in candidates.items():
            if self._set_slot(
                appointment,
                slot_name,
                value=value,
                raw_value=raw_value,
                confidence=confidence,
                source_turn=source_turn,
                timestamp=timestamp,
                temporal_resolution=(parsed_date if slot_name == "date" else None),
            ):
                changed.append(slot_name)

        cancellation = bool(_CANCEL_REQUEST.search(normalized))
        desired_status = self._derive_status(appointment, cancellation=cancellation)
        if self._set_slot(
            appointment,
            "status",
            value=desired_status.value,
            raw_value=(
                "local_appointment_request_cancelled"
                if desired_status is AppointmentActionState.CANCELLED
                else "controller_derived_no_external_action"
            ),
            confidence=1.0,
            source_turn=source_turn,
            timestamp=timestamp,
        ):
            changed.append("status")

        appointment.source_turn = source_turn
        appointment.updated_at = timestamp
        appointment.confidence = min(
            (getattr(appointment, name).confidence for name in changed),
            default=appointment.confidence,
        )
        return self._delta(
            source_turn=source_turn,
            appointment_id=appointment_id,
            updated_slots=tuple(changed),
            created=created,
            resolution_reason=resolution.reason,
        )

    def record_tool_result(
        self,
        appointment_id: str,
        *,
        action: str,
        success: bool,
        source_turn: str,
    ) -> AppointmentStateDelta:
        """The only path allowed to claim an external booking/cancellation result."""

        appointment = self._appointments[appointment_id]
        timestamp = self._timestamp()
        normalized_action = action.casefold().strip()
        if normalized_action not in {"search", "book", "reschedule", "cancel"}:
            raise ValueError("tool action must be search, book, reschedule or cancel")
        if not success:
            status = AppointmentActionState.FAILED
        elif normalized_action == "search":
            status = AppointmentActionState.AVAILABLE
        elif normalized_action == "book":
            status = AppointmentActionState.BOOKED
        elif normalized_action == "reschedule":
            status = AppointmentActionState.BOOKED
        elif normalized_action == "cancel":
            status = AppointmentActionState.CANCELLED
        appointment.external_action_confirmed = bool(
            success and normalized_action in {"book", "reschedule", "cancel"}
        )
        self._set_slot(
            appointment,
            "status",
            value=status.value,
            raw_value=f"tool_{normalized_action}_{'success' if success else 'failed'}",
            confidence=1.0,
            source_turn=source_turn,
            timestamp=timestamp,
        )
        appointment.updated_at = timestamp
        return self._delta(
            source_turn=source_turn,
            appointment_id=appointment_id,
            updated_slots=("status",),
            resolution_reason="explicit_tool_result",
            external_action_performed=True,
        )

    def reasoning_context(self, delta: AppointmentStateDelta) -> str | None:
        if not self._appointments and not delta.clarification_required:
            return None
        compact = {
            appointment_id: {
                name: slot["value"]
                for name, slot in appointment.to_dict().items()
                if isinstance(slot, dict) and "value" in slot
            }
            for appointment_id, appointment in self._appointments.items()
        }
        resolved_state = (
            compact.get(delta.appointment_id, {})
            if delta.appointment_id is not None
            else {}
        )
        known_slots = tuple(
            name
            for name in ("purpose", "specialty", "date", "time", "location", "provider")
            if resolved_state.get(name)
        )
        missing_slots = tuple(
            name
            for name in ("purpose", "date", "time", "location", "provider")
            if not resolved_state.get(name)
        )
        payload = {
            "appointments": compact,
            "active_focus_appointment_id": self.active_focus_appointment_id,
            "resolved_appointment_id_this_turn": delta.appointment_id,
            "updated_slots_this_user_turn": list(delta.updated_slots),
            "clarification_required": delta.clarification_required,
            "clarification_options": list(delta.clarification_options),
            "external_action_performed": delta.external_action_performed,
            "temporal_resolution": _temporal_provenance(delta.state.get("date")),
            "response_contract": {
                "must_use_word": "Termin",
                "must_acknowledge_updated_values": {
                    name: resolved_state[name]
                    for name in delta.updated_slots
                    if name in resolved_state and name != "status"
                },
                "known_slots_never_ask_again": list(known_slots),
                "ask_at_most_one_of_missing_slots": list(missing_slots),
                "clarify_instead_of_guessing": delta.clarification_required,
                "forbidden_without_tool_success": [
                    "Termin gebucht",
                    "Termin eingetragen",
                    "Termin storniert",
                    "Termin abgesagt",
                    "Termin gelöscht",
                ],
                "required_local_action_noun": "Terminwunsch",
            },
            "action_truthfulness": (
                "BOOKED requires successful real tool result. Without tool success say "
                "Terminwunsch notiert, never booked. Local CANCELLED means only the "
                "unbooked request was removed; never claim external cancellation."
            ),
            "state_policy": (
                "AUTHORITATIVE_MULTI_APPOINTMENT_DELTA_STATE; mutate only resolved "
                "appointment_id; unchanged objects and slots remain unchanged; never "
                "reconstruct transaction state from assistant history"
            ),
            "temporal_policy": (
                "DETERMINISTIC_CONTROLLER_RESOLUTION; use only resolved ISO date and "
                "temporal_resolution; never calculate relative dates in the LLM"
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _resolve(
        self,
        *,
        normalized: str,
        references: tuple[_EntityReference, ...],
        explicitly_about_appointment: bool,
        has_slot_delta: bool,
    ) -> _Resolution:
        if _AVAILABILITY_QUERY.search(normalized) and not references and not has_slot_delta:
            return _Resolution(
                self.active_focus_appointment_id,
                "availability_query",
            )
        if (
            _PLURAL_OVERVIEW.search(normalized)
            and not _AVAILABILITY_QUERY.search(normalized)
            and not references
        ):
            options = tuple(
                appointment_id
                for appointment_id, appointment in self._appointments.items()
                if appointment.active
            )
            self.active_focus_appointment_id = None
            return _Resolution(None, "multi_appointment_overview", options)

        if references:
            reference = references[-1]
            matching = self._matching_entity(reference)
            force_new = bool(_NEW_APPOINTMENT.search(normalized))
            if matching and not force_new:
                return _Resolution(matching[-1], "explicit_entity_reference")
            if explicitly_about_appointment:
                return _Resolution(
                    None,
                    "new_explicit_appointment_entity",
                    create_entity=reference,
                )

        active_ids = tuple(
            appointment_id
            for appointment_id, appointment in self._appointments.items()
            if appointment.active
        )
        if re.search(r"\b(?:den|der)\s+anderen\b", normalized):
            alternatives = tuple(
                appointment_id
                for appointment_id in active_ids
                if appointment_id != self.active_focus_appointment_id
            )
            if len(alternatives) == 1:
                return _Resolution(alternatives[0], "explicit_other_reference")
            return _Resolution(None, "ambiguous_other_reference", alternatives or active_ids)

        pronoun = bool(_GENERIC_PRONOUN.search(normalized))
        if pronoun and len(active_ids) > 1:
            if (
                self.active_focus_appointment_id in active_ids
                and self._last_explicit_focus_turn == self._turn_index - 1
            ):
                return _Resolution(
                    self.active_focus_appointment_id,
                    "pronoun_resolved_from_immediately_prior_explicit_focus",
                )
            return _Resolution(None, "ambiguous_pronoun", active_ids)

        wants_new = bool(_NEW_APPOINTMENT.search(normalized)) and explicitly_about_appointment
        if not self._appointments and (explicitly_about_appointment or has_slot_delta):
            return _Resolution(
                None,
                "new_initial_appointment",
                create_entity=_EntityReference("Termin", None, "Termin", 0),
            )
        if wants_new:
            return _Resolution(
                None,
                "new_additional_generic_appointment",
                create_entity=_EntityReference("Termin", None, "Termin", 0),
            )
        if (
            self.active_focus_appointment_id is not None
            and "nein" not in normalized
            and _BOOKING_CONFIRMATION.search(normalized)
        ):
            return _Resolution(
                self.active_focus_appointment_id,
                "booking_confirmation_for_active_focus",
            )
        if self.active_focus_appointment_id is not None and (
            explicitly_about_appointment or has_slot_delta or pronoun
        ):
            return _Resolution(self.active_focus_appointment_id, "active_focus_delta")
        if len(active_ids) == 1 and (explicitly_about_appointment or has_slot_delta):
            return _Resolution(active_ids[0], "single_active_appointment")
        return _Resolution(None, "no_appointment_reference")

    def _matching_entity(self, reference: _EntityReference) -> list[str]:
        matches: list[str] = []
        for appointment_id, appointment in self._appointments.items():
            values = {
                appointment.purpose.value.casefold() if appointment.purpose else "",
                appointment.specialty.value.casefold() if appointment.specialty else "",
                appointment.provider.value.casefold() if appointment.provider else "",
            }
            if reference.purpose.casefold() in values or (
                reference.specialty is not None
                and reference.specialty.casefold() in values
            ):
                matches.append(appointment_id)
        return matches

    @staticmethod
    def _derive_status(
        appointment: AppointmentState, *, cancellation: bool
    ) -> AppointmentActionState:
        current = appointment.status.value if appointment.status else None
        if cancellation:
            if current == AppointmentActionState.BOOKED:
                return AppointmentActionState.TOOL_PENDING
            return AppointmentActionState.CANCELLED
        if current in {
            AppointmentActionState.BOOKED,
            AppointmentActionState.CANCELLED,
            AppointmentActionState.TOOL_PENDING,
        }:
            return AppointmentActionState(current)
        if appointment.date is not None and appointment.time is not None:
            return AppointmentActionState.READY_TO_BOOK
        if any(
            slot is not None
            for slot in (appointment.purpose, appointment.date, appointment.time)
        ):
            return AppointmentActionState.COLLECTING_DETAILS
        return AppointmentActionState.PROPOSED

    @staticmethod
    def _set_slot(
        appointment: AppointmentState,
        slot_name: str,
        *,
        value: str,
        raw_value: str,
        confidence: float,
        source_turn: str,
        timestamp: str,
        temporal_resolution: TemporalResolution | None = None,
    ) -> bool:
        current = getattr(appointment, slot_name)
        if current is not None and current.value == value:
            return False
        setattr(
            appointment,
            slot_name,
            SlotValue(
                value=value,
                source_turn=source_turn,
                updated_at=timestamp,
                confidence=confidence,
                raw_value=raw_value,
                raw_expression=(
                    temporal_resolution.raw_expression
                    if temporal_resolution is not None
                    else None
                ),
                resolved_iso_date=(
                    temporal_resolution.resolved_iso_date
                    if temporal_resolution is not None
                    else None
                ),
                timezone=(
                    temporal_resolution.timezone
                    if temporal_resolution is not None
                    else None
                ),
                resolution_rule=(
                    temporal_resolution.resolution_rule
                    if temporal_resolution is not None
                    else None
                ),
            ),
        )
        return True

    def _delta(
        self,
        *,
        source_turn: str,
        appointment_id: str | None,
        updated_slots: tuple[str, ...],
        created: bool = False,
        clarification_required: bool = False,
        clarification_options: tuple[str, ...] = (),
        resolution_reason: str,
        external_action_performed: bool = False,
    ) -> AppointmentStateDelta:
        state = (
            self._appointments[appointment_id].to_dict()
            if appointment_id is not None
            else {}
        )
        return AppointmentStateDelta(
            source_turn=source_turn,
            appointment_id=appointment_id,
            updated_slots=updated_slots,
            state=state,
            appointments={
                key: appointment.to_dict()
                for key, appointment in self._appointments.items()
            },
            active_focus_appointment_id=self.active_focus_appointment_id,
            created=created,
            clarification_required=clarification_required,
            clarification_options=clarification_options,
            resolution_reason=resolution_reason,
            external_action_performed=external_action_performed,
        )

    def _timestamp(self) -> str:
        return self._now().astimezone(UTC).isoformat().replace("+00:00", "Z")


def _extract_entity_references(text: str) -> tuple[_EntityReference, ...]:
    lowered = text.casefold()
    patterns = (
        (r"orthopäd\w*", "Orthopädie", "Orthopädie"),
        (r"friseur\w*", "Friseur", None),
        (r"zahn(?:arzt|ärzt)\w*", "Zahnarzt", "Zahnmedizin"),
        (r"hausarzt\w*", "Hausarzt", "Allgemeinmedizin"),
        (r"hautärzt\w*", "Hautarzt", "Dermatologie"),
        (r"augenarzt\w*", "Augenarzt", "Augenheilkunde"),
        (r"geschäftstermin\w*", "Geschäftstermin", None),
    )
    references: list[_EntityReference] = []
    for pattern, purpose, specialty in patterns:
        for match in re.finditer(pattern, lowered, re.IGNORECASE):
            references.append(
                _EntityReference(purpose, specialty, match.group(0), match.start())
            )
    return tuple(sorted(references, key=lambda item: item.position))


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


def _extract_generic_purpose(normalized: str) -> str | None:
    if "arzttermin" in normalized or "behandlung" in normalized:
        return "Arzttermin"
    return None


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


def _temporal_provenance(raw_date: object) -> dict[str, object] | None:
    if not isinstance(raw_date, dict):
        return None
    names = ("raw_expression", "resolved_iso_date", "timezone", "resolution_rule")
    payload = {
        name: raw_date.get(name)
        for name in names
        if raw_date.get(name) is not None
    }
    return payload or None
