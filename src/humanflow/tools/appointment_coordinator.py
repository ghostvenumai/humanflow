"""Deterministic bridge from conversational appointment intent to SQLite tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import OperationToken
from humanflow.runtime.appointment_state import (
    AppointmentActionState,
    AppointmentState,
    AppointmentStateDelta,
    AppointmentStateTracker,
)
from humanflow.runtime.temporal import requested_weekday_matches_date
from humanflow.telemetry.events import EventType

from .executor import ResilientToolExecutor
from .providers import ToolProvider


_AVAILABILITY_INTENT = re.compile(
    r"\b(?:welche\s+termine\s+sind\s+frei|was\s+ist\s+frei|"
    r"wann\s+(?:habt|hättet)\s+ihr\s+etwas\s+frei|"
    r"welche\s+zeiten\s+(?:wären|sind)\s+verfügbar|"
    r"freie\s+termine|verfügbare\s+termine)\b",
    re.IGNORECASE,
)
_BOOKED_LIST_INTENT = re.compile(
    r"\b(?:welche\s+termine\s+habe\s+ich|was\s+habe\s+ich\s+gebucht|"
    r"zeig(?:e)?\s+mir\s+meine\s+termine|meine\s+termine|alle\s+termine)\b",
    re.IGNORECASE,
)
_BOOKING_CONFIRMATION = re.compile(
    r"^(?:ja(?:[\s,]+(?:bitte(?:\s+buchen)?|perfekt|super|alles\s+ist\s+super|machen\s+wir|"
    r"nehmen\s+wir))?|super|perfekt|passt|machen\s+wir|nehmen\s+wir|okay|ok|"
    r"bitte\s+buchen|buch(?:e)?(?:n\s+wir)?)$",
    re.IGNORECASE,
)
_BOOKING_CORRECTION = re.compile(
    r"\b(?:nein|nicht|lieber|stattdessen|warte|moment|doch\s+nicht)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AppointmentTransactionOutcome:
    tool_name: str
    value: Mapping[str, Any]
    elapsed_ms: float
    success: bool
    failure_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", MappingProxyType(dict(self.value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "elapsed_ms": self.elapsed_ms,
            "result": dict(self.value),
        }


class AppointmentTransactionCoordinator:
    """Choose only the five approved tools from controller-owned state deltas."""

    def __init__(
        self,
        *,
        conversation_id: str,
        state_machine: ConversationStateMachine,
        provider: ToolProvider,
        timeout_ms: float = 4_000.0,
        timezone: str = "Europe/Berlin",
        today: Callable[[], date] | None = None,
    ) -> None:
        self._conversation_id = conversation_id
        self._state_machine = state_machine
        self._executor = ResilientToolExecutor(
            state_machine=state_machine,
            provider=provider,
            timeout_ms=timeout_ms,
            max_attempts=1,
        )
        self._timezone = ZoneInfo(timezone)
        self._today = today or (lambda: datetime.now(self._timezone).date())
        self._persisted_ids: dict[str, str] = {}
        self._pending_creates: dict[str, dict[str, Any]] = {}

    async def execute(
        self,
        *,
        transcript: str,
        delta: AppointmentStateDelta,
        tracker: AppointmentStateTracker,
        correlation_id: str,
        source_turn: str,
        parent_token: OperationToken,
    ) -> tuple[AppointmentStateDelta, AppointmentTransactionOutcome | None]:
        plan = self._plan(transcript=transcript, delta=delta, tracker=tracker)
        if plan is None:
            return delta, None
        tool_name, arguments, local_appointment_id = plan
        result = await self._executor.execute(
            tool_name,
            arguments,
            fallback=lambda name, _: {
                "success": False,
                "tool": name,
                "result_status": "TECHNICAL_FAILURE",
                "error_code": "TOOL_UNAVAILABLE",
            },
            correlation_id=correlation_id,
        )
        value = dict(result.value)
        success = value.get("success") is True and not result.used_fallback
        outcome = AppointmentTransactionOutcome(
            tool_name=tool_name,
            value=value,
            elapsed_ms=result.elapsed_ms,
            success=success,
            failure_reason=result.failure_reason,
        )
        if not self._state_machine.accept_result(
            parent_token, correlation_id=correlation_id
        ):
            return delta, outcome
        semantic_success = success
        if tool_name == "search_availability":
            semantic_success = value.get("result_status") == "AVAILABLE"
            self._remember_availability(
                local_appointment_id=local_appointment_id,
                arguments=arguments,
                value=value,
                tracker=tracker,
            )
        elif tool_name == "create_appointment" and local_appointment_id is not None:
            self._pending_creates.pop(local_appointment_id, None)
            if success:
                self._persisted_ids[local_appointment_id] = str(
                    value["appointment_id"]
                )
        should_record_action_state = (
            local_appointment_id is not None
            and tool_name != "list_appointments"
            and not (
                tool_name == "search_availability"
                and local_appointment_id in self._persisted_ids
            )
        )
        if should_record_action_state:
            action = {
                "search_availability": "search",
                "create_appointment": "book",
                "reschedule_appointment": "reschedule",
                "cancel_appointment": "cancel",
            }[tool_name]
            delta = tracker.record_tool_result(
                local_appointment_id,
                action=action,
                success=semantic_success,
                source_turn=f"tool:{source_turn}",
            )
        self._record_domain_event(
            outcome=outcome,
            correlation_id=correlation_id,
            appointment_id=value.get("appointment_id"),
        )
        return delta, outcome

    def enrich_reasoning_context(
        self,
        context: str | None,
        outcome: AppointmentTransactionOutcome | None,
    ) -> str | None:
        if outcome is None:
            return context
        try:
            payload = json.loads(context) if context else {}
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["database_tool_result"] = outcome.to_dict()
        payload["database_state_is_authoritative"] = True
        payload["external_action_performed"] = bool(
            outcome.success
            and outcome.tool_name
            in {"create_appointment", "reschedule_appointment", "cancel_appointment"}
        )
        if outcome.tool_name == "list_appointments":
            payload["clarification_required"] = False
            contract = payload.get("response_contract")
            if isinstance(contract, dict):
                contract["clarify_instead_of_guessing"] = False
        payload["database_response_policy"] = (
            "Use only database_tool_result values for availability, booked appointments, "
            "rescheduling and cancellation. Never invent a slot or claim success when "
            "database_tool_result.success is false. Keep the spoken reply to one or two "
            "short German sentences."
        )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _plan(
        self,
        *,
        transcript: str,
        delta: AppointmentStateDelta,
        tracker: AppointmentStateTracker,
    ) -> tuple[str, dict[str, Any], str | None] | None:
        candidate_id = delta.appointment_id or tracker.active_focus_appointment_id
        candidate = (
            tracker.appointments.get(candidate_id) if candidate_id is not None else None
        )
        candidate_date = _slot(candidate, "date") if candidate is not None else None
        if candidate_date and not requested_weekday_matches_date(
            transcript, candidate_date
        ):
            raise RuntimeError("temporal_weekday_date_invariant_violated")
        if _AVAILABILITY_INTENT.search(transcript):
            return self._availability_plan(delta=delta, tracker=tracker)
        if (
            delta.resolution_reason == "multi_appointment_overview"
            or _BOOKED_LIST_INTENT.search(transcript)
        ):
            return "list_appointments", {"include_cancelled": False}, None
        focused_id = tracker.active_focus_appointment_id
        if (
            focused_id is not None
            and focused_id in self._pending_creates
            and _confirms_booking(transcript)
        ):
            return (
                "create_appointment",
                dict(self._pending_creates[focused_id]),
                focused_id,
            )
        if delta.clarification_required or delta.appointment_id is None:
            return None
        appointment = tracker.appointments[delta.appointment_id]
        persisted_id = self._persisted_ids.get(delta.appointment_id)
        status = _slot(appointment, "status")
        if status == AppointmentActionState.CANCELLED.value:
            return None
        if status == AppointmentActionState.TOOL_PENDING.value:
            if persisted_id is None:
                return None
            return (
                "cancel_appointment",
                {"appointment_id": persisted_id},
                delta.appointment_id,
            )
        appointment_type = _appointment_type(appointment)
        date_value = _slot(appointment, "date")
        time_value = _slot(appointment, "time")
        if persisted_id is not None and {"date", "time"}.intersection(delta.updated_slots):
            if date_value and time_value:
                return (
                    "reschedule_appointment",
                    {
                        "appointment_id": persisted_id,
                        "start_datetime": self._local_datetime(date_value, time_value),
                    },
                    delta.appointment_id,
                )
        if persisted_id is None and date_value and time_value and appointment_type:
            return (
                "search_availability",
                {
                    "appointment_type": appointment_type,
                    "provider_name": _slot(appointment, "provider"),
                    "location": _slot(appointment, "location"),
                    "date": date_value,
                    "preferred_time": time_value,
                },
                delta.appointment_id,
            )
        if date_value and not time_value and appointment_type:
            return (
                "search_availability",
                {
                    "appointment_type": appointment_type,
                    "provider_name": _slot(appointment, "provider"),
                    "location": _slot(appointment, "location"),
                    "date": date_value,
                },
                delta.appointment_id,
            )
        return None

    def _availability_plan(
        self,
        *,
        delta: AppointmentStateDelta,
        tracker: AppointmentStateTracker,
    ) -> tuple[str, dict[str, Any], str | None]:
        local_id = delta.appointment_id or tracker.active_focus_appointment_id
        appointment = (
            tracker.appointments.get(local_id) if local_id is not None else None
        )
        date_value = _slot(appointment, "date") if appointment is not None else None
        start_date = date_value or self._today().isoformat()
        end_date = date_value or (self._today() + timedelta(days=21)).isoformat()
        arguments: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
        }
        if appointment is not None:
            optional = {
                "appointment_type": _appointment_type(appointment),
                "provider_name": _slot(appointment, "provider"),
                "location": _slot(appointment, "location"),
                "preferred_time": (
                    _slot(appointment, "time")
                    if "time" in delta.updated_slots
                    else None
                ),
            }
            arguments.update({name: value for name, value in optional.items() if value})
        return "search_availability", arguments, local_id

    def _remember_availability(
        self,
        *,
        local_appointment_id: str | None,
        arguments: Mapping[str, Any],
        value: Mapping[str, Any],
        tracker: AppointmentStateTracker,
    ) -> None:
        if local_appointment_id is None:
            return
        if local_appointment_id in self._persisted_ids:
            return
        exact_date = _requested_exact_date(arguments)
        slots = [
            slot
            for slot in value.get("slots", ())
            if isinstance(slot, Mapping)
            and isinstance(slot.get("start_datetime"), str)
            and _slot_matches_exact_date(slot, exact_date)
        ]
        if value.get("result_status") != "AVAILABLE" or len(slots) != 1:
            self._pending_creates.pop(local_appointment_id, None)
            return
        appointment = tracker.appointments[local_appointment_id]
        start_datetime = slots[0]["start_datetime"]
        appointment_type = _appointment_type(appointment)
        if appointment_type is None:
            return
        self._pending_creates[local_appointment_id] = {
            "appointment_id": self._database_appointment_id(local_appointment_id),
            "appointment_type": appointment_type,
            "provider_name": _slot(appointment, "provider"),
            "location": _slot(appointment, "location"),
            "start_datetime": start_datetime,
        }

    def _database_appointment_id(self, local_appointment_id: str) -> str:
        stable = uuid5(
            NAMESPACE_URL,
            f"humanflow:{self._conversation_id}:{local_appointment_id}",
        )
        return f"appointment-{stable}"

    def _local_datetime(self, iso_date: str, iso_time: str) -> str:
        value = datetime.combine(
            date.fromisoformat(iso_date),
            time.fromisoformat(iso_time),
            tzinfo=self._timezone,
        )
        return value.isoformat(timespec="seconds")

    def _record_domain_event(
        self,
        *,
        outcome: AppointmentTransactionOutcome,
        correlation_id: str,
        appointment_id: object,
    ) -> None:
        if not outcome.success and outcome.value.get("error_code") != "BOOKING_CONFLICT":
            return
        event_type = {
            "search_availability": EventType.AVAILABILITY_QUERIED,
            "create_appointment": (
                EventType.APPOINTMENT_CREATED
                if outcome.success
                else EventType.BOOKING_CONFLICT_DETECTED
            ),
            "reschedule_appointment": (
                EventType.APPOINTMENT_RESCHEDULED
                if outcome.success
                else EventType.BOOKING_CONFLICT_DETECTED
            ),
            "cancel_appointment": EventType.APPOINTMENT_CANCELLED,
            "list_appointments": EventType.APPOINTMENTS_LISTED,
        }[outcome.tool_name]
        self._state_machine.record(
            event_type,
            correlation_id=correlation_id,
            reason_code=("transaction_committed" if outcome.success else "transaction_not_committed"),
            payload={
                "tool_name": outcome.tool_name,
                "appointment_id": appointment_id,
                "duration_ms": outcome.elapsed_ms,
                "success": outcome.success,
                "error_class": outcome.failure_reason,
                "transaction_result": {
                    "status": outcome.value.get("status"),
                    "result_status": outcome.value.get("result_status"),
                    "error_code": outcome.value.get("error_code"),
                    "result_count": len(outcome.value.get("appointments", ()))
                    if isinstance(outcome.value.get("appointments"), list)
                    else len(outcome.value.get("slots", ()))
                    if isinstance(outcome.value.get("slots"), list)
                    else None,
                },
            },
        )


def _slot(appointment: AppointmentState, name: str) -> str | None:
    value = getattr(appointment, name)
    return None if value is None else value.value


def _appointment_type(appointment: AppointmentState) -> str | None:
    purpose = _slot(appointment, "purpose")
    if purpose and purpose != "Termin":
        return purpose
    return _slot(appointment, "specialty")


def _confirms_booking(transcript: str) -> bool:
    normalized = " ".join(transcript.casefold().strip().rstrip(".!?").split())
    return (
        _BOOKING_CORRECTION.search(normalized) is None
        and _BOOKING_CONFIRMATION.fullmatch(normalized) is not None
    )


def _slot_matches_exact_date(slot: Mapping[str, Any], requested_date: object) -> bool:
    if not isinstance(requested_date, str):
        return True
    return str(slot["start_datetime"])[:10] == requested_date


def _requested_exact_date(arguments: Mapping[str, Any]) -> object:
    explicit = arguments.get("date")
    if isinstance(explicit, str):
        return explicit
    start = arguments.get("start_date")
    end = arguments.get("end_date")
    return start if isinstance(start, str) and start == end else None
