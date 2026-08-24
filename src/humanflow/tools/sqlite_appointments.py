"""Persistent SQLite appointment tools for the post-freeze application demo."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .models import ToolProviderResponse
from .providers import ToolProviderError


_OPERATION_GUARD = "_humanflow_operation_is_current"
_SUPPORTED_TOOLS = frozenset(
    {
        "search_availability",
        "create_appointment",
        "reschedule_appointment",
        "cancel_appointment",
        "list_appointments",
    }
)


@dataclass(frozen=True, slots=True)
class DemoAvailability:
    resource_id: str
    start_datetime: str
    end_datetime: str


_DEMO_PROVIDERS = (
    ("demo-ortho-1", "Orthopädie", "Praxis am Stadtpark (Demo)", "Berlin", 1),
    ("demo-friseur-1", "Friseur", "Salon Morgenrot (Demo)", "Berlin", 1),
)

_DEMO_AVAILABILITY = (
    DemoAvailability("demo-ortho-1", "2026-09-03T10:30:00+02:00", "2026-09-03T11:00:00+02:00"),
    DemoAvailability("demo-ortho-1", "2026-09-03T14:00:00+02:00", "2026-09-03T14:30:00+02:00"),
    DemoAvailability("demo-ortho-1", "2026-09-04T09:30:00+02:00", "2026-09-04T10:00:00+02:00"),
    DemoAvailability("demo-ortho-1", "2026-09-10T15:00:00+02:00", "2026-09-10T15:30:00+02:00"),
    DemoAvailability("demo-friseur-1", "2026-08-31T11:00:00+02:00", "2026-08-31T11:30:00+02:00"),
    DemoAvailability("demo-friseur-1", "2026-09-02T14:00:00+02:00", "2026-09-02T14:30:00+02:00"),
    DemoAvailability("demo-friseur-1", "2026-09-03T16:00:00+02:00", "2026-09-03T16:30:00+02:00"),
)


@dataclass(slots=True)
class SQLiteAppointmentToolProvider:
    """Five transactional tools backed by atomic, local SQLite writes."""

    database_path: Path
    delay_ms: float = 0.0
    failure_tool: str | None = None
    call_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.database_path = Path(self.database_path)
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if self.failure_tool not in {None, "*", *_SUPPORTED_TOOLS}:
            raise ValueError("unsupported failure tool")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS providers (
                    resource_id TEXT PRIMARY KEY,
                    appointment_type TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1))
                );
                CREATE TABLE IF NOT EXISTS availability (
                    availability_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_id TEXT NOT NULL REFERENCES providers(resource_id),
                    start_datetime TEXT NOT NULL,
                    end_datetime TEXT NOT NULL,
                    UNIQUE(resource_id, start_datetime, end_datetime)
                );
                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    resource_id TEXT NOT NULL REFERENCES providers(resource_id),
                    appointment_type TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    start_datetime TEXT NOT NULL,
                    end_datetime TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PROPOSED', 'BOOKED', 'CANCELLED')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_booked_appointment_per_slot
                ON appointments(resource_id, start_datetime, end_datetime)
                WHERE status = 'BOOKED';
                """
            )
            connection.executemany(
                """INSERT OR IGNORE INTO providers
                   (resource_id, appointment_type, provider_name, location, is_demo)
                   VALUES (?, ?, ?, ?, ?)""",
                _DEMO_PROVIDERS,
            )
            connection.executemany(
                """INSERT OR IGNORE INTO availability
                   (resource_id, start_datetime, end_datetime) VALUES (?, ?, ?)""",
                [
                    (slot.resource_id, slot.start_datetime, slot.end_datetime)
                    for slot in _DEMO_AVAILABILITY
                ],
            )

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        if name not in _SUPPORTED_TOOLS:
            raise ToolProviderError("unknown_tool")
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        if name == "search_availability" and self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000.0)
        if self.failure_tool in {"*", name}:
            raise ToolProviderError("injected_demo_tool_failure")
        handler = getattr(self, name)
        value = handler(dict(arguments))
        return ToolProviderResponse(response_id=str(uuid4()), value=value)

    def search_availability(self, arguments: dict[str, Any]) -> dict[str, Any]:
        appointment_type = _required(arguments, "appointment_type")
        start_date = str(arguments.get("date") or arguments.get("start_date") or "").strip()
        end_date = str(arguments.get("end_date") or start_date).strip()
        if not start_date or not end_date:
            raise ToolProviderError("missing_date_or_range")
        provider_name = _optional(arguments, "provider_name")
        location = _optional(arguments, "location")
        preferred_time = _optional(arguments, "preferred_time")
        preferred_daypart = _optional(arguments, "preferred_daypart")
        query = """
            SELECT a.resource_id, p.appointment_type, p.provider_name, p.location,
                   a.start_datetime, a.end_datetime
            FROM availability AS a
            JOIN providers AS p ON p.resource_id = a.resource_id
            WHERE lower(p.appointment_type) = lower(?)
              AND substr(a.start_datetime, 1, 10) BETWEEN ? AND ?
              AND NOT EXISTS (
                  SELECT 1 FROM appointments AS booked
                  WHERE booked.resource_id = a.resource_id
                    AND booked.status = 'BOOKED'
                    AND booked.start_datetime < a.end_datetime
                    AND booked.end_datetime > a.start_datetime
              )
        """
        parameters: list[Any] = [appointment_type, start_date, end_date]
        if provider_name:
            query += " AND lower(p.provider_name) = lower(?)"
            parameters.append(provider_name)
        if location:
            query += " AND lower(p.location) = lower(?)"
            parameters.append(location)
        query += " ORDER BY a.start_datetime, p.provider_name"
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, parameters)]
        if preferred_time:
            rows = [row for row in rows if row["start_datetime"][11:16] == preferred_time]
        if preferred_daypart:
            rows = [row for row in rows if _matches_daypart(row["start_datetime"], preferred_daypart)]
        return {
            "success": True,
            "tool": "search_availability",
            "demo_data": True,
            "slots": rows,
        }

    def create_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        appointment_id = _required(arguments, "appointment_id")
        appointment_type = _required(arguments, "appointment_type")
        start_datetime = _required(arguments, "start_datetime")
        operation_guard = _guard(arguments)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
            if existing is not None:
                value = dict(existing)
                value.update(success=True, idempotent_replay=True, tool="create_appointment")
                connection.commit()
                return value
            slot = self._available_slot(
                connection,
                appointment_type=appointment_type,
                start_datetime=start_datetime,
                provider_name=_optional(arguments, "provider_name"),
                location=_optional(arguments, "location"),
            )
            if slot is None:
                connection.rollback()
                return _conflict("create_appointment", appointment_id)
            operation_guard()
            now = _utc_now()
            try:
                connection.execute(
                    """INSERT INTO appointments
                       (appointment_id, resource_id, appointment_type, provider_name,
                        location, start_datetime, end_datetime, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'BOOKED', ?, ?)""",
                    (
                        appointment_id,
                        slot["resource_id"],
                        slot["appointment_type"],
                        slot["provider_name"],
                        slot["location"],
                        slot["start_datetime"],
                        slot["end_datetime"],
                        now,
                        now,
                    ),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                connection.rollback()
                return _conflict("create_appointment", appointment_id)
        return {
            "success": True,
            "tool": "create_appointment",
            "appointment_id": appointment_id,
            "status": "BOOKED",
            **slot,
        }

    def reschedule_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        appointment_id = _required(arguments, "appointment_id")
        target = _required(arguments, "start_datetime")
        operation_guard = _guard(arguments)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
            if current is None or current["status"] != "BOOKED":
                connection.rollback()
                return _failure("reschedule_appointment", "APPOINTMENT_NOT_BOOKED", appointment_id)
            slot = self._available_slot(
                connection,
                appointment_type=current["appointment_type"],
                start_datetime=target,
                provider_name=current["provider_name"],
                location=current["location"],
                exclude_appointment_id=appointment_id,
            )
            if slot is None:
                connection.rollback()
                value = _conflict("reschedule_appointment", appointment_id)
                value["current_slot"] = {
                    "start_datetime": current["start_datetime"],
                    "end_datetime": current["end_datetime"],
                }
                return value
            operation_guard()
            old_slot = {
                "start_datetime": current["start_datetime"],
                "end_datetime": current["end_datetime"],
            }
            try:
                connection.execute(
                    """UPDATE appointments
                       SET start_datetime = ?, end_datetime = ?, updated_at = ?
                       WHERE appointment_id = ?""",
                    (slot["start_datetime"], slot["end_datetime"], _utc_now(), appointment_id),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                connection.rollback()
                return _conflict("reschedule_appointment", appointment_id)
        return {
            "success": True,
            "tool": "reschedule_appointment",
            "appointment_id": appointment_id,
            "status": "BOOKED",
            "old_slot": old_slot,
            "new_slot": {
                "start_datetime": slot["start_datetime"],
                "end_datetime": slot["end_datetime"],
            },
            "provider_name": slot["provider_name"],
            "location": slot["location"],
        }

    def cancel_appointment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        appointment_id = _required(arguments, "appointment_id")
        operation_guard = _guard(arguments)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
            ).fetchone()
            if current is None:
                connection.rollback()
                return _failure("cancel_appointment", "APPOINTMENT_NOT_FOUND", appointment_id)
            if current["status"] == "CANCELLED":
                value = dict(current)
                value.update(success=True, idempotent_replay=True, tool="cancel_appointment")
                connection.commit()
                return value
            operation_guard()
            connection.execute(
                "UPDATE appointments SET status = 'CANCELLED', updated_at = ? WHERE appointment_id = ?",
                (_utc_now(), appointment_id),
            )
            connection.commit()
        return {
            "success": True,
            "tool": "cancel_appointment",
            "appointment_id": appointment_id,
            "appointment_type": current["appointment_type"],
            "start_datetime": current["start_datetime"],
            "status": "CANCELLED",
        }

    def list_appointments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        include_cancelled = bool(arguments.get("include_cancelled", False))
        query = "SELECT * FROM appointments"
        parameters: tuple[Any, ...] = ()
        if not include_cancelled:
            query += " WHERE status = ?"
            parameters = ("BOOKED",)
        query += " ORDER BY start_datetime, appointment_id"
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, parameters)]
        return {"success": True, "tool": "list_appointments", "appointments": rows}

    def _available_slot(
        self,
        connection: sqlite3.Connection,
        *,
        appointment_type: str,
        start_datetime: str,
        provider_name: str | None,
        location: str | None,
        exclude_appointment_id: str | None = None,
    ) -> dict[str, Any] | None:
        query = """
            SELECT a.resource_id, p.appointment_type, p.provider_name, p.location,
                   a.start_datetime, a.end_datetime
            FROM availability AS a
            JOIN providers AS p ON p.resource_id = a.resource_id
            WHERE lower(p.appointment_type) = lower(?) AND a.start_datetime = ?
        """
        parameters: list[Any] = [appointment_type, start_datetime]
        if provider_name:
            query += " AND lower(p.provider_name) = lower(?)"
            parameters.append(provider_name)
        if location:
            query += " AND lower(p.location) = lower(?)"
            parameters.append(location)
        query += " ORDER BY p.provider_name"
        for raw_slot in connection.execute(query, parameters):
            slot = dict(raw_slot)
            conflict = connection.execute(
                """SELECT 1 FROM appointments
                   WHERE resource_id = ? AND status = 'BOOKED'
                     AND start_datetime < ? AND end_datetime > ?
                     AND appointment_id != ? LIMIT 1""",
                (
                    slot["resource_id"],
                    slot["end_datetime"],
                    slot["start_datetime"],
                    exclude_appointment_id or "",
                ),
            ).fetchone()
            if conflict is None:
                return slot
        return None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _required(arguments: dict[str, Any], name: str) -> str:
    raw_value = arguments.get(name)
    value = "" if raw_value is None else str(raw_value).strip()
    if not value:
        raise ToolProviderError(f"missing_required_argument:{name}")
    return value


def _optional(arguments: dict[str, Any], name: str) -> str | None:
    raw_value = arguments.get(name)
    value = "" if raw_value is None else str(raw_value).strip()
    return value or None


def _guard(arguments: dict[str, Any]) -> Callable[[], None]:
    callback = arguments.get(_OPERATION_GUARD)
    if callback is None:
        return lambda: None
    if not callable(callback):
        raise ToolProviderError("invalid_operation_guard")

    def assert_current() -> None:
        if not callback():
            raise ToolProviderError("stale_operation_before_commit")

    return assert_current


def _matches_daypart(start_datetime: str, daypart: str) -> bool:
    hour = int(start_datetime[11:13])
    normalized = daypart.casefold()
    if normalized in {"vormittag", "morning"}:
        return hour < 12
    if normalized in {"nachmittag", "afternoon"}:
        return 12 <= hour < 17
    if normalized in {"abend", "evening"}:
        return hour >= 17
    raise ToolProviderError("unsupported_daypart")


def _failure(tool: str, code: str, appointment_id: str | None = None) -> dict[str, Any]:
    return {
        "success": False,
        "tool": tool,
        "error_code": code,
        "appointment_id": appointment_id,
    }


def _conflict(tool: str, appointment_id: str) -> dict[str, Any]:
    return _failure(tool, "BOOKING_CONFLICT", appointment_id)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
