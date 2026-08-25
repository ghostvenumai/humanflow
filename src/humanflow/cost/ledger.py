"""Append-only SQLite persistence isolated from the realtime conversation path."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from .models import CostEvent
from .pricing import PricingRule


class CostLedgerWriteError(RuntimeError):
    """A persistence failure that callers must isolate from conversation behavior."""


class CostEventWriter(Protocol):
    def append(self, event: CostEvent) -> bool: ...


class CostLedger:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS cost_events (
                    cost_event_id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    timestamp_monotonic_ns INTEGER NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT,
                    response_id TEXT,
                    operation_id TEXT NOT NULL,
                    provider_request_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    usage_source TEXT NOT NULL,
                    cost_source TEXT NOT NULL,
                    input_units TEXT,
                    output_units TEXT,
                    total_units TEXT,
                    unit_type TEXT,
                    audio_input_seconds TEXT,
                    audio_output_seconds TEXT,
                    characters INTEGER,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    credits TEXT,
                    actual_usage INTEGER NOT NULL,
                    estimated_usage INTEGER NOT NULL,
                    actual_cost_micros INTEGER,
                    estimated_cost_micros INTEGER,
                    currency TEXT,
                    pricing_rule_id TEXT,
                    pricing_version TEXT,
                    pricing_effective_date TEXT,
                    played_fraction TEXT,
                    heard_units TEXT,
                    unheard_units TEXT,
                    wasted_cost_estimate_micros INTEGER,
                    fallback INTEGER NOT NULL,
                    retry INTEGER NOT NULL,
                    cancelled INTEGER NOT NULL,
                    tool_success INTEGER,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS cost_events_session_idx
                    ON cost_events(session_id, timestamp_monotonic_ns);
                CREATE INDEX IF NOT EXISTS cost_events_response_idx
                    ON cost_events(response_id);
                CREATE INDEX IF NOT EXISTS cost_events_operation_idx
                    ON cost_events(operation_id);
                CREATE INDEX IF NOT EXISTS cost_events_provider_idx
                    ON cost_events(provider, model);
                CREATE INDEX IF NOT EXISTS cost_events_service_idx
                    ON cost_events(service_type);
                CREATE TRIGGER IF NOT EXISTS cost_events_append_only_update
                    BEFORE UPDATE ON cost_events
                    BEGIN SELECT RAISE(ABORT, 'cost_events_are_append_only'); END;
                CREATE TRIGGER IF NOT EXISTS cost_events_append_only_delete
                    BEFORE DELETE ON cost_events
                    BEGIN SELECT RAISE(ABORT, 'cost_events_are_append_only'); END;
                CREATE TABLE IF NOT EXISTS pricing_rules (
                    pricing_rule_id TEXT PRIMARY KEY,
                    pricing_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    service_type TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    input_rate TEXT,
                    output_rate TEXT,
                    currency TEXT NOT NULL,
                    effective_date TEXT NOT NULL,
                    verified_at TEXT,
                    source_note TEXT NOT NULL,
                    active INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_cost_summary (
                    session_id TEXT PRIMARY KEY,
                    generated_at_utc TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                """
            )

    def store_pricing_rules(self, rules: Iterable[PricingRule]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """INSERT OR IGNORE INTO pricing_rules
                   (pricing_rule_id, pricing_version, provider, model, service_type,
                    unit, input_rate, output_rate, currency, effective_date,
                    verified_at, source_note, active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        rule.pricing_rule_id,
                        rule.pricing_version,
                        rule.provider,
                        rule.model,
                        rule.service.value,
                        rule.unit,
                        None if rule.input_rate is None else str(rule.input_rate),
                        None if rule.output_rate is None else str(rule.output_rate),
                        rule.currency,
                        rule.effective_date,
                        rule.verified_at,
                        rule.source_note,
                        int(rule.active),
                    )
                    for rule in rules
                ],
            )

    def append(self, event: CostEvent) -> bool:
        payload = event.to_dict()
        columns = (
            "cost_event_id",
            "dedupe_key",
            "timestamp_monotonic_ns",
            "timestamp_utc",
            "session_id",
            "conversation_id",
            "turn_id",
            "response_id",
            "operation_id",
            "provider_request_id",
            "provider",
            "model",
            "service_type",
            "operation",
            "usage_source",
            "cost_source",
            "input_units",
            "output_units",
            "total_units",
            "unit_type",
            "audio_input_seconds",
            "audio_output_seconds",
            "characters",
            "tokens_input",
            "tokens_output",
            "credits",
            "actual_usage",
            "estimated_usage",
            "actual_cost_micros",
            "estimated_cost_micros",
            "currency",
            "pricing_rule_id",
            "pricing_version",
            "pricing_effective_date",
            "played_fraction",
            "heard_units",
            "unheard_units",
            "wasted_cost_estimate_micros",
            "fallback",
            "retry",
            "cancelled",
            "tool_success",
            "metadata_json",
        )
        values = [
            (
                json.dumps(payload["metadata"], ensure_ascii=False, sort_keys=True)
                if column == "metadata_json"
                else int(payload[column])
                if column
                in {"actual_usage", "estimated_usage", "fallback", "retry", "cancelled"}
                else (
                    None
                    if payload[column] is None
                    else int(payload[column])
                )
                if column == "tool_success"
                else payload[column]
            )
            for column in columns
        ]
        placeholders = ", ".join("?" for _ in columns)
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO cost_events ({', '.join(columns)}) "
                    f"VALUES ({placeholders})",
                    values,
                )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed" in str(error):
                return False
            raise CostLedgerWriteError("cost_ledger_integrity_error") from error
        except sqlite3.Error as error:
            raise CostLedgerWriteError("cost_ledger_sqlite_error") from error
        return True

    def rows(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cost_events"
        parameters: tuple[Any, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            parameters = (session_id,)
        query += " ORDER BY timestamp_monotonic_ns, cost_event_id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def save_session_summary(
        self,
        session_id: str,
        generated_at_utc: str,
        summary: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO session_cost_summary
                   (session_id, generated_at_utc, summary_json) VALUES (?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       generated_at_utc = excluded.generated_at_utc,
                       summary_json = excluded.summary_json""",
                (
                    session_id,
                    generated_at_utc,
                    json.dumps(summary, ensure_ascii=False, sort_keys=True),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        return connection


FailureCallback = Callable[[str, dict[str, Any]], None]


class AsyncCostRecorder:
    """Non-blocking best-effort recorder; failures are observable and never raised."""

    def __init__(
        self,
        writer: CostEventWriter,
        *,
        on_failure: FailureCallback | None = None,
        queue_size: int = 2_000,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._writer = writer
        self._on_failure = on_failure or (lambda _kind, _payload: None)
        self._queue: asyncio.Queue[CostEvent | None] = asyncio.Queue(queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    def record_nowait(self, event: CostEvent) -> bool:
        if self._closed:
            self._notify("COST_LEDGER_WRITE_FAILED", event, "recorder_closed")
            return False
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name=f"humanflow-cost-ledger-{event.session_id}"
            )
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._notify("COST_LEDGER_WRITE_FAILED", event, "queue_full")
            return False
        return True

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                try:
                    inserted = await asyncio.to_thread(self._writer.append, event)
                except Exception as error:
                    self._notify(
                        "COST_LEDGER_WRITE_FAILED",
                        event,
                        type(error).__name__,
                    )
                else:
                    if not inserted:
                        self._notify(
                            "COST_ANOMALY_DETECTED",
                            event,
                            "duplicate_cost_event",
                        )
            finally:
                self._queue.task_done()

    def _notify(self, kind: str, event: CostEvent, reason: str) -> None:
        self._on_failure(
            kind,
            {
                "cost_event_id": event.cost_event_id,
                "operation_id": event.operation_id,
                "service_type": event.service_type.value,
                "reason": reason,
            },
        )
