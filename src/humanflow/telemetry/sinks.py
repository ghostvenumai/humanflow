"""Telemetry sinks keep event production independent from storage."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Protocol

from .events import TelemetryEvent


class TelemetrySink(Protocol):
    def emit(self, event: TelemetryEvent) -> None: ...


class InMemoryTelemetrySink:
    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []
        self._lock = Lock()

    def emit(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[TelemetryEvent, ...]:
        with self._lock:
            return tuple(self._events)


class JsonlTelemetrySink:
    """Append-only JSONL sink suitable for deterministic timeline replay."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def emit(self, event: TelemetryEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

