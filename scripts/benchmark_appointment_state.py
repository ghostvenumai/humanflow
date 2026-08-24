#!/usr/bin/env python3
"""Reproducible deterministic benchmark for appointment delta handling."""

from __future__ import annotations

import json
import statistics
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter_ns

from humanflow.runtime.appointment_state import AppointmentStateTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "appointment-state-benchmark.json"
ITERATIONS = 1_000
TURNS = (
    ("Ich möchte den Termin nächsten Freitag.", "turn-1"),
    ("Äh, machen wir übernächste Woche Donnerstag.", "turn-2"),
    ("Am besten gegen 14 Uhr.", "turn-3"),
    ("Vielleicht 15 Uhr.", "turn-4"),
)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value)))
    return ordered[rank]


def main() -> None:
    sequence_latencies_ms: list[float] = []
    correct = 0
    unchanged_date_checks = 0
    provenance_checks = 0
    for _ in range(ITERATIONS):
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 24),
            now=lambda: datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        )
        started_ns = perf_counter_ns()
        for text, turn_id in TURNS:
            tracker.apply_user_turn(text, source_turn=turn_id)
            if turn_id in {"turn-3", "turn-4"}:
                date_slot = tracker.state.date
                if date_slot is not None and date_slot.value == "2026-09-10":
                    unchanged_date_checks += 1
        sequence_latencies_ms.append((perf_counter_ns() - started_ns) / 1_000_000.0)
        date_slot = tracker.state.date
        time_slot = tracker.state.time
        if (
            date_slot is not None
            and time_slot is not None
            and date_slot.value == "2026-09-10"
            and time_slot.value == "15:00"
        ):
            correct += 1
        if (
            date_slot is not None
            and time_slot is not None
            and date_slot.source_turn == "turn-2"
            and time_slot.source_turn == "turn-4"
        ):
            provenance_checks += 1

    payload = {
        "benchmark": "appointment_delta_state",
        "iterations": ITERATIONS,
        "turns_per_iteration": len(TURNS),
        "latency_ms": {
            "median": round(statistics.median(sequence_latencies_ms), 6),
            "p95": round(percentile(sequence_latencies_ms, 0.95), 6),
            "max": round(max(sequence_latencies_ms), 6),
        },
        "correct_final_state_count": correct,
        "expected_final_state": {"date": "2026-09-10", "time": "15:00"},
        "unchanged_date_checks": unchanged_date_checks,
        "expected_unchanged_date_checks": ITERATIONS * 2,
        "provenance_checks": provenance_checks,
        "assistant_history_used_as_state_source": False,
        "result": (
            "PASS"
            if correct == ITERATIONS
            and unchanged_date_checks == ITERATIONS * 2
            and provenance_checks == ITERATIONS
            else "FAIL"
        ),
        "measurement_scope": "in-process deterministic German slot parsing; no provider call",
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
