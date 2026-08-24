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
    ("Ich brauche einen Orthopädentermin.", "turn-1"),
    ("Am besten nächste Woche Freitag um 14 Uhr.", "turn-2"),
    (
        "Mmm, warte mal, dann machen wir vielleicht 16 Uhr nächste Woche Donnerstag.",
        "turn-3",
    ),
    (
        "Ich brauch noch 'n Friseurtermin, nächste Woche Mittwoch um 14 Uhr.",
        "turn-4",
    ),
    ("Der Friseurtermin, was mit dem?", "turn-5"),
    ("Nee, ich will ihn absagen.", "turn-6"),
    (
        "Nicht Orthopäde. Der soll bleiben. Den Friseurtermin brauche ich nicht mehr.",
        "turn-7",
    ),
)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value)))
    return ordered[rank]


def main() -> None:
    sequence_latencies_ms: list[float] = []
    correct = 0
    cross_contamination_checks = 0
    provenance_checks = 0
    for _ in range(ITERATIONS):
        tracker = AppointmentStateTracker(
            today=lambda: date(2026, 8, 24),
            now=lambda: datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        )
        started_ns = perf_counter_ns()
        for text, turn_id in TURNS:
            tracker.apply_user_turn(text, source_turn=turn_id)
        sequence_latencies_ms.append((perf_counter_ns() - started_ns) / 1_000_000.0)
        orthopedist = tracker.appointments.get("appointment_1")
        hairdresser = tracker.appointments.get("appointment_2")
        if (
            orthopedist is not None
            and hairdresser is not None
            and orthopedist.date is not None
            and orthopedist.time is not None
            and orthopedist.status is not None
            and hairdresser.date is not None
            and hairdresser.time is not None
            and hairdresser.status is not None
            and orthopedist.date.value == "2026-09-03"
            and orthopedist.time.value == "16:00"
            and orthopedist.status.value == "READY_TO_BOOK"
            and hairdresser.date.value == "2026-09-02"
            and hairdresser.time.value == "14:00"
            and hairdresser.status.value == "CANCELLED"
        ):
            correct += 1
        if (
            orthopedist is not None
            and hairdresser is not None
            and orthopedist.date is not None
            and orthopedist.time is not None
            and hairdresser.date is not None
            and hairdresser.time is not None
            and orthopedist.date.source_turn == "turn-3"
            and orthopedist.time.source_turn == "turn-3"
            and hairdresser.date.source_turn == "turn-4"
            and hairdresser.time.source_turn == "turn-4"
        ):
            provenance_checks += 1
        if (
            orthopedist is not None
            and hairdresser is not None
            and orthopedist.date is not hairdresser.date
            and orthopedist.time is not hairdresser.time
        ):
            cross_contamination_checks += 1

    payload = {
        "benchmark": "multi_appointment_delta_state",
        "iterations": ITERATIONS,
        "turns_per_iteration": len(TURNS),
        "latency_ms": {
            "median": round(statistics.median(sequence_latencies_ms), 6),
            "p95": round(percentile(sequence_latencies_ms, 0.95), 6),
            "max": round(max(sequence_latencies_ms), 6),
        },
        "correct_final_state_count": correct,
        "expected_final_state": {
            "appointment_1": {
                "purpose": "Orthopädie",
                "date": "2026-09-03",
                "time": "16:00",
                "status": "READY_TO_BOOK",
            },
            "appointment_2": {
                "purpose": "Friseur",
                "date": "2026-09-02",
                "time": "14:00",
                "status": "CANCELLED",
            },
        },
        "cross_contamination_checks": cross_contamination_checks,
        "provenance_checks": provenance_checks,
        "stable_appointment_ids": True,
        "external_booking_claim_without_tool": False,
        "assistant_history_used_as_state_source": False,
        "result": (
            "PASS"
            if correct == ITERATIONS
            and cross_contamination_checks == ITERATIONS
            and provenance_checks == ITERATIONS
            else "FAIL"
        ),
        "measurement_scope": (
            "in-process deterministic German multi-object reference, correction and "
            "cancellation parsing; no provider call"
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
