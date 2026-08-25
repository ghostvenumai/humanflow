"""Append-only measurements for the engineering loop itself."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TaskRunMetrics:
    run_id: str
    task_id: str
    agent: str
    task_class: str
    iterations: int
    duration_seconds: float
    result: str
    files_changed: int
    lines_added: int
    lines_removed: int
    regressions_introduced: int
    reviewer_findings: int
    circuit_breaker_events: int
    tokens_input: int | None = None
    tokens_output: int | None = None
    estimated_cost: str | None = None
    currency: str | None = None
    tests_before: str | None = None
    tests_after: str | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.run_id, self.task_id, self.agent, self.task_class, self.result)
        ):
            raise ValueError("metric identity must not be empty")
        counts = (
            self.iterations,
            self.files_changed,
            self.lines_added,
            self.lines_removed,
            self.regressions_introduced,
            self.reviewer_findings,
            self.circuit_breaker_events,
        )
        if any(value < 0 for value in counts) or self.duration_seconds < 0:
            raise ValueError("metric counts and duration must be non-negative")
        if self.tokens_input is not None and self.tokens_input < 0:
            raise ValueError("tokens_input must be non-negative")
        if self.tokens_output is not None and self.tokens_output < 0:
            raise ValueError("tokens_output must be non-negative")
        if (self.estimated_cost is None) != (self.currency is None):
            raise ValueError("estimated_cost and currency must both be set or both be null")
        if self.estimated_cost is not None:
            try:
                cost = Decimal(self.estimated_cost)
            except InvalidOperation as error:
                raise ValueError("estimated_cost must be an exact decimal string") from error
            if cost < 0:
                raise ValueError("estimated_cost must be non-negative")

    @property
    def verified_success(self) -> bool:
        return self.result == "VERIFIED_PASS"

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRunMetrics":
        return cls(**payload)


class HarnessMetricsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, metrics: TaskRunMetrics) -> None:
        with self._lock:
            if any(item.run_id == metrics.run_id for item in self._read_unlocked()):
                raise ValueError(f"duplicate run_id: {metrics.run_id}")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics.to_dict(), sort_keys=True) + "\n")

    def read_all(self) -> tuple[TaskRunMetrics, ...]:
        with self._lock:
            return self._read_unlocked()

    def aggregate(self) -> dict[str, dict[str, Any]]:
        groups: dict[tuple[str, str], list[TaskRunMetrics]] = {}
        for item in self.read_all():
            groups.setdefault((item.agent, item.task_class), []).append(item)
        result: dict[str, dict[str, Any]] = {}
        for (agent, task_class), items in sorted(groups.items()):
            successes = [item for item in items if item.verified_success]
            known_costs = [
                Decimal(item.estimated_cost) for item in items if item.estimated_cost is not None
            ]
            complete_cost_evidence = len(known_costs) == len(items)
            result[f"{agent}:{task_class}"] = {
                "n": len(items),
                "verified_successes": len(successes),
                "success_rate": len(successes) / len(items),
                "median_iterations": _median([item.iterations for item in items]),
                "regression_rate": sum(item.regressions_introduced > 0 for item in items)
                / len(items),
                "reviewer_findings": sum(item.reviewer_findings for item in items),
                "cost_per_verified_success": (
                    str(sum(known_costs) / len(successes))
                    if successes and complete_cost_evidence
                    else None
                ),
                "cost_evidence_complete": complete_cost_evidence,
            }
        return result

    def _read_unlocked(self) -> tuple[TaskRunMetrics, ...]:
        if not self.path.exists():
            return ()
        return tuple(
            TaskRunMetrics.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
