"""Load immutable turn fixtures and score deterministic policies."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from humanflow.turns.models import TurnDecision, TurnDecisionType, TurnSignals


class TurnPolicy(Protocol):
    def decide(self, signals: TurnSignals) -> TurnDecision: ...


@dataclass(frozen=True, slots=True)
class TurnCase:
    case_id: str
    scenario: str
    expected: TurnDecisionType
    signals: TurnSignals


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    policy_name: str
    total: int
    correct: int
    accuracy: float
    confusion: dict[str, dict[str, int]]
    failed_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "confusion": self.confusion,
            "failed_case_ids": list(self.failed_case_ids),
        }


def load_turn_cases(path: Path, *, minimum_cases: int = 50) -> tuple[TurnCase, ...]:
    cases: list[TurnCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            case_id = str(payload["id"])
            scenario = str(payload["scenario"])
            expected = TurnDecisionType(payload["expected"])
            signals = TurnSignals(**payload["signals"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid turn fixture at {path}:{line_number}: {error}") from error
        if case_id in seen:
            raise ValueError(f"duplicate turn fixture id: {case_id}")
        seen.add(case_id)
        cases.append(TurnCase(case_id, scenario, expected, signals))
    if len(cases) < minimum_cases:
        raise ValueError(f"expected at least {minimum_cases} turn fixtures, found {len(cases)}")
    return tuple(cases)


def evaluate_policy(
    policy: TurnPolicy,
    cases: Sequence[TurnCase],
    *,
    policy_name: str | None = None,
) -> PolicyEvaluation:
    confusion_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failed: list[str] = []
    correct = 0
    for case in cases:
        actual = policy.decide(case.signals).decision
        confusion_counts[case.expected.value][actual.value] += 1
        if actual is case.expected:
            correct += 1
        else:
            failed.append(case.case_id)
    total = len(cases)
    confusion = {
        expected: dict(sorted(actuals.items()))
        for expected, actuals in sorted(confusion_counts.items())
    }
    return PolicyEvaluation(
        policy_name=policy_name or type(policy).__name__,
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        confusion=confusion,
        failed_case_ids=tuple(failed),
    )

