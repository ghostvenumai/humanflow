"""Deterministic stuck detection, circuit breaking and diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class HarnessFailureState(StrEnum):
    HEALTHY = "HEALTHY"
    BLOCKED_MANUAL_VALIDATION = "BLOCKED_MANUAL_VALIDATION"
    TEST_DISPUTE = "TEST_DISPUTE"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    NO_PROGRESS = "NO_PROGRESS"
    OSCILLATION_DETECTED = "OSCILLATION_DETECTED"
    OUTPUT_COLLAPSE = "OUTPUT_COLLAPSE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    REPEATED_FAILURE = "REPEATED_FAILURE"
    MERGE_REJECTED = "MERGE_REJECTED"
    RELEASE_REJECTED = "RELEASE_REJECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class FailureFingerprint:
    sha256: str
    test_name: str
    exception_type: str
    normalized_error: str
    component: str

    @classmethod
    def create(
        cls,
        *,
        test_name: str,
        exception_type: str,
        error: str,
        component: str,
    ) -> "FailureFingerprint":
        values = (test_name, exception_type, error, component)
        if any(not value.strip() for value in values):
            raise ValueError("failure fingerprint inputs must not be empty")
        normalized_error = normalize_failure_error(error)
        digest = hashlib.sha256(
            "\x1f".join(
                (test_name.strip(), exception_type.strip(), normalized_error, component.strip())
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            digest, test_name.strip(), exception_type.strip(), normalized_error, component.strip()
        )


@dataclass(frozen=True, slots=True)
class IterationObservation:
    iteration: int
    diff_sha256: str
    failing_fingerprints: tuple[str, ...]
    lines_changed: int
    output_characters: int
    task_resolved: bool = False
    budget_exceeded: bool = False
    scope_violation: bool = False

    def __post_init__(self) -> None:
        if self.iteration < 1 or self.lines_changed < 0 or self.output_characters < 0:
            raise ValueError("iteration and counts must be non-negative")
        if len(self.diff_sha256) != 64:
            raise ValueError("diff_sha256 must be a SHA-256 digest")


class FailureCircuitBreaker:
    def __init__(
        self,
        *,
        repeated_failure_threshold: int = 3,
        no_progress_threshold: int = 3,
        maximum_retries_after_circuit: int = 1,
        output_collapse_characters: int = 32,
    ) -> None:
        if min(repeated_failure_threshold, no_progress_threshold) < 2:
            raise ValueError("failure thresholds must be at least two")
        if maximum_retries_after_circuit < 0 or output_collapse_characters < 0:
            raise ValueError("retry and output thresholds must be non-negative")
        self.repeated_failure_threshold = repeated_failure_threshold
        self.no_progress_threshold = no_progress_threshold
        self.maximum_retries_after_circuit = maximum_retries_after_circuit
        self.output_collapse_characters = output_collapse_characters
        self._failure_counts: dict[str, int] = {}
        self._observations: list[IterationObservation] = []
        self._retries_after_circuit = 0
        self.state = HarnessFailureState.HEALTHY

    def record_failure(self, fingerprint: FailureFingerprint) -> HarnessFailureState:
        count = self._failure_counts.get(fingerprint.sha256, 0) + 1
        self._failure_counts[fingerprint.sha256] = count
        if count >= self.repeated_failure_threshold:
            self.state = HarnessFailureState.CIRCUIT_OPEN
        return self.state

    def record_iteration(self, observation: IterationObservation) -> HarnessFailureState:
        self._observations.append(observation)
        if observation.task_resolved:
            self.state = HarnessFailureState.HEALTHY
            return self.state
        if observation.budget_exceeded:
            self.state = HarnessFailureState.BUDGET_EXCEEDED
        elif observation.scope_violation:
            self.state = HarnessFailureState.SCOPE_VIOLATION
        elif (
            observation.lines_changed == 0
            and observation.output_characters < self.output_collapse_characters
        ):
            self.state = HarnessFailureState.OUTPUT_COLLAPSE
        elif self._oscillates():
            self.state = HarnessFailureState.OSCILLATION_DETECTED
        elif self._no_progress():
            self.state = HarnessFailureState.NO_PROGRESS
        return self.state

    def authorize_retry(self) -> bool:
        if self.state is not HarnessFailureState.CIRCUIT_OPEN:
            return False
        if self._retries_after_circuit >= self.maximum_retries_after_circuit:
            self.state = HarnessFailureState.BLOCKED
            return False
        self._retries_after_circuit += 1
        self.state = HarnessFailureState.REPEATED_FAILURE
        return True

    @property
    def failure_counts(self) -> Mapping[str, int]:
        return dict(self._failure_counts)

    def _no_progress(self) -> bool:
        window = self._observations[-self.no_progress_threshold :]
        return bool(
            len(window) == self.no_progress_threshold
            and len({item.diff_sha256 for item in window}) == 1
            and len({item.failing_fingerprints for item in window}) == 1
        )

    def _oscillates(self) -> bool:
        if len(self._observations) < 4:
            return False
        hashes = [item.diff_sha256 for item in self._observations[-4:]]
        return hashes[0] == hashes[2] and hashes[1] == hashes[3] and hashes[0] != hashes[1]


class DiagnosticPackageWriter:
    FILES = (
        "task.json",
        "git.diff",
        "git.status",
        "test_output.txt",
        "failure_fingerprints.json",
        "agent_log.txt",
        "metrics.json",
        "review_findings.json",
        "evidence_refs.json",
    )

    def write(
        self,
        directory: Path,
        *,
        task: Mapping[str, Any],
        git_diff: str,
        git_status: str,
        test_output: str,
        failure_fingerprints: tuple[FailureFingerprint, ...],
        agent_log: str,
        metrics: Mapping[str, Any],
        review_findings: tuple[str, ...],
        evidence_refs: tuple[str, ...],
    ) -> tuple[Path, ...]:
        directory.mkdir(parents=True, exist_ok=False)
        payloads: dict[str, str] = {
            "task.json": _json(task),
            "git.diff": _redact(git_diff),
            "git.status": _redact(git_status),
            "test_output.txt": _redact(test_output),
            "failure_fingerprints.json": _json(
                [asdict(fingerprint) for fingerprint in failure_fingerprints]
            ),
            "agent_log.txt": _redact(agent_log),
            "metrics.json": _json(metrics),
            "review_findings.json": _json(list(review_findings)),
            "evidence_refs.json": _json(list(evidence_refs)),
        }
        paths: list[Path] = []
        for name in self.FILES:
            path = directory / name
            path.write_text(payloads[name], encoding="utf-8")
            paths.append(path)
        return tuple(paths)


def normalize_failure_error(error: str) -> str:
    normalized = error.casefold()
    normalized = re.sub(r"0x[0-9a-f]+", "<hex>", normalized)
    normalized = re.sub(r"/tmp/[\w./-]+", "<tmp>", normalized)
    normalized = re.sub(r"\bline\s+\d+\b", "line <n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _redact(value: str) -> str:
    patterns = (
        re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(authorization\s*[=:]\s*)\S+"),
        re.compile(r"(?i)(password\s*[=:]\s*)\S+"),
    )
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
