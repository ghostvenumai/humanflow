"""Fail-closed candidate integrity, hidden acceptance and merge gate."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic_ns

from .registry import EngineeringTaskRecord
from .reviewer import ReviewResult, ReviewVerdict
from .scheduler import path_matches


class VerificationStatus(StrEnum):
    VERIFIED_PASS = "VERIFIED_PASS"
    MERGE_REJECTED = "MERGE_REJECTED"


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    passed: bool
    changed_paths: tuple[str, ...]
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    name: str
    argv: tuple[str, ...]
    returncode: int
    duration_ms: float
    output_tail: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class MergeGateEvidence:
    worker_tests_passed: bool
    relevant_regression_passed: bool
    protected_tests_passed: bool
    hidden_acceptance_passed: bool
    integrity: IntegrityReport
    review: ReviewResult
    merge_conflict: bool
    latest_main_revalidated: bool


class CandidateIntegrityGate:
    DEFAULT_PROTECTED_PATHS = (
        "tests/golden/**",
        "eval/golden/**",
        "grader/**",
        "hidden_acceptance/**",
        "config/quality-gates.yaml",
        "docs/METRICS.md",
        "schemas/metric-definitions.json",
        "sprint/**",
    )
    _WEAKENING_PATTERN = re.compile(
        r"(?:@pytest\.mark\.(?:skip|xfail)|pytest\.(?:skip|xfail)\(|"
        r"unittest\.skip|#\s*assert\b)",
        re.IGNORECASE,
    )

    def inspect(
        self,
        *,
        repository: Path,
        baseline_commit: str,
        candidate_commit: str,
        task: EngineeringTaskRecord,
    ) -> IntegrityReport:
        repository = repository.resolve()
        baseline = _git(repository, "rev-parse", f"{baseline_commit}^{{commit}}")
        candidate = _git(repository, "rev-parse", f"{candidate_commit}^{{commit}}")
        changed = tuple(
            line.strip()
            for line in _git(repository, "diff", "--name-only", baseline, candidate).splitlines()
            if line.strip()
        )
        findings: list[str] = []
        protected_rules = (*self.DEFAULT_PROTECTED_PATHS, *task.protected_paths)
        for path in changed:
            if any(path_matches(path, rule) for rule in protected_rules):
                findings.append(f"PROTECTED_PATH_MODIFIED:{path}")
            if not task.allowed_paths or not any(
                path_matches(path, rule) for rule in task.allowed_paths
            ):
                findings.append(f"OUTSIDE_ALLOWED_PATHS:{path}")
        patch = _git(repository, "diff", "--unified=0", baseline, candidate, "--")
        for line in patch.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                if self._WEAKENING_PATTERN.search(line[1:]):
                    findings.append("TEST_WEAKENING_PATTERN_ADDED")
                    break
        return IntegrityReport(
            passed=not findings,
            changed_paths=changed,
            findings=tuple(dict.fromkeys(findings)),
        )


class SupervisorCommandRunner:
    """Run verifier-owned argv without a shell; workers receive only its result."""

    def __init__(self, *, timeout_seconds: float = 900.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def run(self, *, name: str, argv: tuple[str, ...], cwd: Path) -> CommandEvidence:
        if not name.strip() or not argv or any(not part.strip() for part in argv):
            raise ValueError("command name and argv must not be empty")
        started = monotonic_ns()
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "src",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        completed = subprocess.run(
            list(argv),
            cwd=cwd.resolve(),
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        return CommandEvidence(
            name=name,
            argv=argv,
            returncode=completed.returncode,
            duration_ms=(monotonic_ns() - started) / 1_000_000.0,
            output_tail=tuple(completed.stdout.splitlines()[-20:]),
        )


def evaluate_merge_gate(evidence: MergeGateEvidence) -> dict[str, object]:
    reasons: list[str] = []
    checks = (
        (evidence.worker_tests_passed, "WORKER_TESTS_FAILED"),
        (evidence.relevant_regression_passed, "RELEVANT_REGRESSION_FAILED"),
        (evidence.protected_tests_passed, "PROTECTED_TESTS_FAILED"),
        (evidence.hidden_acceptance_passed, "HIDDEN_ACCEPTANCE_FAILED"),
        (evidence.integrity.passed, "INTEGRITY_FAILURE"),
        (evidence.review.verdict is ReviewVerdict.PASS, "REVIEW_FAILED"),
        (not evidence.merge_conflict, "MERGE_CONFLICT"),
        (evidence.latest_main_revalidated, "LATEST_MAIN_REVALIDATION_MISSING"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return {
        "status": (
            VerificationStatus.VERIFIED_PASS.value
            if not reasons
            else VerificationStatus.MERGE_REJECTED.value
        ),
        "reasons": reasons or ["ALL_VERIFICATION_GATES_PASSED"],
    }


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
