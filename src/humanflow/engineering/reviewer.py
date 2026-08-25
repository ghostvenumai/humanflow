"""Independent, read-only reviewer evidence contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class ReviewAssignment:
    task_id: str
    worker_session_id: str
    reviewer_session_id: str
    baseline_commit: str
    candidate_commit: str
    read_only: bool = True

    def __post_init__(self) -> None:
        values = (
            self.task_id,
            self.worker_session_id,
            self.reviewer_session_id,
            self.baseline_commit,
            self.candidate_commit,
        )
        if any(not value.strip() for value in values):
            raise ValueError("review assignment identity must not be empty")
        if self.worker_session_id == self.reviewer_session_id:
            raise ValueError("reviewer must use a fresh independent session")
        if not self.read_only:
            raise ValueError("review assignment must be read-only")


@dataclass(frozen=True, slots=True)
class ReviewResult:
    assignment: ReviewAssignment
    verdict: ReviewVerdict
    findings: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verdict is ReviewVerdict.PASS and self.findings:
            raise ValueError("PASS review cannot retain unresolved findings")
        if self.verdict is not ReviewVerdict.PASS and not self.findings:
            raise ValueError("failed or blocked review requires findings")
