"""Bounded offline task orchestration up to, but never through, deployment."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .registry import ActorRole, TaskRegistry, TaskStatus
from .reviewer import ReviewAssignment, ReviewResult
from .verification import (
    CandidateIntegrityGate,
    CommandEvidence,
    MergeGateEvidence,
    SupervisorCommandRunner,
    evaluate_merge_gate,
)
from .worktrees import WorktreeLease, WorktreeManager


@dataclass(frozen=True, slots=True)
class WorkerResult:
    candidate_commit: str
    worker_tests_passed: bool
    evidence_refs: tuple[str, ...]


class WorkerRunner(Protocol):
    def run(self, *, task_id: str, lease: WorktreeLease) -> WorkerResult: ...


class ReviewerRunner(Protocol):
    def review(
        self,
        *,
        assignment: ReviewAssignment,
        worktree: Path,
    ) -> ReviewResult: ...


@dataclass(frozen=True, slots=True)
class HarnessRunResult:
    task_id: str
    status: str
    baseline_commit: str
    candidate_commit: str | None
    worktree_branch: str | None
    integrity_findings: tuple[str, ...]
    commands: tuple[CommandEvidence, ...]
    review: ReviewResult | None
    reasons: tuple[str, ...]


class EngineeringHarness:
    def __init__(
        self,
        *,
        repository: Path,
        worktree_manager: WorktreeManager,
        registry: TaskRegistry,
        protected_commands: tuple[tuple[str, tuple[str, ...]], ...],
        hidden_acceptance_commands: tuple[tuple[str, tuple[str, ...]], ...],
        command_runner: SupervisorCommandRunner | None = None,
    ) -> None:
        self.repository = repository.resolve()
        self.worktree_manager = worktree_manager
        self.registry = registry
        self.protected_commands = protected_commands
        self.hidden_acceptance_commands = hidden_acceptance_commands
        self.command_runner = command_runner or SupervisorCommandRunner()

    def run_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        worker_session_id: str,
        reviewer_session_id: str,
        worker: WorkerRunner,
        reviewer: ReviewerRunner,
    ) -> HarnessRunResult:
        task = self.registry.get(task_id)
        if task.status is not TaskStatus.READY:
            raise RuntimeError("task must be READY before orchestration")
        if not self.protected_commands or not self.hidden_acceptance_commands:
            raise RuntimeError("protected and hidden acceptance commands must be configured")
        baseline = _git(self.repository, "rev-parse", "HEAD")
        self.registry.transition(task_id, TaskStatus.RUNNING, actor=ActorRole.COORDINATOR)
        self.registry.save()
        lease = self.worktree_manager.create(
            task_id=task_id, worker_id=worker_id, baseline_commit=baseline
        )
        commands: list[CommandEvidence] = []
        review_result: ReviewResult | None = None
        candidate_commit: str | None = None
        try:
            worker_result = worker.run(task_id=task_id, lease=lease)
            candidate_commit = _git(lease.path, "rev-parse", "HEAD")
            if candidate_commit != _git(
                lease.path, "rev-parse", f"{worker_result.candidate_commit}^{{commit}}"
            ):
                raise RuntimeError("worker result does not identify worktree HEAD")
            if _git(lease.path, "status", "--porcelain"):
                raise RuntimeError("worker left an uncommitted working tree")
            self.registry.transition(
                task_id,
                TaskStatus.VERIFICATION,
                actor=ActorRole.WORKER,
                evidence_refs=worker_result.evidence_refs,
            )
            integrity = CandidateIntegrityGate().inspect(
                repository=self.repository,
                baseline_commit=baseline,
                candidate_commit=candidate_commit,
                task=task,
            )
            relevant = tuple(
                self.command_runner.run(
                    name=f"task-verification-{index}", argv=argv, cwd=lease.path
                )
                for index, argv in enumerate(task.verification, start=1)
            )
            protected = tuple(
                self.command_runner.run(name=name, argv=argv, cwd=lease.path)
                for name, argv in self.protected_commands
            )
            hidden = tuple(
                self.command_runner.run(name=name, argv=argv, cwd=lease.path)
                for name, argv in self.hidden_acceptance_commands
            )
            commands.extend((*relevant, *protected, *hidden))
            assignment = ReviewAssignment(
                task_id,
                worker_session_id,
                reviewer_session_id,
                baseline,
                candidate_commit,
            )
            review_result = reviewer.review(assignment=assignment, worktree=lease.path)
            if review_result.assignment != assignment:
                raise RuntimeError("review result does not match independent assignment")
            latest_main = _git(self.repository, "rev-parse", "HEAD")
            latest_main_revalidated = latest_main == baseline and _is_ancestor(
                self.repository, latest_main, candidate_commit
            )
            evidence = MergeGateEvidence(
                worker_tests_passed=worker_result.worker_tests_passed,
                relevant_regression_passed=bool(relevant)
                and all(command.passed for command in relevant),
                protected_tests_passed=all(command.passed for command in protected),
                hidden_acceptance_passed=all(command.passed for command in hidden),
                integrity=integrity,
                review=review_result,
                merge_conflict=not _is_ancestor(self.repository, baseline, candidate_commit),
                latest_main_revalidated=latest_main_revalidated,
            )
            gate = evaluate_merge_gate(evidence)
            if gate["status"] == "VERIFIED_PASS":
                self.registry.transition(
                    task_id,
                    TaskStatus.PASSED,
                    actor=ActorRole.VERIFIER,
                    evidence_refs=review_result.evidence_refs,
                )
                self.registry.transition(
                    task_id,
                    TaskStatus.MERGE_CANDIDATE,
                    actor=ActorRole.VERIFIER,
                )
            else:
                self.registry.transition(
                    task_id,
                    TaskStatus.FAILED,
                    actor=ActorRole.VERIFIER,
                    evidence_refs=review_result.evidence_refs,
                )
            self.registry.save()
            return HarnessRunResult(
                task_id,
                str(gate["status"]),
                baseline,
                candidate_commit,
                lease.branch,
                integrity.findings,
                tuple(commands),
                review_result,
                tuple(str(reason) for reason in gate["reasons"]),
            )
        except Exception:
            current = self.registry.get(task_id)
            if current.status in {TaskStatus.RUNNING, TaskStatus.VERIFICATION}:
                self.registry.transition(task_id, TaskStatus.BLOCKED, actor=ActorRole.COORDINATOR)
                self.registry.save()
            raise
        finally:
            if lease.path.exists() and not _git(lease.path, "status", "--porcelain"):
                self.worktree_manager.cleanup(lease)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
