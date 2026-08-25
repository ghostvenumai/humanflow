"""Bounded supervisor joining scheduling, verification, metrics and release evidence."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Mapping
from uuid import uuid4

from .failures import FailureCircuitBreaker, FailureFingerprint, HarnessFailureState
from .metrics import HarnessMetricsStore, TaskRunMetrics
from .orchestrator import (
    EngineeringHarness,
    HarnessRunResult,
    ReviewerRunner,
    WorkerRunner,
)
from .registry import ActorRole, TaskRegistry, TaskStatus
from .release import ReleaseCandidateBundle
from .scheduler import ConservativeTaskScheduler, ScheduleDecision


@dataclass(frozen=True, slots=True)
class TaskExecution:
    task_id: str
    worker_id: str
    worker_session_id: str
    reviewer_session_id: str
    worker: WorkerRunner
    reviewer: ReviewerRunner


@dataclass(frozen=True, slots=True)
class SupervisorTaskOutcome:
    task_id: str
    final_status: str
    attempts: int
    harness_results: tuple[HarnessRunResult, ...]
    circuit_state: HarnessFailureState
    release_candidate_path: str | None
    failure_fingerprints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupervisorBatchOutcome:
    schedule: ScheduleDecision
    outcomes: Mapping[str, SupervisorTaskOutcome]


class EngineeringSupervisor:
    def __init__(
        self,
        *,
        harness: EngineeringHarness,
        registry: TaskRegistry,
        scheduler: ConservativeTaskScheduler,
        metrics_store: HarnessMetricsStore,
        release_candidate_directory: Path,
        maximum_iterations: int = 4,
    ) -> None:
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if harness.registry is not registry:
            raise ValueError("harness and supervisor must share one authoritative registry")
        self.harness = harness
        self.registry = registry
        self.scheduler = scheduler
        self.metrics_store = metrics_store
        self.release_candidate_directory = release_candidate_directory
        self.maximum_iterations = maximum_iterations

    def run_ready(self, executions: Mapping[str, TaskExecution]) -> SupervisorBatchOutcome:
        ready = tuple(
            task
            for task in self.registry.tasks
            if task.status is TaskStatus.READY and task.task_id in executions
        )
        decision = self.scheduler.plan(ready, known_tasks=self.registry.tasks)
        deferred = dict(decision.deferred)
        for task in self.registry.tasks:
            if task.status is TaskStatus.READY and task.task_id not in executions:
                deferred[task.task_id] = "NO_AUTHORIZED_RUNNER"
        decision = ScheduleDecision(decision.scheduled, deferred)
        outcomes: dict[str, SupervisorTaskOutcome] = {}
        with ThreadPoolExecutor(
            max_workers=self.scheduler.maximum_parallel_workers,
            thread_name_prefix="humanflow-engineering",
        ) as pool:
            futures = {
                pool.submit(self.run_task_bounded, executions[task_id]): task_id
                for task_id in decision.scheduled
            }
            for future in as_completed(futures):
                task_id = futures[future]
                outcomes[task_id] = future.result()
        return SupervisorBatchOutcome(decision, outcomes)

    def run_task_bounded(self, execution: TaskExecution) -> SupervisorTaskOutcome:
        breaker = FailureCircuitBreaker(
            repeated_failure_threshold=3,
            no_progress_threshold=3,
            maximum_retries_after_circuit=1,
        )
        results: list[HarnessRunResult] = []
        fingerprints: list[str] = []
        release_path: Path | None = None
        final_status = HarnessFailureState.BLOCKED.value
        attempts = 0
        for attempt in range(1, self.maximum_iterations + 1):
            attempts = attempt
            self._prepare_attempt(execution.task_id)
            started = monotonic()
            result: HarnessRunResult | None = None
            error: Exception | None = None
            try:
                result = self.harness.run_task(
                    execution.task_id,
                    worker_id=f"{execution.worker_id}-i{attempt}",
                    worker_session_id=f"{execution.worker_session_id}-i{attempt}",
                    reviewer_session_id=f"{execution.reviewer_session_id}-i{attempt}",
                    worker=execution.worker,
                    reviewer=execution.reviewer,
                )
                results.append(result)
                final_status = result.status
            except Exception as caught:
                error = caught
                final_status = _exception_state(caught).value
            if result is not None and result.status == "VERIFIED_PASS":
                self._record_metrics(
                    execution=execution,
                    attempt=attempt,
                    duration_seconds=monotonic() - started,
                    result=result,
                    error=error,
                    circuit_breaker_events=0,
                )
                release_path = self._write_release_candidate(result)
                break
            fingerprint = _failure_fingerprint(execution.task_id, result, error)
            fingerprints.append(fingerprint.sha256)
            breaker.record_failure(fingerprint)
            self._record_metrics(
                execution=execution,
                attempt=attempt,
                duration_seconds=monotonic() - started,
                result=result,
                error=error,
                circuit_breaker_events=int(breaker.state is HarnessFailureState.CIRCUIT_OPEN),
            )
            if final_status in {
                HarnessFailureState.INTEGRITY_FAILURE.value,
                HarnessFailureState.SCOPE_VIOLATION.value,
            }:
                breaker.state = HarnessFailureState(final_status)
                break
            if breaker.state is HarnessFailureState.CIRCUIT_OPEN:
                if not breaker.authorize_retry():
                    final_status = breaker.state.value
                    break
            if attempt == self.maximum_iterations:
                final_status = breaker.state.value
        return SupervisorTaskOutcome(
            task_id=execution.task_id,
            final_status=final_status,
            attempts=attempts,
            harness_results=tuple(results),
            circuit_state=breaker.state,
            release_candidate_path=str(release_path) if release_path is not None else None,
            failure_fingerprints=tuple(fingerprints),
        )

    def _prepare_attempt(self, task_id: str) -> None:
        task = self.registry.get(task_id)
        if task.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
            self.registry.transition(task_id, TaskStatus.READY, actor=ActorRole.COORDINATOR)
            self.registry.save()
        elif task.status is not TaskStatus.READY:
            raise RuntimeError(f"task cannot be retried from {task.status.value}")

    def _record_metrics(
        self,
        *,
        execution: TaskExecution,
        attempt: int,
        duration_seconds: float,
        result: HarnessRunResult | None,
        error: Exception | None,
        circuit_breaker_events: int,
    ) -> None:
        added, removed = _diff_counts(
            self.harness.repository,
            result.baseline_commit if result is not None else None,
            result.candidate_commit if result is not None else None,
        )
        review_findings = (
            len(result.review.findings) if result is not None and result.review is not None else 0
        )
        self.metrics_store.append(
            TaskRunMetrics(
                run_id=f"{execution.task_id}-{attempt}-{uuid4().hex}",
                task_id=execution.task_id,
                agent=execution.worker_id,
                task_class=self.registry.get(execution.task_id).source,
                iterations=attempt,
                duration_seconds=duration_seconds,
                result=result.status if result is not None else _exception_state(error).value,
                files_changed=len(result.changed_paths) if result is not None else 0,
                lines_added=added,
                lines_removed=removed,
                regressions_introduced=(
                    0 if result is not None and result.status == "VERIFIED_PASS" else 1
                ),
                reviewer_findings=review_findings,
                circuit_breaker_events=circuit_breaker_events,
            )
        )

    def _write_release_candidate(self, result: HarnessRunResult) -> Path:
        if result.candidate_commit is None:
            raise RuntimeError("verified result is missing candidate commit")
        evidence_refs = tuple(
            dict.fromkeys(
                (
                    *self.registry.get(result.task_id).evidence_refs,
                    *(result.review.evidence_refs if result.review is not None else ()),
                    *(
                        f"command:{command.name}:{command.returncode}"
                        for command in result.commands
                    ),
                )
            )
        )
        bundle = ReleaseCandidateBundle(
            task_id=result.task_id,
            baseline_commit=result.baseline_commit,
            candidate_commit=result.candidate_commit,
            latest_main_commit=_git(self.harness.repository, "rev-parse", "HEAD"),
            previous_good_commit=result.baseline_commit,
            verification_status=result.status,
            evidence_refs=evidence_refs,
            changed_paths=result.changed_paths,
            protected_hashes_sha256=result.protected_hashes_sha256,
        )
        path = self.release_candidate_directory / f"{result.task_id}.json"
        bundle.write(path)
        return path


def _failure_fingerprint(
    task_id: str,
    result: HarnessRunResult | None,
    error: Exception | None,
) -> FailureFingerprint:
    if error is not None:
        return FailureFingerprint.create(
            test_name="engineering_harness",
            exception_type=type(error).__name__,
            error=str(error) or type(error).__name__,
            component=task_id,
        )
    assert result is not None
    return FailureFingerprint.create(
        test_name="merge_gate",
        exception_type="MergeRejected",
        error=" ".join(result.reasons),
        component=task_id,
    )


def _exception_state(error: Exception | None) -> HarnessFailureState:
    message = str(error).upper() if error is not None else ""
    if "FROZEN_BUILD_INTEGRITY_FAILURE" in message or "INTEGRITY_FAILURE" in message:
        return HarnessFailureState.INTEGRITY_FAILURE
    if "SCOPE" in message:
        return HarnessFailureState.SCOPE_VIOLATION
    return HarnessFailureState.BLOCKED


def _diff_counts(
    repository: Path,
    baseline_commit: str | None,
    candidate_commit: str | None,
) -> tuple[int, int]:
    if baseline_commit is None or candidate_commit is None:
        return 0, 0
    output = _git(repository, "diff", "--numstat", baseline_commit, candidate_commit)
    added = 0
    removed = 0
    for line in output.splitlines():
        before, after, _ = line.split("\t", 2)
        if before.isdigit():
            added += int(before)
        if after.isdigit():
            removed += int(after)
    return added, removed


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
