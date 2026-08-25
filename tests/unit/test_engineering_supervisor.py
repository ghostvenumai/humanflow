from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier

from humanflow.engineering import (
    ActorRole,
    ConservativeTaskScheduler,
    EngineeringHarness,
    EngineeringSupervisor,
    EngineeringTaskRecord,
    HarnessMetricsStore,
    ReviewAssignment,
    ReviewResult,
    ReviewVerdict,
    TaskExecution,
    TaskPriority,
    TaskRegistry,
    TaskRisk,
    TaskStatus,
    WorkerResult,
    WorktreeLease,
    WorktreeManager,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "HumanFlow Test")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("VALUE = 1\n")
    (repo / "src" / "b.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "everlast-72h-build", baseline)
    return repo, baseline


def _registry(path: Path, *tasks: tuple[str, str]) -> TaskRegistry:
    registry = TaskRegistry(path)
    for task_id, allowed_path in tasks:
        registry.add(
            EngineeringTaskRecord(
                task_id,
                f"Change {allowed_path}",
                "deterministic_fixture",
                TaskPriority.P2,
                TaskRisk.LOW,
                allowed_paths=(allowed_path,),
                verification=(("python3", "-c", "assert True"),),
                evidence_refs=(f"problem:{task_id}",),
            )
        )
        registry.transition(task_id, TaskStatus.TRIAGED, actor=ActorRole.COORDINATOR)
        registry.transition(task_id, TaskStatus.READY, actor=ActorRole.COORDINATOR)
    registry.save()
    return registry


@dataclass
class FileWorker:
    relative_path: str
    barrier: Barrier | None = None

    def run(self, *, task_id: str, lease: WorktreeLease) -> WorkerResult:
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        target = lease.path / self.relative_path
        target.write_text(f"TASK = {task_id!r}\n")
        _git(lease.path, "add", self.relative_path)
        _git(lease.path, "commit", "-m", f"candidate {task_id}")
        return WorkerResult(_git(lease.path, "rev-parse", "HEAD"), True, (f"worker:{task_id}",))


@dataclass
class AlwaysFailWorker:
    def run(self, *, task_id: str, lease: WorktreeLease) -> WorkerResult:
        del task_id, lease
        raise RuntimeError("same deterministic worker failure 123")


@dataclass
class PassingReviewer:
    def review(self, *, assignment: ReviewAssignment, worktree: Path) -> ReviewResult:
        assert worktree.is_dir()
        return ReviewResult(assignment, ReviewVerdict.PASS, (), ("review:pass",))


def _supervisor(
    tmp_path: Path, repo: Path, baseline: str, registry: TaskRegistry
) -> EngineeringSupervisor:
    harness = EngineeringHarness(
        repository=repo,
        worktree_manager=WorktreeManager(repository=repo, worktree_root=tmp_path / "worktrees"),
        registry=registry,
        protected_commands=(("protected", ("python3", "-c", "assert True")),),
        hidden_acceptance_commands=(("hidden", ("python3", "-c", "assert True")),),
        frozen_commit=baseline,
        freeze_tag="everlast-72h-build",
    )
    return EngineeringSupervisor(
        harness=harness,
        registry=registry,
        scheduler=ConservativeTaskScheduler(maximum_parallel_workers=2),
        metrics_store=HarnessMetricsStore(tmp_path / "metrics.jsonl"),
        release_candidate_directory=tmp_path / "release-candidates",
    )


def _execution(task_id: str, worker: FileWorker | AlwaysFailWorker) -> TaskExecution:
    return TaskExecution(
        task_id=task_id,
        worker_id=f"worker-{task_id}",
        worker_session_id=f"worker-session-{task_id}",
        reviewer_session_id=f"reviewer-session-{task_id}",
        worker=worker,
        reviewer=PassingReviewer(),
    )


def test_supervisor_records_metrics_and_release_candidate_without_merging(
    tmp_path: Path,
) -> None:
    repo, baseline = _repo(tmp_path)
    registry = _registry(tmp_path / "tasks.json", ("HF-1", "src/a.py"))
    supervisor = _supervisor(tmp_path, repo, baseline, registry)

    outcome = supervisor.run_task_bounded(_execution("HF-1", FileWorker("src/a.py")))

    assert outcome.final_status == "VERIFIED_PASS"
    assert outcome.attempts == 1
    assert registry.get("HF-1").status is TaskStatus.MERGE_CANDIDATE
    assert _git(repo, "rev-parse", "HEAD") == baseline
    assert outcome.release_candidate_path is not None
    bundle = json.loads(Path(outcome.release_candidate_path).read_text())
    assert bundle["human_deployment_required"] is True
    assert bundle["changed_paths"] == ["src/a.py"]
    metrics = supervisor.metrics_store.read_all()
    assert len(metrics) == 1
    assert metrics[0].result == "VERIFIED_PASS"


def test_two_disjoint_tasks_execute_concurrently_in_isolated_worktrees(
    tmp_path: Path,
) -> None:
    repo, baseline = _repo(tmp_path)
    registry = _registry(
        tmp_path / "tasks.json",
        ("HF-1", "src/a.py"),
        ("HF-2", "src/b.py"),
    )
    supervisor = _supervisor(tmp_path, repo, baseline, registry)
    barrier = Barrier(2)

    batch = supervisor.run_ready(
        {
            "HF-1": _execution("HF-1", FileWorker("src/a.py", barrier)),
            "HF-2": _execution("HF-2", FileWorker("src/b.py", barrier)),
        }
    )

    assert batch.schedule.scheduled == ("HF-1", "HF-2")
    assert set(batch.outcomes) == {"HF-1", "HF-2"}
    assert all(item.final_status == "VERIFIED_PASS" for item in batch.outcomes.values())
    commits = {item.harness_results[0].candidate_commit for item in batch.outcomes.values()}
    assert len(commits) == 2
    assert _git(repo, "rev-parse", "HEAD") == baseline


def test_repeated_worker_failure_opens_circuit_and_stops_after_one_retry(
    tmp_path: Path,
) -> None:
    repo, baseline = _repo(tmp_path)
    registry = _registry(tmp_path / "tasks.json", ("HF-1", "src/a.py"))
    supervisor = _supervisor(tmp_path, repo, baseline, registry)

    outcome = supervisor.run_task_bounded(_execution("HF-1", AlwaysFailWorker()))

    assert outcome.final_status == "BLOCKED"
    assert outcome.attempts == 4
    assert len(set(outcome.failure_fingerprints)) == 1
    metrics = supervisor.metrics_store.read_all()
    assert len(metrics) == 4
    assert sum(item.circuit_breaker_events for item in metrics) >= 1
    assert outcome.release_candidate_path is None
