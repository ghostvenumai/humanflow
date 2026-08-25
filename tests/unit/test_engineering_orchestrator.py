from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from humanflow.engineering import (
    ActorRole,
    EngineeringHarness,
    EngineeringTaskRecord,
    ReviewAssignment,
    ReviewResult,
    ReviewVerdict,
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
    (repo / "src" / "value.py").write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


@dataclass
class LocalWorker:
    def run(self, *, task_id: str, lease: WorktreeLease) -> WorkerResult:
        del task_id
        (lease.path / "src" / "value.py").write_text("VALUE = 2\n")
        _git(lease.path, "add", "src/value.py")
        _git(lease.path, "commit", "-m", "candidate")
        return WorkerResult(_git(lease.path, "rev-parse", "HEAD"), True, ("worker-1",))


@dataclass
class PassingReviewer:
    def review(self, *, assignment: ReviewAssignment, worktree: Path) -> ReviewResult:
        assert worktree.is_dir()
        return ReviewResult(assignment, ReviewVerdict.PASS, (), ("review-1",))


def _ready_registry(path: Path) -> TaskRegistry:
    registry = TaskRegistry(path)
    registry.add(
        EngineeringTaskRecord(
            "HF-1",
            "Change deterministic value",
            "test",
            TaskPriority.P2,
            TaskRisk.LOW,
            allowed_paths=("src/value.py",),
            verification=(("python3", "-c", "assert True"),),
        )
    )
    registry.transition("HF-1", TaskStatus.TRIAGED, actor=ActorRole.COORDINATOR)
    registry.transition("HF-1", TaskStatus.READY, actor=ActorRole.COORDINATOR)
    registry.save()
    return registry


def test_harness_prepares_independently_verified_merge_candidate_without_merging(
    tmp_path: Path,
) -> None:
    repo, baseline = _repo(tmp_path)
    registry = _ready_registry(tmp_path / "feature_list.json")
    manager = WorktreeManager(repository=repo, worktree_root=tmp_path / "worktrees")
    harness = EngineeringHarness(
        repository=repo,
        worktree_manager=manager,
        registry=registry,
        protected_commands=(("protected", ("python3", "-c", "assert True")),),
        hidden_acceptance_commands=(("hidden", ("python3", "-c", "assert True")),),
    )

    result = harness.run_task(
        "HF-1",
        worker_id="local-worker",
        worker_session_id="worker-session",
        reviewer_session_id="reviewer-session",
        worker=LocalWorker(),
        reviewer=PassingReviewer(),
    )

    assert result.status == "VERIFIED_PASS"
    assert registry.get("HF-1").status is TaskStatus.MERGE_CANDIDATE
    assert result.baseline_commit == baseline
    assert _git(repo, "rev-parse", "HEAD") == baseline
    assert result.candidate_commit != baseline
    assert not (tmp_path / "worktrees" / "HF-1-local-worker").exists()


def test_harness_fails_closed_when_hidden_acceptance_is_not_configured(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    registry = _ready_registry(tmp_path / "feature_list.json")
    harness = EngineeringHarness(
        repository=repo,
        worktree_manager=WorktreeManager(repository=repo, worktree_root=tmp_path / "worktrees"),
        registry=registry,
        protected_commands=(("protected", ("python3", "-c", "assert True")),),
        hidden_acceptance_commands=(),
    )

    with pytest.raises(RuntimeError, match="hidden acceptance"):
        harness.run_task(
            "HF-1",
            worker_id="local-worker",
            worker_session_id="worker-session",
            reviewer_session_id="reviewer-session",
            worker=LocalWorker(),
            reviewer=PassingReviewer(),
        )

    assert registry.get("HF-1").status is TaskStatus.READY


def test_operator_cli_is_read_only_by_default() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [str(root / "hf-loop"), "run-ready"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["reason"] == "EXTERNAL_EXECUTION_DISABLED"
    assert payload["production_deployment"] == "MANUAL_ONLY"

    parallel = subprocess.run(
        [str(root / "hf-loop"), "parallel", "--workers", "2"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert parallel.returncode == 2
    parallel_payload = json.loads(parallel.stdout)
    assert parallel_payload["workers"] == 2
    assert parallel_payload["reason"] == "EXTERNAL_EXECUTION_DISABLED"
