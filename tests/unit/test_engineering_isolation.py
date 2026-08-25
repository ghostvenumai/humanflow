from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from humanflow.engineering import (
    CandidateIntegrityGate,
    ConservativeTaskScheduler,
    EngineeringTaskRecord,
    IntegrityReport,
    MergeGateEvidence,
    ReviewAssignment,
    ReviewResult,
    ReviewVerdict,
    SupervisorCommandRunner,
    TaskPriority,
    TaskRisk,
    TaskStatus,
    WorktreeManager,
    evaluate_merge_gate,
    tasks_conflict,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "HumanFlow Test")
    (repo / "src").mkdir()
    (repo / "src" / "base.py").write_text("BASE = True\n", encoding="utf-8")
    (repo / "tests" / "golden").mkdir(parents=True)
    (repo / "tests" / "golden" / "fixture.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _task(
    task_id: str,
    *allowed_paths: str,
    status: TaskStatus = TaskStatus.READY,
) -> EngineeringTaskRecord:
    return EngineeringTaskRecord(
        task_id=task_id,
        title=task_id,
        source="test",
        priority=TaskPriority.P1,
        risk=TaskRisk.MEDIUM,
        status=status,
        allowed_paths=allowed_paths,
        protected_paths=("tests/golden/**",),
    )


def test_two_worktrees_start_from_same_commit_and_remain_isolated(tmp_path: Path) -> None:
    repo, baseline = _repository(tmp_path)
    manager = WorktreeManager(repository=repo, worktree_root=tmp_path / "worktrees")
    left = manager.create(task_id="HF-1", worker_id="codex-a", baseline_commit=baseline)
    right = manager.create(task_id="HF-2", worker_id="codex-b", baseline_commit=baseline)

    (left.path / "src" / "left.py").write_text("LEFT = True\n", encoding="utf-8")

    assert manager.inspect(left)["head"] == baseline
    assert manager.inspect(right)["head"] == baseline
    assert not (right.path / "src" / "left.py").exists()
    with pytest.raises(RuntimeError, match="dirty"):
        manager.cleanup(left)
    (left.path / "src" / "left.py").unlink()
    manager.cleanup(left)
    manager.cleanup(right)


def test_scheduler_parallelizes_disjoint_paths_and_serializes_conflicts() -> None:
    date_task = _task("HF-1", "src/humanflow/runtime/temporal.py")
    telemetry_task = _task("HF-2", "src/humanflow/telemetry/**")
    overlapping = _task("HF-3", "src/humanflow/runtime/**")
    scheduler = ConservativeTaskScheduler(maximum_parallel_workers=2)

    parallel = scheduler.plan((date_task, telemetry_task))
    conflict = scheduler.plan((date_task, overlapping))

    assert parallel.scheduled == ("HF-1", "HF-2")
    assert conflict.scheduled == ("HF-1",)
    assert conflict.deferred == {"HF-3": "RESOURCE_CONFLICT"}
    assert tasks_conflict(date_task, overlapping)


def test_integrity_gate_rejects_protected_and_out_of_scope_changes(tmp_path: Path) -> None:
    repo, baseline = _repository(tmp_path)
    (repo / "tests" / "golden" / "fixture.json").write_text('{"weakened": true}\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "bad candidate")
    candidate = _git(repo, "rev-parse", "HEAD")

    report = CandidateIntegrityGate().inspect(
        repository=repo,
        baseline_commit=baseline,
        candidate_commit=candidate,
        task=_task("HF-1", "src/**"),
    )

    assert report.passed is False
    assert "PROTECTED_PATH_MODIFIED:tests/golden/fixture.json" in report.findings
    assert "OUTSIDE_ALLOWED_PATHS:tests/golden/fixture.json" in report.findings


def test_integrity_gate_rejects_added_skip_even_when_test_path_is_allowed(
    tmp_path: Path,
) -> None:
    repo, baseline = _repository(tmp_path)
    test_file = repo / "tests" / "visible_test.py"
    test_file.write_text("import pytest\n\n@pytest.mark.skip\ndef test_x():\n    assert True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "hide failure")

    report = CandidateIntegrityGate().inspect(
        repository=repo,
        baseline_commit=baseline,
        candidate_commit=_git(repo, "rev-parse", "HEAD"),
        task=_task("HF-1", "tests/**"),
    )

    assert report.passed is False
    assert "TEST_WEAKENING_PATTERN_ADDED" in report.findings


def test_engineering_acceptance_fixtures_are_supervisor_protected(
    tmp_path: Path,
) -> None:
    repo, _ = _repository(tmp_path)
    fixture = repo / "eval" / "engineering" / "scenarios.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add engineering acceptance fixture")
    baseline = _git(repo, "rev-parse", "HEAD")
    fixture.write_text('{"weakened": true}\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "weaken engineering acceptance fixture")

    report = CandidateIntegrityGate().inspect(
        repository=repo,
        baseline_commit=baseline,
        candidate_commit=_git(repo, "rev-parse", "HEAD"),
        task=_task("HF-1", "eval/engineering/**"),
    )

    assert report.passed is False
    assert "PROTECTED_PATH_MODIFIED:eval/engineering/scenarios.json" in report.findings


def test_reviewer_is_independent_and_hidden_command_returns_only_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fresh independent"):
        ReviewAssignment("HF-1", "same", "same", "base", "candidate")
    assignment = ReviewAssignment("HF-1", "worker", "reviewer", "base", "candidate")
    review = ReviewResult(assignment, ReviewVerdict.PASS, (), ("review-1",))
    command = SupervisorCommandRunner(timeout_seconds=5).run(
        name="hidden-acceptance",
        argv=(sys.executable, "-c", "print('hidden pass')"),
        cwd=tmp_path,
    )

    assert review.verdict is ReviewVerdict.PASS
    assert command.passed
    assert command.output_tail == ("hidden pass",)


def test_merge_gate_fails_closed_and_requires_latest_main_revalidation() -> None:
    assignment = ReviewAssignment("HF-1", "worker", "reviewer", "base", "candidate")
    review = ReviewResult(assignment, ReviewVerdict.PASS, (), ("review-1",))
    integrity = IntegrityReport(True, ("src/change.py",), ())
    passing = MergeGateEvidence(True, True, True, True, integrity, review, False, True)
    stale = MergeGateEvidence(True, True, True, True, integrity, review, False, False)

    assert evaluate_merge_gate(passing)["status"] == "VERIFIED_PASS"
    assert evaluate_merge_gate(stale) == {
        "status": "MERGE_REJECTED",
        "reasons": ["LATEST_MAIN_REVALIDATION_MISSING"],
    }
