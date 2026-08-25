from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from humanflow.finalization.finish_loop import FinishLoopState, verify_post_freeze_repository


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _post_freeze_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "humanflow/appointment-tools")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "HumanFlow Test")
    (repo / "evidence.txt").write_text("frozen", encoding="utf-8")
    _git(repo, "add", "evidence.txt")
    _git(repo, "commit", "-m", "frozen")
    frozen = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "everlast-72h-build", frozen)
    (repo / "post-freeze.txt").write_text("extension", encoding="utf-8")
    _git(repo, "add", "post-freeze.txt")
    _git(repo, "commit", "-m", "post-freeze")
    return repo, frozen


def test_finish_loop_state_is_resumable_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = FinishLoopState(
        phase="B",
        iteration=4,
        task="aggregation",
        status="KEEP",
        baseline_commit="abc",
        candidate_commit="def",
        tests_run=["pytest cost"],
        test_results=["PASS"],
        paid_provider_calls={"anthropic": 0},
    )
    state.save(path)
    loaded = FinishLoopState.load(path)

    assert loaded.phase == "B"
    assert loaded.iteration == 4
    assert json.loads(path.read_text(encoding="utf-8"))["paid_provider_calls"] == {
        "anthropic": 0
    }
    with pytest.raises(ValueError, match="exceeds"):
        FinishLoopState(
            phase="C", iteration=2, task="too-many", status="KEEP", baseline_commit="abc"
        )
    with pytest.raises(ValueError, match="validation reserve"):
        FinishLoopState(
            phase="A",
            iteration=1,
            task="feature",
            status="IN_PROGRESS",
            baseline_commit="abc",
            resource_remaining_percent=20,
        )


def test_repository_guard_preserves_frozen_tag_and_requires_descendant(
    tmp_path: Path,
) -> None:
    repo, frozen = _post_freeze_repo(tmp_path)
    evidence = verify_post_freeze_repository(
        repo,
        frozen_commit=frozen,
        freeze_tag="everlast-72h-build",
    )

    assert evidence["branch"] == "humanflow/appointment-tools"
    assert evidence["freeze_tag_commit"] == frozen
    assert evidence["head"] != frozen

    (repo / "STOP_LOOP").touch()
    with pytest.raises(RuntimeError, match="STOP_LOOP"):
        verify_post_freeze_repository(
            repo,
            frozen_commit=frozen,
            freeze_tag="everlast-72h-build",
        )
