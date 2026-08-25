"""Small resumable guard for the bounded post-freeze finish loops."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LOOP_LIMITS = {"A": 3, "B": 4, "C": 1}


@dataclass(slots=True)
class FinishLoopState:
    phase: str
    iteration: int
    task: str
    status: str
    baseline_commit: str
    candidate_commit: str | None = None
    tests_run: list[str] = field(default_factory=list)
    test_results: list[str] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    paid_provider_calls: dict[str, int] = field(default_factory=dict)
    remaining_blockers: list[str] = field(default_factory=list)
    resource_remaining_percent: int | None = None
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )

    def __post_init__(self) -> None:
        if self.phase not in LOOP_LIMITS:
            raise ValueError("phase must be A, B, or C")
        if not 1 <= self.iteration <= LOOP_LIMITS[self.phase]:
            raise ValueError(f"iteration exceeds bounded Loop {self.phase} limit")
        if self.status not in {
            "IN_PROGRESS",
            "KEEP",
            "REVERT",
            "STOPPED",
            "HUMAN_VALIDATION_PENDING",
        }:
            raise ValueError("invalid finish-loop status")
        if self.resource_remaining_percent is not None:
            if not 0 <= self.resource_remaining_percent <= 100:
                raise ValueError("resource percentage must be between zero and 100")
            if self.resource_remaining_percent <= 20 and self.status == "IN_PROGRESS":
                raise ValueError("feature work must stop at the validation reserve")

    @classmethod
    def load(cls, path: Path) -> "FinishLoopState":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["updated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def verify_post_freeze_repository(
    repo: Path,
    *,
    frozen_commit: str,
    freeze_tag: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    if (repo / "STOP_LOOP").exists():
        raise RuntimeError("STOP_LOOP exists")
    branch = _git(repo, "branch", "--show-current")
    if not branch or branch == "HEAD":
        raise RuntimeError("post-freeze work requires a named branch")
    head = _git(repo, "rev-parse", "HEAD")
    resolved_frozen = _git(repo, "rev-parse", frozen_commit)
    resolved_tag = _git(repo, "rev-parse", f"{freeze_tag}^{{}}")
    if resolved_tag != resolved_frozen:
        raise RuntimeError("frozen tag no longer resolves to the frozen commit")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_frozen, head],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise RuntimeError("HEAD is not descended from the frozen commit")
    if head == resolved_frozen:
        raise RuntimeError("post-freeze development cannot run on the frozen commit")
    return {
        "branch": branch,
        "head": head,
        "frozen_commit": resolved_frozen,
        "freeze_tag": freeze_tag,
        "freeze_tag_commit": resolved_tag,
        "working_tree_clean": not bool(_git(repo, "status", "--porcelain")),
    }


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
