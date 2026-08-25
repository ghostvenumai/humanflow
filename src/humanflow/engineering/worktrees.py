"""Conservative Git worktree lifecycle for isolated engineering workers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True, slots=True)
class WorktreeLease:
    task_id: str
    worker_id: str
    path: Path
    branch: str
    baseline_commit: str


class WorktreeManager:
    def __init__(self, *, repository: Path, worktree_root: Path) -> None:
        self.repository = repository.resolve()
        self.worktree_root = worktree_root.resolve()
        self._git_admin_lock = Lock()
        if self.worktree_root == self.repository or self.repository in self.worktree_root.parents:
            raise ValueError("worktree_root must not be inside the source repository")

    def create(self, *, task_id: str, worker_id: str, baseline_commit: str) -> WorktreeLease:
        _validate_identifier(task_id, "task_id")
        _validate_identifier(worker_id, "worker_id")
        resolved_baseline = self._git("rev-parse", f"{baseline_commit}^{{commit}}")
        slug = f"{task_id}-{worker_id}"
        path = (self.worktree_root / slug).resolve()
        if path.parent != self.worktree_root:
            raise ValueError("worktree path escaped configured root")
        if path.exists():
            raise FileExistsError(path)
        branch = f"agent/{slug}"
        with self._git_admin_lock:
            if self._branch_exists(branch):
                raise ValueError(f"worker branch already exists: {branch}")
            self.worktree_root.mkdir(parents=True, exist_ok=True)
            self._git("worktree", "add", "-b", branch, str(path), resolved_baseline)
        return WorktreeLease(task_id, worker_id, path, branch, resolved_baseline)

    def inspect(self, lease: WorktreeLease) -> dict[str, object]:
        self._validate_lease_path(lease)
        return {
            "task_id": lease.task_id,
            "worker_id": lease.worker_id,
            "path": str(lease.path),
            "branch": self._git_at(lease.path, "branch", "--show-current"),
            "head": self._git_at(lease.path, "rev-parse", "HEAD"),
            "working_tree_clean": not bool(self._git_at(lease.path, "status", "--porcelain")),
        }

    def cleanup(self, lease: WorktreeLease) -> None:
        """Remove only a clean managed worktree; retain its branch as evidence."""

        self._validate_lease_path(lease)
        if not lease.path.exists():
            return
        if self._git_at(lease.path, "branch", "--show-current") != lease.branch:
            raise RuntimeError("managed worktree branch identity changed")
        if self._git_at(lease.path, "status", "--porcelain"):
            raise RuntimeError("refusing to remove dirty worker worktree")
        with self._git_admin_lock:
            self._git("worktree", "remove", str(lease.path))

    def _validate_lease_path(self, lease: WorktreeLease) -> None:
        path = lease.path.resolve()
        if path.parent != self.worktree_root:
            raise ValueError("lease is outside configured worktree root")

    def _branch_exists(self, branch: str) -> bool:
        completed = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=self.repository,
            check=False,
        )
        return completed.returncode == 0

    def _git(self, *arguments: str) -> str:
        return self._git_at(self.repository, *arguments)

    @staticmethod
    def _git_at(root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()


def _validate_identifier(value: str, name: str) -> None:
    if _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"unsafe {name}: {value}")
