"""Versioned HF task registry with coordinator/verifier-owned transitions."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping


class TaskStatus(StrEnum):
    PROPOSED = "proposed"
    TRIAGED = "triaged"
    READY = "ready"
    RUNNING = "running"
    VERIFICATION = "verification"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    MERGE_CANDIDATE = "merge_candidate"
    MERGED = "merged"
    RELEASED = "released"
    MEASURED = "measured"


class TaskRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskPriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class ActorRole(StrEnum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    VERIFIER = "verifier"
    HUMAN = "human"


_ALLOWED_TRANSITIONS: Mapping[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PROPOSED: frozenset({TaskStatus.TRIAGED, TaskStatus.BLOCKED}),
    TaskStatus.TRIAGED: frozenset({TaskStatus.READY, TaskStatus.BLOCKED}),
    TaskStatus.READY: frozenset({TaskStatus.RUNNING, TaskStatus.BLOCKED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.VERIFICATION, TaskStatus.FAILED, TaskStatus.BLOCKED}),
    TaskStatus.VERIFICATION: frozenset({TaskStatus.PASSED, TaskStatus.FAILED, TaskStatus.BLOCKED}),
    TaskStatus.PASSED: frozenset({TaskStatus.MERGE_CANDIDATE}),
    TaskStatus.FAILED: frozenset({TaskStatus.READY, TaskStatus.BLOCKED}),
    TaskStatus.BLOCKED: frozenset({TaskStatus.TRIAGED, TaskStatus.READY}),
    TaskStatus.MERGE_CANDIDATE: frozenset({TaskStatus.MERGED, TaskStatus.FAILED}),
    TaskStatus.MERGED: frozenset({TaskStatus.RELEASED}),
    TaskStatus.RELEASED: frozenset({TaskStatus.MEASURED, TaskStatus.FAILED}),
    TaskStatus.MEASURED: frozenset(),
}
_VERIFIER_ONLY = frozenset({TaskStatus.PASSED, TaskStatus.FAILED, TaskStatus.MERGE_CANDIDATE})
_TASK_KEYS = frozenset(
    {
        "task_id",
        "title",
        "source",
        "priority",
        "risk",
        "status",
        "passes",
        "problem_fingerprint",
        "dependencies",
        "allowed_paths",
        "protected_paths",
        "verification",
        "target_metrics",
        "evidence_refs",
        "human_approval_required",
        "human_approved",
        "created_at_utc",
        "updated_at_utc",
    }
)
_REQUIRED_TASK_KEYS = frozenset(
    {"task_id", "title", "source", "priority", "risk", "status", "passes"}
)


@dataclass(frozen=True, slots=True)
class EngineeringTaskRecord:
    task_id: str
    title: str
    source: str
    priority: TaskPriority
    risk: TaskRisk
    status: TaskStatus = TaskStatus.PROPOSED
    passes: bool = False
    problem_fingerprint: str | None = None
    dependencies: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    verification: tuple[tuple[str, ...], ...] = ()
    target_metrics: Mapping[str, str] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    human_approval_required: bool = False
    human_approved: bool = False
    created_at_utc: str = field(default_factory=lambda: _timestamp())
    updated_at_utc: str = field(default_factory=lambda: _timestamp())

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.title.strip() or not self.source.strip():
            raise ValueError("task identity, title and source must not be empty")
        if self.passes != (
            self.status
            in {
                TaskStatus.PASSED,
                TaskStatus.MERGE_CANDIDATE,
                TaskStatus.MERGED,
                TaskStatus.RELEASED,
                TaskStatus.MEASURED,
            }
        ):
            raise ValueError("passes must be derived from verified task status")
        for path in (*self.allowed_paths, *self.protected_paths):
            _validate_relative_path(path)
        if set(self.allowed_paths).intersection(self.protected_paths):
            raise ValueError("allowed_paths and protected_paths must not overlap exactly")
        for command in self.verification:
            if not command or any(not part.strip() for part in command):
                raise ValueError("verification commands must be non-empty argv tuples")
        object.__setattr__(self, "target_metrics", MappingProxyType(dict(self.target_metrics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "source": self.source,
            "priority": self.priority.value,
            "risk": self.risk.value,
            "status": self.status.value,
            "passes": self.passes,
            "problem_fingerprint": self.problem_fingerprint,
            "dependencies": list(self.dependencies),
            "allowed_paths": list(self.allowed_paths),
            "protected_paths": list(self.protected_paths),
            "verification": [list(command) for command in self.verification],
            "target_metrics": dict(self.target_metrics),
            "evidence_refs": list(self.evidence_refs),
            "human_approval_required": self.human_approval_required,
            "human_approved": self.human_approved,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineeringTaskRecord":
        status = TaskStatus(str(payload.get("status", TaskStatus.PROPOSED.value)))
        return cls(
            task_id=str(payload["task_id"]),
            title=str(payload["title"]),
            source=str(payload["source"]),
            priority=TaskPriority(str(payload["priority"])),
            risk=TaskRisk(str(payload["risk"])),
            status=status,
            passes=bool(payload.get("passes", False)),
            problem_fingerprint=(
                str(payload["problem_fingerprint"])
                if payload.get("problem_fingerprint") is not None
                else None
            ),
            dependencies=tuple(str(item) for item in payload.get("dependencies", [])),
            allowed_paths=tuple(str(item) for item in payload.get("allowed_paths", [])),
            protected_paths=tuple(str(item) for item in payload.get("protected_paths", [])),
            verification=tuple(
                tuple(str(part) for part in command) for command in payload.get("verification", [])
            ),
            target_metrics=(
                payload.get("target_metrics", {})
                if isinstance(payload.get("target_metrics", {}), Mapping)
                else {}
            ),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", [])),
            human_approval_required=bool(payload.get("human_approval_required", False)),
            human_approved=bool(payload.get("human_approved", False)),
            created_at_utc=str(payload.get("created_at_utc", _timestamp())),
            updated_at_utc=str(payload.get("updated_at_utc", _timestamp())),
        )


class TaskRegistry:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, tasks: tuple[EngineeringTaskRecord, ...] = ()) -> None:
        self.path = path
        self._tasks = {task.task_id: task for task in tasks}
        self._lock = RLock()
        if len(self._tasks) != len(tasks):
            raise ValueError("task IDs must be unique")

    @classmethod
    def load(cls, path: Path) -> "TaskRegistry":
        if not path.exists():
            return cls(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported task registry schema_version")
        raw_tasks = payload.get("features")
        if not isinstance(raw_tasks, list):
            raise ValueError("features must be a list")
        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise ValueError("every feature must be an object")
            missing = _REQUIRED_TASK_KEYS.difference(item)
            unknown = set(item).difference(_TASK_KEYS)
            if missing:
                raise ValueError(f"feature is missing required fields: {sorted(missing)}")
            if unknown:
                raise ValueError(f"feature has unknown fields: {sorted(unknown)}")
        return cls(path, tuple(EngineeringTaskRecord.from_dict(item) for item in raw_tasks))

    @property
    def tasks(self) -> tuple[EngineeringTaskRecord, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def get(self, task_id: str) -> EngineeringTaskRecord:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as error:
                raise KeyError(f"unknown task: {task_id}") from error

    def add(self, task: EngineeringTaskRecord) -> EngineeringTaskRecord:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task_id: {task.task_id}")
        if task.problem_fingerprint is not None:
            duplicate = next(
                (
                    current
                    for current in self._tasks.values()
                    if current.problem_fingerprint == task.problem_fingerprint
                ),
                None,
            )
            if duplicate is not None:
                linked = replace(
                    duplicate,
                    evidence_refs=tuple(
                        dict.fromkeys((*duplicate.evidence_refs, *task.evidence_refs))
                    ),
                    updated_at_utc=_timestamp(),
                )
                self._tasks[duplicate.task_id] = linked
                return linked
        self._tasks[task.task_id] = task
        return task

    def transition(
        self,
        task_id: str,
        target: TaskStatus,
        *,
        actor: ActorRole,
        evidence_refs: tuple[str, ...] = (),
    ) -> EngineeringTaskRecord:
        if target is TaskStatus.MERGED:
            raise PermissionError("merged requires record_verified_merge")
        with self._lock:
            return self._transition_unlocked(
                task_id, target, actor=actor, evidence_refs=evidence_refs
            )

    def _transition_unlocked(
        self,
        task_id: str,
        target: TaskStatus,
        *,
        actor: ActorRole,
        evidence_refs: tuple[str, ...],
    ) -> EngineeringTaskRecord:
        current = self.get(task_id)
        if target not in _ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(f"illegal task transition: {current.status.value}->{target.value}")
        if target in _VERIFIER_ONLY and actor is not ActorRole.VERIFIER:
            raise PermissionError(f"{target.value} requires verifier authority")
        if target is TaskStatus.RELEASED and actor is not ActorRole.HUMAN:
            raise PermissionError("initial release policy requires human authority")
        if target is TaskStatus.RUNNING and actor is not ActorRole.COORDINATOR:
            raise PermissionError("only the coordinator may dispatch a task")
        if (
            target is TaskStatus.READY
            and current.human_approval_required
            and not current.human_approved
        ):
            raise PermissionError("task requires human approval before READY")
        updated = replace(
            current,
            status=target,
            passes=target
            in {
                TaskStatus.PASSED,
                TaskStatus.MERGE_CANDIDATE,
                TaskStatus.MERGED,
                TaskStatus.RELEASED,
                TaskStatus.MEASURED,
            },
            evidence_refs=tuple(dict.fromkeys((*current.evidence_refs, *evidence_refs))),
            updated_at_utc=_timestamp(),
        )
        self._tasks[task_id] = updated
        return updated

    def record_verified_merge(
        self,
        task_id: str,
        *,
        repository: Path,
        candidate_commit: str,
        actor: ActorRole,
        evidence_refs: tuple[str, ...],
    ) -> EngineeringTaskRecord:
        if actor not in {ActorRole.COORDINATOR, ActorRole.HUMAN}:
            raise PermissionError("merged requires coordinator or human authority")
        if not evidence_refs:
            raise PermissionError("merged requires verifier evidence")
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", candidate_commit) is None:
            raise PermissionError("candidate commit must be a full hexadecimal object ID")
        repository = repository.resolve()
        candidate = _git(repository, "rev-parse", f"{candidate_commit}^{{commit}}")
        latest_main = _git(repository, "rev-parse", "HEAD^{commit}")
        if not _is_ancestor(repository, candidate, latest_main):
            raise PermissionError("candidate commit is not contained in latest main")
        with self._lock:
            return self._transition_unlocked(
                task_id,
                TaskStatus.MERGED,
                actor=actor,
                evidence_refs=tuple(
                    dict.fromkeys(
                        (*evidence_refs, f"git-candidate:{candidate}", f"git-merge:{latest_main}")
                    )
                ),
            )

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "features": [task.to_dict() for task in self.tasks],
            }
            encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.path)
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"task path must be a safe repository-relative path: {value}")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
