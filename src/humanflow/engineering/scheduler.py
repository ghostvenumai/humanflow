"""Conflict-aware planning for bounded offline workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from .registry import EngineeringTaskRecord, TaskStatus


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    scheduled: tuple[str, ...]
    deferred: dict[str, str]


class ConservativeTaskScheduler:
    def __init__(self, *, maximum_parallel_workers: int = 2) -> None:
        if not 1 <= maximum_parallel_workers <= 3:
            raise ValueError("initial maximum_parallel_workers must be between 1 and 3")
        self.maximum_parallel_workers = maximum_parallel_workers

    def plan(
        self,
        tasks: tuple[EngineeringTaskRecord, ...],
        *,
        running: tuple[EngineeringTaskRecord, ...] = (),
        known_tasks: tuple[EngineeringTaskRecord, ...] = (),
    ) -> ScheduleDecision:
        scheduled: list[EngineeringTaskRecord] = []
        deferred: dict[str, str] = {}
        by_id = {task.task_id: task for task in (*known_tasks, *tasks, *running)}
        capacity = max(0, self.maximum_parallel_workers - len(running))
        ordered = sorted(tasks, key=lambda task: (task.priority.value, task.task_id))
        for task in ordered:
            if task.status is not TaskStatus.READY:
                deferred[task.task_id] = "TASK_NOT_READY"
                continue
            unresolved = [
                dependency
                for dependency in task.dependencies
                if dependency not in by_id
                or by_id[dependency].status
                not in {
                    TaskStatus.PASSED,
                    TaskStatus.MERGE_CANDIDATE,
                    TaskStatus.MERGED,
                    TaskStatus.RELEASED,
                    TaskStatus.MEASURED,
                }
            ]
            if unresolved:
                deferred[task.task_id] = "DEPENDENCY_NOT_VERIFIED"
                continue
            if len(scheduled) >= capacity:
                deferred[task.task_id] = "WORKER_LIMIT"
                continue
            peers = (*running, *scheduled)
            if any(tasks_conflict(task, peer) for peer in peers):
                deferred[task.task_id] = "RESOURCE_CONFLICT"
                continue
            scheduled.append(task)
        return ScheduleDecision(
            scheduled=tuple(task.task_id for task in scheduled), deferred=deferred
        )


def tasks_conflict(left: EngineeringTaskRecord, right: EngineeringTaskRecord) -> bool:
    if left.task_id == right.task_id:
        return True
    if not left.allowed_paths or not right.allowed_paths:
        return True
    return any(
        _path_rules_may_overlap(left_rule, right_rule)
        for left_rule in left.allowed_paths
        for right_rule in right.allowed_paths
    )


def path_matches(path: str, rule: str) -> bool:
    normalized_path = PurePosixPath(path).as_posix().strip("/")
    normalized_rule = PurePosixPath(rule).as_posix().strip("/")
    if not normalized_path or not normalized_rule:
        return False
    if normalized_rule.endswith("/**"):
        recursive_root = normalized_rule[:-3].rstrip("/")
        return normalized_path == recursive_root or normalized_path.startswith(recursive_root + "/")
    if not any(character in normalized_rule for character in "*?["):
        return normalized_path == normalized_rule or normalized_path.startswith(
            normalized_rule + "/"
        )
    return PurePosixPath(normalized_path).match(normalized_rule)


def _path_rules_may_overlap(left: str, right: str) -> bool:
    left_prefix = _literal_prefix(left)
    right_prefix = _literal_prefix(right)
    if not left_prefix or not right_prefix:
        return True
    return (
        left_prefix == right_prefix
        or left_prefix.startswith(right_prefix + "/")
        or right_prefix.startswith(left_prefix + "/")
    )


def _literal_prefix(rule: str) -> str:
    parts: list[str] = []
    for part in PurePosixPath(rule).parts:
        if any(character in part for character in "*?["):
            break
        parts.append(part)
    return "/".join(parts).rstrip("/")
