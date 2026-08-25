"""Release-candidate evidence and non-executing rollback recommendations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class PostReleaseStatus(StrEnum):
    MEASURED_SUCCESS = "MEASURED_SUCCESS"
    TARGET_NOT_REACHED = "TARGET_NOT_REACHED"
    RELEASE_REJECTED = "RELEASE_REJECTED"


@dataclass(frozen=True, slots=True)
class ReleaseCandidateBundle:
    task_id: str
    baseline_commit: str
    candidate_commit: str
    latest_main_commit: str
    previous_good_commit: str
    verification_status: str
    evidence_refs: tuple[str, ...]
    changed_paths: tuple[str, ...]
    protected_hashes_sha256: str
    human_deployment_required: bool = True

    def __post_init__(self) -> None:
        identities = (
            self.task_id,
            self.baseline_commit,
            self.candidate_commit,
            self.latest_main_commit,
            self.previous_good_commit,
            self.protected_hashes_sha256,
        )
        if any(not value.strip() for value in identities):
            raise ValueError("release candidate identity must not be empty")
        if self.verification_status != "VERIFIED_PASS":
            raise ValueError("only VERIFIED_PASS may become a release candidate")
        if not self.human_deployment_required:
            raise ValueError("initial deployment policy must remain human-gated")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["changed_paths"] = list(self.changed_paths)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ReleaseMeasurement:
    task_id: str
    metric_name: str
    baseline_value: float
    current_value: float
    target_value: float
    higher_is_better: bool
    sample_size: int
    evidence_refs: tuple[str, ...]


def evaluate_post_release(measurement: ReleaseMeasurement) -> dict[str, Any]:
    if measurement.sample_size < 1 or not measurement.evidence_refs:
        raise ValueError("post-release measurement requires samples and evidence")
    meets_target = (
        measurement.current_value >= measurement.target_value
        if measurement.higher_is_better
        else measurement.current_value <= measurement.target_value
    )
    regression = (
        measurement.current_value < measurement.baseline_value
        if measurement.higher_is_better
        else measurement.current_value > measurement.baseline_value
    )
    if regression:
        status = PostReleaseStatus.RELEASE_REJECTED
        rollback_recommended = True
    elif meets_target:
        status = PostReleaseStatus.MEASURED_SUCCESS
        rollback_recommended = False
    else:
        status = PostReleaseStatus.TARGET_NOT_REACHED
        rollback_recommended = False
    return {
        "status": status.value,
        "task_id": measurement.task_id,
        "metric_name": measurement.metric_name,
        "baseline_value": measurement.baseline_value,
        "current_value": measurement.current_value,
        "target_value": measurement.target_value,
        "sample_size": measurement.sample_size,
        "evidence_refs": list(measurement.evidence_refs),
        "rollback_recommended": rollback_recommended,
        "rollback_executed": False,
    }
