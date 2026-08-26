from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from humanflow.engineering import (
    ActorRole,
    AppendOnlyEvidenceStore,
    EngineeringEvidenceRef,
    EngineeringTaskRecord,
    EvidenceKind,
    EvidenceScope,
    TaskPriority,
    TaskRegistry,
    TaskRisk,
    TaskStatus,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _task(
    task_id: str = "HF-241", *, fingerprint: str | None = "date-cluster"
) -> EngineeringTaskRecord:
    return EngineeringTaskRecord(
        task_id=task_id,
        title="Improve relative date interpretation",
        source="telemetry_detector",
        priority=TaskPriority.P1,
        risk=TaskRisk.MEDIUM,
        problem_fingerprint=fingerprint,
        allowed_paths=("src/humanflow/runtime/temporal.py",),
        protected_paths=("tests/golden/", "docs/METRICS.md"),
        verification=(("python3", "-m", "pytest", "-q", "tests/unit/test_temporal_resolver.py"),),
        target_metrics={"relative_date_success": ">=0.99"},
        evidence_refs=("evidence-1",),
    )


def test_evidence_store_is_append_only_deduplicated_and_privacy_safe(
    tmp_path: Path,
) -> None:
    store = AppendOnlyEvidenceStore(tmp_path / "evidence.jsonl")
    evidence = EngineeringEvidenceRef.create(
        evidence_id="evidence-1",
        kind=EvidenceKind.METRIC,
        scope=EvidenceScope.LOCAL_SYNTHETIC,
        uri="reports/runtime-quality-eval.json#/metrics/date_success",
        sample_size=220,
        metadata={"metric_name": "relative_date_success", "value": 0.91},
    )

    store.append(evidence)

    assert store.read_all() == (evidence,)
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        store.append(evidence)
    with pytest.raises(ValueError, match="forbidden"):
        EngineeringEvidenceRef.create(
            evidence_id="raw-data",
            kind=EvidenceKind.TELEMETRY_EVENT,
            scope=EvidenceScope.REAL_BROWSER_SESSION,
            uri="telemetry:event-1",
            metadata={"raw_text": "private conversation"},
        )


def test_task_registry_persists_atomically_and_deduplicates_problem_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feature_list.json"
    registry = TaskRegistry(path)
    registry.add(_task())
    linked = registry.add(
        EngineeringTaskRecord(
            task_id="HF-242",
            title="Duplicate detector output",
            source="telemetry_detector",
            priority=TaskPriority.P1,
            risk=TaskRisk.MEDIUM,
            problem_fingerprint="date-cluster",
            evidence_refs=("evidence-2",),
        )
    )
    registry.save()
    loaded = TaskRegistry.load(path)

    assert linked.task_id == "HF-241"
    assert linked.evidence_refs == ("evidence-1", "evidence-2")
    assert len(loaded.tasks) == 1
    assert loaded.get("HF-241").target_metrics == {"relative_date_success": ">=0.99"}
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_registry_json_schema_matches_runtime_contract(tmp_path: Path) -> None:
    path = tmp_path / "feature_list.json"
    registry = TaskRegistry(path)
    registry.add(_task())
    registry.save()
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_path = Path(__file__).parents[2] / "schemas" / "engineering-task-registry.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(payload)

    payload["features"][0]["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        TaskRegistry.load(path)


def test_worker_cannot_self_verify_and_high_risk_ready_requires_human_approval(
    tmp_path: Path,
) -> None:
    registry = TaskRegistry(tmp_path / "feature_list.json")
    registry.add(_task())
    registry.transition("HF-241", TaskStatus.TRIAGED, actor=ActorRole.COORDINATOR)
    registry.transition("HF-241", TaskStatus.READY, actor=ActorRole.COORDINATOR)
    registry.transition("HF-241", TaskStatus.RUNNING, actor=ActorRole.COORDINATOR)
    registry.transition("HF-241", TaskStatus.VERIFICATION, actor=ActorRole.WORKER)

    with pytest.raises(PermissionError, match="verifier"):
        registry.transition("HF-241", TaskStatus.PASSED, actor=ActorRole.WORKER)
    passed = registry.transition(
        "HF-241",
        TaskStatus.PASSED,
        actor=ActorRole.VERIFIER,
        evidence_refs=("verification-1",),
    )
    assert passed.passes is True

    registry.add(
        EngineeringTaskRecord(
            task_id="HF-900",
            title="Critical state-machine change",
            source="human_report",
            priority=TaskPriority.P0,
            risk=TaskRisk.CRITICAL,
            human_approval_required=True,
        )
    )
    registry.transition("HF-900", TaskStatus.TRIAGED, actor=ActorRole.COORDINATOR)
    with pytest.raises(PermissionError, match="human approval"):
        registry.transition("HF-900", TaskStatus.READY, actor=ActorRole.COORDINATOR)

    with pytest.raises(PermissionError, match="human authority"):
        registry.record_human_approval(
            "HF-900",
            actor=ActorRole.WORKER,
            evidence_refs=("operator-message:2026-08-26",),
        )
    approved = registry.record_human_approval(
        "HF-900",
        actor=ActorRole.HUMAN,
        evidence_refs=("operator-message:2026-08-26",),
    )
    assert approved.human_approved is True
    ready = registry.transition(
        "HF-900", TaskStatus.READY, actor=ActorRole.COORDINATOR
    )
    assert ready.status is TaskStatus.READY


def test_initial_release_transition_requires_human_authority(tmp_path: Path) -> None:
    registry = TaskRegistry(tmp_path / "feature_list.json")
    registry.add(_task())
    for target, actor in (
        (TaskStatus.TRIAGED, ActorRole.COORDINATOR),
        (TaskStatus.READY, ActorRole.COORDINATOR),
        (TaskStatus.RUNNING, ActorRole.COORDINATOR),
        (TaskStatus.VERIFICATION, ActorRole.WORKER),
        (TaskStatus.PASSED, ActorRole.VERIFIER),
        (TaskStatus.MERGE_CANDIDATE, ActorRole.VERIFIER),
    ):
        registry.transition("HF-241", target, actor=actor)

    with pytest.raises(PermissionError, match="record_verified_merge"):
        registry.transition(
            "HF-241",
            TaskStatus.MERGED,
            actor=ActorRole.WORKER,
            evidence_refs=("git-merge:unsafe",),
        )
    with pytest.raises(PermissionError, match="record_verified_merge"):
        registry.transition(
            "HF-241",
            TaskStatus.MERGED,
            actor=ActorRole.COORDINATOR,
            evidence_refs=("not-a-git-object",),
        )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "HumanFlow Test")
    (repo / "candidate.txt").write_text("verified\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "verified candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(PermissionError, match="full hexadecimal"):
        registry.record_verified_merge(
            "HF-241",
            repository=repo,
            candidate_commit="not-a-git-object",
            actor=ActorRole.COORDINATOR,
            evidence_refs=("release-candidate:HF-241",),
        )
    merged = registry.record_verified_merge(
        "HF-241",
        repository=repo,
        candidate_commit=candidate,
        actor=ActorRole.COORDINATOR,
        evidence_refs=("release-candidate:HF-241",),
    )
    assert f"git-candidate:{candidate}" in merged.evidence_refs

    with pytest.raises(PermissionError, match="human"):
        registry.transition("HF-241", TaskStatus.RELEASED, actor=ActorRole.COORDINATOR)
    released = registry.transition("HF-241", TaskStatus.RELEASED, actor=ActorRole.HUMAN)
    assert released.status is TaskStatus.RELEASED


def test_task_paths_and_passes_cannot_be_forged() -> None:
    with pytest.raises(ValueError, match="safe repository-relative"):
        EngineeringTaskRecord(
            task_id="HF-bad-path",
            title="escape",
            source="test",
            priority=TaskPriority.P2,
            risk=TaskRisk.LOW,
            allowed_paths=("../outside",),
        )
    with pytest.raises(ValueError, match="passes"):
        EngineeringTaskRecord(
            task_id="HF-fake-pass",
            title="fake",
            source="worker",
            priority=TaskPriority.P2,
            risk=TaskRisk.LOW,
            passes=True,
        )
