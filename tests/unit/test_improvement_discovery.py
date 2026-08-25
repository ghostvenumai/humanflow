from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanflow.development import EngineeringTask
from humanflow.engineering import (
    ActorRole,
    EvidenceInformedAgentRouter,
    FailureSignal,
    ImprovementDiscoveryEngine,
    MetricDirection,
    MetricSeries,
    ProblemSeverity,
    ReleaseCandidateBundle,
    ReleaseMeasurement,
    TaskRegistry,
    TaskRunMetrics,
    TaskStatus,
    evaluate_post_release,
    register_proposal,
)


ROOT = Path(__file__).resolve().parents[2]


def _series(raw: dict[str, object]) -> MetricSeries:
    return MetricSeries(
        metric_name="relative_date_success",
        affected_component="temporal",
        direction=MetricDirection.HIGHER_IS_BETTER,
        target=float(raw["target"]),
        baseline_values=tuple(float(value) for value in raw["baseline"]),
        current_values=tuple(float(value) for value in raw["current"]),
        sample_size=int(raw["sample_size"]),
        first_seen="2026-08-20T00:00:00Z",
        last_seen="2026-08-25T00:00:00Z",
        evidence_refs=(f"fixture:{raw['id']}",),
        reproduction_hint="run protected relative-date corpus",
    )


def test_metric_discovery_ignores_healthy_noise_and_detects_real_regression() -> None:
    fixture = json.loads((ROOT / "eval" / "engineering" / "discovery_scenarios.json").read_text())
    engine = ImprovementDiscoveryEngine()

    for raw in fixture["scenarios"]:
        candidates = engine.analyze(metrics=(_series(raw),))
        expected = raw["expected"]
        assert (candidates[0].detector_type.value if candidates else None) == expected


def test_rare_critical_failure_creates_candidate_but_noisy_single_low_does_not() -> None:
    critical = FailureSignal(
        "unexpected-hangup",
        "Unexpected hangup",
        "realtime",
        ProblemSeverity.CRITICAL,
        0.99,
        1,
        "2026-08-25T10:00:00Z",
        "2026-08-25T10:00:00Z",
        ("event:hangup",),
        "replay event timeline",
        True,
    )
    noisy = FailureSignal(
        "one-noisy-event",
        "Uncertain noise",
        "realtime",
        ProblemSeverity.LOW,
        0.5,
        1,
        "2026-08-25T10:00:00Z",
        "2026-08-25T10:00:00Z",
        ("event:noise",),
        "none",
        False,
    )

    candidates = ImprovementDiscoveryEngine().analyze(failures=(critical, noisy))

    assert len(candidates) == 1
    assert candidates[0].severity is ProblemSeverity.CRITICAL
    assert candidates[0].detector_type.value == "HIGH_SEVERITY"


def test_sudden_degradation_is_anomaly_but_sudden_improvement_is_not() -> None:
    def latency(current: tuple[float, ...]) -> MetricSeries:
        return MetricSeries(
            "internal_latency_ms",
            "telemetry",
            MetricDirection.LOWER_IS_BETTER,
            200,
            (100, 101, 99),
            current,
            100,
            "2026-08-20T00:00:00Z",
            "2026-08-25T00:00:00Z",
            ("metric:latency",),
            "replay latency fixture",
        )

    degraded = ImprovementDiscoveryEngine().analyze(metrics=(latency((140, 141, 139)),))
    improved = ImprovementDiscoveryEngine().analyze(metrics=(latency((60, 61, 59)),))

    assert degraded[0].detector_type.value == "ANOMALY"
    assert improved == ()


def test_repeated_uncovered_failure_becomes_one_coverage_gap_candidate() -> None:
    signal = FailureSignal(
        "new-phrase-class",
        "Missing phrase-class regression",
        "temporal",
        ProblemSeverity.MEDIUM,
        0.9,
        3,
        "2026-08-20T00:00:00Z",
        "2026-08-25T00:00:00Z",
        ("event:1", "event:2", "event:3"),
        "replay sanitized phrase class",
        False,
    )

    candidates = ImprovementDiscoveryEngine().analyze(failures=(signal, signal))

    assert len(candidates) == 1
    assert candidates[0].detector_type.value == "COVERAGE_GAP"


def test_candidate_becomes_one_deduplicated_task_and_high_risk_needs_approval(
    tmp_path: Path,
) -> None:
    signal = FailureSignal(
        "phantom-final",
        "Reject phantom FINAL",
        "final_admission",
        ProblemSeverity.HIGH,
        0.95,
        3,
        "2026-08-24T00:00:00Z",
        "2026-08-25T00:00:00Z",
        ("evidence-1",),
        "run Final Admission replay",
        True,
    )
    candidate = ImprovementDiscoveryEngine().analyze(failures=(signal,))[0]
    registry = TaskRegistry(tmp_path / "tasks.json")

    first = register_proposal(registry, candidate, task_id="HF-1")
    second = register_proposal(registry, candidate, task_id="HF-2")
    registry.transition(first.task_id, TaskStatus.TRIAGED, actor=ActorRole.COORDINATOR)

    assert first.task_id == second.task_id
    assert len(registry.tasks) == 1
    assert first.human_approval_required is True
    with pytest.raises(PermissionError, match="human approval"):
        registry.transition(first.task_id, TaskStatus.READY, actor=ActorRole.COORDINATOR)


def _history(agent: str, success: int, failures: int) -> tuple[TaskRunMetrics, ...]:
    items = []
    for index in range(success + failures):
        items.append(
            TaskRunMetrics(
                run_id=f"{agent}-{index}",
                task_id=f"HF-{index}",
                agent=agent,
                task_class="date_understanding",
                iterations=2,
                duration_seconds=10,
                result="VERIFIED_PASS" if index < success else "MERGE_REJECTED",
                files_changed=1,
                lines_added=5,
                lines_removed=1,
                regressions_introduced=0 if index < success else 1,
                reviewer_findings=0 if index < success else 2,
                circuit_breaker_events=0,
                estimated_cost="0.50" if agent == "claude" else "0.10",
                currency="USD",
            )
        )
    return tuple(items)


def test_history_router_requires_samples_and_prioritizes_quality_over_lower_cost() -> None:
    task = EngineeringTask(
        "HF-route",
        "date_understanding",
        "date fix",
        0.4,
        0.5,
        0.3,
        0.2,
    )
    router = EvidenceInformedAgentRouter(exploration_percent=0)
    cold = router.route(task, _history("claude", 3, 0))
    informed = router.route(
        task,
        (*_history("codex", 4, 1), *_history("claude", 5, 0)),
    )

    assert cold.used_history is False
    assert cold.selected_agent == "codex"
    assert informed.used_history is True
    assert informed.selected_agent == "claude"
    assert "quality_gates_precede_cost" in informed.evidence


def test_release_bundle_is_human_gated_and_post_release_regression_only_recommends(
    tmp_path: Path,
) -> None:
    bundle = ReleaseCandidateBundle(
        "HF-1",
        "base",
        "candidate",
        "latest-main",
        "previous-good",
        "VERIFIED_PASS",
        ("verification-1",),
        ("src/x.py",),
        "a" * 64,
    )
    path = tmp_path / "release-candidate.json"
    bundle.write(path)
    regression = evaluate_post_release(
        ReleaseMeasurement(
            "HF-1",
            "relative_date_success",
            0.99,
            0.90,
            0.99,
            True,
            220,
            ("post-release-evidence",),
        )
    )

    assert json.loads(path.read_text())["human_deployment_required"] is True
    assert regression["status"] == "RELEASE_REJECTED"
    assert regression["rollback_recommended"] is True
    assert regression["rollback_executed"] is False


def test_post_release_target_improvement_is_measured_success() -> None:
    measured = evaluate_post_release(
        ReleaseMeasurement(
            "HF-2",
            "relative_date_success",
            0.90,
            0.99,
            0.98,
            True,
            220,
            ("post-release-success-evidence",),
        )
    )

    assert measured["status"] == "MEASURED_SUCCESS"
    assert measured["rollback_recommended"] is False
    assert measured["rollback_executed"] is False
