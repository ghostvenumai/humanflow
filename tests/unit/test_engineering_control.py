from __future__ import annotations

from pathlib import Path

from humanflow.development import CandidateSubmission
from humanflow.engineering import (
    DiagnosticPackageWriter,
    FailureCircuitBreaker,
    FailureFingerprint,
    HarnessFailureState,
    HarnessMetricsStore,
    IterationObservation,
    TaskRunMetrics,
    VerifiedTournamentCoordinator,
)


def _digest(value: str) -> str:
    return (value * 64)[:64]


def _failure() -> FailureFingerprint:
    return FailureFingerprint.create(
        test_name="test_booking",
        exception_type="AssertionError",
        error="/tmp/run-123/test.py line 42 expected 1 at 0xABC but got 0",
        component="appointment_coordinator",
    )


def _observation(iteration: int, diff: str, failure: str = "failure") -> IterationObservation:
    return IterationObservation(iteration, _digest(diff), (failure,), 3, 200)


def test_third_identical_failure_opens_circuit_and_allows_only_one_retry() -> None:
    breaker = FailureCircuitBreaker()
    failure = _failure()

    assert breaker.record_failure(failure) is HarnessFailureState.HEALTHY
    assert breaker.record_failure(failure) is HarnessFailureState.HEALTHY
    assert breaker.record_failure(failure) is HarnessFailureState.CIRCUIT_OPEN
    assert breaker.authorize_retry() is True
    breaker.record_failure(failure)
    assert breaker.state is HarnessFailureState.CIRCUIT_OPEN
    assert breaker.authorize_retry() is False
    assert breaker.state is HarnessFailureState.BLOCKED


def test_required_failure_policy_states_are_explicit() -> None:
    expected = {
        "BLOCKED",
        "BLOCKED_MANUAL_VALIDATION",
        "TEST_DISPUTE",
        "RESOURCE_CONFLICT",
        "INTEGRITY_FAILURE",
        "REPEATED_FAILURE",
        "NO_PROGRESS",
        "OSCILLATION_DETECTED",
        "OUTPUT_COLLAPSE",
        "BUDGET_EXCEEDED",
        "SCOPE_VIOLATION",
        "MERGE_REJECTED",
        "RELEASE_REJECTED",
    }

    assert expected.issubset({state.value for state in HarnessFailureState})


def test_no_progress_oscillation_output_budget_and_scope_are_distinct() -> None:
    no_progress = FailureCircuitBreaker()
    for iteration in range(1, 4):
        state = no_progress.record_iteration(_observation(iteration, "a"))
    assert state is HarnessFailureState.NO_PROGRESS

    oscillation = FailureCircuitBreaker(no_progress_threshold=5)
    for iteration, diff in enumerate(("a", "b", "a", "b"), start=1):
        state = oscillation.record_iteration(_observation(iteration, diff))
    assert state is HarnessFailureState.OSCILLATION_DETECTED

    collapse = FailureCircuitBreaker().record_iteration(
        IterationObservation(1, _digest("c"), ("f",), 0, 4)
    )
    budget = FailureCircuitBreaker().record_iteration(
        IterationObservation(1, _digest("d"), ("f",), 2, 200, budget_exceeded=True)
    )
    scope = FailureCircuitBreaker().record_iteration(
        IterationObservation(1, _digest("e"), ("f",), 2, 200, scope_violation=True)
    )
    assert collapse is HarnessFailureState.OUTPUT_COLLAPSE
    assert budget is HarnessFailureState.BUDGET_EXCEEDED
    assert scope is HarnessFailureState.SCOPE_VIOLATION


def test_diagnostic_package_is_complete_and_redacts_credentials(tmp_path: Path) -> None:
    paths = DiagnosticPackageWriter().write(
        tmp_path / "diagnostic",
        task={"task_id": "HF-1"},
        git_diff="api_key=secret-value",
        git_status="M src/x.py",
        test_output="Authorization: bearer-secret",
        failure_fingerprints=(_failure(),),
        agent_log="password=hunter2",
        metrics={"iterations": 3},
        review_findings=("race",),
        evidence_refs=("evidence-1",),
    )

    assert len(paths) == 9
    assert all(path.is_file() for path in paths)
    combined = "".join(path.read_text(encoding="utf-8") for path in paths)
    assert "secret-value" not in combined
    assert "bearer-secret" not in combined
    assert "hunter2" not in combined


def _metrics(run_id: str, result: str, cost: str | None) -> TaskRunMetrics:
    return TaskRunMetrics(
        run_id=run_id,
        task_id="HF-1",
        agent="codex",
        task_class="date_understanding",
        iterations=2,
        duration_seconds=12.5,
        result=result,
        files_changed=2,
        lines_added=10,
        lines_removed=3,
        regressions_introduced=0,
        reviewer_findings=1,
        circuit_breaker_events=0,
        tokens_input=None,
        tokens_output=None,
        estimated_cost=cost,
        currency="USD" if cost is not None else None,
    )


def test_harness_metrics_preserve_nulls_and_do_not_invent_cost(tmp_path: Path) -> None:
    store = HarnessMetricsStore(tmp_path / "metrics.jsonl")
    store.append(_metrics("run-1", "VERIFIED_PASS", None))
    summary = store.aggregate()["codex:date_understanding"]

    assert store.read_all()[0].tokens_input is None
    assert summary["success_rate"] == 1.0
    assert summary["cost_per_verified_success"] is None
    assert summary["cost_evidence_complete"] is False


def test_verified_tournament_rejects_unverified_candidate_and_can_have_no_winner() -> None:
    evaluation = {
        "protected_hashes": {"golden": "same"},
        "commands_passed": True,
        "runtime_quality": {"score": 1.0},
    }
    candidates = (
        CandidateSubmission("codex", "base", "a", evaluation, 10),
        CandidateSubmission("claude", "base", "b", evaluation, 8),
    )
    coordinator = VerifiedTournamentCoordinator()

    one_verified = coordinator.evaluate(
        candidates,
        verification_status={"codex": "VERIFIED_PASS", "claude": "MERGE_REJECTED"},
    )
    none_verified = coordinator.evaluate(
        candidates,
        verification_status={"codex": "MERGE_REJECTED", "claude": "MERGE_REJECTED"},
    )

    assert one_verified["winner"] == "codex"
    assert one_verified["disqualified"]["claude"] == ["independent_verification_not_passed"]
    assert none_verified["status"] == "NO_WINNER"
    assert none_verified["winner"] is None


def test_tournament_never_selects_verified_candidate_with_failed_commands() -> None:
    good_evaluation = {
        "protected_hashes": {"golden": "same"},
        "commands_passed": True,
        "runtime_quality": {"score": 1.0},
    }
    failed_evaluation = {
        **good_evaluation,
        "commands_passed": False,
    }
    candidates = (
        CandidateSubmission("bad", "base", "a", failed_evaluation, 5),
        CandidateSubmission("unverified", "base", "b", good_evaluation, 5),
    )

    result = VerifiedTournamentCoordinator().evaluate(
        candidates,
        verification_status={"bad": "VERIFIED_PASS", "unverified": "MERGE_REJECTED"},
    )

    assert result["status"] == "NO_WINNER"
    assert result["winner"] is None
    assert "critical_regression" in result["disqualified"]["bad"]
