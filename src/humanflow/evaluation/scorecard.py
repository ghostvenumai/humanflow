"""Build a scope-labeled scorecard from immutable fixtures and measured reports."""

from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path
from typing import Any, Callable

import yaml

from humanflow.turns.models import TurnDecisionType
from humanflow.turns.policies import FixedSilencePolicy, HybridTurnPolicy

from .replay import load_turn_cases


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "lt": operator.lt,
    "gt": operator.gt,
    "eq": operator.eq,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scorecard(root: Path) -> dict[str, Any]:
    gates_path = root / "config" / "quality-gates.yaml"
    corpus_path = root / "tests" / "golden" / "turn_cases.jsonl"
    realtime_path = root / "reports" / "realtime-core-benchmark.json"
    recovery_path = root / "reports" / "recovery-benchmark.json"
    torture_path = root / "reports" / "torture-run.json"
    gates_payload = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    gate_definitions = gates_payload["gates"]
    realtime = _load_json(realtime_path)
    recovery = _load_json(recovery_path)
    torture = _load_json(torture_path)

    cases = load_turn_cases(corpus_path)
    policy = HybridTurnPolicy()
    predictions = [(case, policy.decide(case.signals).decision) for case in cases]
    baseline_predictions = [
        (case, FixedSilencePolicy().decide(case.signals).decision) for case in cases
    ]
    non_interruptions = [item for item in predictions if item[0].expected is not TurnDecisionType.INTERRUPTION]
    false_interruptions = sum(
        actual is TurnDecisionType.INTERRUPTION for _, actual in non_interruptions
    )
    premature = sum(
        case.expected in {TurnDecisionType.CONTINUE_LISTENING, TurnDecisionType.UNCERTAIN}
        and actual in {TurnDecisionType.COMPLETE, TurnDecisionType.LIKELY_COMPLETE}
        for case, actual in predictions
    )
    binary_interruption_correct = sum(
        (case.expected is TurnDecisionType.INTERRUPTION)
        == (actual is TurnDecisionType.INTERRUPTION)
        for case, actual in predictions
    )
    baseline_non_interruptions = [
        item
        for item in baseline_predictions
        if item[0].expected is not TurnDecisionType.INTERRUPTION
    ]
    baseline_false_interruptions = sum(
        actual is TurnDecisionType.INTERRUPTION for _, actual in baseline_non_interruptions
    )
    baseline_premature = sum(
        case.expected in {TurnDecisionType.CONTINUE_LISTENING, TurnDecisionType.UNCERTAIN}
        and actual in {TurnDecisionType.COMPLETE, TurnDecisionType.LIKELY_COMPLETE}
        for case, actual in baseline_predictions
    )
    baseline_binary_interruption_correct = sum(
        (case.expected is TurnDecisionType.INTERRUPTION)
        == (actual is TurnDecisionType.INTERRUPTION)
        for case, actual in baseline_predictions
    )
    t20 = next(result for result in torture["results"] if result["scenario_id"] == "T20")
    expected_calls = 1
    completed_calls = int(
        t20["passed"] and t20["evidence"].get("termination_reason") == "caller_goodbye"
    )

    metrics: dict[str, dict[str, Any]] = {
        "ttfa_ms": {
            "value": realtime["metrics"]["ttfa_ms"]["p50"],
            "baseline_value": None,
            "statistic": "p50",
            "sample_count": realtime["metrics"]["ttfa_ms"]["n"],
            "scope": "local_event_loop_transport",
            "source": str(realtime_path.relative_to(root)),
        },
        "audible_barge_in_latency_ms": {
            "value": realtime["metrics"]["audible_barge_in_latency_ms"]["p50"],
            "baseline_value": None,
            "statistic": "p50",
            "sample_count": realtime["metrics"]["audible_barge_in_latency_ms"]["n"],
            "scope": "local_timed_pcm_sink",
            "source": str(realtime_path.relative_to(root)),
        },
        "false_interruption_rate": {
            "value": false_interruptions / len(non_interruptions),
            "baseline_value": baseline_false_interruptions / len(baseline_non_interruptions),
            "statistic": "rate",
            "numerator": false_interruptions,
            "sample_count": len(non_interruptions),
            "scope": "protected_german_fixture_corpus",
            "source": str(corpus_path.relative_to(root)),
        },
        "premature_endpoint_rate": {
            "value": premature / len(cases),
            "baseline_value": baseline_premature / len(cases),
            "statistic": "rate",
            "numerator": premature,
            "sample_count": len(cases),
            "scope": "protected_german_fixture_corpus",
            "source": str(corpus_path.relative_to(root)),
        },
        "call_completion_rate": {
            "value": completed_calls / expected_calls,
            "baseline_value": None,
            "statistic": "rate",
            "numerator": completed_calls,
            "sample_count": expected_calls,
            "scope": "synthetic_torture_termination_scenario",
            "source": str(torture_path.relative_to(root)),
        },
        "tool_failure_recovery_rate": {
            "value": recovery["metrics"]["tool_failure_recovery_rate"]["rate"],
            "baseline_value": None,
            "statistic": "rate",
            "numerator": recovery["metrics"]["tool_failure_recovery_rate"][
                "recovery_completed"
            ],
            "sample_count": recovery["metrics"]["tool_failure_recovery_rate"][
                "injected_failures"
            ],
            "scope": "local_deterministic_fault_injection",
            "source": str(recovery_path.relative_to(root)),
        },
        "unexpected_hangups": {
            "value": expected_calls - completed_calls,
            "baseline_value": None,
            "statistic": "count",
            "sample_count": expected_calls,
            "scope": "synthetic_torture_termination_scenario",
            "source": str(torture_path.relative_to(root)),
        },
        "german_interruption_accuracy": {
            "value": binary_interruption_correct / len(cases),
            "baseline_value": baseline_binary_interruption_correct / len(cases),
            "statistic": "accuracy",
            "numerator": binary_interruption_correct,
            "sample_count": len(cases),
            "scope": "protected_german_fixture_corpus_binary_interruption",
            "source": str(corpus_path.relative_to(root)),
        },
    }

    gate_results: dict[str, dict[str, Any]] = {}
    for name, definition in gate_definitions.items():
        metric = metrics[name]
        if metric["statistic"] != definition["statistic"]:
            raise ValueError(f"statistic mismatch for protected gate {name}")
        operation = OPERATORS[definition["operator"]]
        observed = float(metric["value"])
        target = float(definition["target"])
        gate_results[name] = {
            "observed": metric["value"],
            "operator": definition["operator"],
            "target": definition["target"],
            "passed": operation(observed, target),
            "scope": metric["scope"],
            "sample_count": metric["sample_count"],
        }

    passed = sum(result["passed"] for result in gate_results.values())
    artifacts = (gates_path, corpus_path, realtime_path, recovery_path, torture_path)
    return {
        "schema_version": 1,
        "protected_artifacts_unchanged_by_scorecard": True,
        "artifacts": {
            str(path.relative_to(root)): {"sha256": _sha256(path)} for path in artifacts
        },
        "metrics": metrics,
        "gates": gate_results,
        "summary": {
            "gates_total": len(gate_results),
            "gates_passed": passed,
            "gates_failed": len(gate_results) - passed,
            "engineering_evidence_status": (
                "PASS_LOCAL_EVIDENCE" if passed == len(gate_results) else "FAIL"
            ),
            "production_release_claim": "NOT_ESTABLISHED_NO_REAL_CALL_DATA",
        },
    }
