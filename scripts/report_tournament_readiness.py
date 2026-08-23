#!/usr/bin/env python3
"""Prove tournament evaluation readiness without making external agent calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from humanflow.development import (
    CandidateSubmission,
    ClaudeCliAdapter,
    CodexCliAdapter,
    TournamentEvaluator,
)


ROOT = Path(__file__).resolve().parents[1]
ITERATION = ROOT / "reports" / "quality-loop" / "iteration-001"
OUTPUT = ROOT / "reports" / "tournament-readiness.json"


def main() -> None:
    baseline = json.loads((ITERATION / "baseline.json").read_text(encoding="utf-8"))
    candidate = json.loads((ITERATION / "candidate.json").read_text(encoding="utf-8"))
    decision = json.loads((ITERATION / "decision.json").read_text(encoding="utf-8"))
    codex = CandidateSubmission(
        agent="codex",
        baseline_commit=decision["baseline_commit"],
        patch_commit=decision["candidate_commit"],
        evaluation=candidate,
        lines_changed=64,
    )
    real_readiness = TournamentEvaluator().evaluate((codex,))

    synthetic_reference = {
        "protected_hashes": baseline["protected_hashes"],
        "commands_passed": True,
        "runtime_quality": {"score": 0.5},
    }
    synthetic_better = {
        "protected_hashes": baseline["protected_hashes"],
        "commands_passed": True,
        "runtime_quality": {"score": 0.75},
    }
    evaluator_self_test = TournamentEvaluator().evaluate(
        (
            CandidateSubmission("codex", "same", "a", synthetic_reference, 30),
            CandidateSubmission("claude", "same", "b", synthetic_better, 40),
        )
    )
    adapters = [CodexCliAdapter().status(), ClaudeCliAdapter().status()]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "READY_NOT_EXECUTED",
        "real_candidate_registration": real_readiness,
        "reason_codes": [
            "only_one_real_candidate_available",
            "external_agent_execution_not_authorized",
            "no_tournament_spend_authorized",
        ],
        "evaluator_self_test": {
            "synthetic_inputs_clearly_labeled": True,
            "result": evaluator_self_test,
        },
        "adapters": [adapter.to_dict() for adapter in adapters],
        "external_agent_calls_made": 0,
        "external_spend_usd": 0.0,
        "production_code_selected_by_synthetic_test": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "real": real_readiness}, indent=2))
    print(f"report={OUTPUT}")


if __name__ == "__main__":
    main()
