from __future__ import annotations

from pathlib import Path

from humanflow.evaluation.scorecard import build_scorecard


ROOT = Path(__file__).resolve().parents[2]


def test_scorecard_uses_protected_gates_and_labels_local_evidence_scope() -> None:
    scorecard = build_scorecard(ROOT)
    assert scorecard["summary"]["gates_failed"] == 0
    assert scorecard["summary"]["engineering_evidence_status"] == "PASS_LOCAL_EVIDENCE"
    assert scorecard["summary"]["production_release_claim"] == (
        "NOT_ESTABLISHED_NO_REAL_CALL_DATA"
    )
    assert scorecard["metrics"]["ttfa_ms"]["sample_count"] == 40
    assert scorecard["metrics"]["tool_failure_recovery_rate"]["sample_count"] == 60
