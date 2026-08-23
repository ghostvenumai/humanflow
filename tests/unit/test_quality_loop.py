from __future__ import annotations

from humanflow.quality.loop import compare_candidates


def _snapshot(score: float, *, commands_passed: bool = True) -> dict[str, object]:
    return {
        "git_commit": "commit",
        "protected_hashes": {"tests/golden/x": "abc"},
        "commands_passed": commands_passed,
        "runtime_quality": {"score": score, "immutable_eval_sha256": "fixture"},
    }


def test_quality_loop_keeps_only_strict_improvement_with_immutable_evidence() -> None:
    decision = compare_candidates(_snapshot(0.0), _snapshot(1.0))
    assert decision["decision"] == "KEEP"


def test_quality_loop_reverts_unchanged_score_or_modified_protected_artifact() -> None:
    unchanged = compare_candidates(_snapshot(1.0), _snapshot(1.0))
    modified = _snapshot(1.0)
    modified["protected_hashes"] = {"tests/golden/x": "changed"}
    protected = compare_candidates(_snapshot(0.0), modified)
    assert unchanged["decision"] == "REVERT"
    assert protected["decision"] == "REVERT"
