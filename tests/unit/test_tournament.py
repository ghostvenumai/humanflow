from __future__ import annotations

from humanflow.development import CandidateSubmission, TournamentEvaluator


def _evaluation(
    score: float, *, commands_passed: bool = True, protected: str = "same"
) -> dict[str, object]:
    return {
        "protected_hashes": {"golden": protected},
        "commands_passed": commands_passed,
        "runtime_quality": {"score": score},
    }


def test_tournament_selects_quality_then_cost_and_churn() -> None:
    result = TournamentEvaluator().evaluate(
        (
            CandidateSubmission("codex", "base", "codex-patch", _evaluation(0.9), 30, 2),
            CandidateSubmission("claude", "base", "claude-patch", _evaluation(0.9), 20, 1),
        )
    )
    assert result["status"] == "WINNER"
    assert result["winner"] == "claude"


def test_tournament_disqualifies_regression_and_protected_modification() -> None:
    result = TournamentEvaluator().evaluate(
        (
            CandidateSubmission("codex", "base", "a", _evaluation(0.8), 20),
            CandidateSubmission(
                "claude", "base", "b", _evaluation(1.0, protected="changed"), 10
            ),
        )
    )
    assert result["winner"] == "codex"
    assert result["disqualified"]["claude"] == ["protected_artifact_modified"]


def test_tournament_requires_two_real_candidates() -> None:
    result = TournamentEvaluator().evaluate(
        (CandidateSubmission("codex", "base", "a", _evaluation(1.0), 10),)
    )
    assert result["status"] == "NOT_EXECUTED"
    assert result["winner"] is None
