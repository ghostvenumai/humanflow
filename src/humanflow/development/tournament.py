"""Same-baseline tournament evaluator with anti-gaming disqualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CandidateSubmission:
    agent: str
    baseline_commit: str
    patch_commit: str
    evaluation: Mapping[str, Any]
    lines_changed: int
    runtime_cost_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.agent.strip() or not self.baseline_commit.strip() or not self.patch_commit.strip():
            raise ValueError("candidate identity and commits must not be empty")
        if self.lines_changed < 0 or self.runtime_cost_score < 0:
            raise ValueError("churn and runtime cost must be non-negative")


class TournamentEvaluator:
    def evaluate(
        self, candidates: tuple[CandidateSubmission, ...]
    ) -> dict[str, Any]:
        if len(candidates) < 2:
            return {
                "status": "NOT_EXECUTED",
                "winner": None,
                "reason_codes": ["at_least_two_real_candidates_required"],
                "candidates": [candidate.agent for candidate in candidates],
            }
        baselines = {candidate.baseline_commit for candidate in candidates}
        if len(baselines) != 1:
            raise ValueError("tournament candidates must share an identical baseline")
        reference_hashes = candidates[0].evaluation["protected_hashes"]
        eligible: list[CandidateSubmission] = []
        disqualified: dict[str, list[str]] = {}
        for candidate in candidates:
            reasons: list[str] = []
            if candidate.evaluation["protected_hashes"] != reference_hashes:
                reasons.append("protected_artifact_modified")
            if not candidate.evaluation["commands_passed"]:
                reasons.append("critical_regression")
            if reasons:
                disqualified[candidate.agent] = reasons
            else:
                eligible.append(candidate)
        if not eligible:
            return {
                "status": "NO_WINNER",
                "winner": None,
                "reason_codes": ["all_candidates_disqualified"],
                "disqualified": disqualified,
            }
        ranked = sorted(
            eligible,
            key=lambda candidate: (
                -float(candidate.evaluation["runtime_quality"]["score"]),
                candidate.runtime_cost_score,
                candidate.lines_changed,
                candidate.agent,
            ),
        )
        winner = ranked[0]
        return {
            "status": "WINNER",
            "winner": winner.agent,
            "winner_commit": winner.patch_commit,
            "reason_codes": [
                "highest_immutable_quality_score",
                "lower_runtime_cost_then_churn_breaks_ties",
            ],
            "disqualified": disqualified,
            "ranking": [
                {
                    "agent": candidate.agent,
                    "score": candidate.evaluation["runtime_quality"]["score"],
                    "runtime_cost_score": candidate.runtime_cost_score,
                    "lines_changed": candidate.lines_changed,
                }
                for candidate in ranked
            ],
        }
