"""Verification-gated orchestration around the existing tournament evaluator."""

from __future__ import annotations

from typing import Any, Mapping

from humanflow.development.tournament import CandidateSubmission, TournamentEvaluator


class VerifiedTournamentCoordinator:
    def evaluate(
        self,
        candidates: tuple[CandidateSubmission, ...],
        *,
        verification_status: Mapping[str, str],
    ) -> dict[str, Any]:
        if len(candidates) < 2:
            return TournamentEvaluator().evaluate(candidates)
        disqualified = {
            candidate.agent: ["independent_verification_not_passed"]
            for candidate in candidates
            if verification_status.get(candidate.agent) != "VERIFIED_PASS"
        }
        eligible = tuple(
            candidate for candidate in candidates if candidate.agent not in disqualified
        )
        if not eligible:
            return {
                "status": "NO_WINNER",
                "winner": None,
                "reason_codes": ["no_candidate_passed_independent_verification"],
                "disqualified": disqualified,
            }
        if len(eligible) == 1:
            winner = eligible[0]
            return {
                "status": "WINNER",
                "winner": winner.agent,
                "winner_commit": winner.patch_commit,
                "reason_codes": [
                    "only_candidate_passing_independent_verification",
                    "all_other_candidates_disqualified",
                ],
                "disqualified": disqualified,
                "ranking": [
                    {
                        "agent": winner.agent,
                        "score": winner.evaluation["runtime_quality"]["score"],
                        "runtime_cost_score": winner.runtime_cost_score,
                        "lines_changed": winner.lines_changed,
                    }
                ],
            }
        evaluated = TournamentEvaluator().evaluate(eligible)
        combined = dict(evaluated)
        combined["disqualified"] = {
            **disqualified,
            **dict(evaluated.get("disqualified", {})),
        }
        return combined
