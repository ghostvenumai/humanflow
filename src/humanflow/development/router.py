"""Minimum-sufficient, risk-aware router for offline engineering tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .models import EngineeringTask, ModelTier, RouteDecision


FAST_CATEGORIES = frozenset({"DOCUMENTATION", "FIXTURE", "DETERMINISTIC_SMALL_FIX"})
ADVANCED_CATEGORIES = frozenset(
    {"ASYNC_AUDIO", "AUDIO_PIPELINE", "RACE_CONDITION", "STATE_MACHINE", "TURN_DETECTION"}
)
ARCHITECTURE_CATEGORIES = frozenset({"ARCHITECTURE", "UNRESOLVED_ROOT_CAUSE"})


@dataclass(frozen=True, slots=True)
class DevelopmentModelRouter:
    max_tournament_usd: float = 5.0
    frontier_requires_human_approval: bool = True

    def route(self, task: EngineeringTask) -> RouteDecision:
        category = task.category.strip().upper()
        evidence = (
            f"category={category}",
            f"risk={task.risk:.2f}",
            f"criticality={task.criticality:.2f}",
            f"ambiguity={task.ambiguity:.2f}",
            f"realtime_impact={task.realtime_impact:.2f}",
            f"prior_failures={task.prior_failures}",
            f"context_tokens={task.context_tokens}",
        )

        if category in ARCHITECTURE_CATEGORIES and task.prior_failures >= 3:
            approved = task.human_budget_approved or not self.frontier_requires_human_approval
            return RouteDecision(
                task_id=task.task_id,
                tier=ModelTier.FRONTIER,
                agents=("codex", "claude"),
                reason_codes=(
                    "hard_unresolved_architecture",
                    "repeated_candidate_failure",
                    "human_budget_approved" if approved else "human_budget_approval_required",
                ),
                evidence=evidence,
                execution_allowed=approved,
                maximum_budget_usd=task.budget_usd,
            )

        tournament_evidence = (
            task.prior_failures >= 2
            or (task.criticality >= 0.8 and task.ambiguity >= 0.7)
            or (task.risk >= 0.85 and task.realtime_impact >= 0.8)
        )
        if tournament_evidence:
            budget = min(task.budget_usd, self.max_tournament_usd)
            return RouteDecision(
                task_id=task.task_id,
                tier=ModelTier.TOURNAMENT,
                agents=("codex", "claude"),
                reason_codes=(
                    "independent_candidates_required",
                    "high_impact_or_prior_failures",
                    "budget_capped",
                ),
                evidence=evidence,
                execution_allowed=budget > 0,
                maximum_budget_usd=budget,
            )

        if (
            category in ADVANCED_CATEGORIES
            or task.realtime_impact >= 0.7
            or task.risk >= 0.7
            or task.context_tokens >= 80_000
        ):
            return RouteDecision(
                task_id=task.task_id,
                tier=ModelTier.ADVANCED,
                agents=("codex",),
                reason_codes=("advanced_async_or_realtime_work", "single_candidate_sufficient"),
                evidence=evidence,
                execution_allowed=True,
                maximum_budget_usd=task.budget_usd,
            )

        if (
            category in FAST_CATEGORIES
            and task.risk < 0.3
            and task.criticality < 0.4
            and task.context_tokens < 20_000
        ):
            return RouteDecision(
                task_id=task.task_id,
                tier=ModelTier.FAST,
                agents=("codex",),
                reason_codes=("deterministic_low_risk_task", "minimum_sufficient_tier"),
                evidence=evidence,
                execution_allowed=True,
                maximum_budget_usd=task.budget_usd,
            )

        return RouteDecision(
            task_id=task.task_id,
            tier=ModelTier.STANDARD,
            agents=("codex",),
            reason_codes=("normal_local_engineering", "minimum_sufficient_tier"),
            evidence=evidence,
            execution_allowed=True,
            maximum_budget_usd=task.budget_usd,
        )
