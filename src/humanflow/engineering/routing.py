"""Evidence-informed agent choice with quality-first cold-start behavior."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

from humanflow.development.models import EngineeringTask, RouteDecision
from humanflow.development.router import DevelopmentModelRouter

from .metrics import TaskRunMetrics


@dataclass(frozen=True, slots=True)
class AgentRoutingDecision:
    base_route: RouteDecision
    selected_agent: str
    evidence: tuple[str, ...]
    used_history: bool
    exploration: bool


class EvidenceInformedAgentRouter:
    def __init__(
        self,
        *,
        minimum_samples_per_agent: int = 5,
        minimum_success_rate: float = 0.8,
        maximum_regression_rate: float = 0.1,
        exploration_percent: int = 10,
    ) -> None:
        if minimum_samples_per_agent < 1:
            raise ValueError("minimum_samples_per_agent must be positive")
        if not 0 <= exploration_percent <= 25:
            raise ValueError("exploration_percent must be between zero and 25")
        self.minimum_samples_per_agent = minimum_samples_per_agent
        self.minimum_success_rate = minimum_success_rate
        self.maximum_regression_rate = maximum_regression_rate
        self.exploration_percent = exploration_percent

    def route(
        self,
        task: EngineeringTask,
        history: tuple[TaskRunMetrics, ...],
    ) -> AgentRoutingDecision:
        base = DevelopmentModelRouter().route(task)
        relevant = [item for item in history if item.task_class == task.category.casefold()]
        grouped: dict[str, list[TaskRunMetrics]] = {}
        for item in relevant:
            grouped.setdefault(item.agent, []).append(item)
        qualified: list[tuple[str, float, float, float, Decimal | None, int]] = []
        for agent, items in grouped.items():
            if len(items) < self.minimum_samples_per_agent:
                continue
            success = sum(item.verified_success for item in items) / len(items)
            regression = sum(item.regressions_introduced > 0 for item in items) / len(items)
            findings = sum(item.reviewer_findings for item in items) / len(items)
            known_costs = [
                Decimal(item.estimated_cost) for item in items if item.estimated_cost is not None
            ]
            cost = sum(known_costs) / len(known_costs) if len(known_costs) == len(items) else None
            if success >= self.minimum_success_rate and regression <= self.maximum_regression_rate:
                qualified.append((agent, success, regression, findings, cost, len(items)))
        if not qualified:
            return AgentRoutingDecision(
                base,
                base.agents[0],
                ("insufficient_or_unqualified_history", *base.evidence),
                False,
                False,
            )
        ranked = sorted(
            qualified,
            key=lambda item: (
                -item[1],
                item[2],
                item[3],
                item[4] is None,
                item[4] or Decimal(0),
                item[0],
            ),
        )
        bucket = int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16) % 100
        explore = bucket < self.exploration_percent and len(ranked) > 1
        selected = min(ranked, key=lambda item: (item[5], item[0])) if explore else ranked[0]
        return AgentRoutingDecision(
            base,
            selected[0],
            (
                f"history_n={selected[5]}",
                f"success_rate={selected[1]:.3f}",
                f"regression_rate={selected[2]:.3f}",
                "quality_gates_precede_cost",
            ),
            True,
            explore,
        )
