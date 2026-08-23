"""Explainable inputs and outputs for development model routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelTier(StrEnum):
    FAST = "FAST"
    STANDARD = "STANDARD"
    ADVANCED = "ADVANCED"
    TOURNAMENT = "TOURNAMENT"
    FRONTIER = "FRONTIER"


@dataclass(frozen=True, slots=True)
class EngineeringTask:
    task_id: str
    category: str
    description: str
    risk: float
    criticality: float
    ambiguity: float
    realtime_impact: float
    context_tokens: int = 0
    prior_failures: int = 0
    budget_usd: float = 0.0
    human_budget_approved: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.category.strip() or not self.description.strip():
            raise ValueError("task identity, category and description must not be empty")
        for name in ("risk", "criticality", "ambiguity", "realtime_impact"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.context_tokens < 0 or self.prior_failures < 0 or self.budget_usd < 0:
            raise ValueError("counts and budget must be non-negative")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    task_id: str
    tier: ModelTier
    agents: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence: tuple[str, ...]
    execution_allowed: bool
    maximum_budget_usd: float

    def __post_init__(self) -> None:
        if not self.agents or not self.reason_codes or not self.evidence:
            raise ValueError("route decision must be explainable")
        if self.maximum_budget_usd < 0:
            raise ValueError("maximum_budget_usd must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "tier": self.tier.value,
            "agents": list(self.agents),
            "reason_codes": list(self.reason_codes),
            "evidence": list(self.evidence),
            "execution_allowed": self.execution_allowed,
            "maximum_budget_usd": self.maximum_budget_usd,
        }
