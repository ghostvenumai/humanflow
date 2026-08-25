"""Conservative optional session-cost warning hooks; disabled by default."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CostBudgetPolicy:
    warning_eur: Decimal | None = None
    hard_limit_eur: Decimal | None = None

    def __post_init__(self) -> None:
        for name in ("warning_eur", "hard_limit_eur"):
            value = getattr(self, name)
            if value is not None:
                decimal_value = Decimal(str(value))
                if decimal_value <= 0:
                    raise ValueError(f"{name} must be positive when enabled")
                object.__setattr__(self, name, decimal_value)
        if (
            self.warning_eur is not None
            and self.hard_limit_eur is not None
            and self.warning_eur > self.hard_limit_eur
        ):
            raise ValueError("warning threshold cannot exceed hard limit")

    @property
    def enabled(self) -> bool:
        return self.warning_eur is not None or self.hard_limit_eur is not None

    def evaluate(self, *, estimated_eur: Decimal, estimate_complete: bool) -> str | None:
        if not estimate_complete:
            return None
        if self.hard_limit_eur is not None and estimated_eur >= self.hard_limit_eur:
            return "HARD_LIMIT_REACHED"
        if self.warning_eur is not None and estimated_eur >= self.warning_eur:
            return "WARNING_THRESHOLD_REACHED"
        return None
