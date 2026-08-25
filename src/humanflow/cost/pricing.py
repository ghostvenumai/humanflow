"""Single versioned pricing catalog; inactive rules never produce money."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .models import CostEvent, CostSource, ServiceType, decimal_to_micros


@dataclass(frozen=True, slots=True)
class PricingRule:
    pricing_rule_id: str
    pricing_version: str
    provider: str
    model: str
    service: ServiceType
    unit: str
    currency: str
    effective_date: str
    verified_at: str | None
    source_note: str
    active: bool
    input_rate: Decimal | None = None
    output_rate: Decimal | None = None

    def __post_init__(self) -> None:
        for name in (
            "pricing_rule_id",
            "pricing_version",
            "provider",
            "model",
            "unit",
            "currency",
            "effective_date",
            "source_note",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        for name in ("input_rate", "output_rate"):
            value = getattr(self, name)
            if value is not None:
                decimal_value = Decimal(str(value))
                if decimal_value < 0:
                    raise ValueError("pricing rates must be non-negative")
                object.__setattr__(self, name, decimal_value)
        if self.active and self.verified_at is None:
            raise ValueError("active pricing rule must have verified_at evidence")
        if self.active and self.input_rate is None and self.output_rate is None:
            raise ValueError("active pricing rule must define a rate")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PricingRule":
        return cls(
            pricing_rule_id=str(payload["pricing_rule_id"]),
            pricing_version=str(payload["pricing_version"]),
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            service=ServiceType(str(payload["service"])),
            unit=str(payload["unit"]),
            currency=str(payload["currency"]),
            effective_date=str(payload["effective_date"]),
            verified_at=(
                str(payload["verified_at"])
                if payload.get("verified_at") is not None
                else None
            ),
            source_note=str(payload["source_note"]),
            active=bool(payload["active"]),
            input_rate=(
                Decimal(str(payload["input_rate"]))
                if payload.get("input_rate") is not None
                else None
            ),
            output_rate=(
                Decimal(str(payload["output_rate"]))
                if payload.get("output_rate") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pricing_rule_id": self.pricing_rule_id,
            "pricing_version": self.pricing_version,
            "provider": self.provider,
            "model": self.model,
            "service": self.service.value,
            "unit": self.unit,
            "input_rate": None if self.input_rate is None else str(self.input_rate),
            "output_rate": None if self.output_rate is None else str(self.output_rate),
            "currency": self.currency,
            "effective_date": self.effective_date,
            "verified_at": self.verified_at,
            "source_note": self.source_note,
            "active": self.active,
        }


class PricingCatalog:
    def __init__(self, rules: Iterable[PricingRule]) -> None:
        self.rules = tuple(rules)
        identities = {rule.pricing_rule_id for rule in self.rules}
        if len(identities) != len(self.rules):
            raise ValueError("pricing_rule_id values must be unique")

    @classmethod
    def load(cls, path: Path) -> "PricingCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            raise ValueError("pricing catalog must contain a rules list")
        return cls(PricingRule.from_dict(item) for item in payload["rules"])

    def rule_for(self, event: CostEvent) -> PricingRule | None:
        candidates = [
            rule
            for rule in self.rules
            if rule.active
            and rule.provider == event.provider
            and rule.model == event.model
            and rule.service is event.service_type
        ]
        return max(candidates, key=lambda rule: rule.effective_date, default=None)

    def price(self, event: CostEvent) -> CostEvent:
        rule = self.rule_for(event)
        if rule is None:
            matching_unverified = any(
                rule.provider == event.provider
                and rule.model == event.model
                and rule.service is event.service_type
                for rule in self.rules
            )
            return _replace_cost_source(
                event,
                CostSource.UNVERIFIED_PRICING
                if matching_unverified
                else CostSource.COST_UNAVAILABLE,
            )
        estimated = _estimate(event, rule)
        if estimated is None:
            return _replace_cost_source(event, CostSource.COST_UNAVAILABLE)
        if event.service_type is ServiceType.TOOL and event.provider == "local-sqlite":
            from dataclasses import replace

            return replace(
                event,
                cost_source=CostSource.LOCAL_TOOL_COST,
                actual_cost_micros=decimal_to_micros(estimated),
                estimated_cost_micros=None,
                currency=rule.currency,
                pricing_rule_id=rule.pricing_rule_id,
                pricing_version=rule.pricing_version,
                pricing_effective_date=rule.effective_date,
            )
        return event.with_estimate(
            estimated_cost_micros=decimal_to_micros(estimated),
            currency=rule.currency,
            pricing_rule_id=rule.pricing_rule_id,
            pricing_version=rule.pricing_version,
            pricing_effective_date=rule.effective_date,
        )


def _replace_cost_source(event: CostEvent, source: CostSource) -> CostEvent:
    from dataclasses import replace

    return replace(
        event,
        cost_source=source,
        actual_cost_micros=None,
        estimated_cost_micros=None,
        currency=None,
        pricing_rule_id=None,
        pricing_version=None,
        pricing_effective_date=None,
    )


def _estimate(event: CostEvent, rule: PricingRule) -> Decimal | None:
    input_rate = rule.input_rate or Decimal("0")
    output_rate = rule.output_rate or Decimal("0")
    if rule.unit == "per_1m_tokens":
        if event.tokens_input is None and event.tokens_output is None:
            return None
        return (
            Decimal(event.tokens_input or 0) * input_rate
            + Decimal(event.tokens_output or 0) * output_rate
        ) / Decimal("1000000")
    if rule.unit == "per_audio_minute":
        seconds = (
            event.audio_input_seconds
            if event.service_type is ServiceType.STT
            else event.audio_output_seconds
        )
        if seconds is None:
            return None
        return Decimal(seconds) * (input_rate or output_rate) / Decimal("60")
    if rule.unit == "per_1k_characters":
        if event.characters is None:
            return None
        return Decimal(event.characters) * (output_rate or input_rate) / Decimal("1000")
    if rule.unit == "per_request":
        return output_rate or input_rate
    return None
