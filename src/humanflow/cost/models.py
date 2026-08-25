"""Immutable cost evidence with explicit actual-versus-estimated semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


MICRO_UNITS_PER_CURRENCY = Decimal("1000000")


class ServiceType(StrEnum):
    STT = "STT"
    LLM = "LLM"
    TTS = "TTS"
    TOOL = "TOOL"
    FALLBACK_TTS = "FALLBACK_TTS"
    OTHER_PROVIDER = "OTHER_PROVIDER"


class UsageSource(StrEnum):
    ACTUAL_USAGE = "ACTUAL_USAGE"
    ESTIMATED_USAGE = "ESTIMATED_USAGE"
    LOCAL_OBSERVATION = "LOCAL_OBSERVATION"


class CostSource(StrEnum):
    PROVIDER_REPORTED_COST = "PROVIDER_REPORTED_COST"
    ESTIMATED_COST = "ESTIMATED_COST"
    LOCAL_TOOL_COST = "LOCAL_TOOL_COST"
    COST_UNAVAILABLE = "COST_UNAVAILABLE"
    UNVERIFIED_PRICING = "UNVERIFIED_PRICING"


def decimal_to_micros(value: Decimal | str | int) -> int:
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    return int(
        (decimal_value * MICRO_UNITS_PER_CURRENCY).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def micros_to_decimal(value: int) -> Decimal:
    return Decimal(value) / MICRO_UNITS_PER_CURRENCY


@dataclass(frozen=True, slots=True)
class CostEvent:
    session_id: str
    conversation_id: str
    provider: str
    model: str
    service_type: ServiceType
    operation: str
    usage_source: UsageSource
    cost_source: CostSource
    operation_id: str
    cost_event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp_monotonic_ns: int = 0
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    turn_id: str | None = None
    response_id: str | None = None
    provider_request_id: str | None = None
    input_units: Decimal | None = None
    output_units: Decimal | None = None
    total_units: Decimal | None = None
    unit_type: str | None = None
    audio_input_seconds: Decimal | None = None
    audio_output_seconds: Decimal | None = None
    characters: int | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    credits: Decimal | None = None
    actual_usage: bool = False
    estimated_usage: bool = False
    actual_cost_micros: int | None = None
    estimated_cost_micros: int | None = None
    currency: str | None = None
    pricing_rule_id: str | None = None
    pricing_version: str | None = None
    pricing_effective_date: str | None = None
    played_fraction: Decimal | None = None
    heard_units: Decimal | None = None
    unheard_units: Decimal | None = None
    wasted_cost_estimate_micros: int | None = None
    fallback: bool = False
    retry: bool = False
    cancelled: bool = False
    tool_success: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "operation_id": self.operation_id,
            "cost_event_id": self.cost_event_id,
        }
        for name, value in required.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.timestamp_monotonic_ns < 0:
            raise ValueError("timestamp_monotonic_ns must be non-negative")
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        for name in (
            "input_units",
            "output_units",
            "total_units",
            "audio_input_seconds",
            "audio_output_seconds",
            "credits",
            "heard_units",
            "unheard_units",
        ):
            value = getattr(self, name)
            if value is not None:
                decimal_value = Decimal(str(value))
                if decimal_value < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, decimal_value)
        for name in (
            "characters",
            "tokens_input",
            "tokens_output",
            "actual_cost_micros",
            "estimated_cost_micros",
            "wasted_cost_estimate_micros",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.played_fraction is not None:
            played_fraction = Decimal(str(self.played_fraction))
            if not Decimal("0") <= played_fraction <= Decimal("1"):
                raise ValueError("played_fraction must be between zero and one")
            object.__setattr__(self, "played_fraction", played_fraction)
        if self.actual_cost_micros is not None and self.cost_source not in {
            CostSource.PROVIDER_REPORTED_COST,
            CostSource.LOCAL_TOOL_COST,
        }:
            raise ValueError("actual cost requires provider-reported or local-tool evidence")
        if (
            self.estimated_cost_micros is not None
            and self.cost_source is not CostSource.ESTIMATED_COST
        ):
            raise ValueError("estimated cost requires ESTIMATED_COST evidence")
        if self.cost_source in {
            CostSource.COST_UNAVAILABLE,
            CostSource.UNVERIFIED_PRICING,
        } and (self.actual_cost_micros is not None or self.estimated_cost_micros is not None):
            raise ValueError("unavailable pricing cannot carry a monetary amount")
        if (
            self.actual_cost_micros is not None
            or self.estimated_cost_micros is not None
            or self.wasted_cost_estimate_micros is not None
        ) and (self.currency is None or not self.currency.strip()):
            raise ValueError("monetary values require currency")
        safe_metadata = json.loads(json.dumps(dict(self.metadata), default=str))
        object.__setattr__(self, "metadata", MappingProxyType(safe_metadata))

    @property
    def dedupe_key(self) -> str:
        provider_identity = self.provider_request_id or self.response_id or "none"
        return "|".join(
            (
                self.session_id,
                self.service_type.value,
                self.provider,
                self.operation,
                self.operation_id,
                provider_identity,
            )
        )

    def with_estimate(
        self,
        *,
        estimated_cost_micros: int,
        currency: str,
        pricing_rule_id: str,
        pricing_version: str,
        pricing_effective_date: str,
    ) -> "CostEvent":
        return replace(
            self,
            cost_source=CostSource.ESTIMATED_COST,
            estimated_cost_micros=estimated_cost_micros,
            currency=currency,
            pricing_rule_id=pricing_rule_id,
            pricing_version=pricing_version,
            pricing_effective_date=pricing_effective_date,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, StrEnum):
                payload[name] = value.value
            elif isinstance(value, Decimal):
                payload[name] = str(value)
            elif isinstance(value, datetime):
                payload[name] = value.astimezone(UTC).isoformat().replace("+00:00", "Z")
            elif isinstance(value, Mapping):
                payload[name] = dict(value)
            else:
                payload[name] = value
        payload["dedupe_key"] = self.dedupe_key
        return payload
