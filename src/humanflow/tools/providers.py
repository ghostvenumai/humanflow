"""Local appointment tools and deterministic fault injection adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import uuid4

from .models import FaultPlan, ToolProviderResponse


class ToolProvider(Protocol):
    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse: ...


class ToolProviderError(RuntimeError):
    """A sanitized provider failure that is eligible for recovery."""


@dataclass(slots=True)
class AppointmentToolProvider:
    """Idempotent in-memory demo tools; no external side effects or claims."""

    _bookings: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        await asyncio.sleep(0)
        if name == "get_available_appointments":
            value: dict[str, Any] = {
                "slots": ["2026-08-25T09:00:00+02:00", "2026-08-25T11:30:00+02:00"]
            }
        elif name == "book_appointment":
            idempotency_key = str(arguments.get("idempotency_key", "")).strip()
            slot = str(arguments.get("slot", "")).strip()
            if not idempotency_key or not slot:
                raise ToolProviderError("missing_required_argument")
            booking = self._bookings.setdefault(
                idempotency_key,
                {"booking_id": str(uuid4()), "slot": slot, "status": "booked"},
            )
            value = dict(booking)
        elif name == "cancel_appointment":
            booking_id = str(arguments.get("booking_id", "")).strip()
            match = next(
                (
                    booking
                    for booking in self._bookings.values()
                    if booking["booking_id"] == booking_id
                ),
                None,
            )
            if match is None:
                raise ToolProviderError("booking_not_found")
            match["status"] = "cancelled"
            value = dict(match)
        elif name == "lookup_customer":
            value = {"customer_found": False, "safe_to_create_callback": True}
        elif name == "create_callback":
            value = {"callback_id": str(uuid4()), "status": "requested"}
        else:
            raise ToolProviderError("unknown_tool")
        return ToolProviderResponse(response_id=str(uuid4()), value=value)


@dataclass(slots=True)
class FaultInjectingToolProvider:
    """Wrap a provider with briefing-defined deterministic failure modes."""

    provider: ToolProvider
    plan: FaultPlan
    timeout_hold_ms: float = 4_000.0
    _duplicate_id: str = field(default_factory=lambda: str(uuid4()))

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        if self.plan.latency_ms:
            await asyncio.sleep(self.plan.latency_ms / 1000.0)
        if self.plan.timeout:
            await asyncio.sleep(self.timeout_hold_ms / 1000.0)
        if self.plan.fail:
            raise ToolProviderError("injected_failure")
        response = await self.provider.call(name, arguments)
        if self.plan.invalid_response:
            return cast(ToolProviderResponse, {"malformed": True})
        if self.plan.duplicate_response:
            return ToolProviderResponse(response_id=self._duplicate_id, value=response.value)
        return response
