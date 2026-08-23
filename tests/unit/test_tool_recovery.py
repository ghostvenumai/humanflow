from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.tools.executor import ResilientToolExecutor
from humanflow.tools.models import FaultPlan, ToolProviderResponse
from humanflow.tools.providers import (
    AppointmentToolProvider,
    FaultInjectingToolProvider,
    ToolProviderError,
)


def _thinking_machine() -> tuple[ConversationStateMachine, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    machine = ConversationStateMachine(conversation_id=str(uuid4()), sink=sink)
    machine.transition(
        ConversationState.LISTENING,
        reason_code="test_started",
        correlation_id="setup",
    )
    machine.transition(
        ConversationState.THINKING,
        reason_code="test_turn_complete",
        correlation_id="setup",
    )
    return machine, sink


def _fallback(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    del arguments
    return {"safe_fallback": True, "tool_name": name}


class FailOnceProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        del name, arguments
        self.calls += 1
        if self.calls == 1:
            raise ToolProviderError("transient")
        return ToolProviderResponse(response_id="retry-success", value={"ok": True})


def test_transient_failure_retries_without_leaving_recovering_state() -> None:
    async def scenario() -> None:
        machine, sink = _thinking_machine()
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=FailOnceProvider(),
            max_attempts=2,
        )
        result = await executor.execute("lookup_customer", {}, fallback=_fallback)

        assert result.recovered
        assert not result.used_fallback
        assert result.attempts == 2
        assert result.value == {"ok": True}
        assert machine.state is ConversationState.THINKING
        assert sum(event.event_type is EventType.TOOL_FAILED for event in sink.events) == 1
        assert sum(event.event_type is EventType.RECOVERY_COMPLETED for event in sink.events) == 1

    asyncio.run(scenario())


def test_briefing_fault_modes_reach_safe_fallback() -> None:
    async def scenario(plan: FaultPlan, *, timeout_ms: float = 20.0) -> None:
        machine, sink = _thinking_machine()
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(),
            plan,
            timeout_hold_ms=50,
        )
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            timeout_ms=timeout_ms,
            max_attempts=2,
        )
        result = await executor.execute("lookup_customer", {}, fallback=_fallback)
        assert result.recovered
        assert result.used_fallback
        assert result.attempts == 2
        assert machine.state is ConversationState.THINKING
        assert any(event.event_type is EventType.RECOVERY_STARTED for event in sink.events)
        assert any(event.event_type is EventType.RECOVERY_COMPLETED for event in sink.events)

    for fault in (
        FaultPlan(fail=True),
        FaultPlan(timeout=True),
        FaultPlan(invalid_response=True),
        FaultPlan(latency_ms=30),
    ):
        asyncio.run(scenario(fault))


def test_duplicate_response_is_rejected_and_booking_is_idempotent() -> None:
    async def scenario() -> None:
        base = AppointmentToolProvider()
        provider = FaultInjectingToolProvider(base, FaultPlan(duplicate_response=True))
        machine, sink = _thinking_machine()
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            max_attempts=1,
        )
        arguments = {
            "slot": "2026-08-25T09:00:00+02:00",
            "idempotency_key": "call-1-turn-1",
        }
        first = await executor.execute("book_appointment", arguments, fallback=_fallback)
        second = await executor.execute("book_appointment", arguments, fallback=_fallback)

        assert not first.recovered
        assert second.recovered and second.used_fallback
        assert second.failure_reason == "duplicate_response"
        assert len(base._bookings) == 1
        assert any(
            event.event_type is EventType.TOOL_FAILED
            and event.reason_code == "duplicate_response"
            for event in sink.events
        )

    asyncio.run(scenario())


def test_late_tool_result_is_rejected_after_epoch_invalidation() -> None:
    class SlowProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
            del name, arguments
            self.calls += 1
            await asyncio.sleep(0.01)
            return ToolProviderResponse(
                response_id=f"slow-{self.calls}", value={"call": self.calls}
            )

    async def scenario() -> None:
        machine, sink = _thinking_machine()
        provider = SlowProvider()
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            timeout_ms=100,
            max_attempts=2,
        )
        task = asyncio.create_task(executor.execute("lookup_customer", {}, fallback=_fallback))
        await asyncio.sleep(0.002)
        machine.invalidate_operations(reason_code="newer_turn", correlation_id="interrupt")
        result = await task

        assert result.recovered
        assert result.value == {"call": 2}
        assert any(event.event_type is EventType.STALE_RESULT_REJECTED for event in sink.events)
        assert machine.state is ConversationState.THINKING

    asyncio.run(scenario())
