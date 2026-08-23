#!/usr/bin/env python3
"""Measure bounded recovery against the briefing fault matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import get_clock_info, monotonic_ns
from typing import Any

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


FAULT_MODES = (
    "fail",
    "timeout",
    "invalid_response",
    "duplicate_response",
    "latency_timeout",
    "stale_result",
)


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "mean": round(fmean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p90": round(_percentile(values, 90), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


def _machine() -> tuple[ConversationStateMachine, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    machine = ConversationStateMachine(conversation_id="recovery-benchmark", sink=sink)
    machine.transition(
        ConversationState.LISTENING,
        reason_code="benchmark_started",
        correlation_id="setup",
    )
    machine.transition(
        ConversationState.THINKING,
        reason_code="benchmark_turn_complete",
        correlation_id="setup",
    )
    return machine, sink


def _fallback(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    del arguments
    return {"safe_fallback": True, "tool_name": name}


class SlowUniqueProvider:
    def __init__(self, *, delay_ms: float) -> None:
        self.delay_ms = delay_ms
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        del name, arguments
        self.calls += 1
        await asyncio.sleep(self.delay_ms / 1000.0)
        return ToolProviderResponse(
            response_id=f"slow-{self.calls}", value={"provider_call": self.calls}
        )


class FailOnceProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        del name, arguments
        self.calls += 1
        if self.calls == 1:
            raise ToolProviderError("injected_once")
        return ToolProviderResponse(response_id="retry-ok", value={"ok": True})


async def _execute_with_tick_probe(
    executor: ResilientToolExecutor,
    *,
    machine: ConversationStateMachine,
    mode: str,
) -> tuple[Any, float]:
    finished = asyncio.Event()
    gaps_ms: list[float] = []

    async def ticker() -> None:
        previous_ns = monotonic_ns()
        while not finished.is_set():
            await asyncio.sleep(0.001)
            current_ns = monotonic_ns()
            gaps_ms.append((current_ns - previous_ns) / 1_000_000.0)
            previous_ns = current_ns

    ticker_task = asyncio.create_task(ticker())
    task = asyncio.create_task(
        executor.execute("lookup_customer", {}, fallback=_fallback, correlation_id=mode)
    )
    if mode == "stale_result":
        await asyncio.sleep(0.002)
        machine.invalidate_operations(reason_code="newer_turn", correlation_id=mode)
    result = await task
    finished.set()
    await ticker_task
    return result, max(gaps_ms, default=0.0)


async def _sample(mode: str, repetition: int) -> dict[str, Any]:
    machine, sink = _machine()
    timeout_ms = 8.0
    if mode == "fail":
        provider: Any = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(fail=True)
        )
    elif mode == "timeout":
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(timeout=True), timeout_hold_ms=30
        )
    elif mode == "invalid_response":
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(invalid_response=True)
        )
    elif mode == "duplicate_response":
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(duplicate_response=True)
        )
    elif mode == "latency_timeout":
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(latency_ms=20)
        )
    elif mode == "stale_result":
        provider = SlowUniqueProvider(delay_ms=5)
    else:
        raise ValueError(mode)

    executor = ResilientToolExecutor(
        state_machine=machine,
        provider=provider,
        timeout_ms=timeout_ms,
        max_attempts=2,
    )
    if mode == "duplicate_response":
        await executor.execute("lookup_customer", {}, fallback=_fallback, correlation_id="prime")

    result, maximum_tick_gap_ms = await _execute_with_tick_probe(
        executor, machine=machine, mode=mode
    )
    event_types = [event.event_type for event in sink.events]
    return {
        "mode": mode,
        "repetition": repetition,
        "attempts": result.attempts,
        "elapsed_ms": result.elapsed_ms,
        "maximum_1ms_tick_gap_ms": maximum_tick_gap_ms,
        "recovered": result.recovered,
        "used_fallback": result.used_fallback,
        "failure_reason": result.failure_reason,
        "recovery_started_events": event_types.count(EventType.RECOVERY_STARTED),
        "recovery_completed_events": event_types.count(EventType.RECOVERY_COMPLETED),
        "stale_rejected_events": event_types.count(EventType.STALE_RESULT_REJECTED),
        "final_state": machine.state.value,
    }


async def _run(repetitions: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for mode in FAULT_MODES:
        for repetition in range(repetitions):
            samples.append(await _sample(mode, repetition))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports/recovery-benchmark.json"))
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("--repetitions must be at least 5")

    samples = asyncio.run(_run(args.repetitions))
    elapsed = [float(sample["elapsed_ms"]) for sample in samples]
    tick_gaps = [float(sample["maximum_1ms_tick_gap_ms"]) for sample in samples]
    completed = sum(int(sample["recovery_completed_events"] > 0) for sample in samples)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "measurement_scope": {
            "kind": "local_fault_injection_benchmark",
            "provider": "in-memory appointment provider with deterministic injected faults",
            "tool_timeout_ms": 8.0,
            "max_attempts": 2,
            "fault_modes": list(FAULT_MODES),
            "latency_fault_ms": 20.0,
            "briefing_4000ms_supported": True,
            "claim_limit": (
                "Measures local controller recovery and event-loop scheduling; no network or "
                "production provider was involved."
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "monotonic_clock": vars(get_clock_info("monotonic")),
        },
        "metrics": {
            "tool_failure_recovery_rate": {
                "injected_failures": len(samples),
                "recovery_completed": completed,
                "rate": completed / len(samples),
            },
            "recovery_elapsed_ms": _summary(elapsed),
            "maximum_1ms_tick_gap_ms": _summary(tick_gaps),
        },
        "by_fault_mode": {
            mode: {
                "elapsed_ms": _summary(
                    [float(sample["elapsed_ms"]) for sample in samples if sample["mode"] == mode]
                ),
                "recovery_completed": sum(
                    int(sample["recovery_completed_events"] > 0)
                    for sample in samples
                    if sample["mode"] == mode
                ),
            }
            for mode in FAULT_MODES
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
