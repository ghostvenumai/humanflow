"""Bounded tool execution that keeps failures inside explicit recovery states."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState
from humanflow.telemetry.events import EventType

from .models import ToolExecutionResult, ToolProviderResponse
from .providers import ToolProvider


class ToolProtocolError(RuntimeError):
    """Provider output failed the controller-owned response contract."""


Fallback = Callable[[str, Mapping[str, Any]], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class ResilientToolExecutor:
    def __init__(
        self,
        *,
        state_machine: ConversationStateMachine,
        provider: ToolProvider,
        timeout_ms: float = 1_000.0,
        max_attempts: int = 2,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._state_machine = state_machine
        self._provider = provider
        self._timeout_ms = timeout_ms
        self._max_attempts = max_attempts
        self._clock_ns = clock_ns
        self._seen_response_ids: set[str] = set()

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        fallback: Fallback,
        correlation_id: str | None = None,
    ) -> ToolExecutionResult:
        if not name.strip():
            raise ValueError("name must not be empty")
        if self._state_machine.state is not ConversationState.THINKING:
            raise RuntimeError("tools may only start while THINKING")
        correlation_id = correlation_id or str(uuid4())
        started_ns = self._clock_ns()
        failure_reason: str | None = None
        recovery_started = False

        for attempt in range(1, self._max_attempts + 1):
            if self._state_machine.state is ConversationState.THINKING:
                self._state_machine.transition(
                    ConversationState.TOOL_WAIT,
                    reason_code="tool_call_started",
                    correlation_id=correlation_id,
                    payload={"tool_name": name, "attempt": attempt},
                )
            token = self._state_machine.issue_operation(kind=f"tool:{name}")
            provider_arguments = dict(arguments)
            provider_arguments["_humanflow_operation_is_current"] = lambda: (
                token.conversation_id == self._state_machine.conversation_id
                and token.epoch == self._state_machine.operation_epoch
            )
            attempt_started_ns = self._clock_ns()
            self._state_machine.record(
                EventType.TOOL_STARTED,
                correlation_id=correlation_id,
                reason_code="provider_call_started",
                payload={"tool_name": name, "attempt": attempt},
            )
            try:
                raw_response = await asyncio.wait_for(
                    self._provider.call(name, provider_arguments),
                    timeout=self._timeout_ms / 1000.0,
                )
                response = self._validate_response(raw_response)
                if not self._state_machine.accept_result(
                    token, correlation_id=correlation_id
                ):
                    raise ToolProtocolError("stale_result")
            except Exception as error:
                # CancelledError remains unhandled because it derives from BaseException.
                failure_reason = self._failure_reason(error)
                failure_ns = self._clock_ns()
                self._state_machine.record(
                    EventType.TOOL_FAILED,
                    correlation_id=correlation_id,
                    reason_code=failure_reason,
                    payload={
                        "tool_name": name,
                        "attempt": attempt,
                        "elapsed_ms": (failure_ns - attempt_started_ns) / 1_000_000.0,
                        "exception_type": type(error).__name__,
                    },
                )
                self._state_machine.invalidate_operations(
                    reason_code="tool_attempt_failed",
                    correlation_id=correlation_id,
                )
                self._state_machine.transition(
                    ConversationState.RECOVERING,
                    reason_code=failure_reason,
                    correlation_id=correlation_id,
                    payload={"tool_name": name, "attempt": attempt},
                )
                if not recovery_started:
                    recovery_started = True
                    self._state_machine.record(
                        EventType.RECOVERY_STARTED,
                        correlation_id=correlation_id,
                        reason_code=failure_reason,
                        payload={"tool_name": name},
                    )
                if attempt < self._max_attempts:
                    self._state_machine.transition(
                        ConversationState.THINKING,
                        reason_code="bounded_tool_retry",
                        correlation_id=correlation_id,
                        payload={"next_attempt": attempt + 1},
                    )
                    continue
                return await self._recover_with_fallback(
                    name=name,
                    arguments=arguments,
                    fallback=fallback,
                    attempts=attempt,
                    failure_reason=failure_reason,
                    started_ns=started_ns,
                    correlation_id=correlation_id,
                )

            completed_ns = self._clock_ns()
            self._seen_response_ids.add(response.response_id)
            self._state_machine.record(
                EventType.TOOL_COMPLETED,
                correlation_id=correlation_id,
                reason_code="provider_response_accepted",
                payload={
                    "tool_name": name,
                    "attempt": attempt,
                    "response_id": response.response_id,
                    "elapsed_ms": (completed_ns - attempt_started_ns) / 1_000_000.0,
                },
            )
            self._state_machine.transition(
                ConversationState.THINKING,
                reason_code="tool_result_available",
                correlation_id=correlation_id,
            )
            if recovery_started:
                self._state_machine.record(
                    EventType.RECOVERY_COMPLETED,
                    correlation_id=correlation_id,
                    reason_code="retry_succeeded",
                    payload={"tool_name": name, "attempts": attempt},
                )
            return ToolExecutionResult(
                tool_name=name,
                value=response.value,
                attempts=attempt,
                recovered=recovery_started,
                used_fallback=False,
                failure_reason=failure_reason,
                elapsed_ms=(completed_ns - started_ns) / 1_000_000.0,
            )

        raise AssertionError("unreachable tool execution state")

    def _validate_response(self, response: object) -> ToolProviderResponse:
        if not isinstance(response, ToolProviderResponse):
            raise ToolProtocolError("invalid_response")
        if response.response_id in self._seen_response_ids:
            raise ToolProtocolError("duplicate_response")
        return response

    async def _recover_with_fallback(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        fallback: Fallback,
        attempts: int,
        failure_reason: str,
        started_ns: int,
        correlation_id: str,
    ) -> ToolExecutionResult:
        value = fallback(name, arguments)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, Mapping):
            self._state_machine.transition(
                ConversationState.FAILED,
                reason_code="fallback_invalid_response",
                correlation_id=correlation_id,
            )
            raise ToolProtocolError("fallback_invalid_response")
        completed_ns = self._clock_ns()
        self._state_machine.transition(
            ConversationState.THINKING,
            reason_code="safe_tool_fallback",
            correlation_id=correlation_id,
            payload={"tool_name": name},
        )
        self._state_machine.record(
            EventType.RECOVERY_COMPLETED,
            correlation_id=correlation_id,
            reason_code="safe_fallback_returned",
            payload={"tool_name": name, "attempts": attempts},
        )
        return ToolExecutionResult(
            tool_name=name,
            value=value,
            attempts=attempts,
            recovered=True,
            used_fallback=True,
            failure_reason=failure_reason,
            elapsed_ms=(completed_ns - started_ns) / 1_000_000.0,
        )

    @staticmethod
    def _failure_reason(error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return "tool_timeout"
        if isinstance(error, ToolProtocolError):
            return str(error)
        return "tool_provider_failure"
