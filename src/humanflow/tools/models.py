"""Tool-call and recovery result types."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ToolProviderResponse:
    response_id: str
    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.response_id.strip():
            raise ValueError("response_id must not be empty")
        object.__setattr__(self, "value", MappingProxyType(dict(self.value)))


@dataclass(frozen=True, slots=True)
class FaultPlan:
    latency_ms: float = 0.0
    fail: bool = False
    timeout: bool = False
    invalid_response: bool = False
    duplicate_response: bool = False

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    tool_name: str
    value: Mapping[str, Any]
    attempts: int
    recovered: bool
    used_fallback: bool
    failure_reason: str | None
    elapsed_ms: float

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        object.__setattr__(self, "value", MappingProxyType(dict(self.value)))
