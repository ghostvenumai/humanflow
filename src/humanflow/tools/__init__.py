"""Replaceable tools, fault injection, and bounded recovery."""

from .executor import ResilientToolExecutor
from .appointment_coordinator import (
    AppointmentTransactionCoordinator,
    AppointmentTransactionOutcome,
)
from .models import FaultPlan, ToolExecutionResult, ToolProviderResponse
from .providers import AppointmentToolProvider, FaultInjectingToolProvider
from .sqlite_appointments import SQLiteAppointmentToolProvider

__all__ = [
    "AppointmentToolProvider",
    "AppointmentTransactionCoordinator",
    "AppointmentTransactionOutcome",
    "FaultInjectingToolProvider",
    "FaultPlan",
    "ResilientToolExecutor",
    "SQLiteAppointmentToolProvider",
    "ToolExecutionResult",
    "ToolProviderResponse",
]
