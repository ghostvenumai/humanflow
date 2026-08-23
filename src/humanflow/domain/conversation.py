"""Conversation lifecycle domain types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConversationState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    POSSIBLE_END_OF_TURN = "POSSIBLE_END_OF_TURN"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    POSSIBLE_INTERRUPTION = "POSSIBLE_INTERRUPTION"
    INTERRUPTED = "INTERRUPTED"
    OVERLAP = "OVERLAP"
    TOOL_WAIT = "TOOL_WAIT"
    RECOVERING = "RECOVERING"
    HANDOFF = "HANDOFF"
    DISCONNECTING = "DISCONNECTING"
    FAILED = "FAILED"


class InvalidTransition(ValueError):
    """Raised when a state transition violates the explicit lifecycle graph."""


@dataclass(frozen=True, slots=True)
class OperationToken:
    """Identifies async work and makes late/stale results rejectable."""

    conversation_id: str
    epoch: int
    operation_id: str
    kind: str

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be empty")
        if not self.kind.strip():
            raise ValueError("kind must not be empty")

