"""Inputs and outputs of turn detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TurnDecisionType(StrEnum):
    CONTINUE_LISTENING = "CONTINUE_LISTENING"
    LIKELY_COMPLETE = "LIKELY_COMPLETE"
    COMPLETE = "COMPLETE"
    INTERRUPTION = "INTERRUPTION"
    BACKCHANNEL = "BACKCHANNEL"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class TurnSignals:
    speech_active: bool
    silence_duration_ms: int
    utterance_duration_ms: int
    partial_transcript: str = ""
    final_transcript: str = ""
    semantic_complete: bool = False
    filler_ending: bool = False
    acoustic_completion: float = 0.0
    background_speech_probability: float = 0.0
    interruption_probability: float = 0.0
    agent_speaking: bool = False
    provider_endpointed: bool = False

    def __post_init__(self) -> None:
        if self.silence_duration_ms < 0 or self.utterance_duration_ms < 0:
            raise ValueError("durations must be non-negative")
        for name in (
            "acoustic_completion",
            "background_speech_probability",
            "interruption_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def transcript(self) -> str:
        return (self.final_transcript or self.partial_transcript).strip()


@dataclass(frozen=True, slots=True)
class TurnDecision:
    decision: TurnDecisionType
    confidence: float
    reason_codes: tuple[str, ...]
    signals_used: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason_codes:
            raise ValueError("reason_codes must not be empty")
        if not self.signals_used:
            raise ValueError("signals_used must not be empty")
