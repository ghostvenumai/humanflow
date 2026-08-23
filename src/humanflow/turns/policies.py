"""Deterministic turn policies used as a baseline and first hybrid candidate."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import TurnDecision, TurnDecisionType, TurnSignals


_SPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^a-zäöüß0-9 ]+")


def _normalized(text: str) -> str:
    return _SPACE.sub(" ", _PUNCTUATION.sub(" ", text.casefold())).strip()


BACKCHANNELS = frozenset(
    {
        "aha",
        "genau",
        "ja",
        "ja genau",
        "mhm",
        "okay",
        "stimmt",
        "verstehe",
    }
)
INTERRUPTION_PHRASES = (
    "halt",
    "moment",
    "nein stopp",
    "stopp",
    "warte",
    "warte mal",
)


def _decision(
    kind: TurnDecisionType,
    confidence: float,
    *reasons: str,
    signals: tuple[str, ...],
) -> TurnDecision:
    return TurnDecision(kind, confidence, tuple(reasons), signals)


@dataclass(frozen=True, slots=True)
class FixedSilencePolicy:
    threshold_ms: int = 800

    def decide(self, signals: TurnSignals) -> TurnDecision:
        if signals.speech_active or signals.silence_duration_ms < self.threshold_ms:
            return _decision(
                TurnDecisionType.CONTINUE_LISTENING,
                0.70,
                "fixed_silence_below_threshold",
                signals=("speech_active", "silence_duration_ms"),
            )
        return _decision(
            TurnDecisionType.COMPLETE,
            0.70,
            "fixed_silence_threshold_reached",
            signals=("speech_active", "silence_duration_ms"),
        )


@dataclass(frozen=True, slots=True)
class HybridTurnPolicy:
    minimum_silence_ms: int = 280
    likely_complete_ms: int = 650
    forced_complete_ms: int = 1400
    interruption_threshold: float = 0.72

    def decide(self, signals: TurnSignals) -> TurnDecision:
        text = _normalized(signals.transcript)

        if signals.agent_speaking:
            if self._is_intentional_interruption(text, signals.interruption_probability):
                return _decision(
                    TurnDecisionType.INTERRUPTION,
                    max(0.85, signals.interruption_probability),
                    "explicit_german_interruption",
                    "interruption_probability_high",
                    signals=("transcript", "interruption_probability", "agent_speaking"),
                )
            if text in BACKCHANNELS and signals.utterance_duration_ms <= 1200:
                return _decision(
                    TurnDecisionType.BACKCHANNEL,
                    0.93,
                    "short_german_acknowledgement",
                    signals=("transcript", "utterance_duration_ms", "agent_speaking"),
                )
            if signals.background_speech_probability >= 0.75:
                return _decision(
                    TurnDecisionType.UNCERTAIN,
                    0.78,
                    "probable_background_speech",
                    signals=("background_speech_probability", "agent_speaking"),
                )

        if signals.speech_active:
            return _decision(
                TurnDecisionType.CONTINUE_LISTENING,
                0.98,
                "speech_still_active",
                signals=("speech_active",),
            )

        if (
            signals.provider_endpointed
            and signals.semantic_complete
            and not signals.agent_speaking
        ):
            return _decision(
                TurnDecisionType.COMPLETE,
                0.99,
                "stt_provider_final_endpoint",
                signals=("provider_endpointed", "semantic_complete"),
            )

        if signals.filler_ending and signals.silence_duration_ms < self.forced_complete_ms:
            return _decision(
                TurnDecisionType.CONTINUE_LISTENING,
                0.88,
                "filler_or_unfinished_ending",
                signals=("filler_ending", "silence_duration_ms", "transcript"),
            )

        if signals.silence_duration_ms < self.minimum_silence_ms:
            return _decision(
                TurnDecisionType.CONTINUE_LISTENING,
                0.90,
                "silence_too_short",
                signals=("silence_duration_ms",),
            )

        if (
            signals.semantic_complete
            and signals.acoustic_completion >= 0.5
            and signals.silence_duration_ms >= self.minimum_silence_ms
        ):
            return _decision(
                TurnDecisionType.COMPLETE,
                min(0.99, 0.78 + signals.acoustic_completion * 0.2),
                "semantic_and_acoustic_completion",
                signals=("semantic_complete", "acoustic_completion", "silence_duration_ms"),
            )

        if signals.semantic_complete and signals.silence_duration_ms >= self.likely_complete_ms:
            return _decision(
                TurnDecisionType.LIKELY_COMPLETE,
                0.76,
                "semantic_completion_with_silence",
                signals=("semantic_complete", "silence_duration_ms"),
            )

        if signals.silence_duration_ms >= self.forced_complete_ms and text:
            return _decision(
                TurnDecisionType.LIKELY_COMPLETE,
                0.68,
                "long_silence_after_nonempty_transcript",
                signals=("silence_duration_ms", "transcript"),
            )

        return _decision(
            TurnDecisionType.UNCERTAIN,
            0.55,
            "insufficient_completion_evidence",
            signals=("silence_duration_ms", "semantic_complete", "acoustic_completion"),
        )

    def _is_intentional_interruption(self, text: str, probability: float) -> bool:
        phrase_match = any(
            text == phrase or text.startswith(phrase + " ") for phrase in INTERRUPTION_PHRASES
        )
        return phrase_match and probability >= self.interruption_threshold
