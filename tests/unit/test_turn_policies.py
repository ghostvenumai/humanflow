from __future__ import annotations

import pytest

from humanflow.turns import (
    FixedSilencePolicy,
    HybridTurnPolicy,
    TurnDecisionType,
    TurnSignals,
)


def signals(**overrides: object) -> TurnSignals:
    values: dict[str, object] = {
        "speech_active": False,
        "silence_duration_ms": 400,
        "utterance_duration_ms": 900,
        "partial_transcript": "",
        "final_transcript": "",
        "semantic_complete": False,
        "filler_ending": False,
        "acoustic_completion": 0.0,
        "background_speech_probability": 0.0,
        "interruption_probability": 0.0,
        "agent_speaking": False,
    }
    values.update(overrides)
    return TurnSignals(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["mhm", "Ja.", "ja genau", "Verstehe"])
def test_short_german_acknowledgements_are_backchannels(text: str) -> None:
    decision = HybridTurnPolicy().decide(
        signals(final_transcript=text, agent_speaking=True, utterance_duration_ms=500)
    )
    assert decision.decision is TurnDecisionType.BACKCHANNEL


@pytest.mark.parametrize("text", ["Moment, stopp", "warte mal", "Nein stopp, ich meinte Freitag"])
def test_explicit_german_takeover_is_interruption(text: str) -> None:
    decision = HybridTurnPolicy().decide(
        signals(
            final_transcript=text,
            agent_speaking=True,
            interruption_probability=0.90,
        )
    )
    assert decision.decision is TurnDecisionType.INTERRUPTION


def test_low_probability_stop_word_does_not_force_interruption() -> None:
    decision = HybridTurnPolicy().decide(
        signals(
            final_transcript="Moment",
            agent_speaking=True,
            interruption_probability=0.30,
        )
    )
    assert decision.decision is not TurnDecisionType.INTERRUPTION


def test_active_speech_and_unfinished_filler_continue_listening() -> None:
    policy = HybridTurnPolicy()
    assert policy.decide(signals(speech_active=True)).decision is TurnDecisionType.CONTINUE_LISTENING
    assert (
        policy.decide(
            signals(
                final_transcript="Ich wollte noch äh",
                filler_ending=True,
                silence_duration_ms=900,
            )
        ).decision
        is TurnDecisionType.CONTINUE_LISTENING
    )


def test_semantic_and_acoustic_completion_is_complete() -> None:
    decision = HybridTurnPolicy().decide(
        signals(
            final_transcript="Dann nehme ich Freitag um zehn.",
            semantic_complete=True,
            acoustic_completion=0.8,
            silence_duration_ms=350,
        )
    )
    assert decision.decision is TurnDecisionType.COMPLETE
    assert decision.confidence >= 0.9


def test_long_silence_without_completion_is_only_likely_complete() -> None:
    decision = HybridTurnPolicy().decide(
        signals(final_transcript="Freitag", silence_duration_ms=1500)
    )
    assert decision.decision is TurnDecisionType.LIKELY_COMPLETE


def test_background_speech_while_agent_speaks_is_uncertain() -> None:
    decision = HybridTurnPolicy().decide(
        signals(agent_speaking=True, background_speech_probability=0.9)
    )
    assert decision.decision is TurnDecisionType.UNCERTAIN


def test_fixed_silence_policy_is_a_reproducible_baseline() -> None:
    policy = FixedSilencePolicy(threshold_ms=800)
    assert policy.decide(signals(silence_duration_ms=799)).decision is TurnDecisionType.CONTINUE_LISTENING
    assert policy.decide(signals(silence_duration_ms=800)).decision is TurnDecisionType.COMPLETE


def test_signal_probabilities_are_validated() -> None:
    with pytest.raises(ValueError):
        signals(interruption_probability=1.1)

