from __future__ import annotations

from pathlib import Path

from humanflow.evaluation import evaluate_policy, load_turn_cases
from humanflow.turns import FixedSilencePolicy, HybridTurnPolicy


CORPUS = Path(__file__).with_name("turn_cases.jsonl")


def test_protected_turn_corpus_has_unique_minimum_sample() -> None:
    cases = load_turn_cases(CORPUS, minimum_cases=50)
    assert len(cases) >= 50
    assert len({case.case_id for case in cases}) == len(cases)


def test_hybrid_policy_beats_fixed_silence_on_same_immutable_cases() -> None:
    cases = load_turn_cases(CORPUS)
    fixed = evaluate_policy(FixedSilencePolicy(), cases)
    hybrid = evaluate_policy(HybridTurnPolicy(), cases)

    assert hybrid.total == fixed.total == len(cases)
    assert hybrid.accuracy > fixed.accuracy
    assert hybrid.accuracy >= 0.95

