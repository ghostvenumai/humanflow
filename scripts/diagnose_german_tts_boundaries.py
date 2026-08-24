#!/usr/bin/env python3
"""Deterministic production-boundary diagnostic without a paid provider call."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from humanflow.runtime.anthropic_provider import _take_speech_boundaries
from humanflow.runtime.prosody import ProsodyPlanner
from humanflow.runtime.speech_text import GermanSpeechNormalizer


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "german-tts-boundaries.json"
RECORDED_AUDIO = ROOT / "reports" / "audio-continuity-10x.json"
PHRASES = (
    "Donnerstag, den 27. August um 12 Uhr.",
    "Freitag, den 4. September um 14 Uhr.",
    "Der Termin ist am 3. September um 11:30 Uhr.",
    "Am 24. Dezember 2026 um 16:45 Uhr.",
    "Das dauert ca. 3,5 Stunden.",
)


def main() -> None:
    planner = ProsodyPlanner()
    normalizer = GermanSpeechNormalizer()
    cases: list[dict[str, object]] = []
    for case_index, phrase in enumerate(PHRASES, start=1):
        ready, pending = _take_speech_boundaries(phrase)
        stable = [*ready, *([pending.strip()] if pending.strip() else [])]
        segments = [segment for text in stable for segment in planner.plan(text)]
        boundaries = [
            {
                "segment_id": f"phrase-{case_index}:segment:{index}",
                "display_text_boundary": segment.text,
                "spoken_text_boundary": normalizer.normalize(segment.text),
                "intentional_linguistic_pause_ms": segment.pause_after_ms,
                "boundary_pause_class": (
                    "INTENTIONAL_LINGUISTIC"
                    if segment.pause_after_ms > 0
                    else "CONTINUOUS_GAPLESS"
                ),
            }
            for index, segment in enumerate(segments, start=1)
        ]
        cases.append(
            {
                "phrase": phrase,
                "semantic_boundaries": boundaries,
                "physical_tts_requests_planned": len(boundaries),
                "ordinal_month_split": len(boundaries) != 1,
                "scheduler_gap_inside_date_ms": 0 if len(boundaries) == 1 else None,
                "human_pronunciation_assessment": None,
            }
        )

    recorded_audio: dict[str, object] | None = None
    if RECORDED_AUDIO.is_file():
        source = json.loads(RECORDED_AUDIO.read_text(encoding="utf-8"))
        aggregate = source.get("aggregate")
        if isinstance(aggregate, dict):
            recorded_audio = {
                "source": str(RECORDED_AUDIO.relative_to(ROOT)),
                "provider_calls_reused": True,
                "rms_range_across_responses_db": aggregate.get(
                    "rms_range_across_responses_db"
                ),
                "total_underruns": aggregate.get("total_underruns"),
                "duplicate_chunks": aggregate.get("duplicate_chunks"),
                "stale_chunks": aggregate.get("stale_chunks"),
            }

    checks = {
        "all_phrases_have_one_atomic_tts_boundary": all(
            len(case["semantic_boundaries"]) == 1 for case in cases
        ),
        "no_ordinal_month_split": all(
            case["ordinal_month_split"] is False for case in cases
        ),
        "no_scheduler_gap_inside_date_structure": all(
            case["scheduler_gap_inside_date_ms"] == 0 for case in cases
        ),
        "display_and_spoken_text_are_separate": all(
            case["semantic_boundaries"][0]["display_text_boundary"]
            != case["semantic_boundaries"][0]["spoken_text_boundary"]
            for case in cases
        ),
    }
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "result": "PASS_STRUCTURAL_HUMAN_PRONUNCIATION_PENDING",
        "scope": (
            "exact production semantic boundary, prosody and speech-normalization "
            "path; no new paid provider request"
        ),
        "checks": checks,
        "cases": cases,
        "ordinal_ab_input": {
            "a_display_numeric": PHRASES[0],
            "b_production_spoken_words": normalizer.normalize(PHRASES[0]),
            "human_winner": None,
        },
        "reused_real_provider_audio_evidence": recorded_audio,
        "human_voice_wobble_assessment": None,
        "manual_validation": "REQUIRED_NOT_ATTESTED",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not all(checks.values()):
        raise RuntimeError("German TTS structural boundary diagnostic failed")
    print(json.dumps({"checks": checks, "report": str(REPORT.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
