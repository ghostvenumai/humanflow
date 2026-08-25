#!/usr/bin/env python3
"""Deterministic PCM evidence for the bounded post-freeze noise audit."""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from humanflow.audio.models import AudioFrame
from humanflow.runtime.acoustic_barge_in import (
    AcousticBargeInDetector,
    AcousticEventType,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "noise-robustness.json"


def _frame(sequence: int, amplitude: int) -> AudioFrame:
    return AudioFrame(
        stream_id="local-synthetic-noise-fixture",
        sequence=sequence,
        pcm16=struct.pack("<h", amplitude) * 320,
        sample_rate_hz=16_000,
        captured_ns=1_000_000_000 + sequence * 20_000_000,
    )


def _measure(name: str, amplitudes: Iterable[int], *, source: str) -> dict[str, object]:
    detector = AcousticBargeInDetector()
    events = []
    for sequence, amplitude in enumerate(amplitudes):
        events.extend(
            detector.observe(
                _frame(sequence, amplitude),
                assistant_playback_active=True,
            )
        )
    event_types = [event.event_type for event in events]
    onset_count = event_types.count(AcousticEventType.SPEECH_ONSET)
    hard_cancel_count = event_types.count(AcousticEventType.SUSTAINED_TAKEOVER)
    ended_count = event_types.count(AcousticEventType.SPEECH_ENDED)
    onset = next(
        (event for event in events if event.event_type is AcousticEventType.SPEECH_ONSET),
        None,
    )
    terminal = next(
        (
            event
            for event in events
            if event.event_type
            in {AcousticEventType.SPEECH_ENDED, AcousticEventType.SUSTAINED_TAKEOVER}
        ),
        None,
    )
    duck_duration_ms = (
        None
        if onset is None or terminal is None
        else round((terminal.observed_ns - onset.observed_ns) / 1_000_000.0, 3)
    )
    known_non_user_noise = source in {"quiet", "stationary_low_level_noise"}
    return {
        "scenario": name,
        "scope": "LOCAL_SYNTHETIC_PCM",
        "source_classification": source,
        "vad_speech_onset_count": onset_count,
        "possible_interruption_count": onset_count,
        "soft_yield_count": onset_count,
        "duck_duration_ms": duck_duration_ms,
        "resume_count": ended_count if hard_cancel_count == 0 else 0,
        "hard_cancel_count": hard_cancel_count,
        "false_hard_cancel_count": (
            hard_cancel_count if known_non_user_noise else None
        ),
        "audible_stop_evidence": "NOT_MEASURED_DETECTOR_ONLY",
        "event_types": [event_type.value for event_type in event_types],
    }


def build_report() -> dict[str, object]:
    speech_like = [5_000] * 26 + [0] * 10
    scenarios = [
        _measure("quiet_baseline", [0] * 100, source="quiet"),
        _measure(
            "stationary_low_level_noise",
            [700] * 100,
            source="stationary_low_level_noise",
        ),
        _measure(
            "speech_like_background",
            speech_like,
            source="synthetic_speech_like_unattributed",
        ),
        _measure(
            "intentional_user_interruption",
            speech_like,
            source="synthetic_intentional_user_speech",
        ),
        _measure(
            "mhm_backchannel",
            [5_000] * 10 + [0] * 10,
            source="synthetic_backchannel",
        ),
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "measurement_scope": "LOCAL_SYNTHETIC_PCM_DETECTOR_ONLY",
        "scenarios": scenarios,
        "finding": (
            "Quiet and stationary low-level fixtures cause no duck or hard cancel. "
            "Speech-like background and intentional speech are intentionally identical "
            "at the mono PCM boundary and therefore cannot be safely separated by an "
            "energy-only detector without risking real floor-taking."
        ),
        "decision": "NO_CHANGE_REQUIRED",
        "decision_reason": (
            "The reported browser volume drop is not reproduced by stationary noise. "
            "A detector-only suppression of speech-like input would endanger intentional "
            "barge-in; human browser correlation remains required."
        ),
        "known_limitation": (
            "Synthetic PCM does not establish audible browser volume pumping or classify "
            "unattributed real-world speech-like sound as background noise."
        ),
    }


def main() -> None:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
