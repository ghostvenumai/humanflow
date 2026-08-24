from __future__ import annotations

from array import array

import pytest

from humanflow.audio.analysis import analyze_pcm16
from humanflow.audio.models import AudioFrame


def test_pcm_signal_metrics_are_raw_and_reproducible() -> None:
    samples = array("h", [8_192, -8_192] * 160)
    metrics = analyze_pcm16(
        AudioFrame(
            stream_id="signal",
            sequence=0,
            pcm16=samples.tobytes(),
            sample_rate_hz=16_000,
        )
    )

    assert metrics.rms_dbfs == pytest.approx(-12.041, abs=0.001)
    assert metrics.peak_dbfs == pytest.approx(-12.041, abs=0.001)
    assert metrics.duration_ms == 20.0
    assert metrics.sample_count == 320


def test_silence_reports_no_invented_decibel_value() -> None:
    metrics = analyze_pcm16(
        AudioFrame(
            stream_id="silence",
            sequence=0,
            pcm16=b"\x00\x00" * 160,
            sample_rate_hz=16_000,
        )
    )

    assert metrics.rms_dbfs is None
    assert metrics.peak_dbfs is None
