"""Objective PCM16 signal measurements for response-level playback telemetry."""

from __future__ import annotations

import math
import warnings
from array import array
from dataclasses import dataclass

from .models import AudioFrame

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop as _audioop
    except ImportError:  # pragma: no cover - Python 3.13+ compatibility path
        _audioop = None


@dataclass(frozen=True, slots=True)
class PcmSignalMetrics:
    rms_dbfs: float | None
    peak_dbfs: float | None
    duration_ms: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "duration_ms": round(self.duration_ms, 3),
            "sample_count": self.sample_count,
        }


def analyze_pcm16(frame: AudioFrame) -> PcmSignalMetrics:
    """Measure raw provider amplitude without applying gain or normalization."""

    sample_count = len(frame.pcm16) // 2
    if sample_count == 0:
        return PcmSignalMetrics(None, None, frame.duration_ms, 0)
    if _audioop is not None:
        peak = float(_audioop.max(frame.pcm16, 2))
        rms = float(_audioop.rms(frame.pcm16, 2))
    else:
        samples = array("h")
        samples.frombytes(frame.pcm16)
        peak = float(max(abs(sample) for sample in samples))
        mean_square = sum(float(sample) * sample for sample in samples) / len(samples)
        rms = math.sqrt(mean_square)
    return PcmSignalMetrics(
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(float(peak)),
        duration_ms=frame.duration_ms,
        sample_count=sample_count,
    )


def _dbfs(amplitude: float) -> float | None:
    if amplitude <= 0:
        return None
    return round(20.0 * math.log10(amplitude / 32_768.0), 3)
