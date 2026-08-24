from __future__ import annotations

import struct

from humanflow.audio.models import AudioFrame
from humanflow.runtime.acoustic_barge_in import (
    AcousticBargeInDetector,
    AcousticEventType,
)


def _frame(sequence: int, amplitude: int, *, base_ns: int = 1_000_000_000) -> AudioFrame:
    samples = 320
    return AudioFrame(
        stream_id="authoritative-microphone",
        sequence=sequence,
        pcm16=struct.pack("<h", amplitude) * samples,
        sample_rate_hz=16_000,
        captured_ns=base_ns + sequence * 20_000_000,
    )


def test_short_backchannel_soft_yields_then_recovers_without_hard_cancel() -> None:
    detector = AcousticBargeInDetector()
    events = []
    for sequence in range(10):
        events.extend(
            detector.observe(
                _frame(sequence, 5_000), assistant_playback_active=True
            )
        )
    for sequence in range(10, 20):
        events.extend(
            detector.observe(
                _frame(sequence, 0), assistant_playback_active=True
            )
        )

    assert [event.event_type for event in events] == [
        AcousticEventType.SPEECH_ONSET,
        AcousticEventType.SPEECH_ENDED,
    ]
    assert events[0].detection_latency_ms == 80.0
    assert not detector.speech_active


def test_sustained_takeover_confirms_without_waiting_for_stt() -> None:
    detector = AcousticBargeInDetector()
    events = []
    for sequence in range(32):
        events.extend(
            detector.observe(
                _frame(sequence, 5_000), assistant_playback_active=True
            )
        )

    assert [event.event_type for event in events] == [
        AcousticEventType.SPEECH_ONSET,
        AcousticEventType.SUSTAINED_TAKEOVER,
    ]
    assert events[1].speech_duration_ms == 620.0
    assert events[1].detection_latency_ms == 620.0


def test_noise_transient_never_ducks_or_cancels() -> None:
    detector = AcousticBargeInDetector()
    events = []
    amplitudes = [0, 0, 6_000, 0, 0, 0, 0]
    for sequence, amplitude in enumerate(amplitudes):
        events.extend(
            detector.observe(
                _frame(sequence, amplitude), assistant_playback_active=True
            )
        )

    assert events == []


def test_audio_outside_assistant_playback_cannot_emit_barge_in() -> None:
    detector = AcousticBargeInDetector()
    events = []
    for sequence in range(40):
        events.extend(
            detector.observe(
                _frame(sequence, 8_000), assistant_playback_active=False
            )
        )

    assert events == []
