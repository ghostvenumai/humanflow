from __future__ import annotations

from humanflow.audio.ledger import LedgerState, PlayedAudioLedger
from humanflow.audio.models import AudioChunk, AudioFrame, PlaybackReceipt


def _chunk(chunk_id: str, text: str, *, samples: int = 160) -> AudioChunk:
    return AudioChunk(
        chunk_id=chunk_id,
        response_id="response-1",
        text=text,
        frame=AudioFrame(
            stream_id="response-1",
            sequence=int(chunk_id[-1]),
            pcm16=b"\x00\x00" * samples,
        ),
    )


def test_ledger_delivered_boundary_excludes_partially_played_semantics() -> None:
    ledger = PlayedAudioLedger()
    complete = _chunk("chunk-1", "Ihr Termin")
    partial = _chunk("chunk-2", "ist Donnerstag")
    pending = _chunk("chunk-3", "um 15 Uhr")

    ledger.register_generated(complete, generated_ns=10)
    ledger.mark_queued(complete.chunk_id, queued_ns=11)
    ledger.mark_playback_started(complete.chunk_id, started_ns=12)
    ledger.record_playback(
        PlaybackReceipt(complete.chunk_id, 160, 160, 12, 22, cancelled=False)
    )
    ledger.register_generated(partial, generated_ns=23)
    ledger.mark_queued(partial.chunk_id, queued_ns=24)
    ledger.mark_playback_started(partial.chunk_id, started_ns=25)
    ledger.record_playback(
        PlaybackReceipt(partial.chunk_id, 160, 80, 25, 30, cancelled=True)
    )
    ledger.register_generated(pending, generated_ns=26)
    ledger.mark_queued(pending.chunk_id, queued_ns=27)
    ledger.cancel_unplayed(response_id="response-1", cancelled_ns=30)

    ledger.assert_invariants()
    assert ledger.delivered_text(response_id="response-1") == "Ihr Termin"
    assert ledger.unheard_text(response_id="response-1") == "ist Donnerstag um 15 Uhr"
    assert [entry.state for entry in ledger.entries] == [
        LedgerState.PLAYED,
        LedgerState.CANCELLED,
        LedgerState.CANCELLED,
    ]


def test_audio_frame_duration_uses_samples_per_channel() -> None:
    frame = AudioFrame(
        stream_id="stereo",
        sequence=0,
        pcm16=b"\x00\x00" * 320,
        sample_rate_hz=16_000,
        channels=2,
    )
    assert frame.samples_per_channel == 160
    assert frame.duration_ms == 10.0
