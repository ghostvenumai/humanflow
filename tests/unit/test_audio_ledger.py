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


def test_streamed_pcm_semantic_is_unheard_until_its_final_boundary_finishes() -> None:
    ledger = PlayedAudioLedger()
    first = AudioChunk(
        chunk_id="stream-1",
        response_id="response-1",
        text="",
        semantic_id="segment-1",
        semantic_text="Freitag passt für mich.",
        semantic_boundary=False,
        frame=AudioFrame(
            stream_id="response-1",
            sequence=1,
            pcm16=b"\x00\x00" * 160,
        ),
    )
    final = AudioChunk(
        chunk_id="stream-2",
        response_id="response-1",
        text="Freitag passt für mich.",
        semantic_id="segment-1",
        semantic_text="Freitag passt für mich.",
        semantic_boundary=True,
        frame=AudioFrame(
            stream_id="response-1",
            sequence=2,
            pcm16=b"\x00\x00" * 160,
        ),
    )
    for index, chunk in enumerate((first, final), start=1):
        ledger.register_generated(chunk, generated_ns=index * 10)
        ledger.mark_queued(chunk.chunk_id, queued_ns=index * 10 + 1)
        ledger.mark_playback_started(chunk.chunk_id, started_ns=index * 10 + 2)
        played = 160 if chunk is first else 80
        ledger.record_playback(
            PlaybackReceipt(
                chunk.chunk_id,
                160,
                played,
                index * 10 + 2,
                index * 10 + 3,
                cancelled=chunk is final,
            )
        )

    assert ledger.delivered_text(response_id="response-1") == ""
    assert ledger.unheard_text(response_id="response-1") == "Freitag passt für mich."


def test_queued_chunk_cancelled_before_start_stays_truthfully_unplayed() -> None:
    ledger = PlayedAudioLedger()
    chunk = _chunk("chunk-4", "Noch nicht hörbar")
    ledger.register_generated(chunk, generated_ns=10)
    ledger.mark_queued(chunk.chunk_id, queued_ns=11)
    ledger.cancel_unplayed(response_id=chunk.response_id, cancelled_ns=12)

    ledger.mark_playback_started(chunk.chunk_id, started_ns=13)
    ledger.record_playback(
        PlaybackReceipt(
            chunk_id=chunk.chunk_id,
            requested_samples=chunk.frame.samples_per_channel,
            played_samples=0,
            playback_started_ns=13,
            playback_stopped_ns=13,
            cancelled=True,
        )
    )

    entry = ledger.entries[0]
    assert entry.state is LedgerState.CANCELLED
    assert entry.played_samples == 0
    assert ledger.delivered_text(response_id=chunk.response_id) == ""
    assert ledger.unheard_text(response_id=chunk.response_id) == "Noch nicht hörbar"
