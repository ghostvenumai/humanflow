#!/usr/bin/env python3
"""Measure the pre-fix stop-and-wait browser-output ordering without mic/STT."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

from humanflow.audio.models import AudioChunk, AudioFrame
from humanflow.web.transport import BrowserAcknowledgedAudioOutput


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_REPORT = ROOT / "reports" / "tts-provider-isolation.json"
RAW_STREAM = Path("/tmp/humanflow_tts_diagnostics/stream-reassembled.pcm")
OUTPUT = ROOT / "reports" / "playback-scheduler-isolation-A.json"
TEXT = (
    "Das ist ein einzelner Test der Audioausgabe. Dieser Satz darf nur einmal "
    "und ohne Unterbrechungen abgespielt werden."
)


async def run() -> dict[str, object]:
    provider_report = json.loads(PROVIDER_REPORT.read_text(encoding="utf-8"))
    lengths = provider_report["streaming_reassembled"]["chunk_byte_lengths"]
    pcm = RAW_STREAM.read_bytes()
    offset = 0
    chunks: list[AudioChunk] = []
    response_id = "isolation-a-no-input"
    for sequence, byte_length in enumerate(lengths):
        chunk_pcm = pcm[offset : offset + byte_length]
        offset += byte_length
        chunks.append(
            AudioChunk(
                chunk_id=f"isolation-a-{sequence}",
                response_id=response_id,
                text=TEXT if sequence == len(lengths) - 1 else "",
                semantic_id="isolation-a-segment",
                semantic_text=TEXT,
                semantic_boundary=sequence == len(lengths) - 1,
                frame=AudioFrame(
                    stream_id=response_id,
                    sequence=sequence,
                    pcm16=chunk_pcm,
                    sample_rate_hz=16_000,
                ),
            )
        )
    if offset != len(pcm):
        raise RuntimeError("provider chunk lengths do not reconstruct raw PCM")

    outbound: asyncio.Queue[dict[str, object] | bytes] = asyncio.Queue()
    output = BrowserAcknowledgedAudioOutput(outbound)
    cancel = asyncio.Event()
    accepted_ids: set[str] = set()
    accepted_sequences: set[int] = set()
    schedule: list[dict[str, object]] = []
    previous_completed_ack_ns: int | None = None

    for chunk in chunks:
        play_task = asyncio.create_task(
            output.play(chunk, cancel_event=cancel, on_started=lambda _: None)
        )
        metadata = await outbound.get()
        payload = await outbound.get()
        received_ns = monotonic_ns()
        if not isinstance(metadata, dict) or not isinstance(payload, bytes):
            raise RuntimeError("unexpected browser transport item")
        if chunk.chunk_id in accepted_ids or chunk.frame.sequence in accepted_sequences:
            raise RuntimeError("duplicate chunk accepted by diagnostic")
        accepted_ids.add(chunk.chunk_id)
        accepted_sequences.add(chunk.frame.sequence)
        output.acknowledge(
            {"type": "playback_started", "chunk_id": chunk.chunk_id}
        )
        decoded_duration_ms = len(payload) / 2 / chunk.frame.sample_rate_hz * 1_000
        scheduled_start_ns = monotonic_ns()
        await asyncio.sleep(decoded_duration_ms / 1_000)
        completed_ack_ns = monotonic_ns()
        output.acknowledge(
            {"type": "playback_completed", "chunk_id": chunk.chunk_id}
        )
        await play_task
        schedule.append(
            {
                "response_id": response_id,
                "stream_id": chunk.frame.stream_id,
                "chunk_id": chunk.chunk_id,
                "chunk_sequence": chunk.frame.sequence,
                "chunk_byte_length": len(payload),
                "codec": "pcm_s16le",
                "sample_rate_hz": chunk.frame.sample_rate_hz,
                "decoded_duration_ms": round(decoded_duration_ms, 3),
                "metadata_received_ns": received_ns,
                "scheduled_start_ns": scheduled_start_ns,
                "source_node_id": f"simulated-source-{chunk.frame.sequence}",
                "actual_playback_start": "SIMULATED_ACK_ONLY",
                "next_chunk_available_after_previous_completion_ms": (
                    None
                    if previous_completed_ack_ns is None
                    else round(
                        (received_ns - previous_completed_ack_ns) / 1_000_000,
                        6,
                    )
                ),
            }
        )
        previous_completed_ack_ns = completed_ack_ns

    gaps = [
        item["next_chunk_available_after_previous_completion_ms"]
        for item in schedule
        if item["next_chunk_available_after_previous_completion_ms"] is not None
    ]
    return {
        "status": "FAIL_STOP_AND_WAIT_GAP_AT_EVERY_RAW_CHUNK_BOUNDARY",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "exact_server_browser_output_order_with_simulated_browser_ack_no_mic_no_stt",
        "response_id": response_id,
        "accepted_chunk_count": len(accepted_ids),
        "accepted_sequence_count": len(accepted_sequences),
        "duplicate_chunks_accepted": 0,
        "active_playback_owners_maximum": 1,
        "raw_chunk_boundaries": len(chunks) - 1,
        "boundaries_where_next_chunk_was_unavailable_until_previous_end": len(gaps),
        "post_completion_dispatch_gap_ms": {
            "sample_count": len(gaps),
            "minimum": round(min(gaps), 6),
            "maximum": round(max(gaps), 6),
            "mean": round(sum(gaps) / len(gaps), 6),
        },
        "interpretation": (
            "The next PCM chunk is not sent to the browser until the prior source "
            "has ended and its completion acknowledgement has returned. The real "
            "browser adds WebSocket, decoding, event-loop and AudioContext delay."
        ),
        "schedule": schedule,
        "human_audible_assessment": None,
        "agent_attestation": False,
    }


def main() -> None:
    report = asyncio.run(run())
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "accepted_chunk_count",
                    "duplicate_chunks_accepted",
                    "raw_chunk_boundaries",
                    "boundaries_where_next_chunk_was_unavailable_until_previous_end",
                    "post_completion_dispatch_gap_ms",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
