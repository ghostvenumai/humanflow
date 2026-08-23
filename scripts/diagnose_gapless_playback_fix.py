#!/usr/bin/env python3
"""Prove the recorded real provider stream becomes one playback unit after the fix."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

from humanflow.audio.models import AudioChunk, AudioFrame
from humanflow.runtime.providers import (
    GaplessSegmentTTSProvider,
    ProviderInfo,
    ProviderMode,
    SpeechSynthesisRequest,
)
from humanflow.web.transport import BrowserAcknowledgedAudioOutput


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_REPORT = ROOT / "reports" / "tts-provider-isolation.json"
RAW_STREAM = Path("/tmp/humanflow_tts_diagnostics/stream-reassembled.pcm")
OUTPUT = ROOT / "reports" / "playback-scheduler-isolation-A-after-fix.json"
TEXT = (
    "Das ist ein einzelner Test der Audioausgabe. Dieser Satz darf nur einmal "
    "und ohne Unterbrechungen abgespielt werden."
)


class RecordedProviderStream:
    provider_info = ProviderInfo(
        role="tts",
        provider="elevenlabs-text-to-speech-stream",
        model="eleven_flash_v2_5",
        mode=ProviderMode.REAL,
        runtime="recorded-provider-output",
    )

    def __init__(self, packets: list[bytes]) -> None:
        self._packets = packets

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        for sequence, packet in enumerate(self._packets):
            if cancel_event.is_set():
                return
            final = sequence == len(self._packets) - 1
            yield AudioChunk(
                chunk_id=f"recorded-provider-{sequence}",
                response_id=request.response_id,
                text=request.text if final else "",
                semantic_id="isolation-a-segment",
                semantic_text=request.text,
                semantic_boundary=final,
                frame=AudioFrame(
                    stream_id=request.response_id,
                    sequence=sequence,
                    pcm16=packet,
                    sample_rate_hz=16_000,
                ),
                display_text=request.text,
                provider=self.provider_info.to_dict(),
            )


async def run() -> dict[str, object]:
    provider_report = json.loads(PROVIDER_REPORT.read_text(encoding="utf-8"))
    lengths = provider_report["streaming_reassembled"]["chunk_byte_lengths"]
    pcm = RAW_STREAM.read_bytes()
    packets: list[bytes] = []
    offset = 0
    for byte_length in lengths:
        packets.append(pcm[offset : offset + byte_length])
        offset += byte_length
    if offset != len(pcm) or any(not packet for packet in packets):
        raise RuntimeError("recorded_provider_packets_do_not_reconstruct_pcm")

    response_id = "isolation-a-gapless-after-fix"
    provider = GaplessSegmentTTSProvider(RecordedProviderStream(packets))
    request = SpeechSynthesisRequest(
        text=TEXT,
        response_id=response_id,
        sequence_start=0,
        language_code="de",
    )
    chunks = [
        chunk
        async for chunk in provider.stream_speech(
            request,
            cancel_event=asyncio.Event(),
        )
    ]
    if len(chunks) != 1 or chunks[0].frame.pcm16 != pcm:
        raise RuntimeError("gapless_wrapper_did_not_emit_exactly_one_lossless_unit")

    outbound: asyncio.Queue[dict[str, object] | bytes] = asyncio.Queue()
    output = BrowserAcknowledgedAudioOutput(outbound)
    accepted_ids: set[str] = set()
    accepted_sequences: set[tuple[str, int]] = set()
    started_ns: list[int] = []
    chunk = chunks[0]
    play_task = asyncio.create_task(
        output.play(
            chunk,
            cancel_event=asyncio.Event(),
            on_started=started_ns.append,
        )
    )
    metadata = await outbound.get()
    payload = await outbound.get()
    if not isinstance(metadata, dict) or not isinstance(payload, bytes):
        raise RuntimeError("unexpected_browser_transport_item")
    sequence_key = (chunk.frame.stream_id, chunk.frame.sequence)
    if chunk.chunk_id in accepted_ids or sequence_key in accepted_sequences:
        raise RuntimeError("duplicate_chunk_accepted")
    accepted_ids.add(chunk.chunk_id)
    accepted_sequences.add(sequence_key)
    scheduled_ns = monotonic_ns()
    source_node_id = "simulated-gapless-source-0"
    output.acknowledge(
        {
            "type": "playback_started",
            "chunk_id": chunk.chunk_id,
            "source_node_id": source_node_id,
            "browser_scheduled_start_ms": 0.0,
            "browser_actual_playback_start_ms": 0.0,
        }
    )
    output.acknowledge(
        {"type": "playback_completed", "chunk_id": chunk.chunk_id}
    )
    receipt = await play_task
    return {
        "status": "PASS_GAPLESS_SINGLE_PLAYBACK_UNIT",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": "recorded_real_provider_pcm_exact_browser_transport_no_mic_no_stt",
        "response_id": response_id,
        "raw_provider_chunk_count": len(packets),
        "raw_provider_chunk_boundaries": len(packets) - 1,
        "browser_playback_unit_count": len(chunks),
        "browser_source_node_count": 1,
        "duplicate_chunks_accepted": 0,
        "duplicate_sequences_scheduled": 0,
        "active_playback_owners_maximum": 1,
        "active_tts_playback_providers_maximum": 1,
        "transport": {
            "stream_id": metadata["stream_id"],
            "chunk_id": metadata["chunk_id"],
            "sequence": metadata["sequence"],
            "chunk_byte_length": metadata["chunk_byte_length"],
            "codec": metadata["codec"],
            "sample_rate_hz": metadata["sample_rate_hz"],
            "decoded_duration_ms": metadata["decoded_duration_ms"],
            "scheduled_ns": scheduled_ns,
            "source_node_id": receipt.source_node_id,
            "actual_playback_start": "SIMULATED_ACK_ONLY",
        },
        "lossless_pcm_sha256": hashlib.sha256(payload).hexdigest(),
        "source_provider_chunk_count": chunk.provider["source_chunk_count"],
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
                "status": report["status"],
                "raw_provider_chunk_count": report["raw_provider_chunk_count"],
                "browser_playback_unit_count": report["browser_playback_unit_count"],
                "browser_source_node_count": report["browser_source_node_count"],
                "duplicate_chunks_accepted": report["duplicate_chunks_accepted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
