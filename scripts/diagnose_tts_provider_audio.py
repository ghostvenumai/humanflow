#!/usr/bin/env python3
"""Compare one-shot and streaming ElevenLabs PCM outside browser scheduling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import wave
from array import array
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from typing import Any

import httpx

from humanflow.runtime.elevenlabs_provider import (
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    ELEVENLABS_FLASH_USD_PER_1K_CHARACTERS,
    ElevenLabsProviderError,
    ElevenLabsStreamingTTSProvider,
    SynthesisBudget,
    _safe_provider_error_code,
)
from humanflow.runtime.providers import SpeechSynthesisRequest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("/tmp/humanflow_tts_diagnostics")
REPORT_PATH = ROOT / "reports" / "tts-provider-isolation.json"
TEXT = (
    "Das ist ein einzelner Test der Audioausgabe. Dieser Satz darf nur einmal "
    "und ohne Unterbrechungen abgespielt werden."
)
SAMPLE_RATE_HZ = 16_000


def request_body() -> dict[str, Any]:
    return {
        "text": TEXT,
        "model_id": DEFAULT_ELEVENLABS_MODEL,
        "language_code": "de",
        "voice_settings": {
            "stability": 0.48,
            "similarity_boost": 0.78,
            "style": 0.08,
            "use_speaker_boost": False,
            "speed": 1.0,
        },
        "apply_text_normalization": "auto",
    }


def write_audio_files(stem: str, pcm16: bytes) -> tuple[Path, Path]:
    raw_path = OUTPUT_DIR / f"{stem}.pcm"
    wav_path = OUTPUT_DIR / f"{stem}.wav"
    raw_path.write_bytes(pcm16)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE_HZ)
        output.writeframes(pcm16)
    return raw_path, wav_path


def boundary_jumps(pcm16: bytes, boundaries: list[int]) -> list[int]:
    samples = array("h")
    samples.frombytes(pcm16)
    jumps: list[int] = []
    for byte_offset in boundaries:
        sample_index = byte_offset // 2
        if 0 < sample_index < len(samples):
            jumps.append(abs(samples[sample_index] - samples[sample_index - 1]))
    return jumps


async def run() -> dict[str, Any]:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if not api_key or not voice_id:
        raise RuntimeError("ElevenLabs runtime configuration is missing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    params = {"output_format": DEFAULT_OUTPUT_FORMAT}

    one_shot_started_ns = monotonic_ns()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            endpoint,
            headers=headers,
            params=params,
            json=request_body(),
        )
    one_shot_completed_ns = monotonic_ns()
    if response.status_code >= 400:
        raise ElevenLabsProviderError(
            response.status_code,
            _safe_provider_error_code(response.content),
        )
    one_shot_pcm = response.content
    if not one_shot_pcm or len(one_shot_pcm) % 2:
        raise RuntimeError("one_shot_provider_returned_invalid_pcm")
    one_shot_cost = response.headers.get("character-cost")

    budget = SynthesisBudget(
        maximum_audio_seconds=60.0,
        maximum_input_characters=500,
    )
    provider = ElevenLabsStreamingTTSProvider(
        api_key=api_key,
        voice_id=voice_id,
        model=DEFAULT_ELEVENLABS_MODEL,
        budget=budget,
    )
    stream_started_ns = monotonic_ns()
    chunks = [
        chunk
        async for chunk in provider.stream_speech(
            SpeechSynthesisRequest(
                text=TEXT,
                response_id="provider-isolation-stream",
                sequence_start=0,
                language_code="de",
                speaking_rate=1.0,
                stability=0.48,
                similarity_boost=0.78,
                style=0.08,
                use_speaker_boost=False,
                intent="information",
            ),
            cancel_event=asyncio.Event(),
        )
    ]
    stream_completed_ns = monotonic_ns()
    streaming_pcm = b"".join(chunk.frame.pcm16 for chunk in chunks)
    boundaries: list[int] = []
    running_bytes = 0
    for chunk in chunks[:-1]:
        running_bytes += len(chunk.frame.pcm16)
        boundaries.append(running_bytes)

    one_raw, one_wav = write_audio_files("one-shot", one_shot_pcm)
    stream_raw, stream_wav = write_audio_files("stream-reassembled", streaming_pcm)
    chunk_hashes = [hashlib.sha256(chunk.frame.pcm16).hexdigest() for chunk in chunks]
    reported_stream_cost = budget.reported_billable_characters
    reported_one_shot_cost = (
        int(one_shot_cost)
        if isinstance(one_shot_cost, str) and one_shot_cost.isdigit()
        else None
    )
    total_billable = reported_stream_cost + (reported_one_shot_cost or 0)
    return {
        "status": "PASS_OBJECTIVE_DATA_CAPTURED_HUMAN_LISTENING_PENDING",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "text_characters": len(TEXT),
        "provider": provider.provider_info.to_dict(),
        "format": {
            "codec": "pcm_s16le",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
        },
        "one_shot": {
            "bytes": len(one_shot_pcm),
            "decoded_duration_seconds": round(
                len(one_shot_pcm) / 2 / SAMPLE_RATE_HZ, 6
            ),
            "wall_duration_ms": round(
                (one_shot_completed_ns - one_shot_started_ns) / 1_000_000, 3
            ),
            "pcm_sha256": hashlib.sha256(one_shot_pcm).hexdigest(),
            "reported_billable_characters": reported_one_shot_cost,
            "raw_path": str(one_raw),
            "wav_path": str(one_wav),
        },
        "streaming_reassembled": {
            "bytes": len(streaming_pcm),
            "decoded_duration_seconds": round(
                len(streaming_pcm) / 2 / SAMPLE_RATE_HZ, 6
            ),
            "wall_duration_ms": round(
                (stream_completed_ns - stream_started_ns) / 1_000_000, 3
            ),
            "pcm_sha256": hashlib.sha256(streaming_pcm).hexdigest(),
            "chunk_count": len(chunks),
            "chunk_sequences": [chunk.frame.sequence for chunk in chunks],
            "chunk_byte_lengths": [len(chunk.frame.pcm16) for chunk in chunks],
            "unique_chunk_hashes": len(set(chunk_hashes)),
            "duplicate_chunk_hashes": len(chunk_hashes) - len(set(chunk_hashes)),
            "boundary_sample_jumps": boundary_jumps(streaming_pcm, boundaries),
            "reported_billable_characters": reported_stream_cost,
            "raw_path": str(stream_raw),
            "wav_path": str(stream_wav),
        },
        "budget": {
            "reported_billable_characters_total": total_billable,
            "estimated_base_cost_usd": round(
                total_billable / 1_000 * ELEVENLABS_FLASH_USD_PER_1K_CHARACTERS,
                6,
            ),
            "audio_seconds_generated_total": round(
                (len(one_shot_pcm) + len(streaming_pcm)) / 2 / SAMPLE_RATE_HZ,
                6,
            ),
        },
        "human_assessment": {
            "one_shot_clean": None,
            "streaming_file_clean": None,
            "agent_attestation": False,
        },
    }


def main() -> None:
    try:
        report = asyncio.run(run())
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "exception_type": type(error).__name__,
                    "provider_http_status": getattr(error, "status_code", None),
                    "provider_error_code": getattr(error, "provider_code", None),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "format": report["format"],
                "one_shot": report["one_shot"],
                "streaming_reassembled": report["streaming_reassembled"],
                "budget": report["budget"],
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
