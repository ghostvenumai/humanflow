#!/usr/bin/env python3
"""One tiny, budget-bounded ElevenLabs smoke test with secret-free evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

from humanflow.runtime.elevenlabs_provider import (
    DEFAULT_ELEVENLABS_MODEL,
    ElevenLabsStreamingTTSProvider,
    SynthesisBudget,
)
from humanflow.runtime.providers import SpeechSynthesisRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "live-tts-smoke.json"
SMOKE_TEXT = "Ich bin HumanFlow, ein KI-Assistent."


async def run() -> dict[str, object]:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
    if not api_key.strip() or not voice_id.strip():
        raise RuntimeError("ElevenLabs runtime configuration is missing")
    budget = SynthesisBudget(
        maximum_audio_seconds=60.0,
        maximum_input_characters=200,
    )
    provider = ElevenLabsStreamingTTSProvider(
        api_key=api_key,
        voice_id=voice_id,
        model=DEFAULT_ELEVENLABS_MODEL,
        budget=budget,
    )
    started_ns = monotonic_ns()
    chunks = [
        chunk
        async for chunk in provider.stream_speech(
            SpeechSynthesisRequest(
                text=SMOKE_TEXT,
                response_id="live-tts-smoke",
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
    completed_ns = monotonic_ns()
    pcm = b"".join(chunk.frame.pcm16 for chunk in chunks)
    if not chunks or not pcm or chunks[-1].text != SMOKE_TEXT:
        raise RuntimeError("live TTS smoke returned incomplete streaming audio")
    metrics = provider.last_request_metrics
    return {
        "status": "PASS",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provider": provider.provider_info.to_dict(),
        "request": {
            "text_characters": len(SMOKE_TEXT),
            "language_code": "de",
            "output_format": "pcm_16000",
        },
        "stream": {
            "audio_chunks": len(chunks),
            "audio_bytes": len(pcm),
            "audio_seconds": round(len(pcm) / 2 / provider.sample_rate_hz, 3),
            "wall_duration_ms": round((completed_ns - started_ns) / 1_000_000, 3),
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "request_metrics": metrics.to_dict() if metrics is not None else None,
        },
        "budget": budget.to_dict(),
        "manual_voice_quality_attested": False,
    }


def main() -> None:
    try:
        report = asyncio.run(run())
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        provider_code = getattr(error, "provider_code", None)
        failure_report = {
            "status": "FAIL",
            "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "provider": {
                "role": "tts",
                "provider": "elevenlabs-text-to-speech-stream",
                "model": DEFAULT_ELEVENLABS_MODEL,
                "mode": "REAL",
                "runtime": "server",
            },
            "request": {
                "text_characters": len(SMOKE_TEXT),
                "language_code": "de",
                "output_format": "pcm_16000",
            },
            "failure": {
                "exception_type": type(error).__name__,
                "provider_http_status": status_code,
                "provider_error_code": provider_code,
            },
            "audio_generated_seconds": 0.0,
            "cost_usd": "NOT_REPORTED",
            "manual_voice_quality_attested": False,
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(failure_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "exception_type": type(error).__name__,
                    "provider_http_status": status_code,
                    "provider_error_code": provider_code,
                    "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
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
    sys.stdout.flush()
    print(
        json.dumps(
            {
                "status": report["status"],
                "provider": report["provider"],
                "stream": report["stream"],
                "budget": report["budget"],
                "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
