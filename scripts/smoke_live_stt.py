#!/usr/bin/env python3
"""Run a small real Scribe PCM smoke without claiming microphone quality."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns

from humanflow.audio.models import AudioFrame
from humanflow.runtime.elevenlabs_provider import (
    DEFAULT_ELEVENLABS_MODEL,
    ElevenLabsStreamingTTSProvider,
    SynthesisBudget,
)
from humanflow.runtime.elevenlabs_stt_provider import (
    DEFAULT_ELEVENLABS_STT_MODEL,
    ElevenLabsRealtimeSTTProvider,
)
from humanflow.runtime.providers import SpeechSynthesisRequest
from humanflow.runtime.transcript_events import normalize_transcript


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "live-stt-smoke.json"
SMOKE_TEXT = "Was ist fünfundzwanzig mal siebzehn?"
FRAME_SAMPLES = 1_600
FRAME_BYTES = FRAME_SAMPLES * 2


async def run() -> dict[str, object]:
    tts_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    stt_api_key = os.environ.get("ELEVENLABS_STT_API_KEY", "") or tts_api_key
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
    if not stt_api_key.strip():
        raise RuntimeError("BLOCKER_REAL_STREAMING_STT_PROVIDER_REQUIRED")
    if not tts_api_key.strip() or not voice_id.strip():
        raise RuntimeError("ElevenLabs voice configuration is missing for STT fixture")

    stt = ElevenLabsRealtimeSTTProvider(
        api_key=stt_api_key,
        model=os.environ.get(
            "HUMANFLOW_STT_MODEL", DEFAULT_ELEVENLABS_STT_MODEL
        ),
        language_code="de",
    )
    stt.bind_audio_source(
        audio_capture_id="live-stt-smoke-capture",
        stream_id="live-stt-smoke-pcm",
    )
    try:
        await stt.start()
    except Exception:
        await stt.close()
        raise

    synthesis_budget = SynthesisBudget(
        maximum_audio_seconds=15.0,
        maximum_input_characters=80,
    )
    tts = ElevenLabsStreamingTTSProvider(
        api_key=tts_api_key,
        voice_id=voice_id,
        model=DEFAULT_ELEVENLABS_MODEL,
        budget=synthesis_budget,
    )
    try:
        tts_chunks = [
            chunk
            async for chunk in tts.stream_speech(
                SpeechSynthesisRequest(
                    text=SMOKE_TEXT,
                    response_id="live-stt-fixture",
                    sequence_start=0,
                    language_code="de",
                ),
                cancel_event=asyncio.Event(),
            )
        ]
    except Exception:
        await stt.close()
        raise
    speech_pcm = b"".join(chunk.frame.pcm16 for chunk in tts_chunks)
    if not speech_pcm:
        await stt.close()
        raise RuntimeError("live STT fixture synthesis returned no PCM")

    started_ns = monotonic_ns()
    first_partial_ns: int | None = None
    committed_ns: int | None = None
    partials: list[str] = []
    finals: list[str] = []
    sequence = 0

    async def submit(pcm16: bytes) -> None:
        nonlocal sequence, first_partial_ns, committed_ns
        updates = await stt.ingest(
            AudioFrame(
                stream_id="live-stt-smoke-pcm",
                sequence=sequence,
                pcm16=pcm16,
                sample_rate_hz=16_000,
                captured_ns=monotonic_ns(),
            )
        )
        sequence += 1
        observed_ns = monotonic_ns()
        for update in updates:
            if update.is_final:
                finals.append(update.text)
                committed_ns = committed_ns or observed_ns
            else:
                partials.append(update.text)
                first_partial_ns = first_partial_ns or observed_ns

    try:
        for offset in range(0, len(speech_pcm), FRAME_BYTES):
            frame = speech_pcm[offset : offset + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame += b"\x00" * (FRAME_BYTES - len(frame))
            await submit(frame)
            await asyncio.sleep(FRAME_SAMPLES / 16_000)

        silence = b"\x00" * FRAME_BYTES
        for _ in range(50):
            await submit(silence)
            if finals:
                break
            await asyncio.sleep(FRAME_SAMPLES / 16_000)
    finally:
        await stt.close()

    if not finals:
        raise RuntimeError("Scribe returned no committed transcript")
    transcript = " ".join(finals)
    normalized = normalize_transcript(transcript)
    semantic_preserved = (
        ("fünfundzwanzig" in normalized or "25" in normalized)
        and ("siebzehn" in normalized or "17" in normalized)
    )
    if not semantic_preserved:
        raise RuntimeError("Scribe transcript did not preserve both numbers")

    completed_ns = monotonic_ns()
    return {
        "status": "PASS",
        "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": (
            "real Scribe websocket and exact PCM provider path; synthetic "
            "ElevenLabs speech fixture, not a browser microphone assessment"
        ),
        "source_binding": {
            "audio_capture_id": "live-stt-smoke-capture",
            "stream_id": "live-stt-smoke-pcm",
            "format": "pcm_s16le_16000_mono",
            "recognition_input_binding": "EXACT_GETUSERMEDIA_PCM16_CONTRACT",
            "browser_speech_recognition_production_status": "OFF",
        },
        "provider": stt.provider_info.to_dict(),
        "fixture": {
            "text": SMOKE_TEXT,
            "tts_provider": tts.provider_info.to_dict(),
            "audio_seconds": round(len(speech_pcm) / 2 / 16_000, 3),
            "tts_budget": synthesis_budget.to_dict(),
        },
        "observed": {
            "partial_events_delivered": len(partials),
            "committed_events_delivered": len(finals),
            "committed_transcript": transcript,
            "semantic_numbers_preserved": semantic_preserved,
            "first_partial_ms": (
                None
                if first_partial_ns is None
                else round((first_partial_ns - started_ns) / 1_000_000, 3)
            ),
            "committed_transcript_ms": (
                None
                if committed_ns is None
                else round((committed_ns - started_ns) / 1_000_000, 3)
            ),
            "wall_duration_ms": round((completed_ns - started_ns) / 1_000_000, 3),
            "pcm_frames_sent": sequence,
        },
        "provider_cost_usd": "NOT_REPORTED_BY_PROVIDER_RESPONSE",
        "manual_microphone_quality_attested": False,
    }


def main() -> None:
    try:
        report = asyncio.run(run())
    except Exception as error:
        failure = {
            "status": "FAIL",
            "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "provider": {
                "role": "stt",
                "provider": "elevenlabs-scribe-realtime",
                "model": os.environ.get(
                    "HUMANFLOW_STT_MODEL", DEFAULT_ELEVENLABS_STT_MODEL
                ),
                "mode": "REAL",
                "runtime": "server",
            },
            "failure": {
                "exception_type": type(error).__name__,
                "provider_error_code": getattr(error, "provider_code", None),
            },
            "browser_speech_recognition_production_status": "OFF",
            "manual_microphone_quality_attested": False,
        }
        REPORT.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "exception_type": type(error).__name__,
                    "provider_error_code": getattr(error, "provider_code", None),
                    "report": str(REPORT.relative_to(ROOT)),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None

    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "provider": report["provider"],
                "observed": report["observed"],
                "report": str(REPORT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
