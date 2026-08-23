from __future__ import annotations

import asyncio
import json
from pathlib import Path

from humanflow.audio.models import AudioChunk, AudioFrame
from humanflow.web.app import (
    PROJECT_ROOT,
    STATIC_DIR,
    _acknowledge_transport_message,
    build_evidence_summary,
    build_voice_quality_summary,
    create_app,
    validate_voice_quality_assessment,
)
from humanflow.web.transport import BrowserAcknowledgedAudioOutput


def _chunk() -> AudioChunk:
    return AudioChunk(
        chunk_id="chunk-browser",
        response_id="response-browser",
        text="Guten Tag",
        frame=AudioFrame(
            stream_id="response-browser",
            sequence=0,
            pcm16=b"\x00\x00" * 1_600,
        ),
    )


def test_browser_acknowledgement_defines_partial_playback_receipt() -> None:
    async def scenario() -> None:
        outbound: asyncio.Queue[dict[str, object] | bytes] = asyncio.Queue()
        output = BrowserAcknowledgedAudioOutput(outbound)
        cancel = asyncio.Event()
        starts: list[int] = []
        task = asyncio.create_task(
            output.play(_chunk(), cancel_event=cancel, on_started=starts.append)
        )

        metadata = await outbound.get()
        pcm = await outbound.get()
        assert isinstance(metadata, dict) and metadata["type"] == "audio_chunk"
        assert metadata["playback_mode"] == "pcm"
        assert metadata["semantic_boundary"] is True
        assert isinstance(pcm, bytes) and len(pcm) == 3_200
        assert output.acknowledge(
            {
                "type": "playback_started",
                "chunk_id": "chunk-browser",
                "browser_audio_context_base_latency_ms": 5.2,
                "browser_audio_context_output_latency_ms": 11.4,
            }
        )
        cancel.set()
        cancel_message = await outbound.get()
        assert cancel_message == {"type": "cancel_audio", "chunk_id": "chunk-browser"}
        assert output.acknowledge(
            {
                "type": "playback_stopped",
                "chunk_id": "chunk-browser",
                "played_samples": 640,
                "player_stop_callback_latency_ms": 2.7,
            }
        )
        receipt = await task

        assert len(starts) == 1
        assert receipt.cancelled
        assert receipt.played_samples == 640
        assert receipt.sink_base_latency_ms == 5.2
        assert receipt.sink_output_latency_ms == 11.4
        assert receipt.player_stop_callback_latency_ms == 2.7
        assert receipt.playback_stopped_ns >= receipt.playback_started_ns

    asyncio.run(scenario())


def test_browser_completed_ack_is_full_delivery() -> None:
    async def scenario() -> None:
        outbound: asyncio.Queue[dict[str, object] | bytes] = asyncio.Queue()
        output = BrowserAcknowledgedAudioOutput(outbound)
        task = asyncio.create_task(
            output.play(_chunk(), cancel_event=asyncio.Event(), on_started=lambda _: None)
        )
        await outbound.get()
        await outbound.get()
        output.acknowledge({"type": "playback_started", "chunk_id": "chunk-browser"})
        output.acknowledge({"type": "playback_completed", "chunk_id": "chunk-browser"})
        receipt = await task
        assert not receipt.cancelled
        assert receipt.played_samples == receipt.requested_samples == 1_600

    asyncio.run(scenario())


def test_transport_ack_handler_releases_playback_outside_control_worker() -> None:
    async def scenario() -> None:
        outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
        output = BrowserAcknowledgedAudioOutput(outbound)
        task = asyncio.create_task(
            output.play(_chunk(), cancel_event=asyncio.Event(), on_started=lambda _: None)
        )
        await outbound.get()
        await outbound.get()

        assert _acknowledge_transport_message(
            '{"type":"playback_started","chunk_id":"chunk-browser"}',
            output,
            outbound,
        )
        assert _acknowledge_transport_message(
            '{"type":"playback_completed","chunk_id":"chunk-browser"}',
            output,
            outbound,
        )
        receipt = await asyncio.wait_for(task, timeout=0.2)
        assert not receipt.cancelled

    asyncio.run(scenario())


def test_demo_app_exposes_static_assets_and_websocket_route() -> None:
    application = create_app()
    paths = {route.path for route in application.routes}
    assert {
        "/",
        "/dashboard",
        "/health",
        "/api/scorecard",
        "/api/evidence",
        "/api/timeline",
        "/api/voice-quality",
        "/ws",
        "/static",
    }.issubset(paths)
    assert (STATIC_DIR / "index.html").is_file()
    assert (STATIC_DIR / "app.js").is_file()
    assert (STATIC_DIR / "dashboard.html").is_file()
    source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "function playPcm(buffer, meta)" in source
    assert 'meta.playback_mode === "pcm"' in source
    assert 'meta.playback_mode !== "browser_speech"' in source
    assert "provider_endpointed: result.isFinal" in source
    assert "silence_duration_ms: Math.round(silenceDuration)" in source
    assert "utterance_duration_ms: Math.round(utteranceDuration)" in source
    assert "provider-reasoning" in markup
    assert "provider-tts-fallback" in markup
    assert "kein stiller Mock-Fallback" in markup


def test_voice_quality_summary_only_uses_actual_human_submissions(tmp_path: Path) -> None:
    path = tmp_path / "voice.jsonl"
    assert build_voice_quality_summary(path)["sample_count"] == 0

    payload = {
        "candidate": "candidate-a",
        "ratings": {
            "naturalness": 4,
            "prosody": 4,
            "pacing": 3,
            "voice_pleasantness": 5,
            "turn_timing": 4,
            "interruption_feel": 3,
            "non_mechanical_impression": 4,
            "overall_conversational_realism": 4,
        },
        "notes": "Von einem Menschen im Test eingetragen.",
    }
    assessment = validate_voice_quality_assessment(payload)
    path.write_text(json.dumps(assessment) + "\n", encoding="utf-8")

    summary = build_voice_quality_summary(path)
    assert summary["sample_count"] == 1
    assert summary["averages"]["naturalness"] == 4.0
    assert summary["agent_attestation"] is False


def test_dashboard_evidence_summary_links_real_reports() -> None:
    summary = build_evidence_summary(PROJECT_ROOT)
    assert summary["torture"] == {"failed": 0, "passed": 20, "total": 20}
    assert summary["quality_loop"]["decision"] == "KEEP"
    assert summary["tournament"]["external_agent_calls_made"] == 0
    assert all(len(artifact["sha256"]) == 64 for artifact in summary["artifacts"].values())
