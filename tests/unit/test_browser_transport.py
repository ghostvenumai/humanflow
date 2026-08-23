from __future__ import annotations

import asyncio

from humanflow.audio.models import AudioChunk, AudioFrame
from humanflow.web.app import PROJECT_ROOT, STATIC_DIR, build_evidence_summary, create_app
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
        assert isinstance(pcm, bytes) and len(pcm) == 3_200
        assert output.acknowledge({"type": "playback_started", "chunk_id": "chunk-browser"})
        cancel.set()
        cancel_message = await outbound.get()
        assert cancel_message == {"type": "cancel_audio", "chunk_id": "chunk-browser"}
        assert output.acknowledge(
            {
                "type": "playback_stopped",
                "chunk_id": "chunk-browser",
                "played_samples": 640,
            }
        )
        receipt = await task

        assert len(starts) == 1
        assert receipt.cancelled
        assert receipt.played_samples == 640
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
        "/ws",
        "/static",
    }.issubset(paths)
    assert (STATIC_DIR / "index.html").is_file()
    assert (STATIC_DIR / "app.js").is_file()
    assert (STATIC_DIR / "dashboard.html").is_file()
    source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "playPcm(buffer" not in source
    assert "provider-reasoning" in markup
    assert "kein stiller Mock-Fallback" in markup


def test_dashboard_evidence_summary_links_real_reports() -> None:
    summary = build_evidence_summary(PROJECT_ROOT)
    assert summary["torture"] == {"failed": 0, "passed": 20, "total": 20}
    assert summary["quality_loop"]["decision"] == "KEEP"
    assert summary["tournament"]["external_agent_calls_made"] == 0
    assert all(len(artifact["sha256"]) == 64 for artifact in summary["artifacts"].values())
