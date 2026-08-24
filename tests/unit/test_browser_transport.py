from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from humanflow.audio.models import AudioChunk, AudioFrame
from humanflow.web.app import (
    PcmAudioRouteState,
    BrowserTranscriptRouteState,
    BrowserSessionLease,
    PROJECT_ROOT,
    STATIC_DIR,
    _acknowledge_transport_message,
    _activate_pcm_stream_message,
    _handle_json,
    build_evidence_summary,
    build_voice_quality_summary,
    create_app,
    validate_voice_quality_assessment,
)
from humanflow.web.transport import BrowserAcknowledgedAudioOutput
from humanflow.runtime.elevenlabs_stt_provider import ElevenLabsRealtimeSTTProvider
from humanflow.telemetry.events import EventType
from humanflow.turns.models import TurnDecisionType


def _browser_provenance(
    result_id: str,
    *,
    recognition_session_id: str = "test-session",
    audio_capture_id: str = "test-capture",
) -> dict[str, object]:
    return {
        "transcript_id": f"{result_id}:final",
        "event_kind": "USER_TRANSCRIPT_FINAL",
        "source": "browser_stt",
        "origin": "BROWSER_SPEECH_RECOGNITION",
        "stream_id": f"browser-recognition:{recognition_session_id}",
        "browser_recognition_session_id": recognition_session_id,
        "audio_capture_id": audio_capture_id,
        "browser_timestamp_ms": 1234.5,
        "recognition_input_binding": "UNVERIFIED_INDEPENDENT_BROWSER_CAPTURE",
    }


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
        assert metadata["stream_id"] == "response-browser"
        assert metadata["sequence"] == 0
        assert metadata["chunk_byte_length"] == 3_200
        assert metadata["codec"] == "pcm_s16le"
        assert metadata["decoded_duration_ms"] == 100.0
        assert isinstance(pcm, bytes) and len(pcm) == 3_200
        assert output.acknowledge(
            {
                "type": "playback_started",
                "chunk_id": "chunk-browser",
                "browser_audio_context_base_latency_ms": 5.2,
                "browser_audio_context_output_latency_ms": 11.4,
                "source_node_id": "pcm-response-browser-0-1",
                "browser_scheduled_start_ms": 412.5,
                "browser_actual_playback_start_ms": 412.6,
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
        assert receipt.source_node_id == "pcm-response-browser-0-1"
        assert receipt.browser_scheduled_start_ms == 412.5
        assert receipt.browser_actual_playback_start_ms == 412.6
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
    assert "recognition_result_id: recognitionResultId" in source
    assert "receivedChunkIds.has(meta.chunk_id)" in source
    assert "scheduledChunkSequences.has(sequenceKey)" in source
    assert 'window.speechSynthesis.cancel();' in source
    assert 'active_tts_playback_providers_exceeded' in source
    assert 'event.event_type === "TTS_PROVIDER_DEACTIVATED"' in source
    assert "function activateTtsProvider(provider)" in source
    recognition_handler = source.split("recognition.onresult =", 1)[1].split(
        "recognition.onerror =", 1
    )[0]
    assert "showTranscript(" not in recognition_handler
    assert 'payload.type === "transcript_result"' in source
    assert '? "ElevenLabs Scribe" : "Diagnostic-Text"' in source
    assert "`USER / ${provider}`" in source
    assert "if (browserSttDiagnosticMode) startBrowserStt();" in source
    assert "browser_speech_recognition_production_status" in source
    assert "ASSISTANT / Claude" in source
    assert "SUPPRESSED / Self-Echo" in source
    assert "Provenienz-Debugging" in markup
    assert "debug-history-roles" in markup
    assert "provider-reasoning" in markup
    assert "provider-tts-fallback" in markup
    assert "kein stiller Mock-Fallback" in markup


def test_browser_session_lease_allows_only_one_playback_owner() -> None:
    async def scenario() -> None:
        lease = BrowserSessionLease()
        assert await lease.acquire("conversation-a")
        assert not await lease.acquire("conversation-b")
        await lease.release("conversation-b")
        assert not await lease.acquire("conversation-b")
        await lease.release("conversation-a")
        assert await lease.acquire("conversation-b")

    asyncio.run(scenario())


def test_pcm_stream_binding_pins_real_stt_to_exact_microphone_source() -> None:
    class FakeStateMachine:
        def __init__(self) -> None:
            self.events: list[tuple[EventType, dict[str, object]]] = []

        def record(self, event_type: EventType, **kwargs: object) -> None:
            self.events.append((event_type, kwargs))

    transcriber = ElevenLabsRealtimeSTTProvider(api_key="test-key-never-sent")
    route = PcmAudioRouteState()
    outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
    session = SimpleNamespace(state_machine=FakeStateMachine())

    assert _activate_pcm_stream_message(
        json.dumps(
            {
                "type": "pcm_stream_started",
                "audio_capture_id": "capture-authoritative",
                "microphone_stream_id": "track-authoritative",
            }
        ),
        transcriber=transcriber,
        route_state=route,
        session=session,  # type: ignore[arg-type]
        outbound=outbound,
    )

    assert route.active
    assert session.state_machine.events[0][0] is EventType.PROVIDER_STATUS
    payload = outbound.get_nowait()
    assert payload["accepted"] is True  # type: ignore[index]
    assert payload["browser_speech_recognition_production_status"] == "OFF"  # type: ignore[index]
    try:
        transcriber.bind_audio_source(
            audio_capture_id="another-capture",
            stream_id="another-track",
        )
    except RuntimeError as error:
        assert str(error) == "stt_audio_source_binding_is_immutable"
    else:
        raise AssertionError("PCM/STT binding unexpectedly changed")


def test_duplicate_browser_diagnostic_final_is_rejected_without_controller() -> None:
    class FakeStateMachine:
        def __init__(self) -> None:
            self.events: list[tuple[EventType, dict[str, object]]] = []

        def record(self, event_type: EventType, **kwargs: object) -> None:
            self.events.append((event_type, kwargs))

    class FakeSession:
        def __init__(self) -> None:
            self.state_machine = FakeStateMachine()
            self.submissions = 0

        async def accept_user_transcript(self, update: object) -> SimpleNamespace:
            del update
            self.submissions += 1
            return SimpleNamespace(
                decision=TurnDecisionType.COMPLETE,
                confidence=1.0,
                reason_codes=("test",),
            )

    async def scenario() -> None:
        session = FakeSession()
        outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
        output = BrowserAcknowledgedAudioOutput(outbound)
        seen: set[str] = set()
        payload = json.dumps(
            {
                "type": "transcript",
                "source": "browser_stt",
                "recognition_result_id": "browser-run-1-result-0",
                "provenance": _browser_provenance("browser-run-1-result-0"),
                "text": "Nur einmal verarbeiten.",
                "final": True,
                "signals": {
                    "speech_active": False,
                    "silence_duration_ms": 350,
                    "utterance_duration_ms": 900,
                    "semantic_complete": True,
                    "provider_endpointed": True,
                },
            }
        )

        await _handle_json(  # type: ignore[arg-type]
            payload,
            session,
            output,
            outbound,
            seen_final_transcript_ids=seen,
            input_only=True,
        )
        await _handle_json(  # type: ignore[arg-type]
            payload,
            session,
            output,
            outbound,
            seen_final_transcript_ids=seen,
            input_only=True,
        )

        assert session.submissions == 0
        assert [event[0] for event in session.state_machine.events] == [
            EventType.FINAL_TRANSCRIPT,
            EventType.DUPLICATE_TRANSCRIPT_REJECTED
        ]
        first = await outbound.get()
        assert first["type"] == "input_probe_transcript"  # type: ignore[index]
        duplicate = await outbound.get()
        assert duplicate["type"] == "transcript_result"  # type: ignore[index]
        assert duplicate["accepted"] is False  # type: ignore[index]
        assert duplicate["rejection_reason"] == "duplicate_final_transcript"  # type: ignore[index]

    asyncio.run(scenario())


def test_input_only_mode_records_microphone_transcript_without_reasoning_or_tts() -> None:
    class FakeStateMachine:
        def __init__(self) -> None:
            self.events: list[tuple[EventType, dict[str, object]]] = []

        def record(self, event_type: EventType, **kwargs: object) -> None:
            self.events.append((event_type, kwargs))

    class FakeSession:
        def __init__(self) -> None:
            self.state_machine = FakeStateMachine()
            self.submissions = 0

        async def submit_transcript(self, update: object) -> None:
            del update
            self.submissions += 1

    async def scenario() -> None:
        session = FakeSession()
        outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
        output = BrowserAcknowledgedAudioOutput(outbound)
        seen: set[str] = set()
        texts = (
            "Das ist der erste Mikrofonsatz.",
            "Hier folgt der zweite echte Satz.",
            "Nur meine Sprache darf als Nutzertext erscheinen.",
        )
        for index, text in enumerate(texts):
            payload = json.dumps(
                {
                    "type": "transcript",
                    "source": "browser_stt",
                    "recognition_result_id": f"browser-input-only:{index}",
                    "provenance": _browser_provenance(
                        f"browser-input-only:{index}"
                    ),
                    "text": text,
                    "final": True,
                    "signals": {
                        "speech_active": False,
                        "silence_duration_ms": 350,
                        "utterance_duration_ms": 900,
                        "semantic_complete": True,
                        "provider_endpointed": True,
                    },
                }
            )
            await _handle_json(  # type: ignore[arg-type]
                payload,
                session,
                output,
                outbound,
                seen_final_transcript_ids=seen,
                input_only=True,
            )

        assert session.submissions == 0
        final_events = [
            kwargs
            for event_type, kwargs in session.state_machine.events
            if event_type is EventType.FINAL_TRANSCRIPT
        ]
        assert [event["payload"]["raw_text"] for event in final_events] == list(texts)  # type: ignore[index]
        assert all(
            event["payload"]["reasoning_called"] is False  # type: ignore[index]
            and event["payload"]["tts_called"] is False  # type: ignore[index]
            for event in final_events
        )
        results = [await outbound.get() for _ in texts]
        assert [result["text"] for result in results] == list(texts)  # type: ignore[index]
        assert all(result["type"] == "input_probe_transcript" for result in results)  # type: ignore[index]

    asyncio.run(scenario())


def test_stale_browser_recognition_session_is_rejected() -> None:
    class FakeStateMachine:
        def __init__(self) -> None:
            self.events: list[EventType] = []

        def record(self, event_type: EventType, **kwargs: object) -> None:
            del kwargs
            self.events.append(event_type)

    class FakeSession:
        def __init__(self) -> None:
            self.state_machine = FakeStateMachine()
            self.submissions = 0

        async def accept_user_transcript(self, update: object) -> None:
            del update
            self.submissions += 1

    async def scenario() -> None:
        session = FakeSession()
        outbound: asyncio.Queue[dict[str, object] | bytes | None] = asyncio.Queue()
        route = BrowserTranscriptRouteState()
        route.activate(
            recognition_session_id="current-session",
            audio_capture_id="current-capture",
        )
        payload = json.dumps(
            {
                "type": "transcript",
                "source": "browser_stt",
                "recognition_result_id": "stale-result",
                "text": "Ein verspätetes altes Ergebnis.",
                "final": True,
                "provenance": _browser_provenance(
                    "stale-result",
                    recognition_session_id="old-session",
                    audio_capture_id="old-capture",
                ),
                "signals": {},
            }
        )

        await _handle_json(  # type: ignore[arg-type]
            payload,
            session,
            BrowserAcknowledgedAudioOutput(outbound),
            outbound,
            transcript_route_state=route,
            input_only=True,
        )

        assert session.submissions == 0
        assert session.state_machine.events == [EventType.TRANSCRIPT_REJECTED]
        result = await outbound.get()
        assert result["accepted"] is False  # type: ignore[index]
        assert result["rejection_reason"] == "stale_recognition_session"  # type: ignore[index]

    asyncio.run(scenario())


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
