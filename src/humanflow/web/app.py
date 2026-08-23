"""FastAPI websocket demo wired to the same realtime session used by tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from humanflow.audio.models import AudioFrame
from humanflow.domain.conversation import OperationToken
from humanflow.runtime.providers import (
    NullTranscriber,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.turns.models import TurnSignals

from .transport import (
    BrowserAcknowledgedAudioOutput,
    BrowserTelemetrySink,
    OutboundItem,
)


STATIC_DIR = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports"


class GermanDemoReasoner:
    """One local response chunk gives browser speech one semantic boundary."""

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        yield f"Ich habe Sie verstanden. Sie sagten: {transcript.strip()}."


def create_app() -> FastAPI:
    application = FastAPI(title="HumanFlow Realtime Demo", version="0.1.0")
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "dashboard.html")

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runtime": "humanflow-local-demo",
            "speech_provider": "browser-speech-synthesis-with-pcm-fallback",
            "production_claim": False,
        }

    @application.get("/api/scorecard")
    async def scorecard() -> dict[str, Any]:
        return _load_report("scorecard.json")

    @application.get("/api/evidence")
    async def evidence() -> dict[str, Any]:
        return build_evidence_summary(PROJECT_ROOT)

    @application.get("/api/timeline")
    async def timeline() -> dict[str, Any]:
        path = REPORTS_DIR / "realtime-timeline.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return {
            "synthetic": True,
            "events": events,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @application.websocket("/ws")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()
        outbound: asyncio.Queue[OutboundItem | None] = asyncio.Queue()
        audio_output = BrowserAcknowledgedAudioOutput(
            outbound, acknowledgement_timeout_s=15.0
        )
        sink = BrowserTelemetrySink(outbound)
        session = RealtimeVoiceSession(
            conversation_id=str(uuid4()),
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=GermanDemoReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=1_500),
            audio_output=audio_output,
        )
        sequence = 0

        async def send_loop() -> None:
            while True:
                item = await outbound.get()
                try:
                    if item is None:
                        return
                    if isinstance(item, bytes):
                        await websocket.send_bytes(item)
                    else:
                        await websocket.send_json(item)
                finally:
                    outbound.task_done()

        sender = asyncio.create_task(send_loop(), name="humanflow-browser-sender")
        await session.start()
        outbound.put_nowait(
            {
                "type": "ready",
                "conversation_id": session.state_machine.conversation_id,
                "input_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
                "output_mode": "browser_speech_synthesis_with_pcm_fallback",
                "demo_limit": "Local browser STT/TTS; not a production-provider quality claim.",
            }
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                binary = message.get("bytes")
                if binary is not None:
                    if not binary or len(binary) % 2:
                        outbound.put_nowait({"type": "error", "code": "invalid_pcm16_frame"})
                        continue
                    session.receive_audio(
                        AudioFrame(
                            stream_id="browser-microphone",
                            sequence=sequence,
                            pcm16=binary,
                            sample_rate_hz=16_000,
                            captured_ns=monotonic_ns(),
                        )
                    )
                    sequence += 1
                    continue
                payload = message.get("text")
                if payload is None:
                    continue
                await _handle_json(payload, session, audio_output, outbound)
        except WebSocketDisconnect:
            pass
        finally:
            audio_output.disconnect()
            await session.close(reason_code="browser_disconnected")
            outbound.put_nowait(None)
            await sender

    return application


def _load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not an object: {name}")
    return payload


def build_evidence_summary(root: Path) -> dict[str, Any]:
    reports = root / "reports"
    sprint = json.loads((root / "sprint" / "start.json").read_text(encoding="utf-8"))
    torture = json.loads((reports / "torture-run.json").read_text(encoding="utf-8"))
    router = json.loads((reports / "development-router.json").read_text(encoding="utf-8"))
    tournament = json.loads((reports / "tournament-readiness.json").read_text(encoding="utf-8"))
    quality = json.loads(
        (reports / "quality-loop" / "iteration-001" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    replay = json.loads((reports / "timeline-replay.json").read_text(encoding="utf-8"))
    report_names = (
        "scorecard.json",
        "torture-run.json",
        "timeline-replay.json",
        "development-router.json",
        "tournament-readiness.json",
        "browser-demo-benchmark.json",
    )
    return {
        "sprint": {
            "start_utc": sprint["sprint"]["start_utc"],
            "deadline_utc": sprint["sprint"]["deadline_utc"],
            "tag": sprint["sprint"]["start_tag"],
            "baseline_commit": sprint["sprint"]["baseline_commit"],
        },
        "torture": torture["summary"],
        "quality_loop": quality,
        "router": router["summary"],
        "tournament": {
            "status": tournament["status"],
            "external_agent_calls_made": tournament["external_agent_calls_made"],
        },
        "timeline": replay["replay"],
        "artifacts": {
            name: {
                "sha256": hashlib.sha256((reports / name).read_bytes()).hexdigest(),
                "path": f"reports/{name}",
            }
            for name in report_names
        },
    }


async def _handle_json(
    payload: str,
    session: RealtimeVoiceSession,
    audio_output: BrowserAcknowledgedAudioOutput,
    outbound: asyncio.Queue[OutboundItem | None],
) -> None:
    import json

    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        outbound.put_nowait({"type": "error", "code": "invalid_json"})
        return
    if not isinstance(message, dict):
        outbound.put_nowait({"type": "error", "code": "json_object_required"})
        return
    message_type = message.get("type")
    if message_type in {"playback_started", "playback_completed", "playback_stopped"}:
        if not audio_output.acknowledge(message):
            outbound.put_nowait({"type": "error", "code": "stale_playback_ack"})
        return
    if message_type == "interrupt":
        latency_ms = await session.interrupt()
        outbound.put_nowait({"type": "interrupt_result", "audible_stop_latency_ms": latency_ms})
        return
    if message_type != "transcript":
        outbound.put_nowait({"type": "error", "code": "unsupported_message"})
        return
    text = message.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 4_096:
        outbound.put_nowait({"type": "error", "code": "invalid_transcript"})
        return
    raw_signals = message.get("signals", {})
    if not isinstance(raw_signals, dict):
        outbound.put_nowait({"type": "error", "code": "invalid_signals"})
        return
    try:
        signals = TurnSignals(
            speech_active=bool(raw_signals.get("speech_active", False)),
            silence_duration_ms=int(raw_signals.get("silence_duration_ms", 350)),
            utterance_duration_ms=int(raw_signals.get("utterance_duration_ms", 500)),
            semantic_complete=bool(raw_signals.get("semantic_complete", message.get("final", False))),
            filler_ending=bool(raw_signals.get("filler_ending", False)),
            acoustic_completion=float(raw_signals.get("acoustic_completion", 0.8)),
            background_speech_probability=float(
                raw_signals.get("background_speech_probability", 0.0)
            ),
            interruption_probability=float(raw_signals.get("interruption_probability", 0.0)),
        )
        decision = await session.submit_transcript(
            TranscriptUpdate(
                text=text.strip(), is_final=bool(message.get("final", False)), signals=signals
            )
        )
    except (TypeError, ValueError) as error:
        outbound.put_nowait(
            {"type": "error", "code": "invalid_signal_value", "detail": type(error).__name__}
        )
        return
    outbound.put_nowait(
        {
            "type": "turn_decision",
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "reason_codes": list(decision.reason_codes),
        }
    )


app = create_app()
