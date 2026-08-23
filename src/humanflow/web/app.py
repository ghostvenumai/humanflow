"""FastAPI websocket demo wired to the same realtime session used by tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from humanflow.audio.models import AudioFrame
from humanflow.runtime.anthropic_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicReasoner,
)
from humanflow.runtime.providers import (
    BrowserSpeechSynthesisAdapter,
    NullTranscriber,
    ProviderInfo,
    ProviderMode,
    StreamingReasoner,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.telemetry.events import EventType
from humanflow.turns.models import TurnSignals

from .transport import (
    BrowserAcknowledgedAudioOutput,
    BrowserTelemetrySink,
    OutboundItem,
)


STATIC_DIR = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    info: ProviderInfo
    availability: str

    def to_dict(self) -> dict[str, str]:
        return {**self.info.to_dict(), "availability": self.availability}


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    reasoner_factory: Callable[[], StreamingReasoner] | None
    providers: tuple[ProviderStatus, ...]
    blocker: str | None = None

    @property
    def ready(self) -> bool:
        return self.reasoner_factory is not None and self.blocker is None

    def provider_payload(self) -> list[dict[str, str]]:
        return [status.to_dict() for status in self.providers]


def load_demo_runtime_config(
    environ: Mapping[str, str] | None = None,
) -> DemoRuntimeConfig:
    """Resolve only explicit real providers; never construct a demo fallback."""

    values = os.environ if environ is None else environ
    selected = values.get("HUMANFLOW_REASONING_PROVIDER", "anthropic").strip().lower()
    model = values.get("HUMANFLOW_REASONING_MODEL", DEFAULT_ANTHROPIC_MODEL).strip()
    stt = ProviderStatus(
        ProviderInfo(
            role="stt",
            provider="browser-web-speech-api",
            model="de-DE",
            mode=ProviderMode.REAL,
            runtime="browser",
        ),
        "BROWSER_CHECK_REQUIRED",
    )
    tts = ProviderStatus(
        BrowserSpeechSynthesisAdapter.provider_info,
        "BROWSER_CHECK_REQUIRED",
    )
    if selected != "anthropic":
        reasoning = ProviderStatus(
            ProviderInfo(
                role="reasoning",
                provider=selected or "unset",
                model=model or "unset",
                mode=ProviderMode.REAL,
                runtime="server",
            ),
            "UNSUPPORTED",
        )
        return DemoRuntimeConfig(
            reasoner_factory=None,
            providers=(stt, reasoning, tts),
            blocker=f"unsupported reasoning provider: {selected or 'empty'}",
        )
    api_key = values.get("ANTHROPIC_API_KEY", "")
    reasoning_info = ProviderInfo(
        role="reasoning",
        provider="anthropic-messages-api",
        model=model or DEFAULT_ANTHROPIC_MODEL,
        mode=ProviderMode.REAL,
        runtime="server",
    )
    if not api_key.strip():
        return DemoRuntimeConfig(
            reasoner_factory=None,
            providers=(
                stt,
                ProviderStatus(reasoning_info, "MISSING_API_KEY"),
                tts,
            ),
            blocker="ANTHROPIC_API_KEY is not configured",
        )

    def create_reasoner() -> StreamingReasoner:
        return AnthropicReasoner(
            api_key=api_key,
            model=model or DEFAULT_ANTHROPIC_MODEL,
        )

    return DemoRuntimeConfig(
        reasoner_factory=create_reasoner,
        providers=(stt, ProviderStatus(reasoning_info, "CONFIGURED"), tts),
    )


def create_app(runtime_config: DemoRuntimeConfig | None = None) -> FastAPI:
    runtime = runtime_config or load_demo_runtime_config()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not runtime.ready:
            raise RuntimeError(f"real provider required: {runtime.blocker}")
        LOGGER.info(
            "HumanFlow live providers: %s",
            json.dumps(runtime.provider_payload(), ensure_ascii=False, sort_keys=True),
        )
        yield

    application = FastAPI(
        title="HumanFlow Realtime Demo",
        version="0.2.0",
        lifespan=lifespan,
    )
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
            "runtime": "humanflow-live-provider-demo",
            "providers": runtime.provider_payload(),
            "mock_conversation_provider": False,
            "manual_validation": "REQUIRED_NOT_ATTESTED",
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
        if runtime.reasoner_factory is None:
            await websocket.send_json(
                {"type": "error", "code": "real_reasoning_provider_unavailable"}
            )
            await websocket.close(code=1011)
            return
        outbound: asyncio.Queue[OutboundItem | None] = asyncio.Queue()
        audio_output = BrowserAcknowledgedAudioOutput(
            outbound, acknowledgement_timeout_s=15.0
        )
        sink = BrowserTelemetrySink(outbound)
        session = RealtimeVoiceSession(
            conversation_id=str(uuid4()),
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=runtime.reasoner_factory(),
            synthesizer=BrowserSpeechSynthesisAdapter(),
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
                "output_mode": "mandatory_browser_speech_synthesis",
                "providers": runtime.provider_payload(),
                "manual_validation": "REQUIRED_NOT_ATTESTED",
                "demo_limit": "Browser STT/TTS plus real network reasoning; no Web- oder Kalender-Tool.",
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
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await session.close(reason_code="browser_disconnected")

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
    if message_type == "provider_capabilities":
        stt_available = message.get("stt_available") is True
        tts_available = message.get("tts_available") is True
        session.state_machine.record(
            EventType.PROVIDER_STATUS,
            correlation_id=str(uuid4()),
            reason_code=(
                "browser_voice_providers_available"
                if stt_available and tts_available
                else "browser_voice_provider_missing"
            ),
            payload={
                "stt": {
                    "provider": "browser-web-speech-api",
                    "mode": "REAL",
                    "available": stt_available,
                },
                "tts": {
                    "provider": "browser-web-speech-api",
                    "mode": "REAL",
                    "available": tts_available,
                },
            },
        )
        return
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
                text=text.strip(),
                is_final=bool(message.get("final", False)),
                signals=signals,
                provider=ProviderInfo(
                    role="stt",
                    provider="browser-web-speech-api",
                    model="de-DE",
                    mode=ProviderMode.REAL,
                    runtime="browser",
                ),
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
