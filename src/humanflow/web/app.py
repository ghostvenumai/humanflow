"""FastAPI websocket demo wired to the same realtime session used by tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from collections.abc import AsyncIterator
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from humanflow.audio.models import AudioFrame
from humanflow.runtime.anthropic_provider import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicReasoner,
)
from humanflow.runtime.elevenlabs_provider import (
    DEFAULT_ELEVENLABS_MODEL,
    ElevenLabsStreamingTTSProvider,
    FallbackStreamingTTSProvider,
    SynthesisBudget,
)
from humanflow.runtime.providers import (
    BrowserSpeechSynthesisAdapter,
    GaplessSegmentTTSProvider,
    NullTranscriber,
    ProviderInfo,
    ProviderMode,
    StreamingReasoner,
    StreamingTTSProvider,
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
VOICE_ASSESSMENTS_PATH = REPORTS_DIR / "human-voice-quality-assessments.jsonl"
LOGGER = logging.getLogger(__name__)
VOICE_RATING_FIELDS = (
    "naturalness",
    "prosody",
    "pacing",
    "voice_pleasantness",
    "turn_timing",
    "interruption_feel",
    "non_mechanical_impression",
    "overall_conversational_realism",
)


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    info: ProviderInfo
    availability: str

    def to_dict(self) -> dict[str, str]:
        return {**self.info.to_dict(), "availability": self.availability}


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    reasoner_factory: Callable[[], StreamingReasoner] | None
    synthesizer_factory: Callable[[], StreamingTTSProvider] | None
    providers: tuple[ProviderStatus, ...]
    blocker: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.reasoner_factory is not None
            and self.synthesizer_factory is not None
            and self.blocker is None
        )

    def provider_payload(self) -> list[dict[str, str]]:
        return [status.to_dict() for status in self.providers]


class BrowserSessionLease:
    """Allow exactly one browser demo to own device playback at a time."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_conversation_id: str | None = None

    async def acquire(self, conversation_id: str) -> bool:
        async with self._lock:
            if self._active_conversation_id is not None:
                return False
            self._active_conversation_id = conversation_id
            return True

    async def release(self, conversation_id: str) -> None:
        async with self._lock:
            if self._active_conversation_id == conversation_id:
                self._active_conversation_id = None


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
    tts_model = values.get("HUMANFLOW_TTS_MODEL", DEFAULT_ELEVENLABS_MODEL).strip()
    tts_info = ProviderInfo(
        role="tts",
        provider="elevenlabs-text-to-speech-stream",
        model=tts_model or DEFAULT_ELEVENLABS_MODEL,
        mode=ProviderMode.REAL,
        runtime="server",
    )
    tts_fallback_info = ProviderInfo(
        role="tts-fallback",
        provider=BrowserSpeechSynthesisAdapter.provider_info.provider,
        model=BrowserSpeechSynthesisAdapter.provider_info.model,
        mode=ProviderMode.REAL,
        runtime="browser",
    )
    tts_fallback = ProviderStatus(
        tts_fallback_info,
        "BROWSER_CHECK_REQUIRED",
    )
    tts_api_key = values.get("ELEVENLABS_API_KEY", "")
    tts_voice_id = values.get("ELEVENLABS_VOICE_ID", "")
    tts_missing = []
    if not tts_api_key.strip():
        tts_missing.append("ELEVENLABS_API_KEY")
    if not tts_voice_id.strip():
        tts_missing.append("ELEVENLABS_VOICE_ID")
    tts_status = ProviderStatus(
        tts_info,
        "CONFIGURED" if not tts_missing else "MISSING_CONFIGURATION",
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
            synthesizer_factory=None,
            providers=(stt, reasoning, tts_status, tts_fallback),
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
            synthesizer_factory=None,
            providers=(
                stt,
                ProviderStatus(reasoning_info, "MISSING_API_KEY"),
                tts_status,
                tts_fallback,
            ),
            blocker=(
                "ANTHROPIC_API_KEY is not configured"
                if not tts_missing
                else "ANTHROPIC_API_KEY and ElevenLabs TTS configuration are required"
            ),
        )

    if tts_missing:
        return DemoRuntimeConfig(
            reasoner_factory=None,
            synthesizer_factory=None,
            providers=(
                stt,
                ProviderStatus(reasoning_info, "CONFIGURED"),
                tts_status,
                tts_fallback,
            ),
            blocker=f"missing TTS configuration: {', '.join(tts_missing)}",
        )

    def create_reasoner() -> StreamingReasoner:
        return AnthropicReasoner(
            api_key=api_key,
            model=model or DEFAULT_ANTHROPIC_MODEL,
        )

    tts_budget = SynthesisBudget()

    def create_synthesizer() -> StreamingTTSProvider:
        primary = ElevenLabsStreamingTTSProvider(
            api_key=tts_api_key,
            voice_id=tts_voice_id,
            model=tts_model or DEFAULT_ELEVENLABS_MODEL,
            budget=tts_budget,
        )
        return GaplessSegmentTTSProvider(
            FallbackStreamingTTSProvider(
                primary=primary,
                fallback=BrowserSpeechSynthesisAdapter(),
            )
        )

    return DemoRuntimeConfig(
        reasoner_factory=create_reasoner,
        synthesizer_factory=create_synthesizer,
        providers=(
            stt,
            ProviderStatus(reasoning_info, "CONFIGURED"),
            tts_status,
            tts_fallback,
        ),
    )


def create_app(runtime_config: DemoRuntimeConfig | None = None) -> FastAPI:
    runtime = runtime_config or load_demo_runtime_config()
    assessment_lock = asyncio.Lock()
    browser_session_lease = BrowserSessionLease()

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

    @application.get("/api/voice-quality")
    async def voice_quality() -> dict[str, Any]:
        return build_voice_quality_summary(VOICE_ASSESSMENTS_PATH)

    @application.post("/api/voice-quality")
    async def record_voice_quality(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            assessment = validate_voice_quality_assessment(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        async with assessment_lock:
            VOICE_ASSESSMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with VOICE_ASSESSMENTS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(assessment, ensure_ascii=False, sort_keys=True) + "\n")
            return build_voice_quality_summary(VOICE_ASSESSMENTS_PATH)

    @application.websocket("/ws")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()
        if runtime.reasoner_factory is None or runtime.synthesizer_factory is None:
            await websocket.send_json(
                {"type": "error", "code": "real_reasoning_provider_unavailable"}
            )
            await websocket.close(code=1011)
            return
        conversation_id = str(uuid4())
        input_only = websocket.query_params.get("mode") == "input-only"
        if not await browser_session_lease.acquire(conversation_id):
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "another_browser_session_owns_playback",
                }
            )
            await websocket.close(code=1008)
            return
        outbound: asyncio.Queue[OutboundItem | None] = asyncio.Queue()
        audio_output = BrowserAcknowledgedAudioOutput(
            outbound, acknowledgement_timeout_s=15.0
        )
        sink = BrowserTelemetrySink(outbound)
        session = RealtimeVoiceSession(
            conversation_id=conversation_id,
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=runtime.reasoner_factory(),
            synthesizer=runtime.synthesizer_factory(),
            audio_output=audio_output,
        )
        sequence = 0
        controls: asyncio.Queue[str] = asyncio.Queue()
        seen_final_transcript_ids: set[str] = set()

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

        async def control_loop() -> None:
            while True:
                payload = await controls.get()
                try:
                    await _handle_json(
                        payload,
                        session,
                        audio_output,
                        outbound,
                        seen_final_transcript_ids=seen_final_transcript_ids,
                        input_only=input_only,
                    )
                finally:
                    controls.task_done()

        sender = asyncio.create_task(send_loop(), name="humanflow-browser-sender")
        controller = asyncio.create_task(
            control_loop(), name="humanflow-browser-controller"
        )
        await session.start()
        outbound.put_nowait(
            {
                "type": "ready",
                "conversation_id": session.state_machine.conversation_id,
                "input_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
                "output_mode": (
                    "disabled_input_routing_isolation"
                    if input_only
                    else "streaming_pcm_with_visible_browser_speech_fallback"
                ),
                "providers": runtime.provider_payload(),
                "manual_validation": "REQUIRED_NOT_ATTESTED",
                "demo_limit": (
                    "INPUT-ONLY: kein Reasoning- oder TTS-Aufruf."
                    if input_only
                    else "Browser STT, echtes ElevenLabs Streaming-TTS und reales Reasoning; kein Web- oder Kalender-Tool."
                ),
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
                if not _acknowledge_transport_message(payload, audio_output, outbound):
                    controls.put_nowait(payload)
        except WebSocketDisconnect:
            pass
        finally:
            audio_output.disconnect()
            controller.cancel()
            sender.cancel()
            await asyncio.gather(controller, sender, return_exceptions=True)
            try:
                await session.close(reason_code="browser_disconnected")
            finally:
                await browser_session_lease.release(conversation_id)

    return application


def _load_report(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report is not an object: {name}")
    return payload


def validate_voice_quality_assessment(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a human-submitted assessment; never synthesize subjective scores."""

    ratings = payload.get("ratings")
    if not isinstance(ratings, Mapping):
        raise ValueError("ratings object required")
    validated: dict[str, int] = {}
    for field in VOICE_RATING_FIELDS:
        value = ratings.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{field} must be an integer between 1 and 5")
        validated[field] = value
    candidate = payload.get("candidate", "live-candidate")
    notes = payload.get("notes", "")
    if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 160:
        raise ValueError("candidate must be a short non-empty string")
    if not isinstance(notes, str) or len(notes) > 2_000:
        raise ValueError("notes must be a string with at most 2000 characters")
    return {
        "assessment_id": str(uuid4()),
        "submitted_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "human_browser_submission",
        "candidate": candidate.strip(),
        "ratings": validated,
        "notes": notes.strip(),
        "manual_validation": "HUMAN_SUBMITTED_NOT_AGENT_ATTESTED",
    }


def build_voice_quality_summary(path: Path) -> dict[str, Any]:
    assessments: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    assessments.append(item)
    averages: dict[str, float] = {}
    if assessments:
        for field in VOICE_RATING_FIELDS:
            values = [item["ratings"][field] for item in assessments]
            averages[field] = round(sum(values) / len(values), 3)
    return {
        "status": "HUMAN_ASSESSMENTS_RECORDED" if assessments else "AWAITING_HUMAN_ASSESSMENT",
        "sample_count": len(assessments),
        "scale": {"minimum": 1, "maximum": 5},
        "fields": list(VOICE_RATING_FIELDS),
        "averages": averages,
        "agent_attestation": False,
    }


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


def _acknowledge_transport_message(
    payload: str,
    audio_output: BrowserAcknowledgedAudioOutput,
    outbound: asyncio.Queue[OutboundItem | None],
) -> bool:
    """Handle playback receipts outside the ordered control worker.

    An interruption waits for the audible-stop receipt, so processing that
    receipt in the same serialized worker would deadlock until timeout.
    """

    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(message, dict):
        return False
    if message.get("type") not in {
        "playback_started",
        "playback_completed",
        "playback_stopped",
    }:
        return False
    if not audio_output.acknowledge(message):
        outbound.put_nowait({"type": "error", "code": "stale_playback_ack"})
    return True


async def _handle_json(
    payload: str,
    session: RealtimeVoiceSession,
    audio_output: BrowserAcknowledgedAudioOutput,
    outbound: asyncio.Queue[OutboundItem | None],
    *,
    seen_final_transcript_ids: set[str] | None = None,
    input_only: bool = False,
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
                "browser_stt_and_tts_fallback_available"
                if stt_available and tts_available
                else "browser_capability_missing"
            ),
            payload={
                "stt": {
                    "provider": "browser-web-speech-api",
                    "mode": "REAL",
                    "available": stt_available,
                },
                "tts_fallback": {
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
    is_final = bool(message.get("final", False))
    source = message.get("source")
    if source == "browser_stt" and is_final:
        recognition_result_id = message.get("recognition_result_id")
        if not isinstance(recognition_result_id, str) or not recognition_result_id.strip():
            outbound.put_nowait(
                {"type": "error", "code": "missing_recognition_result_id"}
            )
            return
        final_ids = seen_final_transcript_ids if seen_final_transcript_ids is not None else set()
        if recognition_result_id in final_ids:
            session.state_machine.record(
                EventType.DUPLICATE_TRANSCRIPT_REJECTED,
                correlation_id=str(uuid4()),
                reason_code="browser_final_result_id_already_processed",
                payload={"recognition_result_id": recognition_result_id},
            )
            outbound.put_nowait(
                {"type": "error", "code": "duplicate_final_transcript_rejected"}
            )
            return
        if len(final_ids) >= 4_096:
            raise RuntimeError("browser_transcript_dedupe_capacity_exceeded")
        final_ids.add(recognition_result_id)
    raw_signals = message.get("signals", {})
    if not isinstance(raw_signals, dict):
        outbound.put_nowait({"type": "error", "code": "invalid_signals"})
        return
    try:
        if source == "browser_stt":
            transcript_provider = ProviderInfo(
                role="stt",
                provider="browser-web-speech-api",
                model="de-DE",
                mode=ProviderMode.REAL,
                runtime="browser",
            )
        else:
            transcript_provider = ProviderInfo(
                role="stt",
                provider="diagnostic-text-input",
                model=str(source or "unspecified"),
                mode=ProviderMode.MOCK,
                runtime="client",
            )
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
            provider_endpointed=bool(raw_signals.get("provider_endpointed", False)),
        )
        update = TranscriptUpdate(
            text=text.strip(),
            is_final=is_final,
            signals=signals,
            provider=transcript_provider,
        )
        if input_only:
            session.state_machine.record(
                EventType.FINAL_TRANSCRIPT if is_final else EventType.PARTIAL_TRANSCRIPT,
                correlation_id=str(uuid4()),
                reason_code="input_only_browser_routing_probe",
                payload={
                    "text": update.text,
                    "provider": transcript_provider.to_dict(),
                    "recognition_result_id": message.get("recognition_result_id"),
                    "reasoning_called": False,
                    "tts_called": False,
                },
            )
            outbound.put_nowait(
                {
                    "type": "input_probe_transcript",
                    "text": update.text,
                    "final": update.is_final,
                    "recognition_result_id": message.get("recognition_result_id"),
                }
            )
            return
        decision = await session.submit_transcript(update)
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
