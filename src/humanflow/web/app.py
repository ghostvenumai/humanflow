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
from dataclasses import dataclass, field as dataclass_field
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
from humanflow.runtime.elevenlabs_stt_provider import (
    DEFAULT_ELEVENLABS_STT_MODEL,
    ElevenLabsRealtimeSTTProvider,
)
from humanflow.runtime.providers import (
    BrowserSpeechSynthesisAdapter,
    GaplessSegmentTTSProvider,
    ProviderInfo,
    ProviderMode,
    StreamingReasoner,
    StreamingSTTProvider,
    StreamingTTSProvider,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
    TranscriptProvenance,
    TranscriptRejected,
    normalize_transcript,
)
from humanflow.telemetry.events import EventType, TelemetryEvent
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
    transcriber_factory: Callable[[], StreamingSTTProvider] | None
    reasoner_factory: Callable[[], StreamingReasoner] | None
    synthesizer_factory: Callable[[], StreamingTTSProvider] | None
    providers: tuple[ProviderStatus, ...]
    blocker: str | None = None
    tts_candidate_factory: Callable[[], StreamingTTSProvider] | None = None

    @property
    def ready(self) -> bool:
        return (
            self.transcriber_factory is not None
            and self.reasoner_factory is not None
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


@dataclass(slots=True)
class BrowserTranscriptRouteState:
    current_recognition_session_id: str | None = None
    current_audio_capture_id: str | None = None
    seen_final_result_ids: set[str] = dataclass_field(default_factory=set)

    def activate(self, *, recognition_session_id: str, audio_capture_id: str) -> None:
        self.current_recognition_session_id = recognition_session_id
        self.current_audio_capture_id = audio_capture_id


@dataclass(slots=True)
class PcmAudioRouteState:
    audio_capture_id: str | None = None
    microphone_stream_id: str | None = None

    @property
    def active(self) -> bool:
        return self.audio_capture_id is not None and self.microphone_stream_id is not None

    def activate(self, *, audio_capture_id: str, microphone_stream_id: str) -> None:
        binding = (audio_capture_id.strip(), microphone_stream_id.strip())
        if not all(binding):
            raise ValueError("PCM source identifiers must not be empty")
        current = (self.audio_capture_id, self.microphone_stream_id)
        if current != (None, None) and current != binding:
            raise RuntimeError("pcm_audio_source_binding_is_immutable")
        self.audio_capture_id, self.microphone_stream_id = binding


@dataclass(slots=True)
class LiveBargeInMetrics:
    """Latest real browser-session values; never substitutes human judgement."""

    values: dict[str, float] = dataclass_field(default_factory=dict)
    sample_counts: dict[str, int] = dataclass_field(default_factory=dict)
    false_interruption_count: int = 0
    conversation_id: str | None = None

    _EVENT_FIELDS = {
        EventType.USER_AUDIO_STARTED: (
            "acoustic_speech_onset_latency_ms",
            "acoustic_speech_onset_latency_ms",
        ),
        EventType.PLAYBACK_DUCK_STARTED: (
            "speech_onset_to_soft_duck_ms",
            "speech_onset_to_soft_duck_ms",
        ),
        EventType.INTERRUPTION_CONFIRMED: (
            "speech_onset_to_hard_cancel_ms",
            "speech_onset_to_hard_cancel_ms",
        ),
        EventType.AUDIBLE_STOP_ACK: (
            "speech_onset_to_audible_stop_ms",
            "speech_onset_to_audible_stop_ms",
        ),
        EventType.BACKCHANNEL_RECOVERY: (
            "backchannel_recovery_latency_ms",
            "backchannel_recovery_latency_ms",
        ),
        EventType.PARTIAL_TRANSCRIPT: (
            "first_stt_partial_ms",
            "first_stt_partial_ms",
        ),
        EventType.FINAL_TRANSCRIPT: ("final_stt_ms", "final_stt_ms"),
    }

    def reset(self, conversation_id: str) -> None:
        self.values.clear()
        self.sample_counts.clear()
        self.false_interruption_count = 0
        self.conversation_id = conversation_id

    def observe(self, event: TelemetryEvent) -> None:
        if self.conversation_id != event.conversation_id:
            self.reset(event.conversation_id)
        if event.event_type is EventType.FALSE_INTERRUPTION_DETECTED:
            self.false_interruption_count += 1
        mapping = self._EVENT_FIELDS.get(event.event_type)
        if mapping is None:
            return
        payload_name, metric_name = mapping
        value = event.payload.get(payload_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        self.values[metric_name] = round(float(value), 3)
        self.sample_counts[metric_name] = self.sample_counts.get(metric_name, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        names = (
            "acoustic_speech_onset_latency_ms",
            "speech_onset_to_soft_duck_ms",
            "speech_onset_to_hard_cancel_ms",
            "speech_onset_to_audible_stop_ms",
            "backchannel_recovery_latency_ms",
            "first_stt_partial_ms",
            "final_stt_ms",
        )
        return {
            "status": "LIVE_SESSION_EVIDENCE" if self.conversation_id else "NO_LIVE_SAMPLE",
            "conversation_id": self.conversation_id,
            "metrics": {
                name: {
                    "last": self.values.get(name),
                    "sample_count": self.sample_counts.get(name, 0),
                }
                for name in names
            },
            "false_interruption_count": self.false_interruption_count,
            "manual_validation": "REQUIRED_NOT_ATTESTED",
            "measurement_scope": (
                "authoritative server PCM onset; browser duck/stop acknowledgements "
                "are upper bounds, not self-attested perception"
            ),
        }


def load_demo_runtime_config(
    environ: Mapping[str, str] | None = None,
) -> DemoRuntimeConfig:
    """Resolve only explicit real providers; never construct a demo fallback."""

    values = os.environ if environ is None else environ
    selected = values.get("HUMANFLOW_REASONING_PROVIDER", "anthropic").strip().lower()
    model = values.get("HUMANFLOW_REASONING_MODEL", DEFAULT_ANTHROPIC_MODEL).strip()
    stt_model = values.get(
        "HUMANFLOW_STT_MODEL", DEFAULT_ELEVENLABS_STT_MODEL
    ).strip()
    stt_api_key = values.get("ELEVENLABS_STT_API_KEY", "") or values.get(
        "ELEVENLABS_API_KEY", ""
    )
    stt = ProviderStatus(
        ProviderInfo(
            role="stt",
            provider="elevenlabs-scribe-realtime",
            model=stt_model or DEFAULT_ELEVENLABS_STT_MODEL,
            mode=ProviderMode.REAL,
            runtime="server",
        ),
        "CONFIGURED_UNVERIFIED" if stt_api_key.strip() else "MISSING_API_KEY",
    )
    browser_stt_diagnostic = ProviderStatus(
        ProviderInfo(
            role="stt-browser-diagnostic",
            provider="browser-web-speech-api",
            model="de-DE",
            mode=ProviderMode.MOCK,
            runtime="browser",
        ),
        "OFF_PRODUCTION",
    )
    tts_model = values.get("HUMANFLOW_TTS_MODEL", DEFAULT_ELEVENLABS_MODEL).strip()
    tts_candidate_model = values.get(
        "HUMANFLOW_TTS_AB_MODEL", "eleven_v3_conversational"
    ).strip()
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
    tts_candidate_status = ProviderStatus(
        ProviderInfo(
            role="tts-ab-candidate",
            provider="elevenlabs-text-to-speech-stream",
            model=tts_candidate_model or "eleven_v3_conversational",
            mode=ProviderMode.REAL,
            runtime="server",
        ),
        "CONFIGURED_UNVERIFIED" if not tts_missing else "MISSING_CONFIGURATION",
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
            transcriber_factory=None,
            reasoner_factory=None,
            synthesizer_factory=None,
            providers=(
                stt,
                browser_stt_diagnostic,
                reasoning,
                tts_status,
                tts_candidate_status,
                tts_fallback,
            ),
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
    if not stt_api_key.strip():
        return DemoRuntimeConfig(
            transcriber_factory=None,
            reasoner_factory=None,
            synthesizer_factory=None,
            providers=(
                stt,
                browser_stt_diagnostic,
                ProviderStatus(
                    reasoning_info,
                    "CONFIGURED" if api_key.strip() else "MISSING_API_KEY",
                ),
                tts_status,
                tts_candidate_status,
                tts_fallback,
            ),
            blocker=(
                "BLOCKER_REAL_STREAMING_STT_PROVIDER_REQUIRED: "
                "ELEVENLABS_STT_API_KEY or ELEVENLABS_API_KEY with STT scope"
            ),
        )
    if not api_key.strip():
        return DemoRuntimeConfig(
            transcriber_factory=None,
            reasoner_factory=None,
            synthesizer_factory=None,
            providers=(
                stt,
                browser_stt_diagnostic,
                ProviderStatus(reasoning_info, "MISSING_API_KEY"),
                tts_status,
                tts_candidate_status,
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
            transcriber_factory=None,
            reasoner_factory=None,
            synthesizer_factory=None,
            providers=(
                stt,
                browser_stt_diagnostic,
                ProviderStatus(reasoning_info, "CONFIGURED"),
                tts_status,
                tts_candidate_status,
                tts_fallback,
            ),
            blocker=f"missing TTS configuration: {', '.join(tts_missing)}",
        )

    def create_transcriber() -> StreamingSTTProvider:
        return ElevenLabsRealtimeSTTProvider(
            api_key=stt_api_key,
            model=stt_model or DEFAULT_ELEVENLABS_STT_MODEL,
            language_code="de",
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
        return FallbackStreamingTTSProvider(
            primary=GaplessSegmentTTSProvider(primary),
            fallback=BrowserSpeechSynthesisAdapter(),
        )

    def create_candidate_synthesizer() -> StreamingTTSProvider:
        candidate = ElevenLabsStreamingTTSProvider(
            api_key=tts_api_key,
            voice_id=tts_voice_id,
            model=tts_candidate_model or "eleven_v3_conversational",
            budget=tts_budget,
        )
        return FallbackStreamingTTSProvider(
            primary=GaplessSegmentTTSProvider(candidate),
            fallback=BrowserSpeechSynthesisAdapter(),
        )

    return DemoRuntimeConfig(
        transcriber_factory=create_transcriber,
        reasoner_factory=create_reasoner,
        synthesizer_factory=create_synthesizer,
        providers=(
            stt,
            browser_stt_diagnostic,
            ProviderStatus(reasoning_info, "CONFIGURED"),
            tts_status,
            tts_candidate_status,
            tts_fallback,
        ),
        tts_candidate_factory=create_candidate_synthesizer,
    )


def create_app(runtime_config: DemoRuntimeConfig | None = None) -> FastAPI:
    runtime = runtime_config or load_demo_runtime_config()
    assessment_lock = asyncio.Lock()
    browser_session_lease = BrowserSessionLease()
    live_barge_in_metrics = LiveBargeInMetrics()

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

    @application.get("/api/live-barge-in")
    async def live_barge_in() -> dict[str, Any]:
        return live_barge_in_metrics.to_dict()

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
        if (
            runtime.transcriber_factory is None
            or runtime.reasoner_factory is None
            or runtime.synthesizer_factory is None
        ):
            await websocket.send_json(
                {"type": "error", "code": "real_reasoning_provider_unavailable"}
            )
            await websocket.close(code=1011)
            return
        conversation_id = str(uuid4())
        input_only = websocket.query_params.get("mode") == "input-only"
        requested_tts = websocket.query_params.get("tts", "baseline")
        if requested_tts not in {"baseline", "candidate"}:
            await websocket.send_json(
                {"type": "error", "code": "unknown_tts_ab_candidate"}
            )
            await websocket.close(code=1008)
            return
        selected_synthesizer_factory = runtime.synthesizer_factory
        if requested_tts == "candidate":
            selected_synthesizer_factory = runtime.tts_candidate_factory
        if selected_synthesizer_factory is None:
            await websocket.send_json(
                {"type": "error", "code": "tts_ab_candidate_unavailable"}
            )
            await websocket.close(code=1011)
            return
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
        sink = BrowserTelemetrySink(
            outbound, observer=live_barge_in_metrics.observe
        )
        transcriber = runtime.transcriber_factory()
        session = RealtimeVoiceSession(
            conversation_id=conversation_id,
            sink=sink,
            transcriber=transcriber,
            reasoner=runtime.reasoner_factory(),
            synthesizer=selected_synthesizer_factory(),
            audio_output=audio_output,
        )
        sequence = 0
        stt_input_failed = False
        controls: asyncio.Queue[str] = asyncio.Queue()
        transcript_route_state = BrowserTranscriptRouteState()
        pcm_audio_route_state = PcmAudioRouteState()

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
                        transcript_route_state=transcript_route_state,
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
                "tts_ab_selection": requested_tts,
                "conversation_history": {
                    "status": "CLEAN_NEW_SESSION",
                    "roles": [],
                },
                "input_topology": {
                    "microphone_source": "getUserMedia",
                    "pcm_source": "mono pcm_s16le 16000 Hz over HumanFlow websocket",
                    "transcript_source": "same PCM -> ElevenLabs Scribe realtime",
                    "recognition_input_binding": "EXACT_GETUSERMEDIA_PCM16",
                    "browser_speech_recognition_production_status": "OFF",
                },
                "demo_limit": (
                    "INPUT-ONLY: kein Reasoning- oder TTS-Aufruf."
                    if input_only
                    else "Scribe Realtime, echtes ElevenLabs Streaming-TTS und reales Reasoning; kein Web- oder Kalender-Tool."
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
                    if not pcm_audio_route_state.active:
                        outbound.put_nowait(
                            {"type": "error", "code": "pcm_audio_source_not_bound"}
                        )
                        continue
                    if not binary or len(binary) % 2:
                        outbound.put_nowait({"type": "error", "code": "invalid_pcm16_frame"})
                        continue
                    if stt_input_failed:
                        continue
                    try:
                        received_ns = monotonic_ns()
                        frame_duration_ns = round(
                            (len(binary) // 2) / 16_000 * 1_000_000_000
                        )
                        session.receive_audio(
                            AudioFrame(
                                stream_id=str(
                                    pcm_audio_route_state.microphone_stream_id
                                ),
                                sequence=sequence,
                                pcm16=binary,
                                sample_rate_hz=16_000,
                                captured_ns=max(0, received_ns - frame_duration_ns),
                            )
                        )
                    except RuntimeError:
                        stt_input_failed = True
                        outbound.put_nowait(
                            {
                                "type": "error",
                                "code": "real_streaming_stt_unavailable",
                                "browser_speech_recognition_fallback": "FORBIDDEN",
                            }
                        )
                        continue
                    sequence += 1
                    continue
                payload = message.get("text")
                if payload is None:
                    continue
                if _activate_pcm_stream_message(
                    payload,
                    transcriber=transcriber,
                    route_state=pcm_audio_route_state,
                    session=session,
                    outbound=outbound,
                ):
                    continue
                if not _acknowledge_transport_message(
                    payload, audio_output, outbound, session=session
                ):
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
    *,
    session: RealtimeVoiceSession | None = None,
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
    message_type = message.get("type")
    if message_type not in {
        "playback_started",
        "playback_completed",
        "playback_stopped",
        "playback_ducked",
        "playback_resumed",
    }:
        return False
    if message_type in {"playback_ducked", "playback_resumed"}:
        if session is None or not session.acknowledge_playback_control(message):
            outbound.put_nowait(
                {"type": "error", "code": "stale_playback_control_ack"}
            )
        return True
    if not audio_output.acknowledge(message):
        outbound.put_nowait({"type": "error", "code": "stale_playback_ack"})
    return True


def _activate_pcm_stream_message(
    payload: str,
    *,
    transcriber: StreamingSTTProvider,
    route_state: PcmAudioRouteState,
    session: RealtimeVoiceSession,
    outbound: asyncio.Queue[OutboundItem | None],
) -> bool:
    """Bind the production STT provider before accepting the first PCM frame."""

    try:
        message = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(message, dict) or message.get("type") != "pcm_stream_started":
        return False
    audio_capture_id = message.get("audio_capture_id")
    microphone_stream_id = message.get("microphone_stream_id")
    if not isinstance(audio_capture_id, str) or not isinstance(
        microphone_stream_id, str
    ):
        outbound.put_nowait(
            {"type": "error", "code": "invalid_pcm_audio_source_binding"}
        )
        return True
    try:
        route_state.activate(
            audio_capture_id=audio_capture_id,
            microphone_stream_id=microphone_stream_id,
        )
        bind_audio_source = getattr(transcriber, "bind_audio_source", None)
        if not callable(bind_audio_source):
            raise RuntimeError("stt_provider_is_not_source_bindable")
        bind_audio_source(
            audio_capture_id=audio_capture_id,
            stream_id=microphone_stream_id,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        outbound.put_nowait(
            {
                "type": "error",
                "code": "pcm_audio_source_binding_rejected",
                "detail": type(error).__name__,
            }
        )
        return True
    provider = transcriber.provider_info
    session.state_machine.record(
        EventType.PROVIDER_STATUS,
        correlation_id=str(uuid4()),
        reason_code="production_stt_bound_to_authoritative_pcm_source",
        payload={
            "provider": provider.to_dict(),
            "capabilities": transcriber.capabilities.to_dict(),
            "audio_capture_id": audio_capture_id,
            "microphone_stream_id": microphone_stream_id,
            "recognition_input_binding": "EXACT_GETUSERMEDIA_PCM16",
            "browser_speech_recognition_production_status": "OFF",
        },
    )
    outbound.put_nowait(
        {
            "type": "pcm_stream_result",
            "accepted": True,
            "audio_capture_id": audio_capture_id,
            "microphone_stream_id": microphone_stream_id,
            "stt_provider": provider.to_dict(),
            "browser_speech_recognition_production_status": "OFF",
        }
    )
    return True


async def _handle_json(
    payload: str,
    session: RealtimeVoiceSession,
    audio_output: BrowserAcknowledgedAudioOutput,
    outbound: asyncio.Queue[OutboundItem | None],
    *,
    seen_final_transcript_ids: set[str] | None = None,
    transcript_route_state: BrowserTranscriptRouteState | None = None,
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
    if message_type == "recognition_session_started":
        recognition_session_id = message.get("browser_recognition_session_id")
        audio_capture_id = message.get("audio_capture_id")
        if (
            transcript_route_state is None
            or not isinstance(recognition_session_id, str)
            or not recognition_session_id.strip()
            or not isinstance(audio_capture_id, str)
            or not audio_capture_id.strip()
        ):
            outbound.put_nowait(
                {"type": "error", "code": "invalid_recognition_session"}
            )
            return
        transcript_route_state.activate(
            recognition_session_id=recognition_session_id,
            audio_capture_id=audio_capture_id,
        )
        session.state_machine.record(
            EventType.PROVIDER_STATUS,
            correlation_id=str(uuid4()),
            reason_code="browser_recognition_session_activated",
            payload={
                "browser_recognition_session_id": recognition_session_id,
                "audio_capture_id": audio_capture_id,
                "recognition_input_binding": message.get(
                    "recognition_input_binding"
                ),
            },
        )
        return
    if message_type == "provider_capabilities":
        microphone_available = message.get("microphone_available") is True
        tts_available = message.get("tts_available") is True
        session.state_machine.record(
            EventType.PROVIDER_STATUS,
            correlation_id=str(uuid4()),
            reason_code=(
                "browser_pcm_capture_and_tts_fallback_available"
                if microphone_available and tts_available
                else "browser_capability_missing"
            ),
            payload={
                "microphone_pcm": {
                    "source": "getUserMedia",
                    "available": microphone_available,
                    "format": "pcm_s16le_16000_mono",
                },
                "tts_fallback": {
                    "provider": "browser-web-speech-api",
                    "mode": "REAL",
                    "available": tts_available,
                },
                "audio_capture_id": message.get("audio_capture_id"),
                "microphone_stream_id": message.get("microphone_stream_id"),
                "recognition_input_binding": message.get(
                    "recognition_input_binding"
                ),
                "browser_speech_recognition_production_status": "OFF",
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
    try:
        provenance = _parse_transcript_provenance(
            message,
            source=source,
            is_final=is_final,
        )
    except (TypeError, ValueError) as error:
        raw_provenance = message.get("provenance")
        session.state_machine.record(
            EventType.TRANSCRIPT_REJECTED,
            correlation_id=str(uuid4()),
            reason_code="invalid_transcript_provenance",
            payload={
                "raw_provenance": (
                    dict(raw_provenance)
                    if isinstance(raw_provenance, Mapping)
                    else None
                ),
                "source": source,
                "raw_text": text.strip(),
                "normalized_text": normalize_transcript(text),
                "is_partial": not is_final,
                "is_final": is_final,
                "accepted_by_user_ingestion": False,
                "accepted_as_user_turn": False,
                "rejection_reason": "invalid_transcript_provenance",
            },
        )
        outbound.put_nowait(
            {
                "type": "transcript_result",
                "accepted": False,
                "raw_text": text.strip(),
                "normalized_text": "",
                "rejection_reason": "invalid_transcript_provenance",
                "detail": type(error).__name__,
            }
        )
        return
    if provenance.origin is TranscriptOrigin.STREAMING_STT_PROVIDER:
        reason = "client_cannot_emit_streaming_stt_transcript"
        session.state_machine.record(
            EventType.TRANSCRIPT_REJECTED,
            correlation_id=str(uuid4()),
            reason_code=reason,
            payload={
                **provenance.to_dict(),
                "raw_text": text,
                "accepted_as_user_turn": False,
                "rejection_reason": reason,
            },
        )
        outbound.put_nowait(
            {
                "type": "transcript_result",
                "accepted": False,
                "accepted_as_user_turn": False,
                "raw_text": text,
                "normalized_text": normalize_transcript(text),
                "rejection_reason": reason,
                "provenance": provenance.to_dict(),
            }
        )
        return
    if (
        provenance.origin is TranscriptOrigin.BROWSER_SPEECH_RECOGNITION
        and not input_only
    ):
        reason = "browser_speech_recognition_off_production"
        session.state_machine.record(
            EventType.TRANSCRIPT_REJECTED,
            correlation_id=str(uuid4()),
            reason_code=reason,
            payload={
                **provenance.to_dict(),
                "raw_text": text,
                "accepted_as_user_turn": False,
                "browser_speech_recognition_production_status": "OFF",
                "rejection_reason": reason,
            },
        )
        outbound.put_nowait(
            {
                "type": "transcript_result",
                "accepted": False,
                "accepted_as_user_turn": False,
                "raw_text": text,
                "normalized_text": normalize_transcript(text),
                "rejection_reason": reason,
                "provenance": provenance.to_dict(),
            }
        )
        return
    if source == "browser_stt" and transcript_route_state is not None and (
        provenance.browser_recognition_session_id
        != transcript_route_state.current_recognition_session_id
        or provenance.audio_capture_id
        != transcript_route_state.current_audio_capture_id
    ):
        session.state_machine.record(
            EventType.TRANSCRIPT_REJECTED,
            correlation_id=str(uuid4()),
            reason_code="stale_recognition_session",
            payload={
                **provenance.to_dict(),
                "raw_text": text,
                "accepted_as_user_turn": False,
                "rejection_reason": "stale_recognition_session",
            },
        )
        outbound.put_nowait(
            {
                "type": "transcript_result",
                "accepted": False,
                "raw_text": text,
                "normalized_text": "",
                "rejection_reason": "stale_recognition_session",
                "provenance": provenance.to_dict(),
            }
        )
        return
    if source == "browser_stt" and is_final:
        recognition_result_id = message.get("recognition_result_id")
        if not isinstance(recognition_result_id, str) or not recognition_result_id.strip():
            outbound.put_nowait(
                {"type": "error", "code": "missing_recognition_result_id"}
            )
            return
        if transcript_route_state is not None:
            final_ids = transcript_route_state.seen_final_result_ids
        else:
            final_ids = (
                seen_final_transcript_ids
                if seen_final_transcript_ids is not None
                else set()
            )
        if recognition_result_id in final_ids:
            session.state_machine.record(
                EventType.DUPLICATE_TRANSCRIPT_REJECTED,
                correlation_id=str(uuid4()),
                reason_code="browser_final_result_id_already_processed",
                payload={
                    **provenance.to_dict(),
                    "recognition_result_id": recognition_result_id,
                    "raw_text": text,
                    "normalized_text": normalize_transcript(text),
                    "is_partial": False,
                    "is_final": True,
                    "accepted_by_user_ingestion": False,
                    "accepted_as_user_turn": False,
                    "rejection_reason": "duplicate_final_transcript",
                },
            )
            outbound.put_nowait(
                {
                    "type": "transcript_result",
                    "accepted": False,
                    "raw_text": text,
                    "normalized_text": normalize_transcript(text),
                    "rejection_reason": "duplicate_final_transcript",
                    "provenance": provenance.to_dict(),
                }
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
                role="stt-browser-diagnostic",
                provider="browser-web-speech-api",
                model="de-DE",
                mode=ProviderMode.MOCK,
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
            provenance=provenance,
            provider=transcript_provider,
            raw_text=text,
        )
        if input_only:
            diagnostic_browser_input = (
                provenance.origin is TranscriptOrigin.BROWSER_SPEECH_RECOGNITION
            )
            if not provenance.is_allowlisted_user_input and not diagnostic_browser_input:
                reason = (
                    "assistant_origin_event_forbidden_from_user_history"
                    if provenance.is_assistant_origin
                    else "transcript_source_not_allowlisted"
                )
                session.state_machine.record(
                    EventType.TRANSCRIPT_REJECTED,
                    correlation_id=str(uuid4()),
                    reason_code=reason,
                    payload={
                        **provenance.to_dict(),
                        "raw_text": update.raw_text,
                        "normalized_text": update.normalized_text,
                        "accepted_by_user_ingestion": False,
                        "accepted_as_user_turn": False,
                        "rejection_reason": reason,
                    },
                )
                outbound.put_nowait(
                    {
                        "type": "transcript_result",
                        "accepted": False,
                        "raw_text": update.raw_text,
                        "normalized_text": update.normalized_text,
                        "rejection_reason": reason,
                        "provenance": provenance.to_dict(),
                    }
                )
                return
            session.state_machine.record(
                EventType.FINAL_TRANSCRIPT if is_final else EventType.PARTIAL_TRANSCRIPT,
                correlation_id=str(uuid4()),
                reason_code="input_only_browser_routing_probe",
                payload={
                    **provenance.to_dict(),
                    "raw_text": update.raw_text,
                    "normalized_text": update.normalized_text,
                    "is_partial": not update.is_final,
                    "is_final": update.is_final,
                    "accepted_by_user_ingestion": True,
                    "accepted_as_user_turn": update.is_final,
                    "rejection_reason": None,
                    "provider": transcript_provider.to_dict(),
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
                    "provenance": provenance.to_dict(),
                }
            )
            return
        decision = await session.accept_user_transcript(update)
    except TranscriptRejected as error:
        outbound.put_nowait(
            {
                "type": "transcript_result",
                "accepted": False,
                "raw_text": text,
                "normalized_text": update.normalized_text,
                "rejection_reason": error.reason_code,
                "provenance": provenance.to_dict(),
            }
        )
        return
    except (TypeError, ValueError) as error:
        outbound.put_nowait(
            {"type": "error", "code": "invalid_signal_value", "detail": type(error).__name__}
        )
        return
    outbound.put_nowait(
        {
            "type": "transcript_result",
            "accepted": True,
            "raw_text": update.raw_text,
            "normalized_text": update.normalized_text,
            "rejection_reason": None,
            "provenance": provenance.to_dict(),
        }
    )
    outbound.put_nowait(
        {
            "type": "turn_decision",
            "decision": decision.decision.value,
            "confidence": decision.confidence,
            "reason_codes": list(decision.reason_codes),
        }
    )


def _parse_transcript_provenance(
    message: Mapping[str, Any],
    *,
    source: Any,
    is_final: bool,
) -> TranscriptProvenance:
    raw = message.get("provenance")
    if not isinstance(raw, Mapping):
        raise ValueError("provenance object required")
    if not isinstance(source, str) or raw.get("source") != source:
        raise ValueError("source and provenance source must match")
    transcript_id = raw.get("transcript_id")
    stream_id = raw.get("stream_id")
    if not isinstance(transcript_id, str) or not isinstance(stream_id, str):
        raise TypeError("transcript_id and stream_id must be strings")
    browser_timestamp_ms = raw.get("browser_timestamp_ms")
    if browser_timestamp_ms is not None and (
        isinstance(browser_timestamp_ms, bool)
        or not isinstance(browser_timestamp_ms, (int, float))
    ):
        raise TypeError("browser_timestamp_ms must be numeric")
    expected_kind = (
        ConversationEventKind.USER_TRANSCRIPT_FINAL
        if is_final
        else ConversationEventKind.USER_TRANSCRIPT_PARTIAL
    )
    event_kind = ConversationEventKind(str(raw.get("event_kind")))
    if event_kind in {
        ConversationEventKind.USER_TRANSCRIPT_FINAL,
        ConversationEventKind.USER_TRANSCRIPT_PARTIAL,
    } and event_kind is not expected_kind:
        raise ValueError("event kind and finality mismatch")
    return TranscriptProvenance(
        transcript_id=transcript_id,
        event_kind=event_kind,
        source=source,
        origin=TranscriptOrigin(str(raw.get("origin"))),
        stream_id=stream_id,
        timestamp_ns=monotonic_ns(),
        browser_recognition_session_id=_optional_string(
            raw.get("browser_recognition_session_id")
        ),
        audio_capture_id=_optional_string(raw.get("audio_capture_id")),
        response_id=_optional_string(raw.get("response_id")),
        browser_timestamp_ms=(
            None if browser_timestamp_ms is None else float(browser_timestamp_ms)
        ),
        recognition_input_binding=_optional_string(
            raw.get("recognition_input_binding")
        ),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional provenance value must be a string")
    return value or None


app = create_app()
