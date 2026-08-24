"""Modular ElevenLabs PCM streaming with cancellation and bounded demo spend."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from humanflow.audio.models import AudioChunk, AudioFrame, AudioPlaybackMode

from .providers import (
    ProviderInfo,
    ProviderMode,
    SpeechSynthesisRequest,
    StreamingTTSProvider,
)


DEFAULT_ELEVENLABS_MODEL = "eleven_flash_v2_5"
DEFAULT_OUTPUT_FORMAT = "pcm_16000"
ELEVENLABS_FLASH_USD_PER_1K_CHARACTERS = 0.05


@dataclass(slots=True)
class SynthesisBudget:
    """Process-local hard limits for the human validation runtime."""

    maximum_audio_seconds: float = 600.0
    maximum_input_characters: int = 5_000
    submitted_characters: int = 0
    reported_billable_characters: int = 0
    generated_audio_seconds: float = 0.0

    def reserve(self, text: str) -> None:
        characters = len(text)
        if characters < 1:
            raise ValueError("text must not be empty")
        if self.submitted_characters + characters > self.maximum_input_characters:
            raise RuntimeError("tts_character_budget_exhausted")
        self.submitted_characters += characters

    def record_reported_cost(self, characters: int | None) -> None:
        if characters is not None and characters >= 0:
            self.reported_billable_characters += characters

    def can_emit(self, pcm_bytes: int, *, sample_rate_hz: int = 16_000) -> bool:
        seconds = pcm_bytes / 2 / sample_rate_hz
        return self.generated_audio_seconds + seconds <= self.maximum_audio_seconds

    def record_audio(self, pcm_bytes: int, *, sample_rate_hz: int = 16_000) -> None:
        self.generated_audio_seconds += pcm_bytes / 2 / sample_rate_hz

    @property
    def estimated_base_cost_usd(self) -> float:
        billable = self.reported_billable_characters or self.submitted_characters
        return billable / 1_000 * ELEVENLABS_FLASH_USD_PER_1K_CHARACTERS

    def to_dict(self) -> dict[str, float | int]:
        return {
            "submitted_characters": self.submitted_characters,
            "reported_billable_characters": self.reported_billable_characters,
            "generated_audio_seconds": round(self.generated_audio_seconds, 3),
            "estimated_base_cost_usd": round(self.estimated_base_cost_usd, 6),
            "maximum_audio_seconds": self.maximum_audio_seconds,
            "maximum_input_characters": self.maximum_input_characters,
        }


@dataclass(frozen=True, slots=True)
class TTSRequestMetrics:
    text_characters: int
    audio_seconds: float
    first_pcm_latency_ms: float | None
    reported_billable_characters: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "text_characters": self.text_characters,
            "audio_seconds": round(self.audio_seconds, 3),
            "first_pcm_latency_ms": self.first_pcm_latency_ms,
            "reported_billable_characters": self.reported_billable_characters,
        }


@dataclass(frozen=True, slots=True)
class _PcmPacket:
    pcm16: bytes
    final: bool


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: Exception


class ElevenLabsProviderError(RuntimeError):
    """A credential-safe HTTP failure containing only the response status."""

    def __init__(self, status_code: int, provider_code: str | None = None) -> None:
        self.status_code = status_code
        self.provider_code = provider_code
        super().__init__(f"elevenlabs_http_{status_code}")


class ElevenLabsStreamingTTSProvider:
    """Stream PCM16 from ElevenLabs without exposing credentials or voice IDs."""

    sample_rate_hz = 16_000

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = DEFAULT_ELEVENLABS_MODEL,
        budget: SynthesisBudget | None = None,
        first_chunk_ms: int = 140,
        following_chunk_ms: int = 360,
        request_timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip() or not voice_id.strip():
            raise ValueError("ElevenLabs credentials are required")
        if not model.strip():
            raise ValueError("model must not be empty")
        if first_chunk_ms < 20 or following_chunk_ms < 20:
            raise ValueError("stream chunk durations must be at least 20ms")
        if request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        self._api_key = api_key.strip()
        self._voice_id = voice_id.strip()
        self._model = model
        self._budget = budget or SynthesisBudget()
        self._first_chunk_bytes = first_chunk_ms * self.sample_rate_hz * 2 // 1_000
        self._following_chunk_bytes = following_chunk_ms * self.sample_rate_hz * 2 // 1_000
        self._request_timeout_seconds = request_timeout_seconds
        self._client = client
        self._last_request_metrics: TTSRequestMetrics | None = None

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            role="tts",
            provider="elevenlabs-text-to-speech-stream",
            model=self._model,
            mode=ProviderMode.REAL,
            runtime="server",
        )

    @property
    def budget(self) -> SynthesisBudget:
        return self._budget

    @property
    def last_request_metrics(self) -> TTSRequestMetrics | None:
        return self._last_request_metrics

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        self._budget.reserve(request.text)
        queue: asyncio.Queue[_PcmPacket | _StreamFailure | None] = asyncio.Queue(maxsize=12)
        started_ns = monotonic_ns()
        producer = asyncio.create_task(
            self._produce_pcm(request, cancel_event, queue, started_ns),
            name=f"humanflow-elevenlabs-{request.response_id}",
        )
        sequence = request.sequence_start
        first = True
        deadline = asyncio.get_running_loop().time() + self._request_timeout_seconds
        try:
            while not cancel_event.is_set():
                item_task = asyncio.create_task(queue.get())
                cancel_task = asyncio.create_task(cancel_event.wait())
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    item_task.cancel()
                    cancel_task.cancel()
                    await asyncio.gather(
                        item_task, cancel_task, return_exceptions=True
                    )
                    raise TimeoutError("elevenlabs_request_timeout")
                done, _ = await asyncio.wait(
                    {item_task, cancel_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    item_task.cancel()
                    cancel_task.cancel()
                    await asyncio.gather(
                        item_task, cancel_task, return_exceptions=True
                    )
                    raise TimeoutError("elevenlabs_request_timeout")
                if cancel_task in done:
                    item_task.cancel()
                    await asyncio.gather(item_task, return_exceptions=True)
                    break
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)
                item = item_task.result()
                if item is None:
                    break
                if isinstance(item, _StreamFailure):
                    raise item.error
                provider = self.provider_info.to_dict()
                yield AudioChunk(
                    chunk_id=str(uuid4()),
                    response_id=request.response_id,
                    text=request.text if item.final else "",
                    semantic_id=f"{request.response_id}:{request.sequence_start}",
                    semantic_text=request.text,
                    frame=AudioFrame(
                        stream_id=request.response_id,
                        sequence=sequence,
                        pcm16=item.pcm16,
                        sample_rate_hz=self.sample_rate_hz,
                        captured_ns=monotonic_ns(),
                    ),
                    playback_mode=AudioPlaybackMode.PCM,
                    semantic_boundary=item.final,
                    display_text=request.text if first else "",
                    pause_after_ms=request.pause_after_ms if item.final else 0,
                    speaking_rate=request.speaking_rate,
                    provider=provider,
                )
                first = False
                sequence += 1
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _produce_pcm(
        self,
        request: SpeechSynthesisRequest,
        cancel_event: asyncio.Event,
        queue: asyncio.Queue[_PcmPacket | _StreamFailure | None],
        started_ns: int,
    ) -> None:
        try:
            if self._client is None:
                import httpx

                timeout = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    await self._read_response(client, request, cancel_event, queue, started_ns)
            else:
                await self._read_response(
                    self._client, request, cancel_event, queue, started_ns
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await queue.put(_StreamFailure(error))
        finally:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _read_response(
        self,
        client: Any,
        request: SpeechSynthesisRequest,
        cancel_event: asyncio.Event,
        queue: asyncio.Queue[_PcmPacket | _StreamFailure | None],
        started_ns: int,
    ) -> None:
        voice = request.voice or self._voice_id
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
        voice_settings = {
            "stability": request.stability if request.stability is not None else 0.48,
            "similarity_boost": (
                request.similarity_boost if request.similarity_boost is not None else 0.78
            ),
            "style": request.style if request.style is not None else 0.08,
            "use_speaker_boost": (
                request.use_speaker_boost
                if request.use_speaker_boost is not None
                else False
            ),
            "speed": request.speaking_rate,
        }
        body: dict[str, Any] = {
            "text": request.text,
            "model_id": self._model,
            "language_code": request.language_code,
            "voice_settings": voice_settings,
            "apply_text_normalization": "auto",
        }
        if request.previous_text.strip():
            body["previous_text"] = request.previous_text[-1_000:]

        buffer = bytearray()
        first_pcm_ns: int | None = None
        emitted_bytes = 0
        reported_cost: int | None = None
        async with client.stream(
            "POST",
            endpoint,
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
            params={"output_format": DEFAULT_OUTPUT_FORMAT},
            json=body,
        ) as response:
            if response.status_code >= 400:
                raw_error = await response.aread()
                provider_code = _safe_provider_error_code(raw_error)
                raise ElevenLabsProviderError(response.status_code, provider_code)
            raw_cost = response.headers.get("character-cost")
            if isinstance(raw_cost, str) and raw_cost.isdigit():
                reported_cost = int(raw_cost)
                self._budget.record_reported_cost(reported_cost)
            target_bytes = self._first_chunk_bytes
            async for data in response.aiter_bytes():
                if cancel_event.is_set():
                    break
                if not data:
                    continue
                if first_pcm_ns is None:
                    first_pcm_ns = monotonic_ns()
                buffer.extend(data)
                if len(buffer) % 2:
                    continue
                while len(buffer) > target_bytes:
                    packet = bytes(buffer[:target_bytes])
                    del buffer[:target_bytes]
                    if not self._budget.can_emit(len(packet), sample_rate_hz=self.sample_rate_hz):
                        raise RuntimeError("tts_audio_budget_exhausted")
                    self._budget.record_audio(len(packet), sample_rate_hz=self.sample_rate_hz)
                    emitted_bytes += len(packet)
                    await queue.put(_PcmPacket(packet, final=False))
                    target_bytes = self._following_chunk_bytes

        if not cancel_event.is_set() and buffer:
            if len(buffer) % 2:
                buffer.pop()
            if buffer:
                packet = bytes(buffer)
                if not self._budget.can_emit(len(packet), sample_rate_hz=self.sample_rate_hz):
                    raise RuntimeError("tts_audio_budget_exhausted")
                self._budget.record_audio(len(packet), sample_rate_hz=self.sample_rate_hz)
                emitted_bytes += len(packet)
                await queue.put(_PcmPacket(packet, final=True))
        if not cancel_event.is_set() and emitted_bytes == 0:
            raise RuntimeError("tts_provider_returned_empty_audio")

        self._last_request_metrics = TTSRequestMetrics(
            text_characters=len(request.text),
            audio_seconds=emitted_bytes / 2 / self.sample_rate_hz,
            first_pcm_latency_ms=(
                None
                if first_pcm_ns is None
                else max(0.0, (first_pcm_ns - started_ns) / 1_000_000.0)
            ),
            reported_billable_characters=reported_cost,
        )


class FallbackStreamingTTSProvider:
    """Use a visible fallback only if the primary fails before emitting audio."""

    def __init__(
        self,
        primary: StreamingTTSProvider,
        fallback: StreamingTTSProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active_provider_info = primary.provider_info  # type: ignore[attr-defined]
        self._last_fallback_reason: str | None = None
        self._response_provider: dict[str, str] = {}

    @property
    def provider_info(self) -> ProviderInfo:
        return self._primary.provider_info  # type: ignore[attr-defined]

    @property
    def fallback_info(self) -> ProviderInfo:
        return self._fallback.provider_info  # type: ignore[attr-defined]

    @property
    def active_provider_info(self) -> ProviderInfo:
        return self._active_provider_info

    @property
    def last_fallback_reason(self) -> str | None:
        return self._last_fallback_reason

    async def stream_speech(
        self,
        request: SpeechSynthesisRequest,
        *,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[AudioChunk]:
        primary_name = self.provider_info.provider
        fallback_name = self.fallback_info.provider
        pinned_provider = self._response_provider.get(request.response_id)
        if pinned_provider == fallback_name:
            self._active_provider_info = self.fallback_info
            async for chunk in self._fallback.stream_speech(
                request, cancel_event=cancel_event
            ):
                yield chunk
            return
        yielded = False
        self._last_fallback_reason = None
        self._active_provider_info = self.provider_info
        try:
            async for chunk in self._primary.stream_speech(
                request, cancel_event=cancel_event
            ):
                yielded = True
                self._pin_response_provider(request.response_id, primary_name)
                yield chunk
            return
        except Exception as error:
            if yielded or cancel_event.is_set():
                raise
            if pinned_provider == primary_name:
                raise
            if (
                isinstance(error, ElevenLabsProviderError)
                and 400 <= error.status_code < 500
                and error.status_code != 429
            ):
                raise
            self._last_fallback_reason = type(error).__name__
        self._active_provider_info = self.fallback_info
        async for chunk in self._fallback.stream_speech(
            request, cancel_event=cancel_event
        ):
            self._pin_response_provider(request.response_id, fallback_name)
            yield chunk

    def _pin_response_provider(self, response_id: str, provider_name: str) -> None:
        existing = self._response_provider.get(response_id)
        if existing is not None and existing != provider_name:
            raise RuntimeError("tts_provider_changed_within_response")
        self._response_provider[response_id] = provider_name
        if len(self._response_provider) > 128:
            oldest = next(iter(self._response_provider))
            del self._response_provider[oldest]


def _safe_provider_error_code(raw_error: bytes) -> str | None:
    """Extract only a bounded machine code, never provider error messages or IDs."""

    try:
        payload = json.loads(raw_error)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    code = (
        detail.get("code") or detail.get("status")
        if isinstance(detail, dict)
        else None
    )
    if not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", code):
        return None
    return code
