from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from humanflow.audio.models import AudioChunk, AudioFrame, AudioPlaybackMode
from humanflow.runtime.elevenlabs_provider import (
    ElevenLabsProviderError,
    ElevenLabsStreamingTTSProvider,
    FallbackStreamingTTSProvider,
    SynthesisBudget,
    _safe_provider_error_code,
)
from humanflow.runtime.prosody import ProsodyPlanner, SpeechIntent
from humanflow.runtime.providers import (
    BrowserSpeechSynthesisAdapter,
    GaplessSegmentTTSProvider,
    ProviderInfo,
    ProviderMode,
    SpeechSynthesisRequest,
)


class FakeResponse:
    def __init__(self, packets: list[bytes], *, character_cost: str = "12") -> None:
        self._packets = packets
        self.headers = {"character-cost": character_cost}
        self.status_code = 200

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for packet in self._packets:
            await asyncio.sleep(0)
            yield packet

    async def aread(self) -> bytes:
        return b""


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, endpoint: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "endpoint": endpoint, **kwargs})
        return self.response


class HangingResponse(FakeResponse):
    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        yield b"\x01\x00" * 3_000
        await asyncio.Event().wait()


def _request(text: str = "Ja, klar. Freitag passt.") -> SpeechSynthesisRequest:
    return SpeechSynthesisRequest(
        text=text,
        response_id="response-streaming",
        sequence_start=4,
        speaking_rate=1.02,
        stability=0.42,
        similarity_boost=0.78,
        style=0.14,
        use_speaker_boost=False,
        pause_after_ms=220,
        intent="confirmation",
    )


def test_elevenlabs_streams_pcm_and_only_final_chunk_advances_text_ledger() -> None:
    async def scenario() -> None:
        pcm = b"\x01\x00" * 5_000
        client = FakeClient(FakeResponse([pcm[:3_000], pcm[3_000:]]))
        budget = SynthesisBudget(maximum_audio_seconds=10, maximum_input_characters=100)
        provider = ElevenLabsStreamingTTSProvider(
            api_key="test-key-never-sent",
            voice_id="test-voice-never-logged",
            client=client,
            budget=budget,
        )

        chunks = [
            chunk
            async for chunk in provider.stream_speech(
                _request(), cancel_event=asyncio.Event()
            )
        ]

        assert len(chunks) == 2
        assert chunks[0].playback_mode is AudioPlaybackMode.PCM
        assert chunks[0].text == ""
        assert not chunks[0].semantic_boundary
        assert chunks[0].display_text == _request().text
        assert chunks[1].text == _request().text
        assert chunks[1].semantic_boundary
        assert chunks[1].pause_after_ms == 220
        assert [chunk.frame.sequence for chunk in chunks] == [4, 5]
        call = client.calls[0]
        assert call["method"] == "POST"
        assert call["params"] == {"output_format": "pcm_16000"}
        assert call["json"]["model_id"] == "eleven_flash_v2_5"
        assert call["json"]["language_code"] == "de"
        assert call["json"]["voice_settings"]["speed"] == 1.02
        assert budget.reported_billable_characters == 12
        assert budget.generated_audio_seconds == pytest.approx(0.3125)
        assert provider.provider_info.to_dict() == {
            "role": "tts",
            "provider": "elevenlabs-text-to-speech-stream",
            "model": "eleven_flash_v2_5",
            "mode": "REAL",
            "runtime": "server",
        }

    asyncio.run(scenario())


def test_gapless_wrapper_turns_raw_provider_packets_into_one_playback_source() -> None:
    async def scenario() -> None:
        pcm = b"\x01\x00" * 5_000
        raw_provider = ElevenLabsStreamingTTSProvider(
            api_key="test-key-never-sent",
            voice_id="test-voice-never-logged",
            client=FakeClient(FakeResponse([pcm[:3_000], pcm[3_000:]])),
            budget=SynthesisBudget(
                maximum_audio_seconds=10,
                maximum_input_characters=100,
            ),
        )
        provider = GaplessSegmentTTSProvider(raw_provider)

        chunks = [
            chunk
            async for chunk in provider.stream_speech(
                _request(), cancel_event=asyncio.Event()
            )
        ]

        assert len(chunks) == 1
        assert chunks[0].frame.pcm16 == pcm
        assert chunks[0].frame.sequence == 4
        assert chunks[0].text == _request().text
        assert chunks[0].provider["delivery"] == "gapless-semantic-buffer"
        assert chunks[0].provider["source_chunk_count"] == "2"
        assert chunks[0].provider["source_sequence_first"] == "4"
        assert chunks[0].provider["source_sequence_last"] == "5"

    asyncio.run(scenario())


def test_gapless_wrapper_rejects_duplicate_raw_sequence_before_playback() -> None:
    class DuplicateSequenceProvider:
        provider_info = ProviderInfo(
            role="tts",
            provider="duplicate-test-provider",
            model="test",
            mode=ProviderMode.MOCK,
            runtime="test",
        )

        async def stream_speech(
            self, request: SpeechSynthesisRequest, *, cancel_event: asyncio.Event
        ) -> AsyncIterator[AudioChunk]:
            del cancel_event
            for chunk_id in ("chunk-a", "chunk-b"):
                yield AudioChunk(
                    chunk_id=chunk_id,
                    response_id=request.response_id,
                    text="",
                    semantic_id="semantic-a",
                    semantic_text=request.text,
                    semantic_boundary=False,
                    frame=AudioFrame(
                        stream_id=request.response_id,
                        sequence=request.sequence_start,
                        pcm16=b"\x00\x00" * 16,
                    ),
                    provider=self.provider_info.to_dict(),
                )

    async def scenario() -> None:
        provider = GaplessSegmentTTSProvider(DuplicateSequenceProvider())
        with pytest.raises(RuntimeError, match="duplicate_tts_chunk_sequence"):
            async for _ in provider.stream_speech(
                _request(), cancel_event=asyncio.Event()
            ):
                pass

    asyncio.run(scenario())


def test_primary_failure_before_audio_uses_explicit_real_browser_fallback() -> None:
    class FailingPrimary:
        provider_info = ProviderInfo(
            role="tts",
            provider="primary-real",
            model="primary-model",
            mode=ProviderMode.REAL,
            runtime="server",
        )

        async def stream_speech(
            self, request: SpeechSynthesisRequest, *, cancel_event: asyncio.Event
        ) -> AsyncIterator[Any]:
            del request, cancel_event
            raise RuntimeError("provider_unavailable")
            yield  # pragma: no cover

    async def scenario() -> None:
        provider = FallbackStreamingTTSProvider(
            primary=FailingPrimary(),
            fallback=BrowserSpeechSynthesisAdapter(),
        )
        chunks = [
            chunk
            async for chunk in provider.stream_speech(
                _request("Kurzer Fallback."), cancel_event=asyncio.Event()
            )
        ]

        assert len(chunks) == 1
        assert chunks[0].playback_mode is AudioPlaybackMode.BROWSER_SPEECH
        assert chunks[0].provider["provider"] == "browser-web-speech-api"
        assert provider.active_provider_info.provider == "browser-web-speech-api"
        assert provider.last_fallback_reason == "RuntimeError"

    asyncio.run(scenario())


def test_incomplete_provider_stream_times_out_before_playback_and_uses_fallback() -> None:
    async def scenario() -> None:
        primary = GaplessSegmentTTSProvider(
            ElevenLabsStreamingTTSProvider(
                api_key="test-key-never-sent",
                voice_id="test-voice-never-logged",
                client=FakeClient(HangingResponse([])),
                request_timeout_seconds=0.01,
            )
        )
        provider = FallbackStreamingTTSProvider(
            primary=primary,
            fallback=BrowserSpeechSynthesisAdapter(),
        )

        chunks = [
            chunk
            async for chunk in provider.stream_speech(
                _request("Zeitlich begrenzter Fallback."),
                cancel_event=asyncio.Event(),
            )
        ]

        assert len(chunks) == 1
        assert chunks[0].playback_mode is AudioPlaybackMode.BROWSER_SPEECH
        assert chunks[0].provider["provider"] == "browser-web-speech-api"
        assert provider.last_fallback_reason == "TimeoutError"

    asyncio.run(scenario())


def test_configuration_error_fails_closed_instead_of_hiding_behind_fallback() -> None:
    class InvalidVoicePrimary:
        provider_info = ProviderInfo(
            role="tts",
            provider="primary-real",
            model="primary-model",
            mode=ProviderMode.REAL,
            runtime="server",
        )

        async def stream_speech(
            self, request: SpeechSynthesisRequest, *, cancel_event: asyncio.Event
        ) -> AsyncIterator[Any]:
            del request, cancel_event
            raise ElevenLabsProviderError(404, "voice_not_found")
            yield  # pragma: no cover

    async def scenario() -> None:
        provider = FallbackStreamingTTSProvider(
            primary=InvalidVoicePrimary(),
            fallback=BrowserSpeechSynthesisAdapter(),
        )
        with pytest.raises(ElevenLabsProviderError) as captured:
            async for _ in provider.stream_speech(
                _request(), cancel_event=asyncio.Event()
            ):
                pass
        assert captured.value.status_code == 404
        assert provider.last_fallback_reason is None

    asyncio.run(scenario())


def test_prosody_planner_is_deterministic_and_does_not_rewrite_words() -> None:
    planner = ProsodyPlanner(maximum_segment_characters=80)
    text = "Ja, klar. Freitag passt. Soll ich zehn Uhr vormerken?"

    first = planner.plan(text)
    second = planner.plan(text)

    assert first == second
    assert " ".join(segment.text for segment in first) == text
    assert [segment.intent for segment in first] == [
        SpeechIntent.CONFIRMATION,
        SpeechIntent.INFORMATION,
        SpeechIntent.QUESTION,
    ]
    assert all(0.5 <= segment.speaking_rate <= 2.0 for segment in first)


def test_budget_stops_before_an_unbounded_provider_request() -> None:
    budget = SynthesisBudget(maximum_audio_seconds=10, maximum_input_characters=10)
    budget.reserve("1234567890")

    with pytest.raises(RuntimeError, match="tts_character_budget_exhausted"):
        budget.reserve("x")


def test_provider_error_parser_prefers_specific_safe_code() -> None:
    raw = b'{"detail":{"code":"insufficient_credits","status":"payment_required"}}'

    assert _safe_provider_error_code(raw) == "insufficient_credits"
