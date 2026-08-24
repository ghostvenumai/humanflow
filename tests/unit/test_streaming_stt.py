from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

import pytest

from humanflow.audio.models import AudioFrame
from humanflow.runtime.elevenlabs_stt_provider import (
    ElevenLabsRealtimeSTTError,
    ElevenLabsRealtimeSTTProvider,
)
from humanflow.runtime.providers import ProviderMode
from humanflow.runtime.transcript_events import (
    ConversationEventKind,
    TranscriptOrigin,
)


_CLOSED = object()


class FakeRealtimeConnection:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | object] = asyncio.Queue()
        self.sent: list[dict[str, object]] = []
        self.closed = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        item = await self.incoming.get()
        if item is _CLOSED:
            raise StopAsyncIteration
        assert isinstance(item, str)
        return item

    async def send(self, payload: str) -> None:
        parsed = json.loads(payload)
        assert isinstance(parsed, dict)
        self.sent.append(parsed)

    async def close(self) -> None:
        self.closed = True
        self.incoming.put_nowait(_CLOSED)

    def emit(self, payload: dict[str, object]) -> None:
        self.incoming.put_nowait(json.dumps(payload))


def _frame(sequence: int, pcm16: bytes = b"\x01\x00" * 1_600) -> AudioFrame:
    return AudioFrame(
        stream_id="browser-track-1",
        sequence=sequence,
        pcm16=pcm16,
        sample_rate_hz=16_000,
    )


def test_scribe_streams_exact_bound_pcm_and_separates_partial_from_committed() -> None:
    async def scenario() -> None:
        connection = FakeRealtimeConnection()
        connection.emit(
            {
                "message_type": "session_started",
                "session_id": "scribe-session-1",
            }
        )
        connect_calls: list[tuple[str, dict[str, object]]] = []

        async def connect(uri: str, **kwargs: object) -> FakeRealtimeConnection:
            connect_calls.append((uri, kwargs))
            return connection

        provider = ElevenLabsRealtimeSTTProvider(
            api_key="test-key-never-logged",
            connect_factory=connect,
        )
        provider.bind_audio_source(
            audio_capture_id="capture-1",
            stream_id="browser-track-1",
        )

        assert await provider.ingest(_frame(0)) == ()
        connection.emit(
            {"message_type": "partial_transcript", "text": "Moment, stopp"}
        )
        await asyncio.sleep(0)
        partials = await provider.ingest(_frame(1))
        assert len(partials) == 1
        partial = partials[0]
        assert not partial.is_final
        assert partial.provenance.event_kind is ConversationEventKind.USER_TRANSCRIPT_PARTIAL
        assert partial.provenance.origin is TranscriptOrigin.STREAMING_STT_PROVIDER
        assert partial.provenance.audio_capture_id == "capture-1"
        assert partial.provenance.stream_id == "browser-track-1"
        assert partial.provenance.stt_session_id == "scribe-session-1"
        assert partial.provenance.audio_frame_sequence == 0
        assert partial.signals.interruption_probability == 0.98

        connection.emit(
            {
                "message_type": "committed_transcript",
                "text": "Moment, stopp. Was ist sieben mal acht?",
            }
        )
        await asyncio.sleep(0)
        finals = await provider.ingest(_frame(2))
        assert len(finals) == 1
        final = finals[0]
        assert final.is_final
        assert final.provenance.event_kind is ConversationEventKind.USER_TRANSCRIPT_FINAL
        assert final.signals.provider_endpointed
        assert final.signals.semantic_complete

        sent_pcm = base64.b64decode(str(connection.sent[0]["audio_base_64"]))
        assert sent_pcm == _frame(0).pcm16
        assert connection.sent[0]["message_type"] == "input_audio_chunk"
        uri, kwargs = connect_calls[0]
        assert "model_id=scribe_v2_realtime" in uri
        assert "audio_format=pcm_16000" in uri
        assert "language_code=de" in uri
        assert "commit_strategy=vad" in uri
        assert kwargs["additional_headers"] == {
            "xi-api-key": "test-key-never-logged"
        }
        assert provider.provider_info.mode is ProviderMode.REAL
        assert provider.capabilities.source_bound
        await provider.close()
        assert connection.closed

    asyncio.run(scenario())


def test_scribe_coalesces_many_partials_and_never_promotes_them_to_final() -> None:
    async def scenario() -> None:
        connection = FakeRealtimeConnection()
        connection.emit(
            {"message_type": "session_started", "session_id": "scribe-session-2"}
        )

        async def connect(*args: object, **kwargs: object) -> FakeRealtimeConnection:
            del args, kwargs
            return connection

        provider = ElevenLabsRealtimeSTTProvider(
            api_key="test-key-never-logged", connect_factory=connect
        )
        provider.bind_audio_source(
            audio_capture_id="capture-2", stream_id="browser-track-1"
        )
        await provider.ingest(_frame(0))
        for text in ("Was", "Was ist", "Was ist fünfundzwanzig"):
            connection.emit({"message_type": "partial_transcript", "text": text})
        await asyncio.sleep(0)

        updates = await provider.ingest(_frame(1))

        assert len(updates) == 1
        assert updates[0].text == "Was ist fünfundzwanzig"
        assert not updates[0].is_final
        assert not updates[0].signals.provider_endpointed
        await provider.close()

    asyncio.run(scenario())


def test_scribe_rejects_unbound_duplicate_and_wrong_format_pcm() -> None:
    async def scenario() -> None:
        provider = ElevenLabsRealtimeSTTProvider(
            api_key="test-key-never-logged",
            connect_factory=lambda *args, **kwargs: None,  # type: ignore[arg-type]
        )
        with pytest.raises(RuntimeError, match="stt_audio_source_not_bound"):
            await provider.ingest(_frame(0))
        provider.bind_audio_source(
            audio_capture_id="capture-3", stream_id="browser-track-1"
        )
        with pytest.raises(RuntimeError, match="stt_received_unbound_pcm_stream"):
            await provider.ingest(
                AudioFrame(
                    stream_id="other-track",
                    sequence=0,
                    pcm16=b"\x00\x00" * 160,
                )
            )

    asyncio.run(scenario())


def test_scribe_surfaces_only_safe_provider_error_codes() -> None:
    error = ElevenLabsRealtimeSTTError("quota_exceeded")

    assert error.provider_code == "quota_exceeded"
    assert str(error) == "elevenlabs_stt_quota_exceeded"
    assert "key" not in str(error)


def test_scribe_auth_failure_is_fast_and_close_does_not_hang() -> None:
    async def scenario() -> None:
        connection = FakeRealtimeConnection()
        connection.emit({"message_type": "auth_error", "error": "secret omitted"})

        async def connect(*args: object, **kwargs: object) -> FakeRealtimeConnection:
            del args, kwargs
            return connection

        provider = ElevenLabsRealtimeSTTProvider(
            api_key="test-key-never-logged", connect_factory=connect
        )
        provider.bind_audio_source(
            audio_capture_id="capture-auth", stream_id="browser-track-1"
        )

        with pytest.raises(ElevenLabsRealtimeSTTError, match="auth_error"):
            await provider.start()
        await asyncio.wait_for(provider.close(), timeout=0.2)
        assert connection.closed

    asyncio.run(scenario())
