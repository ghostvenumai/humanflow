from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from humanflow.domain.conversation import OperationToken
from humanflow.runtime.anthropic_provider import DEFAULT_SYSTEM_PROMPT, AnthropicReasoner
from humanflow.runtime.elevenlabs_provider import FallbackStreamingTTSProvider
from humanflow.runtime.providers import ProviderMode
from humanflow.web.app import load_demo_runtime_config


class FakeMessageStream:
    def __init__(self, deltas: list[str], *, input_tokens: int, output_tokens: int) -> None:
        self._deltas = deltas
        self._usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def __aenter__(self) -> "FakeMessageStream":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        for delta in self._deltas:
            yield delta

    async def get_final_message(self) -> Any:
        return SimpleNamespace(usage=self._usage)


class FakeMessages:
    def __init__(self, responses: list[list[str]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeMessageStream:
        self.calls.append(kwargs)
        return FakeMessageStream(
            self.responses.pop(0),
            input_tokens=21,
            output_tokens=13,
        )


class FakeClient:
    def __init__(self, responses: list[list[str]]) -> None:
        self.messages = FakeMessages(responses)


def _token(operation_id: str) -> OperationToken:
    return OperationToken(
        conversation_id="browser-live-test",
        epoch=0,
        operation_id=operation_id,
        kind="reasoning_and_speech",
    )


async def _collect(reasoner: AnthropicReasoner, text: str, turn: int) -> str:
    chunks = [chunk async for chunk in reasoner.stream_response(text, _token(str(turn)))]
    return " ".join(chunks)


def test_real_reasoner_retains_multi_turn_context_and_semantic_output() -> None:
    async def scenario() -> None:
        client = FakeClient(
            [
                ["Ein ERP-System bündelt zentrale Geschäftsprozesse. ", "Es schafft eine gemeinsame Datenbasis."],
                ["Damit meine ich die gemeinsame Datenbasis aus der ersten Antwort."],
            ]
        )
        reasoner = AnthropicReasoner(
            api_key="test-key-never-sent",
            model="claude-test-real-adapter",
            client=client,
        )

        first = await _collect(reasoner, "Was ist ein ERP-System?", 1)
        second = await _collect(reasoner, "Was meinst du mit Datenbasis?", 2)

        assert first.startswith("Ein ERP-System")
        assert second.startswith("Damit meine ich")
        second_request = client.messages.calls[1]["messages"]
        assert second_request == [
            {"role": "user", "content": "Was ist ein ERP-System?"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Was meinst du mit Datenbasis?"},
        ]
        assert [message["role"] for message in reasoner.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert reasoner.provider_info.mode is ProviderMode.REAL
        assert reasoner.last_usage is not None
        assert reasoner.last_usage.to_dict() == {"input_tokens": 21, "output_tokens": 13}

    asyncio.run(scenario())


def test_demo_fails_closed_without_real_reasoning_credentials() -> None:
    runtime = load_demo_runtime_config({})

    assert not runtime.ready
    assert runtime.reasoner_factory is None
    reasoning = next(item for item in runtime.providers if item.info.role == "reasoning")
    assert reasoning.info.mode is ProviderMode.REAL
    assert reasoning.availability == "MISSING_API_KEY"


def test_demo_factory_can_only_build_declared_real_reasoner() -> None:
    runtime = load_demo_runtime_config(
        {
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "ELEVENLABS_API_KEY": "test-eleven-key-never-sent",
            "ELEVENLABS_VOICE_ID": "test-voice-id-never-sent",
        }
    )

    assert runtime.ready
    assert runtime.reasoner_factory is not None
    assert runtime.transcriber_factory is not None
    transcriber = runtime.transcriber_factory()
    reasoner = runtime.reasoner_factory()
    assert runtime.synthesizer_factory is not None
    synthesizer = runtime.synthesizer_factory()
    assert isinstance(reasoner, AnthropicReasoner)
    assert transcriber.provider_info.provider == "elevenlabs-scribe-realtime"
    assert transcriber.provider_info.mode is ProviderMode.REAL
    assert isinstance(synthesizer, FallbackStreamingTTSProvider)
    assert runtime.tts_candidate_factory is not None
    candidate = runtime.tts_candidate_factory()
    assert candidate.provider_info.model == "eleven_v3_conversational"
    assert reasoner.provider_info.mode is ProviderMode.REAL
    browser_diagnostic = next(
        status for status in runtime.providers if status.info.role == "stt-browser-diagnostic"
    )
    assert browser_diagnostic.info.mode is ProviderMode.MOCK
    assert browser_diagnostic.availability == "OFF_PRODUCTION"
    tts_candidate = next(
        status for status in runtime.providers if status.info.role == "tts-ab-candidate"
    )
    assert tts_candidate.info.model == "eleven_v3_conversational"
    assert tts_candidate.availability == "CONFIGURED_UNVERIFIED"


def test_demo_accepts_dedicated_scribe_key_without_exposing_it() -> None:
    runtime = load_demo_runtime_config(
        {
            "ANTHROPIC_API_KEY": "reasoning-test-key",
            "ELEVENLABS_API_KEY": "tts-test-key",
            "ELEVENLABS_STT_API_KEY": "stt-test-key",
            "ELEVENLABS_VOICE_ID": "voice-test-id",
        }
    )

    assert runtime.ready
    stt = next(status for status in runtime.providers if status.info.role == "stt")
    assert stt.availability == "CONFIGURED_UNVERIFIED"
    assert "stt-test-key" not in str(runtime.provider_payload())


def test_dedicated_scribe_key_does_not_hide_missing_tts_key() -> None:
    runtime = load_demo_runtime_config(
        {
            "ANTHROPIC_API_KEY": "reasoning-test-key",
            "ELEVENLABS_STT_API_KEY": "stt-test-key",
            "ELEVENLABS_VOICE_ID": "voice-test-id",
        }
    )

    assert not runtime.ready
    assert runtime.blocker == "missing TTS configuration: ELEVENLABS_API_KEY"


def test_demo_fails_closed_when_high_quality_tts_configuration_is_missing() -> None:
    runtime = load_demo_runtime_config({"ANTHROPIC_API_KEY": "test-key-never-sent"})

    assert not runtime.ready
    assert runtime.synthesizer_factory is None
    tts = next(item for item in runtime.providers if item.info.role == "tts")
    assert tts.info.provider == "elevenlabs-text-to-speech-stream"
    assert tts.info.mode is ProviderMode.REAL
    assert tts.availability == "MISSING_CONFIGURATION"


def test_unsupported_provider_does_not_fall_back_to_acknowledgement_stub() -> None:
    runtime = load_demo_runtime_config(
        {
            "HUMANFLOW_REASONING_PROVIDER": "echo",
            "ANTHROPIC_API_KEY": "test-key-never-sent",
        }
    )

    assert not runtime.ready
    assert runtime.reasoner_factory is None
    assert runtime.blocker == "unsupported reasoning provider: echo"


def test_live_prompt_requires_ai_disclosure_and_rejects_canned_acknowledgement() -> None:
    assert "Ich bin HumanFlow, ein KI-Assistent" in DEFAULT_SYSTEM_PROMPT
    assert "niemals behaupten, ein Mensch zu sein" in DEFAULT_SYSTEM_PROMPT
    assert "Beginne nicht mit einer generischen Empfangsbestätigung" in DEFAULT_SYSTEM_PROMPT
    assert "Ich habe Sie verstanden" in DEFAULT_SYSTEM_PROMPT
    assert "ein bis zwei kurzen Sätzen" in DEFAULT_SYSTEM_PROMPT
    assert "stellst genau eine relevante nächste" in DEFAULT_SYSTEM_PROMPT
    assert "ungefragten Erklärungen oder langen Disclaimer" in DEFAULT_SYSTEM_PROMPT
    assert "Rechenresultate gibst du immer als Ziffern" in DEFAULT_SYSTEM_PROMPT
    assert "„Das Ergebnis ist 425.“" in DEFAULT_SYSTEM_PROMPT


def test_authoritative_transaction_context_does_not_rewrite_provider_history() -> None:
    async def scenario() -> None:
        client = FakeClient([["Donnerstag bleibt bestehen. ", "Fünfzehn Uhr passt."]])
        reasoner = AnthropicReasoner(
            api_key="test-key-never-sent",
            model="claude-test-real-adapter",
            client=client,
        )
        context = (
            '{"appointment_state":{"date":"2026-09-10","time":"15:00"},'
            '"updated_slots_this_user_turn":["time"]}'
        )
        reasoner.set_authoritative_transaction_context(context)

        await _collect(reasoner, "Vielleicht 15 Uhr.", 1)

        call = client.messages.calls[0]
        assert context in call["system"]
        assert "Ältere widersprüchliche Werte" in call["system"]
        assert "Frage nie erneut nach einem bereits gesetzten Slot" in call["system"]
        assert "musst du ausdrücklich das Wort „Termin“ verwenden" in call["system"]
        assert "höchstens eine Frage zum nächsten" in call["system"]
        assert call["messages"] == [
            {"role": "user", "content": "Vielleicht 15 Uhr."}
        ]
        assert reasoner.history[0]["content"] == "Vielleicht 15 Uhr."

    asyncio.run(scenario())
