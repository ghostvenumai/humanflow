from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from humanflow.domain.conversation import OperationToken
from humanflow.runtime.anthropic_provider import AnthropicReasoner
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
    runtime = load_demo_runtime_config({"ANTHROPIC_API_KEY": "test-key-never-sent"})

    assert runtime.ready
    assert runtime.reasoner_factory is not None
    reasoner = runtime.reasoner_factory()
    assert isinstance(reasoner, AnthropicReasoner)
    assert reasoner.provider_info.mode is ProviderMode.REAL
    assert all(status.info.mode is ProviderMode.REAL for status in runtime.providers)


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
