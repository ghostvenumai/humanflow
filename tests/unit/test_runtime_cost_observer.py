from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

from humanflow.audio.ledger import LedgerEntrySnapshot, LedgerState
from humanflow.cost import (
    AsyncCostRecorder,
    CostLedger,
    CostSource,
    PricingCatalog,
    PricingRule,
    RuntimeCostObserver,
    ServiceType,
    aggregate_cost_rows,
)
from humanflow.domain.conversation import OperationToken
from humanflow.runtime.anthropic_provider import ReasoningUsage
from humanflow.runtime.providers import (
    NullTranscriber,
    ProviderInfo,
    ProviderMode,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.turns.models import TurnSignals


def _catalog() -> PricingCatalog:
    return PricingCatalog(
        [
            PricingRule(
                pricing_rule_id="test-tts-v1",
                pricing_version="v1",
                provider="elevenlabs-text-to-speech-stream",
                model="eleven_flash_v2_5",
                service=ServiceType.TTS,
                unit="per_1k_characters",
                output_rate=Decimal("0.05"),
                currency="USD",
                effective_date="2026-08-25",
                verified_at="deterministic-test-fixture",
                source_note="TEST ONLY",
                active=True,
            ),
            PricingRule(
                pricing_rule_id="test-local-tool",
                pricing_version="v1",
                provider="local-sqlite",
                model="humanflow-appointments-v1",
                service=ServiceType.TOOL,
                unit="per_request",
                input_rate=Decimal("0"),
                currency="EUR",
                effective_date="2026-08-25",
                verified_at="deterministic-test-fixture",
                source_note="TEST ONLY",
                active=True,
            ),
        ]
    )


def _entry(*, played_samples: int) -> LedgerEntrySnapshot:
    return LedgerEntrySnapshot(
        chunk_id="chunk-1",
        response_id="response-1",
        text="Zehn Sekunden Test.",
        semantic_id="semantic-1",
        semantic_text="Zehn Sekunden Test.",
        total_samples=160_000,
        played_samples=played_samples,
        sample_rate_hz=16_000,
        generated_ns=1,
        queued_ns=2,
        playback_started_ns=3,
        playback_stopped_ns=4,
        cancelled_ns=4 if played_samples < 160_000 else None,
        state=LedgerState.CANCELLED if played_samples < 160_000 else LedgerState.PLAYED,
    )


def test_played_audio_economics_preserves_full_provider_cost_and_estimates_waste(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        ledger = CostLedger(tmp_path / "costs.sqlite3")
        recorder = AsyncCostRecorder(ledger)
        observer = RuntimeCostObserver(
            session_id="session-1",
            conversation_id="conversation-1",
            recorder=recorder,
            pricing=_catalog(),
        )
        observer.record_tts(
            operation_id="tts-1",
            turn_id="turn-1",
            response_id="response-1",
            provider="elevenlabs-text-to-speech-stream",
            model="eleven_flash_v2_5",
            characters=1_000,
            audio_seconds=Decimal("10"),
            reported_billable_characters=1_000,
            fallback=False,
            cancelled=True,
            latency_ms=90.0,
        )
        observer.record_played_audio(
            operation_id="playback-1",
            turn_id="turn-1",
            response_id="response-1",
            entries=[_entry(played_samples=64_000)],
        )
        await observer.close()

        rows = ledger.rows(session_id="session-1")
        generated = next(row for row in rows if row["operation"] == "streaming_synthesis")
        allocation = next(row for row in rows if row["operation"] == "played_audio_allocation")
        assert generated["estimated_cost_micros"] == 50_000
        assert allocation["played_fraction"] == "0.4"
        assert allocation["heard_units"] == "4"
        assert allocation["unheard_units"] == "6"
        assert allocation["wasted_cost_estimate_micros"] == 30_000
        summary = aggregate_cost_rows(rows, session_id="session-1")
        assert summary["estimated_cost"]["value"] == "0.05"
        assert summary["played_audio_economics"]["wasted_cost_estimate"] == "0.03"

    asyncio.run(scenario())


def test_primary_failure_and_fallback_are_separate_without_duplicate_billing(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        ledger = CostLedger(tmp_path / "costs.sqlite3")
        observer = RuntimeCostObserver(
            session_id="session-1",
            conversation_id="conversation-1",
            recorder=AsyncCostRecorder(ledger),
            pricing=_catalog(),
        )
        observer.record_primary_tts_failure(
            operation_id="primary-failure",
            turn_id="turn-1",
            response_id="response-1",
            provider="elevenlabs-text-to-speech-stream",
            model="eleven_flash_v2_5",
            failure_class="TimeoutError",
        )
        observer.record_tts(
            operation_id="fallback-success",
            turn_id="turn-1",
            response_id="response-1",
            provider="browser-web-speech-api",
            model="system-voice",
            characters=40,
            audio_seconds=Decimal("2.5"),
            reported_billable_characters=None,
            fallback=True,
            cancelled=False,
            latency_ms=None,
        )
        await observer.close()

        rows = ledger.rows(session_id="session-1")
        assert len(rows) == 2
        assert [row["service_type"] for row in rows] == ["TTS", "FALLBACK_TTS"]
        assert rows[0]["estimated_cost_micros"] is None
        assert rows[1]["estimated_cost_micros"] is None
        assert rows[1]["fallback"] == 1

    asyncio.run(scenario())


def test_local_appointment_tool_sequence_has_zero_external_cost_and_visible_failures(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        ledger = CostLedger(tmp_path / "costs.sqlite3")
        observer = RuntimeCostObserver(
            session_id="session-1",
            conversation_id="conversation-1",
            recorder=AsyncCostRecorder(ledger),
            pricing=_catalog(),
        )
        for index, (tool_name, success) in enumerate(
            (
                ("search_availability", True),
                ("create_appointment", True),
                ("list_appointments", True),
                ("reschedule_appointment", True),
                ("list_appointments", True),
                ("cancel_appointment", True),
                ("list_appointments", True),
                ("create_appointment", False),
            )
        ):
            observer.record_tool(
                operation_id=f"tool-{index}",
                turn_id=f"turn-{index}",
                tool_name=tool_name,
                duration_ms=5.0,
                success=success,
                retry=False,
                appointment_id="appointment-1",
                transaction_result="BOOKED" if success else "BOOKING_CONFLICT",
                failure_class=None,
            )
        await observer.close()
        rows = ledger.rows(session_id="session-1")
        assert all(row["cost_source"] == CostSource.LOCAL_TOOL_COST.value for row in rows)
        assert all(row["actual_cost_micros"] == 0 for row in rows)
        summary = aggregate_cost_rows(rows, session_id="session-1")
        assert summary["tools"]["call_count"] == 8
        assert summary["tools"]["successful_actions"] == 7
        assert summary["tools"]["failed_actions"] == 1

    asyncio.run(scenario())


class _UsageReasoner:
    provider_info = ProviderInfo(
        role="reasoning",
        provider="anthropic-messages-api",
        model="claude-test",
        mode=ProviderMode.MOCK,
        runtime="test",
    )

    def __init__(self) -> None:
        self.last_usage = ReasoningUsage(input_tokens=12, output_tokens=4)

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del transcript, token
        yield "Kurze Antwort."


def _complete() -> TranscriptUpdate:
    return TranscriptUpdate(
        text="Was kannst du?",
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(final=True),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=400,
            utterance_duration_ms=800,
            semantic_complete=True,
            acoustic_completion=0.9,
        ),
    )


def test_cost_database_failure_never_breaks_live_response() -> None:
    class BrokenWriter:
        def append(self, event: object) -> bool:
            del event
            raise sqlite3.OperationalError("simulated ledger outage")

    async def scenario() -> None:
        failures: list[str] = []
        recorder = AsyncCostRecorder(
            BrokenWriter(),
            on_failure=lambda kind, _: failures.append(kind),
        )
        observer = RuntimeCostObserver(
            session_id="session-isolation",
            conversation_id="conversation-isolation",
            recorder=recorder,
            pricing=PricingCatalog([]),
        )
        sink = InMemoryTelemetrySink()
        session = RealtimeVoiceSession(
            conversation_id="conversation-isolation",
            sink=sink,
            transcriber=NullTranscriber(),
            reasoner=_UsageReasoner(),
            synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=10),
            audio_output=TimedPcmOutput(quantum_ms=1),
            cost_observer=observer,
        )
        await session.start()
        await session.submit_transcript(_complete())
        await session.wait_for_response()
        assert any(
            event.event_type is EventType.AGENT_AUDIO_COMPLETED for event in sink.events
        )
        await session.close()
        assert failures
        assert session.state.value == "IDLE"

    asyncio.run(scenario())
