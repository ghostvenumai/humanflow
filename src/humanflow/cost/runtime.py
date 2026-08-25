"""Best-effort runtime usage attribution; never participates in voice correctness."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from time import monotonic_ns
from typing import Any, Callable, Iterable

from humanflow.audio.ledger import LedgerEntrySnapshot

from .ledger import AsyncCostRecorder
from .budget import CostBudgetPolicy
from .models import CostEvent, CostSource, ServiceType, UsageSource
from .pricing import PricingCatalog


@dataclass(slots=True)
class RuntimeCostObserver:
    session_id: str
    conversation_id: str
    recorder: AsyncCostRecorder
    pricing: PricingCatalog
    clock_ns: Callable[[], int] = monotonic_ns
    budget_policy: CostBudgetPolicy = field(default_factory=CostBudgetPolicy)
    on_budget_event: Callable[[str, dict[str, Any]], None] = field(
        default=lambda _kind, _payload: None
    )
    _response_tts_events: dict[str, list[CostEvent]] = field(default_factory=dict)
    _estimated_total_micros: int = 0
    _budget_events_emitted: set[str] = field(default_factory=set)

    def record_stt(
        self,
        *,
        operation_id: str,
        turn_id: str,
        provider: str,
        model: str,
        audio_seconds: Decimal,
        partial_count: int,
        committed_count: int = 1,
        provider_session_id: str | None = None,
    ) -> bool:
        return self._record(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                provider=provider,
                model=model,
                service_type=ServiceType.STT,
                operation="streaming_transcription",
                operation_id=operation_id,
                provider_request_id=provider_session_id,
                usage_source=UsageSource.ACTUAL_USAGE,
                cost_source=CostSource.COST_UNAVAILABLE,
                audio_input_seconds=audio_seconds,
                total_units=audio_seconds,
                unit_type="audio_seconds",
                actual_usage=True,
                metadata={
                    "partial_transcript_count": partial_count,
                    "committed_transcript_count": committed_count,
                    "billing_basis": "audio_duration_not_partial_event_count",
                },
            )
        )

    def record_llm(
        self,
        *,
        operation_id: str,
        turn_id: str,
        response_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        retry: bool = False,
    ) -> bool:
        return self._record(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                response_id=response_id,
                provider=provider,
                model=model,
                service_type=ServiceType.LLM,
                operation="reasoning",
                operation_id=operation_id,
                usage_source=UsageSource.ACTUAL_USAGE,
                cost_source=CostSource.COST_UNAVAILABLE,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                input_units=Decimal(input_tokens),
                output_units=Decimal(output_tokens),
                total_units=Decimal(input_tokens + output_tokens),
                unit_type="tokens",
                actual_usage=True,
                retry=retry,
                metadata={"latency_ms": latency_ms, "success": success},
            )
        )

    def record_tts(
        self,
        *,
        operation_id: str,
        turn_id: str,
        response_id: str,
        provider: str,
        model: str,
        characters: int,
        audio_seconds: Decimal,
        reported_billable_characters: int | None,
        fallback: bool,
        cancelled: bool,
        latency_ms: float | None,
    ) -> bool:
        service = ServiceType.FALLBACK_TTS if fallback else ServiceType.TTS
        usage_characters = reported_billable_characters or characters
        event = self.pricing.price(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                response_id=response_id,
                provider=provider,
                model=model,
                service_type=service,
                operation="streaming_synthesis",
                operation_id=operation_id,
                usage_source=UsageSource.ACTUAL_USAGE,
                cost_source=CostSource.COST_UNAVAILABLE,
                output_units=Decimal(usage_characters),
                total_units=Decimal(usage_characters),
                unit_type="characters",
                audio_output_seconds=audio_seconds,
                characters=usage_characters,
                actual_usage=True,
                fallback=fallback,
                cancelled=cancelled,
                metadata={
                    "submitted_characters": characters,
                    "provider_billable_characters": reported_billable_characters,
                    "first_audio_latency_ms": latency_ms,
                },
            )
        )
        self._response_tts_events.setdefault(response_id, []).append(event)
        self._observe_budget(event)
        return self.recorder.record_nowait(event)

    def record_primary_tts_failure(
        self,
        *,
        operation_id: str,
        turn_id: str,
        response_id: str,
        provider: str,
        model: str,
        failure_class: str,
    ) -> bool:
        return self._record(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                response_id=response_id,
                provider=provider,
                model=model,
                service_type=ServiceType.TTS,
                operation="primary_synthesis_failure",
                operation_id=operation_id,
                usage_source=UsageSource.LOCAL_OBSERVATION,
                cost_source=CostSource.COST_UNAVAILABLE,
                fallback=True,
                metadata={
                    "success": False,
                    "failure_class": failure_class,
                    "known_primary_billable_usage": False,
                },
            )
        )

    def record_tool(
        self,
        *,
        operation_id: str,
        turn_id: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        retry: bool,
        appointment_id: str | None,
        transaction_result: str | None,
        failure_class: str | None,
    ) -> bool:
        return self._record(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                provider="local-sqlite",
                model="humanflow-appointments-v1",
                service_type=ServiceType.TOOL,
                operation=tool_name,
                operation_id=operation_id,
                usage_source=UsageSource.ACTUAL_USAGE,
                cost_source=CostSource.COST_UNAVAILABLE,
                total_units=Decimal("1"),
                unit_type="request",
                actual_usage=True,
                retry=retry,
                tool_success=success,
                metadata={
                    "duration_ms": duration_ms,
                    "appointment_id": appointment_id,
                    "transaction_result": transaction_result,
                    "failure_class": failure_class,
                },
            )
        )

    def record_played_audio(
        self,
        *,
        operation_id: str,
        turn_id: str,
        response_id: str,
        entries: Iterable[LedgerEntrySnapshot],
    ) -> bool | None:
        relevant = [entry for entry in entries if entry.response_id == response_id]
        generated_samples = sum(entry.total_samples for entry in relevant)
        if generated_samples <= 0:
            return None
        heard_samples = sum(entry.played_samples for entry in relevant)
        unheard_samples = generated_samples - heard_samples
        played_fraction = Decimal(heard_samples) / Decimal(generated_samples)
        generated_seconds = sum(
            Decimal(entry.total_samples) / Decimal(entry.sample_rate_hz)
            for entry in relevant
        )
        heard_seconds = sum(
            Decimal(entry.played_samples) / Decimal(entry.sample_rate_hz)
            for entry in relevant
        )
        unheard_seconds = generated_seconds - heard_seconds
        tts_events = self._response_tts_events.pop(response_id, [])
        known_estimated_micros = sum(
            event.estimated_cost_micros or 0 for event in tts_events
        )
        wasted_micros = (
            round(Decimal(known_estimated_micros) * (Decimal("1") - played_fraction))
            if tts_events and all(event.estimated_cost_micros is not None for event in tts_events)
            else None
        )
        provider = tts_events[-1].provider if tts_events else "tts-playback-ledger"
        model = tts_events[-1].model if tts_events else "unknown"
        currency = tts_events[-1].currency if wasted_micros is not None else None
        return self.recorder.record_nowait(
            CostEvent(
                **self._identity(),
                turn_id=turn_id,
                response_id=response_id,
                provider=provider,
                model=model,
                service_type=ServiceType.TTS,
                operation="played_audio_allocation",
                operation_id=operation_id,
                usage_source=UsageSource.LOCAL_OBSERVATION,
                cost_source=(
                    CostSource.ESTIMATED_COST
                    if wasted_micros is not None
                    else CostSource.COST_UNAVAILABLE
                ),
                unit_type="audio_seconds",
                played_fraction=played_fraction,
                heard_units=heard_seconds,
                unheard_units=unheard_seconds,
                wasted_cost_estimate_micros=wasted_micros,
                currency=currency,
                cancelled=unheard_samples > 0,
                metadata={
                    "generated_audio_seconds": str(generated_seconds),
                    "allocation_is_estimated": True,
                    "provider_billing_not_reduced_by_played_fraction": True,
                },
            )
        )

    def _record(self, event: CostEvent) -> bool:
        priced = self.pricing.price(event)
        self._observe_budget(priced)
        return self.recorder.record_nowait(priced)

    def _observe_budget(self, event: CostEvent) -> None:
        if event.estimated_cost_micros is None or event.currency != "EUR":
            return
        self._estimated_total_micros += event.estimated_cost_micros
        decision = self.budget_policy.evaluate(
            estimated_eur=Decimal(self._estimated_total_micros) / Decimal("1000000"),
            estimate_complete=True,
        )
        if decision is None or decision in self._budget_events_emitted:
            return
        self._budget_events_emitted.add(decision)
        self.on_budget_event(
            "COST_BUDGET_WARNING",
            {
                "severity": decision,
                "estimated_session_cost_eur": str(
                    Decimal(self._estimated_total_micros) / Decimal("1000000")
                ),
                "conversation_action": "OBSERVE_ONLY",
            },
        )

    async def close(self) -> None:
        await self.recorder.flush()
        await self.recorder.close()

    def _identity(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "timestamp_monotonic_ns": self.clock_ns(),
        }
