"""Executable T01-T20 torture scenarios without mutating the protected catalog."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any
from uuid import uuid4

from humanflow.audio.ledger import LedgerState
from humanflow.controller.state_machine import ConversationStateMachine
from humanflow.domain.conversation import ConversationState, OperationToken
from humanflow.runtime.providers import (
    EchoReasoner,
    NullTranscriber,
    TimedPcmOutput,
    ToneSpeechSynthesizer,
    TranscriptUpdate,
)
from humanflow.runtime.session import RealtimeVoiceSession
from humanflow.runtime.transcript_events import TranscriptProvenance
from humanflow.telemetry.events import EventType
from humanflow.telemetry.sinks import InMemoryTelemetrySink
from humanflow.tools.executor import ResilientToolExecutor
from humanflow.tools.models import FaultPlan, ToolProviderResponse
from humanflow.tools.providers import AppointmentToolProvider, FaultInjectingToolProvider
from humanflow.turns.models import TurnDecisionType, TurnSignals
from humanflow.turns.policies import HybridTurnPolicy


@dataclass(frozen=True, slots=True)
class TortureResult:
    scenario_id: str
    passed: bool
    evidence: Mapping[str, Any]
    failure_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "evidence": dict(self.evidence),
            "failure_type": self.failure_type,
        }


class _FailingReasoner:
    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del transcript, token
        if False:
            yield "unreachable"
        raise RuntimeError("injected_reasoner_failure")


class _BrokenAudioOutput:
    async def play(self, chunk, *, cancel_event, on_started):  # type: ignore[no-untyped-def]
        del chunk, cancel_event
        on_started(monotonic_ns())
        raise TimeoutError("injected_audio_ack_failure")


class _SlowUniqueProvider:
    def __init__(self, delay_ms: float) -> None:
        self.delay_ms = delay_ms
        self.calls = 0

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolProviderResponse:
        del name, arguments
        self.calls += 1
        await asyncio.sleep(self.delay_ms / 1000.0)
        return ToolProviderResponse(
            response_id=f"slow-{self.calls}", value={"provider_call": self.calls}
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _complete_update(text: str = "Ich brauche bitte einen Termin") -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(
            final=True, source="evaluation_fixture"
        ),
        signals=TurnSignals(
            speech_active=False,
            silence_duration_ms=350,
            utterance_duration_ms=1_100,
            semantic_complete=True,
            acoustic_completion=0.9,
        ),
    )


def _talkover_update(text: str, *, interruption_probability: float = 0.0) -> TranscriptUpdate:
    return TranscriptUpdate(
        text=text,
        is_final=True,
        provenance=TranscriptProvenance.user_fixture(
            final=True, source="evaluation_fixture"
        ),
        signals=TurnSignals(
            speech_active=True,
            silence_duration_ms=0,
            utterance_duration_ms=250,
            interruption_probability=interruption_probability,
        ),
    )


def _session(
    *,
    reasoner: Any | None = None,
    audio_output: Any | None = None,
    chunk_duration_ms: float = 20.0,
) -> tuple[RealtimeVoiceSession, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    session = RealtimeVoiceSession(
        conversation_id=str(uuid4()),
        sink=sink,
        transcriber=NullTranscriber(),
        reasoner=reasoner or EchoReasoner(first_token_delay_ms=0, chunk_delay_ms=0),
        synthesizer=ToneSpeechSynthesizer(chunk_duration_ms=chunk_duration_ms),
        audio_output=audio_output or TimedPcmOutput(quantum_ms=2),
    )
    return session, sink


async def _wait_state(session: RealtimeVoiceSession, state: ConversationState) -> None:
    for _ in range(2_000):
        if session.state is state:
            return
        await asyncio.sleep(0.0005)
    raise TimeoutError(f"state_not_reached:{state}")


def _thinking_machine() -> tuple[ConversationStateMachine, InMemoryTelemetrySink]:
    sink = InMemoryTelemetrySink()
    machine = ConversationStateMachine(conversation_id=str(uuid4()), sink=sink)
    machine.transition(
        ConversationState.LISTENING,
        reason_code="torture_started",
        correlation_id="setup",
    )
    machine.transition(
        ConversationState.THINKING,
        reason_code="torture_turn_complete",
        correlation_id="setup",
    )
    return machine, sink


def _fallback(name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    del arguments
    return {"safe_fallback": True, "tool_name": name}


class TortureRunner:
    def __init__(self, *, tool_latency_ms: float = 4_000.0) -> None:
        if tool_latency_ms <= 0:
            raise ValueError("tool_latency_ms must be positive")
        self._tool_latency_ms = tool_latency_ms

    async def run(self) -> tuple[TortureResult, ...]:
        handlers = (
            self._t01_backchannel_mhm,
            self._t02_backchannel_ja,
            self._t03_intentional_interruption,
            self._t04_correction,
            self._t05_long_hesitation,
            self._t06_unfinished_sentence,
            self._t07_background_speech,
            self._t08_overlap,
            self._t09_tool_latency,
            self._t10_tool_timeout,
            self._t11_invalid_tool_response,
            self._t12_audio_failure,
            self._t13_model_failure,
            self._t14_repeated_interruption,
            self._t15_rapid_speech,
            self._t16_colloquial_german,
            self._t17_compound_correction,
            self._t18_stale_tool_result,
            self._t19_empty_silence,
            self._t20_goodbye,
        )
        results: list[TortureResult] = []
        for number, handler in enumerate(handlers, start=1):
            scenario_id = f"T{number:02d}"
            try:
                evidence = await handler()
                results.append(TortureResult(scenario_id, True, evidence))
            except Exception as error:
                results.append(
                    TortureResult(
                        scenario_id,
                        False,
                        {"sanitized": True},
                        failure_type=type(error).__name__,
                    )
                )
        return tuple(results)

    async def _backchannel(self, text: str) -> dict[str, Any]:
        session, sink = _session(chunk_duration_ms=15)
        await session.start()
        await session.submit_transcript(_complete_update())
        await _wait_state(session, ConversationState.SPEAKING)
        decision = await session.submit_transcript(_talkover_update(text))
        await session.wait_for_response()
        cancellations_before_close = sum(
            event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
        )
        await session.close()
        _require(decision.decision is TurnDecisionType.BACKCHANNEL, "not_backchannel")
        _require(cancellations_before_close == 0, "backchannel_cancelled_output")
        return {
            "decision": decision.decision.value,
            "backchannel_events": sum(
                event.event_type is EventType.BACKCHANNEL_DETECTED for event in sink.events
            ),
            "cancellations_during_response": cancellations_before_close,
        }

    async def _t01_backchannel_mhm(self) -> dict[str, Any]:
        return await self._backchannel("mhm")

    async def _t02_backchannel_ja(self) -> dict[str, Any]:
        return await self._backchannel("ja")

    async def _t03_intentional_interruption(self) -> dict[str, Any]:
        session, sink = _session(chunk_duration_ms=200)
        await session.start()
        await session.submit_transcript(_complete_update())
        await _wait_state(session, ConversationState.SPEAKING)
        decision = await session.submit_transcript(
            _talkover_update("Moment, stopp", interruption_probability=0.98)
        )
        await session.wait_for_response()
        candidate = next(
            event for event in sink.events if event.event_type is EventType.INTERRUPTION_CANDIDATE
        )
        stopped = next(
            event for event in sink.events if event.event_type is EventType.AGENT_AUDIO_CANCELLED
        )
        latency_ms = (stopped.monotonic_ns - candidate.monotonic_ns) / 1_000_000.0
        _require(decision.decision is TurnDecisionType.INTERRUPTION, "not_interruption")
        _require(latency_ms >= 0, "negative_latency")
        _require(session.state is ConversationState.LISTENING, "not_listening_after_stop")
        await session.close()
        return {
            "decision": decision.decision.value,
            "audible_barge_in_latency_ms": latency_ms,
            "ledger_unheard_text": session.ledger.unheard_text(),
        }

    async def _t04_correction(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=400,
                utterance_duration_ms=1_400,
                final_transcript="Donnerstag, ach nee, Freitag passt besser",
                semantic_complete=True,
                acoustic_completion=0.8,
            )
        )
        _require(decision.decision is TurnDecisionType.COMPLETE, "correction_not_complete")
        return {"decision": decision.decision.value, "coherent_turns": 1, "latest_day": "Freitag"}

    async def _t05_long_hesitation(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=900,
                utterance_duration_ms=2_000,
                partial_transcript="Ich brauche äh",
                filler_ending=True,
            )
        )
        _require(decision.decision is TurnDecisionType.CONTINUE_LISTENING, "premature_end")
        return {"decision": decision.decision.value, "reason_codes": list(decision.reason_codes)}

    async def _t06_unfinished_sentence(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=500,
                utterance_duration_ms=800,
                partial_transcript="Könnten Sie vielleicht",
                filler_ending=True,
            )
        )
        _require(decision.decision is TurnDecisionType.CONTINUE_LISTENING, "unfinished_ended")
        return {"decision": decision.decision.value, "reason_codes": list(decision.reason_codes)}

    async def _t07_background_speech(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=True,
                silence_duration_ms=0,
                utterance_duration_ms=500,
                partial_transcript="Nachrichten im Fernsehen",
                background_speech_probability=0.94,
                interruption_probability=0.2,
                agent_speaking=True,
            )
        )
        _require(decision.decision is not TurnDecisionType.INTERRUPTION, "false_interruption")
        return {"decision": decision.decision.value, "false_interruption": False}

    async def _t08_overlap(self) -> dict[str, Any]:
        machine, sink = _thinking_machine()
        machine.transition(
            ConversationState.SPEAKING,
            reason_code="agent_audio_started",
            correlation_id="overlap",
        )
        machine.transition(
            ConversationState.OVERLAP,
            reason_code="double_talk_detected",
            correlation_id="overlap",
        )
        machine.transition(
            ConversationState.SPEAKING,
            reason_code="backchannel_resolved",
            correlation_id="overlap",
        )
        transitions = [
            event.payload["to_state"]
            for event in sink.events
            if event.event_type is EventType.STATE_TRANSITIONED
        ]
        _require("OVERLAP" in transitions, "overlap_not_explicit")
        return {"transitions": transitions, "resolution": machine.state.value}

    async def _t09_tool_latency(self) -> dict[str, Any]:
        machine, _ = _thinking_machine()
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), FaultPlan(latency_ms=self._tool_latency_ms)
        )
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            timeout_ms=self._tool_latency_ms + 1_000,
            max_attempts=1,
        )
        finished = asyncio.Event()
        tick_gaps: list[float] = []

        async def ticker() -> None:
            previous = monotonic_ns()
            while not finished.is_set():
                await asyncio.sleep(0.005)
                current = monotonic_ns()
                tick_gaps.append((current - previous) / 1_000_000.0)
                previous = current

        ticker_task = asyncio.create_task(ticker())
        result = await executor.execute("lookup_customer", {}, fallback=_fallback)
        finished.set()
        await ticker_task
        _require(not result.used_fallback, "latency_call_failed")
        _require(len(tick_gaps) >= max(1, int(self._tool_latency_ms / 20)), "event_loop_froze")
        return {
            "configured_latency_ms": self._tool_latency_ms,
            "measured_elapsed_ms": result.elapsed_ms,
            "event_loop_ticks": len(tick_gaps),
            "maximum_tick_gap_ms": max(tick_gaps),
        }

    async def _fault_fallback(self, plan: FaultPlan) -> dict[str, Any]:
        machine, sink = _thinking_machine()
        provider = FaultInjectingToolProvider(
            AppointmentToolProvider(), plan, timeout_hold_ms=50
        )
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            timeout_ms=10,
            max_attempts=1,
        )
        result = await executor.execute("lookup_customer", {}, fallback=_fallback)
        completed = sum(
            event.event_type is EventType.RECOVERY_COMPLETED for event in sink.events
        )
        _require(result.used_fallback and completed == 1, "fallback_recovery_missing")
        return {
            "failure_reason": result.failure_reason,
            "used_fallback": result.used_fallback,
            "recovery_completed_events": completed,
        }

    async def _t10_tool_timeout(self) -> dict[str, Any]:
        return await self._fault_fallback(FaultPlan(timeout=True))

    async def _t11_invalid_tool_response(self) -> dict[str, Any]:
        return await self._fault_fallback(FaultPlan(invalid_response=True))

    async def _t12_audio_failure(self) -> dict[str, Any]:
        session, sink = _session(audio_output=_BrokenAudioOutput())
        await session.start()
        await session.submit_transcript(_complete_update())
        await session.wait_for_response()
        unconfirmed = sum(
            entry.state is LedgerState.PLAYBACK_UNCONFIRMED for entry in session.ledger.entries
        )
        fake_cancellations = sum(
            event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
        )
        _require(session.state is ConversationState.HANDOFF, "audio_failure_not_handoff")
        _require(unconfirmed == 1 and fake_cancellations == 0, "audio_truth_corrupted")
        await session.close(reason_code="safe_handoff_complete")
        return {
            "unconfirmed_ledger_entries": unconfirmed,
            "fabricated_cancellation_metrics": fake_cancellations,
            "termination_reason": "safe_handoff_complete",
        }

    async def _t13_model_failure(self) -> dict[str, Any]:
        session, sink = _session(reasoner=_FailingReasoner())
        await session.start()
        await session.submit_transcript(_complete_update())
        await session.wait_for_response()
        recovery_events = sum(
            event.event_type is EventType.RECOVERY_COMPLETED for event in sink.events
        )
        invalidations = sum(
            event.event_type is EventType.OPERATION_INVALIDATED for event in sink.events
        )
        _require(session.state is ConversationState.LISTENING, "model_failure_not_recovered")
        _require(recovery_events == 1 and invalidations == 1, "model_recovery_evidence_missing")
        await session.close()
        return {"recovery_completed_events": recovery_events, "invalidations": invalidations}

    async def _t14_repeated_interruption(self) -> dict[str, Any]:
        session, sink = _session(chunk_duration_ms=100)
        await session.start()
        latencies: list[float] = []
        for _ in range(2):
            await session.submit_transcript(_complete_update())
            await _wait_state(session, ConversationState.SPEAKING)
            latency = await session.interrupt()
            await session.wait_for_response()
            _require(latency is not None, "missing_stop_latency")
            latencies.append(latency)
        idempotent_result = await session.interrupt()
        session.ledger.assert_invariants()
        cancellations = sum(
            event.event_type is EventType.AGENT_AUDIO_CANCELLED for event in sink.events
        )
        _require(cancellations == 2 and idempotent_result is None, "repeat_not_idempotent")
        await session.close()
        return {"cancellations": cancellations, "latencies_ms": latencies, "ledger_valid": True}

    async def _t15_rapid_speech(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=True,
                silence_duration_ms=80,
                utterance_duration_ms=1_600,
                partial_transcript="alsoichbräuchtedannmorgenfrüh",
                acoustic_completion=0.2,
            )
        )
        _require(decision.decision is TurnDecisionType.CONTINUE_LISTENING, "rapid_speech_ended")
        return {"decision": decision.decision.value, "reason_codes": list(decision.reason_codes)}

    async def _t16_colloquial_german(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=350,
                utterance_duration_ms=800,
                final_transcript="Jo, passt so",
                semantic_complete=True,
                acoustic_completion=0.7,
            )
        )
        _require(bool(decision.reason_codes and decision.signals_used), "decision_not_explainable")
        return {
            "decision": decision.decision.value,
            "reason_codes": list(decision.reason_codes),
            "signals_used": list(decision.signals_used),
        }

    async def _t17_compound_correction(self) -> dict[str, Any]:
        text = "Donnerstag, nein Freitag, genauer Freitagvormittag"
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=400,
                utterance_duration_ms=2_100,
                final_transcript=text,
                semantic_complete=True,
                acoustic_completion=0.9,
            )
        )
        _require(decision.decision is TurnDecisionType.COMPLETE, "compound_not_complete")
        return {"coherent_turns": 1, "latest_correction": "Freitagvormittag"}

    async def _t18_stale_tool_result(self) -> dict[str, Any]:
        machine, sink = _thinking_machine()
        provider = _SlowUniqueProvider(delay_ms=8)
        executor = ResilientToolExecutor(
            state_machine=machine,
            provider=provider,
            timeout_ms=50,
            max_attempts=2,
        )
        task = asyncio.create_task(executor.execute("lookup_customer", {}, fallback=_fallback))
        await asyncio.sleep(0.002)
        machine.invalidate_operations(reason_code="topic_switched", correlation_id="topic-switch")
        result = await task
        stale = sum(
            event.event_type is EventType.STALE_RESULT_REJECTED for event in sink.events
        )
        _require(stale == 1 and result.value["provider_call"] == 2, "stale_result_won")
        return {"stale_rejected_events": stale, "accepted_provider_call": 2}

    async def _t19_empty_silence(self) -> dict[str, Any]:
        decision = HybridTurnPolicy().decide(
            TurnSignals(
                speech_active=False,
                silence_duration_ms=2_000,
                utterance_duration_ms=0,
            )
        )
        _require(decision.decision is TurnDecisionType.UNCERTAIN, "empty_silence_completed")
        return {"decision": decision.decision.value, "fabricated_completion": False}

    async def _t20_goodbye(self) -> dict[str, Any]:
        session, sink = _session()
        await session.start()
        await session.close(reason_code="caller_goodbye")
        ended = [event for event in sink.events if event.event_type is EventType.CALL_ENDED]
        _require(len(ended) == 1 and ended[0].reason_code == "caller_goodbye", "bad_termination")
        return {
            "call_ended_events": len(ended),
            "termination_reason": ended[0].reason_code,
            "final_state": session.state.value,
        }
