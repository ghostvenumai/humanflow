"""Browser-acknowledged audio output and websocket telemetry transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any, Callable

from humanflow.audio.models import AudioChunk, PlaybackReceipt
from humanflow.runtime.providers import PlaybackStarted
from humanflow.telemetry.events import TelemetryEvent
from humanflow.telemetry.sinks import InMemoryTelemetrySink


OutboundItem = dict[str, Any] | bytes


@dataclass(slots=True)
class _PendingPlayback:
    chunk: AudioChunk
    started: asyncio.Future[int]
    stopped: asyncio.Future[tuple[int, int, bool]]
    sink_base_latency_ms: float | None = None
    sink_output_latency_ms: float | None = None
    player_stop_callback_latency_ms: float | None = None
    source_node_id: str | None = None
    browser_scheduled_start_ms: float | None = None
    browser_actual_playback_start_ms: float | None = None
    browser_actual_playback_end_ms: float | None = None
    previous_segment_end_ms: float | None = None
    inter_segment_gap_ms: float | None = None
    queue_depth_ms: float | None = None
    underrun_count: int = 0


class BrowserTelemetrySink(InMemoryTelemetrySink):
    """Retain evidence in memory and mirror ordered events to the browser queue."""

    def __init__(
        self,
        outbound: asyncio.Queue[OutboundItem],
        *,
        observer: Callable[[TelemetryEvent], None] | None = None,
    ) -> None:
        super().__init__()
        self._outbound = outbound
        self._observer = observer

    def emit(self, event: TelemetryEvent) -> None:
        super().emit(event)
        if self._observer is not None:
            self._observer(event)
        self._outbound.put_nowait({"type": "telemetry", "event": event.to_dict()})


class BrowserAcknowledgedAudioOutput:
    """Audio output whose receipts end at browser playback acknowledgements."""

    def __init__(
        self,
        outbound: asyncio.Queue[OutboundItem],
        *,
        acknowledgement_timeout_s: float = 3.0,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if acknowledgement_timeout_s <= 0:
            raise ValueError("acknowledgement_timeout_s must be positive")
        self._outbound = outbound
        self._timeout = acknowledgement_timeout_s
        self._clock_ns = clock_ns
        self._pending: dict[str, _PendingPlayback] = {}
        self._playback_epoch = 0
        self._invalidated_response_ids: set[str] = set()

    @property
    def playback_epoch(self) -> int:
        return self._playback_epoch

    def soft_duck(self, *, response_id: str, speech_onset_ns: int) -> bool:
        if response_id in self._invalidated_response_ids:
            return False
        self._outbound.put_nowait(
            {
                "type": "playback_duck",
                "response_id": response_id,
                "speech_onset_ns": speech_onset_ns,
                "playback_epoch": self._playback_epoch,
                "target_gain": 0.08,
            }
        )
        return True

    def resume_playback(self, *, response_id: str, speech_onset_ns: int) -> bool:
        if response_id in self._invalidated_response_ids:
            return False
        self._outbound.put_nowait(
            {
                "type": "playback_resume",
                "response_id": response_id,
                "speech_onset_ns": speech_onset_ns,
                "playback_epoch": self._playback_epoch,
            }
        )
        return True

    def invalidate_response(self, *, response_id: str, speech_onset_ns: int) -> int:
        if response_id in self._invalidated_response_ids:
            return self._playback_epoch
        self._invalidated_response_ids.add(response_id)
        self._playback_epoch += 1
        self._outbound.put_nowait(
            {
                "type": "invalidate_playback",
                "response_id": response_id,
                "speech_onset_ns": speech_onset_ns,
                "playback_epoch": self._playback_epoch,
                "future_audio_from_invalidated_epoch": "FORBIDDEN",
            }
        )
        return self._playback_epoch

    async def play(
        self,
        chunk: AudioChunk,
        *,
        cancel_event: asyncio.Event,
        on_started: PlaybackStarted,
    ) -> PlaybackReceipt:
        if chunk.response_id in self._invalidated_response_ids:
            stopped_ns = self._clock_ns()
            on_started(stopped_ns)
            return PlaybackReceipt(
                chunk_id=chunk.chunk_id,
                requested_samples=chunk.frame.samples_per_channel,
                played_samples=0,
                playback_started_ns=stopped_ns,
                playback_stopped_ns=stopped_ns,
                cancelled=True,
            )
        loop = asyncio.get_running_loop()
        pending = _PendingPlayback(
            chunk=chunk,
            started=loop.create_future(),
            stopped=loop.create_future(),
        )
        if chunk.chunk_id in self._pending:
            raise RuntimeError("browser output received duplicate pending chunk")
        self._pending[chunk.chunk_id] = pending
        self._outbound.put_nowait(
            {
                "type": "audio_chunk",
                "chunk_id": chunk.chunk_id,
                "response_id": chunk.response_id,
                "stream_id": chunk.frame.stream_id,
                "sequence": chunk.frame.sequence,
                "chunk_byte_length": len(chunk.frame.pcm16),
                "codec": "pcm_s16le",
                "sample_rate_hz": chunk.frame.sample_rate_hz,
                "channels": chunk.frame.channels,
                "samples": chunk.frame.samples_per_channel,
                "decoded_duration_ms": chunk.frame.duration_ms,
                "playback_mode": chunk.playback_mode.value,
                "text_boundary": chunk.display_text,
                "ledger_boundary": chunk.text,
                "semantic_boundary": chunk.semantic_boundary,
                "pause_after_ms": chunk.pause_after_ms,
                "speaking_rate": chunk.speaking_rate,
                "tts_provider": dict(chunk.provider),
                "playback_epoch": self._playback_epoch,
            }
        )
        self._outbound.put_nowait(chunk.frame.pcm16)
        cancel_wait = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {pending.started, cancel_wait},
                timeout=self._timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if pending.started in done:
                started_ns = pending.started.result()
                on_started(started_ns)
            elif cancel_wait in done:
                started_ns = self._clock_ns()
                on_started(started_ns)
                self._outbound.put_nowait(
                    {"type": "cancel_audio", "chunk_id": chunk.chunk_id}
                )
            else:
                raise TimeoutError("browser_playback_start_ack_timeout")

            if cancel_event.is_set() and not pending.stopped.done():
                self._outbound.put_nowait(
                    {"type": "cancel_audio", "chunk_id": chunk.chunk_id}
                )
            if not cancel_event.is_set():
                done, _ = await asyncio.wait(
                    {pending.stopped, cancel_wait},
                    timeout=self._timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_wait in done and not pending.stopped.done():
                    self._outbound.put_nowait(
                        {"type": "cancel_audio", "chunk_id": chunk.chunk_id}
                    )
            try:
                stopped_ns, played_samples, browser_cancelled = await asyncio.wait_for(
                    asyncio.shield(pending.stopped), timeout=self._timeout
                )
            except TimeoutError:
                raise TimeoutError("browser_playback_stop_ack_timeout") from None
            cancelled = browser_cancelled or played_samples < chunk.frame.samples_per_channel
            return PlaybackReceipt(
                chunk_id=chunk.chunk_id,
                requested_samples=chunk.frame.samples_per_channel,
                played_samples=played_samples,
                playback_started_ns=started_ns,
                playback_stopped_ns=max(started_ns, stopped_ns),
                cancelled=cancelled,
                sink_base_latency_ms=pending.sink_base_latency_ms,
                sink_output_latency_ms=pending.sink_output_latency_ms,
                player_stop_callback_latency_ms=(
                    pending.player_stop_callback_latency_ms
                ),
                source_node_id=pending.source_node_id,
                browser_scheduled_start_ms=pending.browser_scheduled_start_ms,
                browser_actual_playback_start_ms=(
                    pending.browser_actual_playback_start_ms
                ),
                browser_actual_playback_end_ms=pending.browser_actual_playback_end_ms,
                previous_segment_end_ms=pending.previous_segment_end_ms,
                inter_segment_gap_ms=pending.inter_segment_gap_ms,
                queue_depth_ms=pending.queue_depth_ms,
                underrun_count=pending.underrun_count,
            )
        finally:
            cancel_wait.cancel()
            self._pending.pop(chunk.chunk_id, None)

    def acknowledge(self, message: dict[str, Any]) -> bool:
        chunk_id = message.get("chunk_id")
        if not isinstance(chunk_id, str):
            return False
        pending = self._pending.get(chunk_id)
        if pending is None:
            return False
        message_type = message.get("type")
        now_ns = self._clock_ns()
        if message_type == "playback_started" and not pending.started.done():
            pending.sink_base_latency_ms = _nonnegative_float(
                message.get("browser_audio_context_base_latency_ms")
            )
            pending.sink_output_latency_ms = _nonnegative_float(
                message.get("browser_audio_context_output_latency_ms")
            )
            pending.browser_scheduled_start_ms = _nonnegative_float(
                message.get("browser_scheduled_start_ms")
            )
            pending.browser_actual_playback_start_ms = _nonnegative_float(
                message.get("browser_actual_playback_start_ms")
            )
            pending.previous_segment_end_ms = _nonnegative_float(
                message.get("previous_segment_end_ms")
            )
            pending.inter_segment_gap_ms = _nonnegative_float(
                message.get("inter_segment_gap_ms")
            )
            pending.queue_depth_ms = _nonnegative_float(
                message.get("queue_depth_ms")
            )
            raw_underruns = message.get("underrun_count", 0)
            if isinstance(raw_underruns, int) and raw_underruns >= 0:
                pending.underrun_count = raw_underruns
            source_node_id = message.get("source_node_id")
            if isinstance(source_node_id, str) and source_node_id.strip():
                pending.source_node_id = source_node_id[:256]
            pending.started.set_result(now_ns)
            return True
        if message_type in {"playback_completed", "playback_stopped"}:
            if not pending.started.done():
                pending.started.set_result(now_ns)
            if pending.stopped.done():
                return False
            total = pending.chunk.frame.samples_per_channel
            pending.browser_actual_playback_end_ms = _nonnegative_float(
                message.get("browser_actual_playback_end_ms")
            )
            if message_type == "playback_completed":
                played_samples = total
                cancelled = False
            else:
                raw_samples = message.get("played_samples", 0)
                if not isinstance(raw_samples, int):
                    return False
                played_samples = max(0, min(total, raw_samples))
                cancelled = True
                pending.player_stop_callback_latency_ms = _nonnegative_float(
                    message.get("player_stop_callback_latency_ms")
                )
            pending.stopped.set_result((now_ns, played_samples, cancelled))
            return True
        return False

    def disconnect(self) -> None:
        """Release a pending play immediately when its browser transport disappears."""
        now_ns = self._clock_ns()
        for pending in tuple(self._pending.values()):
            if not pending.started.done():
                pending.started.set_result(now_ns)
            if not pending.stopped.done():
                pending.stopped.set_result((now_ns, 0, True))


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if converted >= 0 else None
