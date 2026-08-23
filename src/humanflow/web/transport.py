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


class BrowserTelemetrySink(InMemoryTelemetrySink):
    """Retain evidence in memory and mirror ordered events to the browser queue."""

    def __init__(self, outbound: asyncio.Queue[OutboundItem]) -> None:
        super().__init__()
        self._outbound = outbound

    def emit(self, event: TelemetryEvent) -> None:
        super().emit(event)
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
        self._pending: _PendingPlayback | None = None

    async def play(
        self,
        chunk: AudioChunk,
        *,
        cancel_event: asyncio.Event,
        on_started: PlaybackStarted,
    ) -> PlaybackReceipt:
        if self._pending is not None:
            raise RuntimeError("browser output accepts one ordered chunk at a time")
        loop = asyncio.get_running_loop()
        pending = _PendingPlayback(
            chunk=chunk,
            started=loop.create_future(),
            stopped=loop.create_future(),
        )
        self._pending = pending
        self._outbound.put_nowait(
            {
                "type": "audio_chunk",
                "chunk_id": chunk.chunk_id,
                "response_id": chunk.response_id,
                "sample_rate_hz": chunk.frame.sample_rate_hz,
                "channels": chunk.frame.channels,
                "samples": chunk.frame.samples_per_channel,
                "text_boundary": chunk.text,
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
            )
        finally:
            cancel_wait.cancel()
            self._pending = None

    def acknowledge(self, message: dict[str, Any]) -> bool:
        pending = self._pending
        if pending is None or message.get("chunk_id") != pending.chunk.chunk_id:
            return False
        message_type = message.get("type")
        now_ns = self._clock_ns()
        if message_type == "playback_started" and not pending.started.done():
            pending.started.set_result(now_ns)
            return True
        if message_type in {"playback_completed", "playback_stopped"}:
            if not pending.started.done():
                pending.started.set_result(now_ns)
            if pending.stopped.done():
                return False
            total = pending.chunk.frame.samples_per_channel
            if message_type == "playback_completed":
                played_samples = total
                cancelled = False
            else:
                raw_samples = message.get("played_samples", 0)
                if not isinstance(raw_samples, int):
                    return False
                played_samples = max(0, min(total, raw_samples))
                cancelled = True
            pending.stopped.set_result((now_ns, played_samples, cancelled))
            return True
        return False

    def disconnect(self) -> None:
        """Release a pending play immediately when its browser transport disappears."""
        pending = self._pending
        if pending is None:
            return
        now_ns = self._clock_ns()
        if not pending.started.done():
            pending.started.set_result(now_ns)
        if not pending.stopped.done():
            pending.stopped.set_result((now_ns, 0, True))
