"""Truthful accounting for generated, queued, played, and cancelled speech."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock

from .models import AudioChunk, PlaybackReceipt


class LedgerState(StrEnum):
    GENERATED = "GENERATED"
    QUEUED = "QUEUED"
    PLAYING = "PLAYING"
    PLAYED = "PLAYED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class LedgerEntrySnapshot:
    chunk_id: str
    response_id: str
    text: str
    total_samples: int
    played_samples: int
    sample_rate_hz: int
    generated_ns: int
    queued_ns: int | None = None
    playback_started_ns: int | None = None
    playback_stopped_ns: int | None = None
    cancelled_ns: int | None = None
    state: LedgerState = LedgerState.GENERATED

    @property
    def fully_delivered(self) -> bool:
        return self.played_samples == self.total_samples

    @property
    def played_duration_ms(self) -> float:
        return self.played_samples * 1000.0 / self.sample_rate_hz

    @property
    def unheard_text(self) -> str:
        return "" if self.fully_delivered else self.text


class PlayedAudioLedger:
    """Append-only chunk ledger with a conservative delivered-text boundary."""

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntrySnapshot] = {}
        self._order: list[str] = []
        self._lock = Lock()

    def register_generated(self, chunk: AudioChunk, *, generated_ns: int) -> None:
        if generated_ns < 0:
            raise ValueError("generated_ns must be non-negative")
        with self._lock:
            if chunk.chunk_id in self._entries:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            self._entries[chunk.chunk_id] = LedgerEntrySnapshot(
                chunk_id=chunk.chunk_id,
                response_id=chunk.response_id,
                text=chunk.text.strip(),
                total_samples=chunk.frame.samples_per_channel,
                played_samples=0,
                sample_rate_hz=chunk.frame.sample_rate_hz,
                generated_ns=generated_ns,
            )
            self._order.append(chunk.chunk_id)

    def mark_queued(self, chunk_id: str, *, queued_ns: int) -> None:
        with self._lock:
            entry = self._require(chunk_id)
            if entry.state is not LedgerState.GENERATED:
                raise ValueError(f"cannot queue chunk in state {entry.state}")
            if queued_ns < entry.generated_ns:
                raise ValueError("queued_ns must not precede generated_ns")
            self._entries[chunk_id] = replace(
                entry, queued_ns=queued_ns, state=LedgerState.QUEUED
            )

    def mark_playback_started(self, chunk_id: str, *, started_ns: int) -> None:
        with self._lock:
            entry = self._require(chunk_id)
            if entry.state is not LedgerState.QUEUED or entry.queued_ns is None:
                raise ValueError(f"cannot start chunk in state {entry.state}")
            if started_ns < entry.queued_ns:
                raise ValueError("started_ns must not precede queued_ns")
            self._entries[chunk_id] = replace(
                entry, playback_started_ns=started_ns, state=LedgerState.PLAYING
            )

    def record_playback(self, receipt: PlaybackReceipt) -> None:
        with self._lock:
            entry = self._require(receipt.chunk_id)
            if entry.state is not LedgerState.PLAYING:
                raise ValueError(f"cannot record playback in state {entry.state}")
            if receipt.requested_samples != entry.total_samples:
                raise ValueError("playback receipt sample count does not match ledger")
            if receipt.playback_started_ns != entry.playback_started_ns:
                raise ValueError("playback receipt start does not match ledger")
            state = LedgerState.CANCELLED if receipt.cancelled else LedgerState.PLAYED
            self._entries[receipt.chunk_id] = replace(
                entry,
                played_samples=receipt.played_samples,
                playback_stopped_ns=receipt.playback_stopped_ns,
                cancelled_ns=receipt.playback_stopped_ns if receipt.cancelled else None,
                state=state,
            )

    def cancel_unplayed(self, *, response_id: str, cancelled_ns: int) -> None:
        """Cancel generated/queued chunks; never rewrite already played truth."""
        with self._lock:
            for chunk_id in self._order:
                entry = self._entries[chunk_id]
                if entry.response_id != response_id:
                    continue
                if entry.state in {LedgerState.GENERATED, LedgerState.QUEUED}:
                    if cancelled_ns < entry.generated_ns:
                        raise ValueError("cancelled_ns must not precede generated_ns")
                    self._entries[chunk_id] = replace(
                        entry, cancelled_ns=cancelled_ns, state=LedgerState.CANCELLED
                    )

    @property
    def entries(self) -> tuple[LedgerEntrySnapshot, ...]:
        with self._lock:
            return tuple(self._entries[chunk_id] for chunk_id in self._order)

    def delivered_text(self, *, response_id: str | None = None) -> str:
        """Return only complete semantic chunks that the sink actually played."""
        entries = self.entries
        return " ".join(
            entry.text
            for entry in entries
            if entry.fully_delivered and (response_id is None or entry.response_id == response_id)
        )

    def unheard_text(self, *, response_id: str | None = None) -> str:
        entries = self.entries
        return " ".join(
            entry.text
            for entry in entries
            if entry.unheard_text and (response_id is None or entry.response_id == response_id)
        )

    def assert_invariants(self) -> None:
        for entry in self.entries:
            if not 0 <= entry.played_samples <= entry.total_samples:
                raise AssertionError("played samples exceed generated samples")
            if entry.state is LedgerState.PLAYED and not entry.fully_delivered:
                raise AssertionError("PLAYED entry is not fully delivered")
            if entry.state is LedgerState.GENERATED and entry.queued_ns is not None:
                raise AssertionError("generated entry unexpectedly has queue timestamp")
            if entry.playback_stopped_ns is not None and entry.playback_started_ns is None:
                raise AssertionError("stopped playback has no start timestamp")

    def _require(self, chunk_id: str) -> LedgerEntrySnapshot:
        try:
            return self._entries[chunk_id]
        except KeyError as error:
            raise KeyError(f"unknown chunk_id: {chunk_id}") from error
