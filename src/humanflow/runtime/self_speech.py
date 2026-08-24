"""Conservative timing-plus-content protection against assistant self-transcription."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher

from .transcript_events import TranscriptOrigin, normalize_transcript


_EXPLICIT_HUMAN_CONTROL = frozenset(
    {"mhm", "hm", "ja", "okay", "ok", "stopp", "moment", "warte", "halt"}
)


@dataclass(slots=True)
class SpokenSegment:
    chunk_id: str
    response_id: str
    text: str
    normalized_text: str
    started_ns: int | None = None
    stopped_ns: int | None = None


@dataclass(frozen=True, slots=True)
class SelfSpeechAssessment:
    candidate: bool
    suppress: bool
    confidence: float
    rejection_reason: str | None
    matched_response_id: str | None
    matched_chunk_id: str | None
    signals: dict[str, float | bool | int | str | None]


class SelfSpeechGuard:
    def __init__(
        self,
        *,
        recent_window_ms: float = 900.0,
        maximum_segments: int = 32,
    ) -> None:
        if recent_window_ms < 0 or maximum_segments < 1:
            raise ValueError("invalid self-speech guard configuration")
        self._recent_window_ns = round(recent_window_ms * 1_000_000)
        self._segments: deque[SpokenSegment] = deque(maxlen=maximum_segments)

    @property
    def recent_segments(self) -> tuple[SpokenSegment, ...]:
        return tuple(self._segments)

    @property
    def current_response_id(self) -> str | None:
        segment = next(
            (
                item
                for item in reversed(self._segments)
                if item.started_ns is not None and item.stopped_ns is None
            ),
            None,
        )
        return None if segment is None else segment.response_id

    @property
    def current_tts_text(self) -> str:
        segment = next(
            (
                item
                for item in reversed(self._segments)
                if item.started_ns is not None and item.stopped_ns is None
            ),
            None,
        )
        return "" if segment is None else segment.text

    def register_pending(self, *, chunk_id: str, response_id: str, text: str) -> None:
        normalized = normalize_transcript(text)
        if not normalized:
            return
        self._segments.append(
            SpokenSegment(
                chunk_id=chunk_id,
                response_id=response_id,
                text=text,
                normalized_text=normalized,
            )
        )

    def mark_started(self, *, chunk_id: str, started_ns: int) -> None:
        segment = self._find(chunk_id)
        if segment is not None:
            segment.started_ns = started_ns

    def mark_stopped(self, *, chunk_id: str, stopped_ns: int) -> None:
        segment = self._find(chunk_id)
        if segment is not None:
            segment.stopped_ns = stopped_ns

    def assess(
        self,
        *,
        text: str,
        observed_ns: int,
        origin: TranscriptOrigin,
        playback_active: bool,
    ) -> SelfSpeechAssessment:
        normalized = normalize_transcript(text)
        tokens = normalized.split()
        if origin not in {
            TranscriptOrigin.BROWSER_SPEECH_RECOGNITION,
            TranscriptOrigin.STREAMING_STT_PROVIDER,
        } or not tokens:
            return self._not_candidate("non_browser_or_empty")
        if len(tokens) <= 3 and any(token in _EXPLICIT_HUMAN_CONTROL for token in tokens):
            return self._not_candidate("explicit_human_control_phrase")

        best: tuple[float, SpokenSegment, dict[str, float | bool | int | str | None]] | None = None
        candidates = list(self._segments)
        relevant_response_ids = {
            segment.response_id
            for segment in candidates
            if segment.started_ns is not None
            and (
                (segment.stopped_ns is None and playback_active)
                or (
                    segment.stopped_ns is not None
                    and observed_ns - segment.stopped_ns <= self._recent_window_ns
                )
            )
        }
        for response_id in relevant_response_ids:
            response_segments = [
                segment
                for segment in candidates
                if segment.response_id == response_id and segment.started_ns is not None
            ]
            normalized_response = " ".join(
                segment.normalized_text for segment in response_segments
            ).strip()
            if len(response_segments) > 1 and normalized_response:
                timing_segment = response_segments[-1]
                candidates.append(
                    SpokenSegment(
                        chunk_id=timing_segment.chunk_id,
                        response_id=response_id,
                        text=" ".join(segment.text for segment in response_segments),
                        normalized_text=normalized_response,
                        started_ns=timing_segment.started_ns,
                        stopped_ns=timing_segment.stopped_ns,
                    )
                )

        for segment in reversed(candidates):
            if segment.started_ns is None:
                continue
            active = segment.stopped_ns is None and playback_active
            age_ns = (
                0
                if active
                else max(0, observed_ns - (segment.stopped_ns or segment.started_ns))
            )
            recent = active or age_ns <= self._recent_window_ns
            if not recent:
                continue
            spoken_tokens = segment.normalized_text.split()
            character_similarity = SequenceMatcher(
                None, normalized, segment.normalized_text
            ).ratio()
            token_overlap = sum(token in spoken_tokens for token in tokens) / len(tokens)
            phrase_order = SequenceMatcher(None, tokens, spoken_tokens).ratio()
            contiguous_fragment = normalized in segment.normalized_text
            compact_fragment = normalized.replace(" ", "") in segment.normalized_text.replace(
                " ", ""
            )
            timing_score = 1.0 if active else max(
                0.0, 1.0 - age_ns / max(1, self._recent_window_ns)
            )
            score = (
                0.38 * character_similarity
                + 0.27 * token_overlap
                + 0.20 * phrase_order
                + 0.15 * timing_score
            )
            if (contiguous_fragment or compact_fragment) and len(tokens) >= 4:
                score = min(1.0, score + 0.12)
            signals: dict[str, float | bool | int | str | None] = {
                "character_similarity": round(character_similarity, 6),
                "token_overlap": round(token_overlap, 6),
                "phrase_order": round(phrase_order, 6),
                "contiguous_fragment": contiguous_fragment,
                "compact_fragment": compact_fragment,
                "playback_active": active,
                "milliseconds_since_playback_stop": (
                    None if active else round(age_ns / 1_000_000, 3)
                ),
                "incoming_token_count": len(tokens),
            }
            if best is None or score > best[0]:
                best = (score, segment, signals)

        if best is None:
            return self._not_candidate("no_recent_spoken_segment")
        score, segment, signals = best
        strong_content_match = (
            float(signals["token_overlap"]) >= 0.78
            and (
                float(signals["character_similarity"]) >= 0.72
                or bool(signals["contiguous_fragment"])
                or bool(signals["compact_fragment"])
                or float(signals["phrase_order"]) >= 0.72
            )
        )
        timing_support = bool(signals["playback_active"]) or (
            signals["milliseconds_since_playback_stop"] is not None
            and float(signals["milliseconds_since_playback_stop"]) <= 650.0
        )
        exact_active_fragment = (
            bool(signals["playback_active"])
            and len(tokens) >= 4
            and (
                bool(signals["contiguous_fragment"])
                or bool(signals["compact_fragment"])
            )
        )
        suppress = len(tokens) >= 4 and timing_support and (
            (strong_content_match and score >= 0.72) or exact_active_fragment
        )
        if exact_active_fragment:
            score = max(score, 0.86)
        candidate = score >= 0.58
        return SelfSpeechAssessment(
            candidate=candidate,
            suppress=suppress,
            confidence=round(score, 6),
            rejection_reason="probable_assistant_self_speech" if suppress else None,
            matched_response_id=segment.response_id,
            matched_chunk_id=segment.chunk_id,
            signals=signals,
        )

    def _find(self, chunk_id: str) -> SpokenSegment | None:
        return next(
            (segment for segment in reversed(self._segments) if segment.chunk_id == chunk_id),
            None,
        )

    @staticmethod
    def _not_candidate(reason: str) -> SelfSpeechAssessment:
        return SelfSpeechAssessment(
            candidate=False,
            suppress=False,
            confidence=0.0,
            rejection_reason=None,
            matched_response_id=None,
            matched_chunk_id=None,
            signals={"classification_reason": reason},
        )
