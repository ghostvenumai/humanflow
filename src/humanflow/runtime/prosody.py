"""Deterministic, wording-preserving prosody plans for conversational German."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .speech_text import safe_word_split, split_tts_sentences

_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:])\s+")


class SpeechIntent(StrEnum):
    QUESTION = "question"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"
    EMPATHY = "empathy"
    UNCERTAINTY = "uncertainty"
    TOOL_WAIT = "tool_wait"
    INFORMATION = "information"


@dataclass(frozen=True, slots=True)
class ProsodySegment:
    text: str
    intent: SpeechIntent
    pause_after_ms: int
    speaking_rate: float
    stability: float
    similarity_boost: float
    style: float
    use_speaker_boost: bool


@dataclass(frozen=True, slots=True)
class ProsodyPlanner:
    """Map stable semantic text to speech controls without rewriting its words."""

    maximum_segment_characters: int = 170

    def plan(self, text: str) -> tuple[ProsodySegment, ...]:
        clean = " ".join(text.split())
        if not clean:
            raise ValueError("text must not be empty")
        pieces: list[str] = []
        for sentence in split_tts_sentences(clean):
            pieces.extend(self._split_long_segment(sentence.strip()))
        return tuple(self._segment(piece) for piece in pieces if piece)

    def _split_long_segment(self, text: str) -> list[str]:
        if len(text) <= self.maximum_segment_characters:
            return [text]
        clauses = [item.strip() for item in _CLAUSE_BOUNDARY.split(text) if item.strip()]
        if len(clauses) > 1:
            packed: list[str] = []
            pending = ""
            for clause in clauses:
                candidate = f"{pending} {clause}".strip()
                if pending and len(candidate) > self.maximum_segment_characters:
                    packed.extend(self._split_at_word(pending))
                    pending = clause
                else:
                    pending = candidate
            if pending:
                packed.extend(self._split_at_word(pending))
            return packed
        return self._split_at_word(text)

    def _split_at_word(self, text: str) -> list[str]:
        remaining = text
        pieces: list[str] = []
        while len(remaining) > self.maximum_segment_characters:
            split_at = safe_word_split(
                remaining,
                minimum=min(80, self.maximum_segment_characters // 2),
                maximum=self.maximum_segment_characters,
            )
            pieces.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    def _segment(self, text: str) -> ProsodySegment:
        intent = _infer_intent(text)
        controls = {
            SpeechIntent.QUESTION: (0.98, 0.43, 0.78, 0.16),
            SpeechIntent.CONFIRMATION: (1.02, 0.42, 0.78, 0.14),
            SpeechIntent.CORRECTION: (0.96, 0.47, 0.79, 0.13),
            SpeechIntent.EMPATHY: (0.94, 0.46, 0.80, 0.18),
            SpeechIntent.UNCERTAINTY: (0.96, 0.49, 0.78, 0.10),
            SpeechIntent.TOOL_WAIT: (0.98, 0.50, 0.78, 0.08),
            SpeechIntent.INFORMATION: (1.00, 0.48, 0.78, 0.08),
        }
        rate, stability, similarity, style = controls[intent]
        if text.endswith("?"):
            pause = 220
        elif text.endswith(("!", ".")):
            pause = 190
        elif text.endswith(","):
            pause = 80
        elif text.endswith((";", ":")):
            pause = 110
        else:
            pause = 0
        return ProsodySegment(
            text=text,
            intent=intent,
            pause_after_ms=pause,
            speaking_rate=rate,
            stability=stability,
            similarity_boost=similarity,
            style=style,
            use_speaker_boost=False,
        )


def _infer_intent(text: str) -> SpeechIntent:
    normalized = text.casefold().strip()
    if normalized.endswith("?"):
        return SpeechIntent.QUESTION
    if any(word in normalized for word in ("tut mir leid", "verstehe, dass", "schwierig")):
        return SpeechIntent.EMPATHY
    if normalized.startswith(("nein", "korrektur", "stattdessen")) or any(
        marker in normalized for marker in ("ich meinte", "doch lieber", "korrigiere")
    ):
        return SpeechIntent.CORRECTION
    if normalized.startswith(("ja", "klar", "genau", "gut", "passt")):
        return SpeechIntent.CONFIRMATION
    if any(word in normalized for word in ("vielleicht", "vermutlich", "ich bin nicht sicher")):
        return SpeechIntent.UNCERTAINTY
    if any(word in normalized for word in ("einen moment", "ich prüfe", "kurz warten")):
        return SpeechIntent.TOOL_WAIT
    return SpeechIntent.INFORMATION
