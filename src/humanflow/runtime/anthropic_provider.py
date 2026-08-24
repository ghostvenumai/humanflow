"""Real, session-scoped Anthropic reasoning provider for the browser demo."""

from __future__ import annotations

import re
from collections.abc import Mapping
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from humanflow.domain.conversation import OperationToken

from .providers import ProviderInfo, ProviderMode


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SYSTEM_PROMPT = """\
Du bist HumanFlow, ein deutschsprachiger KI-Voice-Agent in einer Live-Demo.
Beginne deinen ersten Assistentenbeitrag einmalig und knapp mit der Offenlegung:
„Ich bin HumanFlow, ein KI-Assistent.“ Sprich danach natürlich weiter und wiederhole
diese Offenlegung nicht ungefragt. Du darfst niemals behaupten, ein Mensch zu sein.
Antworte direkt, semantisch relevant, natürlich, informell und gut sprechbar. Eine
normale Voice-Antwort besteht aus ein bis zwei kurzen Sätzen und bleibt meist unter
dreißig Wörtern. Nur wenn der Nutzer ausdrücklich eine Erklärung oder Details möchte,
darfst du länger antworten; strukturiere sie dann in kurze, unterbrechbare Gedanken.
Bei transaktionalen Abläufen bestätigst du knapp, stellst genau eine relevante nächste
Frage und hörst dann zu. Gib keine ungefragten Erklärungen oder langen Disclaimer. Verwende
ausschließlich reinen gesprochenen Text: kein Markdown, keine Überschriften und keine
Listenzeichen. Vermeide schriftsprachliche Monologe, unnötig vollständige Formulierungen,
formelhafte Hilfsangebote und Wiederholungen dessen, was der Nutzer gerade gesagt hat.
Nutze den bisherigen
Gesprächsverlauf für Rückfragen und Verweise. Erfinde keine Live-Daten und keine
ausgeführten Aktionen. Bei Fragen zum aktuellen Wetter sage ausdrücklich, dass du
keinen Zugriff auf aktuelle Wetterdaten hast, und biete Hilfe anhand vom Nutzer
genannter Daten an. Du hast derzeit auch keinen echten Kalender. Erkläre diese
Grenzen natürlich und hilf anschließend so konkret wie möglich weiter. Behaupte bei
Terminwünschen niemals, ein Termin sei bereits gebucht.
Beginne nicht mit einer generischen Empfangsbestätigung wie „Ich habe Sie verstanden“.
Schreibe Daten und Uhrzeiten so, dass eine deutsche Sprachausgabe sie eindeutig und
natürlich vorlesen kann. Rechenresultate gibst du immer als Ziffern in einem kurzen
gesprochenen Satz aus, zum Beispiel „Das Ergebnis ist 425.“, damit weder Schreibfehler
noch Mehrdeutigkeit in den Gesprächskontext gelangen.
"""

_SEMANTIC_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")


@dataclass(frozen=True, slots=True)
class ReasoningUsage:
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class AnthropicReasoner:
    """Stream sentence-sized German output while retaining per-session context."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 200,
        max_history_turns: int = 12,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("ANTHROPIC_API_KEY is required")
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be positive")
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._max_history_messages = max_history_turns * 2
        self._history: list[dict[str, str]] = []
        self._last_usage: ReasoningUsage | None = None
        self._authoritative_transaction_context: str | None = None

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            role="reasoning",
            provider="anthropic-messages-api",
            model=self._model,
            mode=ProviderMode.REAL,
            runtime="server",
        )

    @property
    def last_usage(self) -> ReasoningUsage | None:
        return self._last_usage

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        """A copy for diagnostics/tests; no credentials or hidden provider state."""

        return tuple(dict(message) for message in self._history)

    def set_authoritative_transaction_context(
        self,
        context: str | None,
        *,
        state: Mapping[str, object] | None = None,
    ) -> None:
        """Set controller-owned state without rewriting the human transcript/history."""

        del state
        self._authoritative_transaction_context = context

    async def stream_response(
        self, transcript: str, token: OperationToken
    ) -> AsyncIterator[str]:
        del token
        user_text = transcript.strip()
        if not user_text:
            raise ValueError("transcript must not be empty")
        _assert_history_roles(self._history)

        request_messages = [
            *self._history,
            {"role": "user", "content": user_text},
        ]
        complete_text = ""
        pending = ""
        final_message: Any | None = None

        system_prompt = self._system_prompt
        if self._authoritative_transaction_context:
            system_prompt += (
                "\n\nAUTORITATIVER TRANSAKTIONSSTATUS (vom Conversation Controller, "
                "nicht aus Freitext-Historie rekonstruiert):\n"
                f"{self._authoritative_transaction_context}\n"
                "Die appointments-Objekte sind die einzige Wahrheit für Termine. "
                "Ändere gedanklich nur resolved_appointment_id_this_turn; alle nicht "
                "in updated_slots_this_user_turn genannten Slots und alle anderen "
                "Terminobjekte bleiben unverändert. Wenn clarification_required wahr "
                "ist, frage knapp nach, welcher der genannten Termine gemeint ist, "
                "und triff keine Annahme. Frage nie erneut nach einem bereits gesetzten "
                "Slot. Bei aktivem Terminstatus musst du ausdrücklich das Wort „Termin“ verwenden. "
                "Bestätige Änderungen knapp und stelle höchstens eine Frage zum "
                "nächsten fehlenden Slot. ISO-Daten sprichst du natürlich auf Deutsch. "
                "Behaupte niemals, dass ein Termin gebucht oder extern abgesagt wurde, "
                "wenn external_action_performed falsch ist. Sage dann nur, dass der "
                "Terminwunsch notiert beziehungsweise entfernt wurde. Nur ein echter "
                "erfolgreicher Tool-Aufruf erlaubt BOOKED oder die Bestätigung einer "
                "externen Absage. Ältere widersprüchliche Werte im Chatverlauf dürfen "
                "niemals wiederaufleben."
            )

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=request_messages,
        ) as stream:
            async for delta in stream.text_stream:
                if not isinstance(delta, str) or not delta:
                    continue
                complete_text += delta
                pending += delta
                ready, pending = _take_speech_boundaries(pending)
                for fragment in ready:
                    yield fragment
            final_message = await stream.get_final_message()

        if pending.strip():
            yield pending.strip()

        assistant_text = complete_text.strip()
        if not assistant_text:
            raise RuntimeError("reasoning_provider_returned_empty_text")
        self._history.extend(
            (
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            )
        )
        self._history = self._history[-self._max_history_messages :]
        _assert_history_roles(self._history)
        usage = getattr(final_message, "usage", None)
        self._last_usage = ReasoningUsage(
            input_tokens=int(getattr(usage, "input_tokens", 0)),
            output_tokens=int(getattr(usage, "output_tokens", 0)),
        )


def _take_speech_boundaries(text: str) -> tuple[list[str], str]:
    """Split complete sentences; cap long clauses at a whitespace boundary."""

    fragments: list[str] = []
    pending = text
    while True:
        match = _SEMANTIC_BOUNDARY.search(pending)
        if match is not None:
            fragment = pending[: match.start()].strip()
            pending = pending[match.end() :]
            if fragment:
                fragments.append(fragment)
            continue
        if len(pending) >= 220:
            split_at = pending.rfind(" ", 120, 220)
            if split_at > 0:
                fragments.append(pending[:split_at].strip())
                pending = pending[split_at + 1 :]
                continue
        return fragments, pending


def _assert_history_roles(history: list[dict[str, str]]) -> None:
    for index, message in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not message.get("content", "").strip():
            raise RuntimeError("conversation_history_role_invariant_violated")
