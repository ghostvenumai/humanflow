"""Real, session-scoped Anthropic reasoning provider for the browser demo."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from humanflow.domain.conversation import OperationToken

from .providers import ProviderInfo, ProviderMode
from .speech_text import safe_word_split, take_stable_speech_boundaries


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
genannter Daten an. Relative Datumsangaben löst HumanFlow deterministisch in
Europe/Berlin auf; verwende dafür ausschließlich den autoritativen Terminstatus.
HumanFlow kann die hinterlegten lokalen Demo-Verfügbarkeiten prüfen, aber nicht die
Live-Kalender beliebiger echter Praxen. Erkläre diese Grenze nur, wenn sie relevant
ist, und hilf anschließend so konkret wie möglich weiter. Behaupte bei
Terminwünschen niemals, ein Termin sei bereits gebucht, solange kein erfolgreicher
lokaler SQLite-Toolaufruf vorliegt. Die SQLite-Terminwerkzeuge enthalten ausschließlich
klar gekennzeichnete lokale Demo-Daten.
Beginne nicht mit einer generischen Empfangsbestätigung wie „Ich habe Sie verstanden“.
Schreibe Daten und Uhrzeiten so, dass eine deutsche Sprachausgabe sie eindeutig und
natürlich vorlesen kann. Rechenresultate gibst du immer als Ziffern in einem kurzen
gesprochenen Satz aus, zum Beispiel „Das Ergebnis ist 425.“, damit weder Schreibfehler
noch Mehrdeutigkeit in den Gesprächskontext gelangen.
"""

_GERMAN_MONTHS = (
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
)
_GERMAN_WEEKDAYS = (
    "Montag",
    "Dienstag",
    "Mittwoch",
    "Donnerstag",
    "Freitag",
    "Samstag",
    "Sonntag",
)
_SPOKEN_DATE = re.compile(
    rf"\b(\d{{1,2}})\.?(?:\s+|\s*,\s*)({'|'.join(_GERMAN_MONTHS)})\b",
    re.IGNORECASE,
)
_SPOKEN_WEEKDAY = re.compile(
    rf"\b({'|'.join(_GERMAN_WEEKDAYS)})\b",
    re.IGNORECASE,
)
_UNVERIFIED_EXTERNAL_ACTION = re.compile(
    r"\b(?:gebucht|eingetragen|storniert|abgesagt|gelöscht)\b",
    re.IGNORECASE,
)
_AI_DISCLOSURE = re.compile(
    r"\bich\s+bin\s+human\s*flow,?\s+ein\s+ki[- ]assistent(?:in)?[.!]?\s*",
    re.IGNORECASE,
)
_IDENTITY_REQUEST = re.compile(
    r"\b(?:wer|was)\s+bist\s+du\b|\bbist\s+du\s+(?:eine?\s+)?ki\b|"
    r"\bwer\s+ist\s+human\s*flow\b",
    re.IGNORECASE,
)
_DATE_QUERY = re.compile(
    r"\b(?:welches(?:\s+genaue)?\s+datum|wann\s+ist\s+(?:mein|der|dieser)\s+"
    r"(?:\w+-?)?termin|an\s+welchem\s+tag\s+ist\s+(?:mein|der|dieser)\s+"
    r"(?:\w+-?)?termin)\b",
    re.IGNORECASE,
)


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
        self._ai_disclosure_count = 0

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

    @property
    def ai_disclosure_count(self) -> int:
        return self._ai_disclosure_count

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
        pending = ""
        delivered_text: list[str] = []
        final_message: Any | None = None
        transaction_contract = _parse_transaction_contract(
            self._authoritative_transaction_context
        )
        authoritative_reply = _authoritative_database_reply(transaction_contract)
        if authoritative_reply is None:
            authoritative_reply = _authoritative_temporal_reply(
                transaction_contract, user_text
            )
        identity_requested = _IDENTITY_REQUEST.search(user_text) is not None

        if authoritative_reply is not None:
            guarded = self._enforce_ai_disclosure(
                authoritative_reply,
                identity_requested=identity_requested,
            )
            self._history.extend(
                (
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": guarded},
                )
            )
            self._history = self._history[-self._max_history_messages :]
            _assert_history_roles(self._history)
            self._last_usage = ReasoningUsage(input_tokens=0, output_tokens=0)
            yield guarded
            return

        system_prompt = self._system_prompt
        if self._ai_disclosure_count == 0:
            system_prompt += (
                "\n\nSESSION-INVARIANTE: Dies ist die erste Assistentenantwort. "
                "Die KI-Offenlegung muss jetzt genau einmal erfolgen."
            )
        elif not identity_requested:
            system_prompt += (
                "\n\nSESSION-INVARIANTE: Die KI-Offenlegung ist bereits erfolgt. "
                "Wiederhole die Vorstellung in dieser Antwort nicht."
            )
        if self._authoritative_transaction_context:
            system_prompt += (
                "\n\nAUTORITATIVER TRANSAKTIONSSTATUS (vom Conversation Controller, "
                "nicht aus Freitext-Historie rekonstruiert):\n"
                f"{self._authoritative_transaction_context}\n"
                "Die appointments-Objekte sind die einzige Wahrheit für Termine. "
                "Der response_contract ist eine zwingende Ausgabeinvariante: Verwende "
                "must_use_word, bestätige must_acknowledge_updated_values und frage "
                "niemals nach known_slots_never_ask_again. "
                "Ändere gedanklich nur resolved_appointment_id_this_turn; alle nicht "
                "in updated_slots_this_user_turn genannten Slots und alle anderen "
                "Terminobjekte bleiben unverändert. Wenn clarification_required wahr "
                "ist, frage knapp nach, welcher der genannten Termine gemeint ist, "
                "und triff keine Annahme. Frage nie erneut nach einem bereits gesetzten "
                "Slot. In jeder Antwort mit aktivem Terminstatus musst du ausdrücklich "
                "das Wort „Termin“ verwenden. "
                "Bestätige Änderungen knapp und stelle höchstens eine Frage zum "
                "nächsten fehlenden Slot. ISO-Daten sprichst du natürlich auf Deutsch. "
                "Behaupte niemals, dass ein Termin gebucht oder extern abgesagt wurde, "
                "wenn external_action_performed falsch ist. Sage dann nur, dass der "
                "Terminwunsch notiert beziehungsweise entfernt wurde. Nur ein echter "
                "erfolgreicher Tool-Aufruf erlaubt BOOKED oder die Bestätigung einer "
                "externen Absage. Ältere widersprüchliche Werte im Chatverlauf dürfen "
                "niemals wiederaufleben. Prüfe vor der Ausgabe jedes Satzes außerdem "
                "gegen forbidden_without_tool_success und verwende für rein lokalen "
                "Status immer das Substantiv „Terminwunsch“."
                " Relative Datumsangaben rechnest du niemals selbst aus; verwende "
                "ausschließlich das vom Controller gelieferte resolved_iso_date."
                " Wenn database_tool_result vorhanden ist, stammen Verfügbarkeit und "
                "gebuchte Termine ausschließlich daraus. Erfinde keine Slots oder "
                "Termine. Nur database_tool_result.success=true erlaubt eine bestätigte "
                "Buchung, Verschiebung oder Absage; bei false meldest du den Fehlschlag "
                "knapp und wahrheitsgemäß."
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
                pending += delta
                ready, pending = _take_speech_boundaries(pending)
                for fragment in ready:
                    guarded = _guard_transaction_fragment(fragment, transaction_contract)
                    guarded = self._enforce_ai_disclosure(
                        guarded, identity_requested=identity_requested
                    )
                    if not guarded:
                        continue
                    delivered_text.append(guarded)
                    yield guarded
            final_message = await stream.get_final_message()

        if pending.strip():
            guarded = _guard_transaction_fragment(pending.strip(), transaction_contract)
            guarded = self._enforce_ai_disclosure(
                guarded, identity_requested=identity_requested
            )
            if guarded:
                delivered_text.append(guarded)
                yield guarded

        assistant_text = " ".join(delivered_text).strip()
        if not assistant_text:
            assistant_text = "Was möchtest du als Nächstes wissen?"
            delivered_text.append(assistant_text)
            yield assistant_text
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

    def _enforce_ai_disclosure(
        self, text: str, *, identity_requested: bool
    ) -> str:
        stripped = text.strip()
        if self._ai_disclosure_count == 0:
            remainder = _AI_DISCLOSURE.sub("", stripped).strip(" ,")
            self._ai_disclosure_count = 1
            return (
                "Ich bin HumanFlow, ein KI-Assistent."
                if not remainder
                else f"Ich bin HumanFlow, ein KI-Assistent. {remainder}"
            )
        if identity_requested:
            return stripped
        return _AI_DISCLOSURE.sub("", stripped).strip(" ,")


def _take_speech_boundaries(text: str) -> tuple[list[str], str]:
    """Split complete sentences; cap long clauses at a whitespace boundary."""

    stable, pending = take_stable_speech_boundaries(text)
    fragments: list[str] = []
    for fragment in stable:
        fragments.extend(_split_long_speech_fragment(fragment))
    while len(pending) >= 220:
        split_at = safe_word_split(pending, minimum=120, maximum=220)
        fragments.append(pending[:split_at].strip())
        pending = pending[split_at:].lstrip()
    return fragments, pending


def _split_long_speech_fragment(text: str) -> list[str]:
    fragments: list[str] = []
    pending = text
    while len(pending) > 220:
        split_at = safe_word_split(pending, minimum=120, maximum=220)
        fragments.append(pending[:split_at].strip())
        pending = pending[split_at:].lstrip()
    if pending:
        fragments.append(pending)
    return fragments


def _parse_transaction_contract(context: str | None) -> dict[str, object] | None:
    if context is None:
        return None
    try:
        payload = json.loads(context)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _authoritative_database_reply(context: Mapping[str, object] | None) -> str | None:
    if context is None:
        return None
    raw_outcome = context.get("database_tool_result")
    if not isinstance(raw_outcome, Mapping):
        return None
    tool_name = raw_outcome.get("tool_name")
    result = raw_outcome.get("result")
    if not isinstance(tool_name, str) or not isinstance(result, Mapping):
        return None
    result_status = result.get("result_status")
    if raw_outcome.get("success") is not True:
        if result_status == "BOOKING_CONFLICT":
            return "Der gewünschte Termin ist leider nicht frei."
        if result_status == "UNAVAILABLE":
            return "Diesen Termin finde ich nicht als aktive Buchung."
        return "Das hat gerade nicht geklappt. Ich kann es noch einmal versuchen."
    if tool_name == "search_availability":
        slots = result.get("slots")
        alternatives = result.get("alternative_slots")
        requested_time = result.get("requested_time")
        if result_status == "UNAVAILABLE":
            prefix = (
                f"{_spoken_clock(requested_time)} ist leider nicht frei."
                if isinstance(requested_time, str)
                else "Dafür ist leider kein Termin frei."
            )
            alternative_rows = alternatives if isinstance(alternatives, list) else []
            alternative_starts = [
                slot.get("start_datetime")
                for slot in alternative_rows
                if isinstance(slot, Mapping)
                and isinstance(slot.get("start_datetime"), str)
            ]
            if alternative_starts:
                spoken = [
                    _spoken_datetime(value, include_date=False)
                    for value in alternative_starts[:3]
                ]
                return f"{prefix} Ich hätte {_join_german(spoken)} etwas frei."
            return prefix
        if not isinstance(slots, list) or not slots:
            return "Für diesen Tag ist in den Demo-Daten kein Termin frei."
        if isinstance(requested_time, str):
            return (
                f"{_spoken_clock(requested_time)} ist in den hinterlegten "
                "Demo-Terminen frei. "
                "Soll ich den Termin buchen?"
            )
        starts = [
            slot.get("start_datetime")
            for slot in slots
            if isinstance(slot, Mapping) and isinstance(slot.get("start_datetime"), str)
        ]
        if not starts:
            return "Für diesen Tag ist in den Demo-Daten kein Termin frei."
        appointment_types = {
            slot.get("appointment_type")
            for slot in slots
            if isinstance(slot, Mapping) and isinstance(slot.get("appointment_type"), str)
        }
        if len(appointment_types) > 1:
            descriptions = [
                f"{slot['appointment_type']} {_spoken_datetime(slot['start_datetime'])}"
                for slot in slots[:3]
                if isinstance(slot, Mapping)
                and isinstance(slot.get("appointment_type"), str)
                and isinstance(slot.get("start_datetime"), str)
            ]
            return f"In den hinterlegten Demo-Terminen sind {_join_german(descriptions)} frei."
        spoken = [_spoken_datetime(value, include_date=False) for value in starts[:3]]
        day = _spoken_datetime(starts[0], include_time=False)
        return (
            f"In den hinterlegten Demo-Terminen hätte ich {day} "
            f"{_join_german(spoken)} frei."
        )
    if tool_name == "create_appointment":
        start = result.get("start_datetime")
        appointment_type = result.get("appointment_type")
        if isinstance(start, str):
            subject = (
                f"dein {appointment_type}-Termin"
                if isinstance(appointment_type, str)
                else "dein Termin"
            )
            return (
                f"Alles klar, {subject} ist {_spoken_datetime(start)} "
                "in HumanFlow gebucht."
            )
    if tool_name == "reschedule_appointment":
        new_slot = result.get("new_slot")
        if isinstance(new_slot, Mapping) and isinstance(new_slot.get("start_datetime"), str):
            return (
                "Okay, ich habe den Termin auf "
                f"{_spoken_datetime(new_slot['start_datetime'])} "
                "in HumanFlow verschoben."
            )
    if tool_name == "cancel_appointment":
        appointment_type = result.get("appointment_type")
        subject = (
            f"der {appointment_type}-Termin"
            if isinstance(appointment_type, str)
            else "der Termin"
        )
        return f"Okay, {subject} ist in HumanFlow abgesagt."
    if tool_name == "list_appointments":
        appointments = result.get("appointments")
        if not isinstance(appointments, list) or not appointments:
            return "Du hast aktuell keine gebuchten Demo-Termine."
        descriptions = []
        for appointment in appointments:
            if not isinstance(appointment, Mapping):
                continue
            start = appointment.get("start_datetime")
            appointment_type = appointment.get("appointment_type")
            if isinstance(start, str) and isinstance(appointment_type, str):
                descriptions.append(
                    f"einen {appointment_type}-Termin {_spoken_datetime(start)}"
                )
        if descriptions:
            return f"Du hast {_join_german(descriptions, connector='und')}."
    return None


def _authoritative_temporal_reply(
    context: Mapping[str, object] | None,
    user_text: str,
) -> str | None:
    if context is None or not _DATE_QUERY.search(user_text):
        return None
    if bool(context.get("clarification_required")):
        return None
    appointment = _resolved_appointment(context)
    if appointment is None:
        return None
    raw_date = appointment.get("date")
    if not isinstance(raw_date, str):
        return None
    try:
        resolved = date.fromisoformat(raw_date)
    except ValueError:
        return None
    weekday = _GERMAN_WEEKDAYS[resolved.weekday()]
    month = _GERMAN_MONTHS[resolved.month - 1]
    return f"Der Termin ist am {weekday}, dem {resolved.day}. {month}."


def _spoken_datetime(
    value: str, *, include_date: bool = True, include_time: bool = True
) -> str:
    parsed = datetime.fromisoformat(value)
    parts: list[str] = []
    if include_date:
        parts.append(f"am {_GERMAN_WEEKDAYS[parsed.weekday()]}")
    if include_time:
        spoken_time = (
            f"{parsed.hour} Uhr"
            if parsed.minute == 0
            else f"{parsed.hour}:{parsed.minute:02d} Uhr"
        )
        parts.append(f"um {spoken_time}")
    return " ".join(parts)


def _spoken_clock(value: str) -> str:
    hour, minute = (int(part) for part in value.split(":"))
    return f"{hour} Uhr" if minute == 0 else f"{hour}:{minute:02d} Uhr"


def _join_german(values: list[str], *, connector: str = "oder") -> str:
    if len(values) < 2:
        return values[0] if values else ""
    return ", ".join(values[:-1]) + f" {connector} {values[-1]}"


def _guard_transaction_fragment(
    fragment: str, context: Mapping[str, object] | None
) -> str:
    """Repair only objectively false transaction claims before they reach TTS.

    The reasoner still owns wording in the normal case. The controller intervenes only
    when a sentence contradicts an authoritative slot or claims an unexecuted action.
    """

    text = fragment.strip()
    if not text or context is None:
        return text
    if bool(context.get("clarification_required")):
        if _UNVERIFIED_EXTERNAL_ACTION.search(text):
            return "Welchen Termin meinst du genau?"
        return text

    appointment = _resolved_appointment(context)
    if appointment is None:
        return text
    external_action_performed = bool(context.get("external_action_performed"))
    if not external_action_performed and _UNVERIFIED_EXTERNAL_ACTION.search(text):
        return _safe_appointment_acknowledgement(appointment)
    if _contradicts_authoritative_date(text, context, appointment):
        return _safe_appointment_acknowledgement(appointment)
    return text


def _resolved_appointment(
    context: Mapping[str, object],
) -> Mapping[str, object] | None:
    resolved_id = context.get("resolved_appointment_id_this_turn")
    appointments = context.get("appointments")
    if not isinstance(resolved_id, str) or not isinstance(appointments, Mapping):
        return None
    appointment = appointments.get(resolved_id)
    return appointment if isinstance(appointment, Mapping) else None


def _contradicts_authoritative_date(
    text: str,
    context: Mapping[str, object],
    appointment: Mapping[str, object],
) -> bool:
    response_contract = context.get("response_contract")
    if not isinstance(response_contract, Mapping):
        return False
    required = response_contract.get("must_acknowledge_updated_values")
    if not isinstance(required, Mapping) or "date" not in required:
        return False
    raw_date = appointment.get("date")
    if not isinstance(raw_date, str):
        return False
    try:
        expected = date.fromisoformat(raw_date)
    except ValueError:
        return False
    spoken_dates = tuple(_SPOKEN_DATE.finditer(text))
    if spoken_dates:
        for match in spoken_dates:
            month = next(
                (
                    index
                    for index, name in enumerate(_GERMAN_MONTHS, start=1)
                    if name.casefold() == match.group(2).casefold()
                ),
                None,
            )
            if int(match.group(1)) != expected.day or month != expected.month:
                return True
    spoken_weekdays = tuple(_SPOKEN_WEEKDAY.finditer(text))
    return any(
        match.group(1).casefold()
        != _GERMAN_WEEKDAYS[expected.weekday()].casefold()
        for match in spoken_weekdays
    )


def _safe_appointment_acknowledgement(appointment: Mapping[str, object]) -> str:
    purpose = appointment.get("purpose")
    subject = f"{purpose}-Terminwunsch" if isinstance(purpose, str) else "Terminwunsch"
    if appointment.get("status") == "CANCELLED":
        return f"Okay, den {subject} habe ich entfernt."

    details: list[str] = []
    raw_date = appointment.get("date")
    if isinstance(raw_date, str):
        try:
            parsed = date.fromisoformat(raw_date)
        except ValueError:
            pass
        else:
            details.append(
                f"{_GERMAN_WEEKDAYS[parsed.weekday()]}, den {parsed.day}. "
                f"{_GERMAN_MONTHS[parsed.month - 1]}"
            )
    raw_time = appointment.get("time")
    if isinstance(raw_time, str) and re.fullmatch(r"\d{2}:\d{2}", raw_time):
        hour, minute = raw_time.split(":", 1)
        details.append(
            f"{int(hour)} Uhr" if minute == "00" else f"{int(hour)}:{minute} Uhr"
        )
    suffix = f" für {' um '.join(details)}" if details else ""
    return f"Deinen {subject} habe ich{suffix} notiert."


def _assert_history_roles(history: list[dict[str, str]]) -> None:
    for index, message in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not message.get("content", "").strip():
            raise RuntimeError("conversation_history_role_invariant_violated")
