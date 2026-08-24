"""TTS-aware German boundaries and pronunciation-only text normalization."""

from __future__ import annotations

import re


GERMAN_MONTHS = (
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
_MONTH_ALTERNATION = "|".join(GERMAN_MONTHS)
_BOUNDARY_CANDIDATE = re.compile(r"[.!?](?=\s|$)")
_ORDINAL_BEFORE_MONTH = re.compile(r"(?:^|\s)(\d{1,2})\.$")
_PROTECTED_ABBREVIATION = re.compile(
    r"(?:\bz\.|\bz\.\s*b\.|\bdr\.|\bbzw\.|\bca\.)$", re.IGNORECASE
)
_DATE = re.compile(
    rf"(?P<prefix>\b(?:am|den|zum|der)\s+)?(?P<day>[1-9]|[12]\d|3[01])\.\s+"
    rf"(?P<month>{_MONTH_ALTERNATION})\b",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"\b(?P<hour>[01]?\d|2[0-3])(?P<separator>[:.])(?P<minute>[0-5]\d)\s*Uhr\b",
    re.IGNORECASE,
)
_HOUR = re.compile(r"\b(?P<hour>[01]?\d|2[0-3])\s+Uhr\b", re.IGNORECASE)
_GROUPED_DECIMAL = re.compile(r"\b(?P<integer>\d{1,3}(?:\.\d{3})+),(?P<fraction>\d+)\b")
_DECIMAL = re.compile(r"\b(?P<integer>\d+),(?P<fraction>\d+)\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def take_stable_speech_boundaries(text: str) -> tuple[list[str], str]:
    """Return only streaming-safe complete sentences and a pending suffix."""

    fragments: list[str] = []
    pending = text
    while True:
        boundary = _first_boundary(pending, streaming=True)
        if boundary is None:
            return fragments, pending
        end, next_start = boundary
        fragment = pending[:end].strip()
        pending = pending[next_start:]
        if fragment:
            fragments.append(fragment)


def split_tts_sentences(text: str) -> tuple[str, ...]:
    """Split complete text without breaking dates, times or abbreviations."""

    fragments: list[str] = []
    pending = text.strip()
    while pending:
        boundary = _first_boundary(pending, streaming=False)
        if boundary is None:
            fragments.append(pending)
            break
        end, next_start = boundary
        fragment = pending[:end].strip()
        if fragment:
            fragments.append(fragment)
        pending = pending[next_start:].strip()
    return tuple(fragments)


def safe_word_split(text: str, *, minimum: int, maximum: int) -> int:
    """Find a whitespace split that does not separate a protected speech token."""

    positions = [
        match.start()
        for match in re.finditer(r"\s+", text[: maximum + 1])
        if match.start() >= minimum
    ]
    for position in reversed(positions):
        if not _protected_join(text[:position], text[position:]):
            return position
    return maximum


class GermanSpeechNormalizer:
    """Create provider speech text while preserving the displayed assistant text."""

    def normalize(self, text: str) -> str:
        spoken = " ".join(text.split())
        spoken = re.sub(r"\bz\.\s*B\.", "zum Beispiel", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bDr\.", "Doktor", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bbzw\.", "beziehungsweise", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bca\.", "circa", spoken, flags=re.IGNORECASE)
        spoken = _GROUPED_DECIMAL.sub(_replace_grouped_decimal, spoken)
        spoken = _DECIMAL.sub(_replace_decimal, spoken)
        spoken = _TIME.sub(_replace_time, spoken)
        spoken = _HOUR.sub(_replace_hour, spoken)
        spoken = _DATE.sub(_replace_date, spoken)
        spoken = _YEAR.sub(lambda match: german_cardinal(int(match.group(0))), spoken)
        return " ".join(spoken.split())


def _first_boundary(text: str, *, streaming: bool) -> tuple[int, int] | None:
    for match in _BOUNDARY_CANDIDATE.finditer(text):
        punctuation_index = match.start()
        end = match.end()
        next_start = end
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        if text[punctuation_index] == "." and _protected_period(
            text, punctuation_index, next_start, streaming=streaming
        ):
            continue
        return end, next_start
    return None


def _protected_period(
    text: str, period_index: int, next_start: int, *, streaming: bool
) -> bool:
    prefix = text[: period_index + 1]
    suffix = text[next_start:]
    if _PROTECTED_ABBREVIATION.search(prefix):
        return True
    ordinal = _ORDINAL_BEFORE_MONTH.search(prefix)
    if ordinal is None:
        return False
    if not suffix:
        return streaming
    return re.match(rf"(?:{_MONTH_ALTERNATION})\b", suffix, re.IGNORECASE) is not None


def _protected_join(left: str, right: str) -> bool:
    left = left.rstrip()
    right = right.lstrip()
    if _ORDINAL_BEFORE_MONTH.search(left) and re.match(
        rf"(?:{_MONTH_ALTERNATION})\b", right, re.IGNORECASE
    ):
        return True
    if re.search(r"\b\d{1,2}(?::|\.)\d{2}$", left) and re.match(
        r"Uhr\b", right, re.IGNORECASE
    ):
        return True
    if re.search(r"\bz\.$", left, re.IGNORECASE) and re.match(
        r"B\.\b", right, re.IGNORECASE
    ):
        return True
    return bool(
        re.search(r"\bDr\.$", left, re.IGNORECASE)
        and re.match(r"[A-ZÄÖÜ]", right)
    )


def _replace_grouped_decimal(match: re.Match[str]) -> str:
    integer = int(match.group("integer").replace(".", ""))
    fraction = " ".join(german_cardinal(int(digit)) for digit in match.group("fraction"))
    return f"{german_cardinal(integer)} Komma {fraction}"


def _replace_decimal(match: re.Match[str]) -> str:
    fraction = " ".join(german_cardinal(int(digit)) for digit in match.group("fraction"))
    return f"{german_cardinal(int(match.group('integer')))} Komma {fraction}"


def _replace_time(match: re.Match[str]) -> str:
    hour = german_cardinal(int(match.group("hour")))
    minute = int(match.group("minute"))
    return f"{hour} Uhr" if minute == 0 else f"{hour} Uhr {german_cardinal(minute)}"


def _replace_hour(match: re.Match[str]) -> str:
    return f"{german_cardinal(int(match.group('hour')))} Uhr"


def _replace_date(match: re.Match[str]) -> str:
    prefix = match.group("prefix") or ""
    normalized_prefix = prefix.casefold().strip()
    stem = german_ordinal_stem(int(match.group("day")))
    if normalized_prefix in {"am", "den", "zum"}:
        ordinal = f"{stem}en"
    elif normalized_prefix == "der":
        ordinal = f"{stem}e"
    else:
        ordinal = f"{stem}er"
    return f"{prefix}{ordinal} {match.group('month')}"


def german_ordinal_stem(value: int) -> str:
    special = {1: "erst", 3: "dritt", 7: "siebt", 8: "acht"}
    if value in special:
        return special[value]
    suffix = "t" if value < 20 else "st"
    return f"{german_cardinal(value, compound_one=True)}{suffix}"


def german_cardinal(value: int, *, compound_one: bool = False) -> str:
    if value < 0 or value > 999_999:
        return str(value)
    units = (
        "null",
        "ein" if compound_one else "eins",
        "zwei",
        "drei",
        "vier",
        "fünf",
        "sechs",
        "sieben",
        "acht",
        "neun",
        "zehn",
        "elf",
        "zwölf",
        "dreizehn",
        "vierzehn",
        "fünfzehn",
        "sechzehn",
        "siebzehn",
        "achtzehn",
        "neunzehn",
    )
    if value < 20:
        return units[value]
    if value < 100:
        tens = {
            2: "zwanzig",
            3: "dreißig",
            4: "vierzig",
            5: "fünfzig",
            6: "sechzig",
            7: "siebzig",
            8: "achtzig",
            9: "neunzig",
        }
        ten, unit = divmod(value, 10)
        if unit == 0:
            return tens[ten]
        one = "ein" if unit == 1 else german_cardinal(unit)
        return f"{one}und{tens[ten]}"
    if value < 1_000:
        hundred, rest = divmod(value, 100)
        prefix = "einhundert" if hundred == 1 else f"{german_cardinal(hundred)}hundert"
        return prefix if rest == 0 else f"{prefix}{german_cardinal(rest)}"
    thousand, rest = divmod(value, 1_000)
    prefix = "eintausend" if thousand == 1 else f"{german_cardinal(thousand)}tausend"
    return prefix if rest == 0 else f"{prefix}{german_cardinal(rest)}"
