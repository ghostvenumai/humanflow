"""Privacy-minimizing evidence references for the offline engineering loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping


class EvidenceKind(StrEnum):
    TEST_RESULT = "TEST_RESULT"
    METRIC = "METRIC"
    TELEMETRY_EVENT = "TELEMETRY_EVENT"
    HUMAN_REPORT = "HUMAN_REPORT"
    REVIEW_FINDING = "REVIEW_FINDING"
    RELEASE_MEASUREMENT = "RELEASE_MEASUREMENT"


class EvidenceScope(StrEnum):
    LOCAL_SYNTHETIC = "LOCAL_SYNTHETIC"
    LOCAL_TIMED_PCM = "LOCAL_TIMED_PCM"
    LOCAL_EVENT_LOOP = "LOCAL_EVENT_LOOP"
    REAL_BROWSER_SESSION = "REAL_BROWSER_SESSION"
    REAL_BROWSER_AUDIO_VALIDATED = "REAL_BROWSER_AUDIO_VALIDATED"
    PRODUCTION_TELEPHONY = "PRODUCTION_TELEPHONY"


_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "raw_audio",
        "raw_text",
        "secret",
        "token",
        "transcript",
        "user_text",
    }
)
_FORBIDDEN_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_credential",
    "_password",
    "_secret",
)


@dataclass(frozen=True, slots=True)
class EngineeringEvidenceRef:
    evidence_id: str
    kind: EvidenceKind
    scope: EvidenceScope
    uri: str
    observed_at_utc: str
    sha256: str | None = None
    sample_size: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.uri.strip():
            raise ValueError("evidence_id and uri must not be empty")
        observed = datetime.fromisoformat(self.observed_at_utc.replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("observed_at_utc must be timezone-aware")
        if self.sha256 is not None and (
            len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        if self.sample_size is not None and self.sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        safe_metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(safe_metadata))

    @classmethod
    def create(
        cls,
        *,
        evidence_id: str,
        kind: EvidenceKind,
        scope: EvidenceScope,
        uri: str,
        sha256: str | None = None,
        sample_size: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        observed_at_utc: datetime | None = None,
    ) -> "EngineeringEvidenceRef":
        timestamp = (observed_at_utc or datetime.now(UTC)).astimezone(UTC)
        return cls(
            evidence_id=evidence_id,
            kind=kind,
            scope=scope,
            uri=uri,
            observed_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
            sha256=sha256,
            sample_size=sample_size,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "scope": self.scope.value,
            "uri": self.uri,
            "observed_at_utc": self.observed_at_utc,
            "sha256": self.sha256,
            "sample_size": self.sample_size,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineeringEvidenceRef":
        return cls(
            evidence_id=str(payload["evidence_id"]),
            kind=EvidenceKind(str(payload["kind"])),
            scope=EvidenceScope(str(payload["scope"])),
            uri=str(payload["uri"]),
            observed_at_utc=str(payload["observed_at_utc"]),
            sha256=(str(payload["sha256"]) if payload.get("sha256") is not None else None),
            sample_size=(
                int(payload["sample_size"]) if payload.get("sample_size") is not None else None
            ),
            metadata=(
                payload.get("metadata", {})
                if isinstance(payload.get("metadata", {}), Mapping)
                else {}
            ),
        )


class AppendOnlyEvidenceStore:
    """Append references exactly once; never copy raw conversation content."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, evidence: EngineeringEvidenceRef) -> None:
        with self._lock:
            if any(item.evidence_id == evidence.evidence_id for item in self._read_unlocked()):
                raise ValueError(f"duplicate evidence_id: {evidence.evidence_id}")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()

    def read_all(self) -> tuple[EngineeringEvidenceRef, ...]:
        with self._lock:
            return self._read_unlocked()

    def _read_unlocked(self) -> tuple[EngineeringEvidenceRef, ...]:
        if not self.path.exists():
            return ()
        items: list[EngineeringEvidenceRef] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(EngineeringEvidenceRef.from_dict(json.loads(line)))
        return tuple(items)


def _validate_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for raw_key, child in value.items():
        key = str(raw_key)
        normalized = key.casefold().replace("-", "_")
        if normalized in _FORBIDDEN_METADATA_KEYS or normalized.endswith(_FORBIDDEN_SUFFIXES):
            raise ValueError(f"sensitive or raw evidence metadata is forbidden: {key}")
        if isinstance(child, Mapping):
            safe[key] = _validate_metadata(child)
        elif isinstance(child, (list, tuple)):
            safe[key] = [
                _validate_metadata(item) if isinstance(item, Mapping) else _scalar(item)
                for item in child
            ]
        else:
            safe[key] = _scalar(child)
    return safe


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"unsupported evidence metadata type: {type(value).__name__}")
