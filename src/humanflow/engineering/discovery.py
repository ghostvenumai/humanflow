"""Conservative problem discovery over sanitized offline evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from statistics import fmean, pstdev


class DetectorType(StrEnum):
    REGRESSION = "REGRESSION"
    THRESHOLD = "THRESHOLD"
    CLUSTER = "CLUSTER"
    ANOMALY = "ANOMALY"
    HIGH_SEVERITY = "HIGH_SEVERITY"
    COVERAGE_GAP = "COVERAGE_GAP"
    DRIFT = "DRIFT"


class ProblemSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


@dataclass(frozen=True, slots=True)
class MetricSeries:
    metric_name: str
    affected_component: str
    direction: MetricDirection
    target: float
    baseline_values: tuple[float, ...]
    current_values: tuple[float, ...]
    sample_size: int
    first_seen: str
    last_seen: str
    evidence_refs: tuple[str, ...]
    reproduction_hint: str

    def __post_init__(self) -> None:
        if not self.metric_name.strip() or not self.affected_component.strip():
            raise ValueError("metric identity must not be empty")
        if not self.baseline_values or not self.current_values or self.sample_size < 1:
            raise ValueError("metric evidence requires baseline, current and sample size")
        if not self.evidence_refs:
            raise ValueError("metric evidence_refs must not be empty")


@dataclass(frozen=True, slots=True)
class FailureSignal:
    fingerprint: str
    title: str
    affected_component: str
    severity: ProblemSeverity
    confidence: float
    occurrences: int
    first_seen: str
    last_seen: str
    evidence_refs: tuple[str, ...]
    reproduction_hint: str
    has_regression_test: bool

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1 or self.occurrences < 1:
            raise ValueError("failure confidence/count is invalid")
        if not self.fingerprint.strip() or not self.evidence_refs:
            raise ValueError("failure signal needs fingerprint and evidence")


@dataclass(frozen=True, slots=True)
class ProblemCandidate:
    candidate_id: str
    fingerprint: str
    detector_type: DetectorType
    title: str
    affected_component: str
    severity: ProblemSeverity
    confidence: float
    sample_size: int
    first_seen: str
    last_seen: str
    evidence_refs: tuple[str, ...]
    reproduction_hint: str
    baseline_metric: float | None
    current_metric: float | None
    proposed_target: str
    status: str = "proposed"


@dataclass(frozen=True, slots=True)
class DiscoveryPolicy:
    minimum_metric_sample_size: int = 20
    minimum_cluster_occurrences: int = 3
    minimum_confidence: float = 0.8
    anomaly_standard_deviations: float = 3.0
    anomaly_relative_change: float = 0.2


class ImprovementDiscoveryEngine:
    def __init__(self, policy: DiscoveryPolicy | None = None) -> None:
        self.policy = policy or DiscoveryPolicy()

    def analyze(
        self,
        *,
        metrics: tuple[MetricSeries, ...] = (),
        failures: tuple[FailureSignal, ...] = (),
    ) -> tuple[ProblemCandidate, ...]:
        candidates = [
            candidate
            for series in metrics
            if (candidate := self._metric_candidate(series)) is not None
        ]
        candidates.extend(
            candidate
            for signal in failures
            if (candidate := self._failure_candidate(signal)) is not None
        )
        by_fingerprint: dict[str, ProblemCandidate] = {}
        for candidate in candidates:
            current = by_fingerprint.get(candidate.fingerprint)
            if current is None or _severity_rank(candidate.severity) > _severity_rank(
                current.severity
            ):
                by_fingerprint[candidate.fingerprint] = candidate
        return tuple(by_fingerprint[key] for key in sorted(by_fingerprint))

    def _metric_candidate(self, series: MetricSeries) -> ProblemCandidate | None:
        if series.sample_size < self.policy.minimum_metric_sample_size:
            return None
        baseline = fmean(series.baseline_values)
        current = fmean(series.current_values)
        baseline_healthy = _meets_target(baseline, series.target, series.direction)
        current_healthy = _meets_target(current, series.target, series.direction)
        detector: DetectorType | None = None
        severity = ProblemSeverity.MEDIUM
        confidence = min(0.99, 0.8 + min(series.sample_size, 200) / 1_000)
        if baseline_healthy and not current_healthy:
            detector = DetectorType.REGRESSION
            severity = ProblemSeverity.HIGH
        elif _is_drift(series.current_values, series.direction) and not current_healthy:
            detector = DetectorType.DRIFT
        elif not current_healthy:
            detector = DetectorType.THRESHOLD
        elif _is_anomaly(series, baseline, current, self.policy):
            detector = DetectorType.ANOMALY
        if detector is None:
            return None
        fingerprint = f"metric:{series.affected_component}:{series.metric_name}"
        return _candidate(
            fingerprint=fingerprint,
            detector=detector,
            title=f"Investigate {series.metric_name} {detector.value.casefold()}",
            component=series.affected_component,
            severity=severity,
            confidence=confidence,
            sample_size=series.sample_size,
            first_seen=series.first_seen,
            last_seen=series.last_seen,
            evidence_refs=series.evidence_refs,
            reproduction_hint=series.reproduction_hint,
            baseline=baseline,
            current=current,
            target=f"{series.direction.value}:{series.target}",
        )

    def _failure_candidate(self, signal: FailureSignal) -> ProblemCandidate | None:
        high_severity = signal.severity in {
            ProblemSeverity.HIGH,
            ProblemSeverity.CRITICAL,
        }
        if high_severity and signal.confidence >= self.policy.minimum_confidence:
            detector = DetectorType.HIGH_SEVERITY
        elif (
            signal.occurrences >= self.policy.minimum_cluster_occurrences
            and signal.confidence >= self.policy.minimum_confidence
        ):
            detector = (
                DetectorType.COVERAGE_GAP
                if not signal.has_regression_test
                else DetectorType.CLUSTER
            )
        else:
            return None
        return _candidate(
            fingerprint=f"failure:{signal.fingerprint}",
            detector=detector,
            title=signal.title,
            component=signal.affected_component,
            severity=signal.severity,
            confidence=signal.confidence,
            sample_size=signal.occurrences,
            first_seen=signal.first_seen,
            last_seen=signal.last_seen,
            evidence_refs=signal.evidence_refs,
            reproduction_hint=signal.reproduction_hint,
            baseline=None,
            current=None,
            target="zero_recurrence_after_verified_fix",
        )


def _candidate(
    *,
    fingerprint: str,
    detector: DetectorType,
    title: str,
    component: str,
    severity: ProblemSeverity,
    confidence: float,
    sample_size: int,
    first_seen: str,
    last_seen: str,
    evidence_refs: tuple[str, ...],
    reproduction_hint: str,
    baseline: float | None,
    current: float | None,
    target: str,
) -> ProblemCandidate:
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
    return ProblemCandidate(
        candidate_id=f"PC-{digest}",
        fingerprint=fingerprint,
        detector_type=detector,
        title=title,
        affected_component=component,
        severity=severity,
        confidence=confidence,
        sample_size=sample_size,
        first_seen=first_seen,
        last_seen=last_seen,
        evidence_refs=evidence_refs,
        reproduction_hint=reproduction_hint,
        baseline_metric=baseline,
        current_metric=current,
        proposed_target=target,
    )


def _meets_target(value: float, target: float, direction: MetricDirection) -> bool:
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return value >= target
    return value <= target


def _is_drift(values: tuple[float, ...], direction: MetricDirection) -> bool:
    if len(values) < 3:
        return False
    recent = values[-3:]
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return recent[0] > recent[1] > recent[2]
    return recent[0] < recent[1] < recent[2]


def _is_anomaly(
    series: MetricSeries,
    baseline: float,
    current: float,
    policy: DiscoveryPolicy,
) -> bool:
    degraded = (
        current < baseline
        if series.direction is MetricDirection.HIGHER_IS_BETTER
        else current > baseline
    )
    deviation = pstdev(series.baseline_values)
    absolute_change = abs(current - baseline)
    relative_change = absolute_change / max(abs(baseline), 1e-12)
    return bool(
        degraded
        and relative_change >= policy.anomaly_relative_change
        and (deviation == 0 or absolute_change >= policy.anomaly_standard_deviations * deviation)
    )


def _severity_rank(severity: ProblemSeverity) -> int:
    return {
        ProblemSeverity.LOW: 0,
        ProblemSeverity.MEDIUM: 1,
        ProblemSeverity.HIGH: 2,
        ProblemSeverity.CRITICAL: 3,
    }[severity]
