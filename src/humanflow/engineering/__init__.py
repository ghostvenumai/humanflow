"""Offline autonomous-engineering contracts, isolated from the live runtime."""

from .evidence import (
    AppendOnlyEvidenceStore,
    EngineeringEvidenceRef,
    EvidenceKind,
    EvidenceScope,
)
from .failures import (
    DiagnosticPackageWriter,
    FailureCircuitBreaker,
    FailureFingerprint,
    HarnessFailureState,
    IterationObservation,
    normalize_failure_error,
)
from .metrics import HarnessMetricsStore, TaskRunMetrics
from .registry import (
    ActorRole,
    EngineeringTaskRecord,
    TaskPriority,
    TaskRegistry,
    TaskRisk,
    TaskStatus,
)
from .reviewer import ReviewAssignment, ReviewResult, ReviewVerdict
from .scheduler import ConservativeTaskScheduler, ScheduleDecision, tasks_conflict
from .verification import (
    CandidateIntegrityGate,
    CommandEvidence,
    IntegrityReport,
    MergeGateEvidence,
    SupervisorCommandRunner,
    VerificationStatus,
    evaluate_merge_gate,
)
from .tournament import VerifiedTournamentCoordinator
from .worktrees import WorktreeLease, WorktreeManager

__all__ = [
    "ActorRole",
    "AppendOnlyEvidenceStore",
    "CandidateIntegrityGate",
    "CommandEvidence",
    "ConservativeTaskScheduler",
    "DiagnosticPackageWriter",
    "EngineeringEvidenceRef",
    "EngineeringTaskRecord",
    "EvidenceKind",
    "EvidenceScope",
    "FailureCircuitBreaker",
    "FailureFingerprint",
    "HarnessFailureState",
    "HarnessMetricsStore",
    "IntegrityReport",
    "IterationObservation",
    "MergeGateEvidence",
    "ReviewAssignment",
    "ReviewResult",
    "ReviewVerdict",
    "ScheduleDecision",
    "SupervisorCommandRunner",
    "TaskPriority",
    "TaskRegistry",
    "TaskRisk",
    "TaskStatus",
    "TaskRunMetrics",
    "VerifiedTournamentCoordinator",
    "VerificationStatus",
    "WorktreeLease",
    "WorktreeManager",
    "evaluate_merge_gate",
    "normalize_failure_error",
    "tasks_conflict",
]
