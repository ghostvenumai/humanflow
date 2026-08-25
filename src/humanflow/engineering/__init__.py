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
from .discovery import (
    DetectorType,
    DiscoveryPolicy,
    FailureSignal,
    ImprovementDiscoveryEngine,
    MetricDirection,
    MetricSeries,
    ProblemCandidate,
    ProblemSeverity,
)
from .metrics import HarnessMetricsStore, TaskRunMetrics
from .proposals import TaskProposalPolicy, register_proposal
from .release import (
    PostReleaseStatus,
    ReleaseCandidateBundle,
    ReleaseMeasurement,
    evaluate_post_release,
)
from .registry import (
    ActorRole,
    EngineeringTaskRecord,
    TaskPriority,
    TaskRegistry,
    TaskRisk,
    TaskStatus,
)
from .reviewer import ReviewAssignment, ReviewResult, ReviewVerdict
from .routing import AgentRoutingDecision, EvidenceInformedAgentRouter
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
    "AgentRoutingDecision",
    "AppendOnlyEvidenceStore",
    "CandidateIntegrityGate",
    "CommandEvidence",
    "ConservativeTaskScheduler",
    "DetectorType",
    "DiagnosticPackageWriter",
    "DiscoveryPolicy",
    "EngineeringEvidenceRef",
    "EngineeringTaskRecord",
    "EvidenceKind",
    "EvidenceInformedAgentRouter",
    "EvidenceScope",
    "FailureCircuitBreaker",
    "FailureFingerprint",
    "FailureSignal",
    "HarnessFailureState",
    "HarnessMetricsStore",
    "ImprovementDiscoveryEngine",
    "IntegrityReport",
    "IterationObservation",
    "MetricDirection",
    "MetricSeries",
    "MergeGateEvidence",
    "ReviewAssignment",
    "ReviewResult",
    "ReviewVerdict",
    "PostReleaseStatus",
    "ProblemCandidate",
    "ProblemSeverity",
    "ReleaseCandidateBundle",
    "ReleaseMeasurement",
    "ScheduleDecision",
    "SupervisorCommandRunner",
    "TaskPriority",
    "TaskRegistry",
    "TaskRisk",
    "TaskStatus",
    "TaskRunMetrics",
    "TaskProposalPolicy",
    "VerifiedTournamentCoordinator",
    "VerificationStatus",
    "WorktreeLease",
    "WorktreeManager",
    "evaluate_merge_gate",
    "evaluate_post_release",
    "normalize_failure_error",
    "register_proposal",
    "tasks_conflict",
]
