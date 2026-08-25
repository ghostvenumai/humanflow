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
from .orchestrator import (
    EngineeringHarness,
    HarnessRunResult,
    ReviewerRunner,
    WorkerResult,
    WorkerRunner,
)
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
from .scheduler import (
    ConservativeTaskScheduler,
    ScheduleDecision,
    path_matches,
    tasks_conflict,
)
from .supervisor import (
    EngineeringSupervisor,
    SupervisorBatchOutcome,
    SupervisorTaskOutcome,
    TaskExecution,
)
from .verification import (
    CandidateIntegrityGate,
    CommandEvidence,
    FrozenBuildIntegrityGuard,
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
    "EngineeringHarness",
    "EngineeringSupervisor",
    "EngineeringTaskRecord",
    "EvidenceKind",
    "EvidenceInformedAgentRouter",
    "EvidenceScope",
    "FailureCircuitBreaker",
    "FailureFingerprint",
    "FailureSignal",
    "FrozenBuildIntegrityGuard",
    "HarnessFailureState",
    "HarnessRunResult",
    "HarnessMetricsStore",
    "ImprovementDiscoveryEngine",
    "IntegrityReport",
    "IterationObservation",
    "MetricDirection",
    "MetricSeries",
    "MergeGateEvidence",
    "ReviewAssignment",
    "ReviewerRunner",
    "ReviewResult",
    "ReviewVerdict",
    "PostReleaseStatus",
    "ProblemCandidate",
    "ProblemSeverity",
    "ReleaseCandidateBundle",
    "ReleaseMeasurement",
    "ScheduleDecision",
    "SupervisorCommandRunner",
    "SupervisorBatchOutcome",
    "SupervisorTaskOutcome",
    "TaskPriority",
    "TaskRegistry",
    "TaskRisk",
    "TaskStatus",
    "TaskRunMetrics",
    "TaskProposalPolicy",
    "TaskExecution",
    "VerifiedTournamentCoordinator",
    "VerificationStatus",
    "WorktreeLease",
    "WorktreeManager",
    "WorkerResult",
    "WorkerRunner",
    "evaluate_merge_gate",
    "evaluate_post_release",
    "normalize_failure_error",
    "path_matches",
    "register_proposal",
    "tasks_conflict",
]
