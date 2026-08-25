"""Offline autonomous-engineering contracts, isolated from the live runtime."""

from .evidence import (
    AppendOnlyEvidenceStore,
    EngineeringEvidenceRef,
    EvidenceKind,
    EvidenceScope,
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
from .worktrees import WorktreeLease, WorktreeManager

__all__ = [
    "ActorRole",
    "AppendOnlyEvidenceStore",
    "CandidateIntegrityGate",
    "CommandEvidence",
    "ConservativeTaskScheduler",
    "EngineeringEvidenceRef",
    "EngineeringTaskRecord",
    "EvidenceKind",
    "EvidenceScope",
    "IntegrityReport",
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
    "VerificationStatus",
    "WorktreeLease",
    "WorktreeManager",
    "evaluate_merge_gate",
    "tasks_conflict",
]
