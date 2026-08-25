"""Policy-controlled conversion of evidence-backed problems into HF tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .discovery import ProblemCandidate, ProblemSeverity
from .registry import EngineeringTaskRecord, TaskPriority, TaskRegistry, TaskRisk


_COMPONENT_PATHS = {
    "temporal": ("src/humanflow/runtime/temporal.py",),
    "appointment": (
        "src/humanflow/runtime/appointment_state.py",
        "src/humanflow/tools/appointment_coordinator.py",
        "src/humanflow/tools/sqlite_appointments.py",
    ),
    "final_admission": ("src/humanflow/runtime/final_admission.py",),
    "realtime": ("src/humanflow/runtime/**", "src/humanflow/audio/**"),
    "telemetry": ("src/humanflow/telemetry/**",),
}
_COMPONENT_TESTS = {
    "temporal": ("tests/unit/test_temporal_resolver.py",),
    "appointment": (
        "tests/unit/test_appointment_state.py",
        "tests/unit/test_sqlite_appointment_tools.py",
    ),
    "final_admission": ("tests/unit/test_final_admission_matrix.py",),
    "realtime": ("tests/unit/test_realtime_session.py",),
    "telemetry": ("tests/unit/test_telemetry.py",),
}


@dataclass(frozen=True, slots=True)
class TaskProposalPolicy:
    minimum_confidence: float = 0.8
    protected_paths: tuple[str, ...] = (
        "tests/golden/**",
        "eval/golden/**",
        "config/quality-gates.yaml",
        "docs/METRICS.md",
        "schemas/metric-definitions.json",
        "sprint/**",
    )

    def propose(
        self,
        candidate: ProblemCandidate,
        *,
        task_id: str,
    ) -> EngineeringTaskRecord:
        if candidate.confidence < self.minimum_confidence:
            raise ValueError("problem candidate confidence is below proposal threshold")
        severity = candidate.severity
        risk = TaskRisk(severity.value)
        priority = {
            ProblemSeverity.CRITICAL: TaskPriority.P0,
            ProblemSeverity.HIGH: TaskPriority.P0,
            ProblemSeverity.MEDIUM: TaskPriority.P1,
            ProblemSeverity.LOW: TaskPriority.P2,
        }[severity]
        component = candidate.affected_component
        allowed = _COMPONENT_PATHS.get(component, ())
        tests = _COMPONENT_TESTS.get(component, ())
        if not allowed or not tests:
            raise ValueError(f"component requires human path/verification triage: {component}")
        return EngineeringTaskRecord(
            task_id=task_id,
            title=candidate.title,
            source=f"problem_candidate:{candidate.candidate_id}",
            priority=priority,
            risk=risk,
            problem_fingerprint=candidate.fingerprint,
            allowed_paths=allowed,
            protected_paths=self.protected_paths,
            verification=tuple(("python3", "-m", "pytest", "-q", test) for test in tests),
            target_metrics={"primary": candidate.proposed_target},
            evidence_refs=candidate.evidence_refs,
            human_approval_required=risk in {TaskRisk.HIGH, TaskRisk.CRITICAL},
        )


def register_proposal(
    registry: TaskRegistry,
    candidate: ProblemCandidate,
    *,
    task_id: str,
    policy: TaskProposalPolicy | None = None,
) -> EngineeringTaskRecord:
    return registry.add((policy or TaskProposalPolicy()).propose(candidate, task_id=task_id))
