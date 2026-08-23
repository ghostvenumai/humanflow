from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanflow.development import (
    ClaudeCliAdapter,
    CodexCliAdapter,
    DevelopmentModelRouter,
    EngineeringTask,
)


ROOT = Path(__file__).resolve().parents[2]


def test_router_matches_fixed_explainable_task_matrix() -> None:
    fixture = json.loads((ROOT / "eval" / "router" / "tasks.json").read_text())
    router = DevelopmentModelRouter()
    for raw in fixture["tasks"]:
        task_data = {key: value for key, value in raw.items() if key != "expected_tier"}
        decision = router.route(EngineeringTask(**task_data))
        assert decision.tier.value == raw["expected_tier"]
        assert decision.reason_codes
        assert decision.evidence


def test_tournament_budget_is_capped_and_frontier_requires_human_approval() -> None:
    router = DevelopmentModelRouter(max_tournament_usd=5)
    tournament = router.route(
        EngineeringTask(
            task_id="tournament",
            category="RACE_CONDITION",
            description="hard race",
            risk=0.9,
            criticality=0.9,
            ambiguity=0.9,
            realtime_impact=0.9,
            prior_failures=2,
            budget_usd=20,
        )
    )
    frontier = router.route(
        EngineeringTask(
            task_id="frontier",
            category="ARCHITECTURE",
            description="unresolved",
            risk=1,
            criticality=1,
            ambiguity=1,
            realtime_impact=1,
            prior_failures=3,
            budget_usd=10,
        )
    )
    assert tournament.maximum_budget_usd == 5
    assert tournament.execution_allowed
    assert not frontier.execution_allowed
    assert "human_budget_approval_required" in frontier.reason_codes


def test_cli_adapters_build_commands_but_refuse_unapproved_execution(tmp_path: Path) -> None:
    prompt = tmp_path / "task.md"
    prompt.write_text("bounded task", encoding="utf-8")
    for adapter in (CodexCliAdapter(), ClaudeCliAdapter()):
        command = adapter.command(prompt_file=prompt, worktree=tmp_path)
        assert command[-1] == "-"
        with pytest.raises(PermissionError, match="not authorized"):
            adapter.execute(prompt_file=prompt, worktree=tmp_path)
