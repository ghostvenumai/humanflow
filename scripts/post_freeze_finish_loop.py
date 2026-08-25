#!/usr/bin/env python3
"""Inspect or persist bounded HumanFlow post-freeze loop state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanflow.finalization.finish_loop import FinishLoopState, verify_post_freeze_repository


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--state", type=Path, default=Path("reports/post-freeze-finish-loop.json"))
    parser.add_argument("--frozen-commit", default="798256e")
    parser.add_argument("--freeze-tag", default="everlast-72h-build")
    parser.add_argument("--phase", choices=("A", "B", "C"))
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--task")
    parser.add_argument(
        "--status",
        choices=("IN_PROGRESS", "KEEP", "REVERT", "STOPPED", "HUMAN_VALIDATION_PENDING"),
    )
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--resource-remaining-percent", type=int)
    arguments = parser.parse_args()

    evidence = verify_post_freeze_repository(
        arguments.repo,
        frozen_commit=arguments.frozen_commit,
        freeze_tag=arguments.freeze_tag,
    )
    if arguments.phase is None:
        payload = {"repository": evidence}
        if arguments.state.exists():
            payload["loop_state"] = json.loads(arguments.state.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    missing = [
        name
        for name, value in (
            ("--iteration", arguments.iteration),
            ("--task", arguments.task),
            ("--status", arguments.status),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required with --phase: {', '.join(missing)}")
    state = FinishLoopState(
        phase=arguments.phase,
        iteration=arguments.iteration,
        task=arguments.task,
        status=arguments.status,
        baseline_commit=evidence["head"],
        candidate_commit=(
            evidence["head"] if arguments.status != "IN_PROGRESS" else None
        ),
        tests_run=arguments.test,
        test_results=arguments.result,
        resource_remaining_percent=arguments.resource_remaining_percent,
    )
    state.save(arguments.state)
    print(arguments.state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
