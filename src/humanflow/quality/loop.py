"""Worktree evaluation and KEEP/REVERT decision evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic_ns
from typing import Any, Mapping, Sequence

import yaml


EVALUATION_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python3", "-m", "pytest", "-q"),
    ("ruff", "check", "."),
    ("python3", "scripts/benchmark_turns.py"),
    ("python3", "scripts/benchmark_realtime_core.py"),
    ("python3", "scripts/run_torture.py"),
    ("python3", "scripts/evaluate_runtime_quality.py"),
    ("python3", "scripts/build_scorecard.py"),
)


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    root: str
    git_commit: str
    generated_at_utc: str
    protected_hashes: Mapping[str, str]
    commands: tuple[Mapping[str, Any], ...]
    runtime_quality_score: float
    runtime_quality_passed: int
    runtime_quality_total: int
    immutable_eval_sha256: str

    @property
    def commands_passed(self) -> bool:
        return all(command["returncode"] == 0 for command in self.commands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "root": self.root,
            "git_commit": self.git_commit,
            "generated_at_utc": self.generated_at_utc,
            "protected_hashes": dict(self.protected_hashes),
            "commands": [dict(command) for command in self.commands],
            "commands_passed": self.commands_passed,
            "runtime_quality": {
                "score": self.runtime_quality_score,
                "passed": self.runtime_quality_passed,
                "total": self.runtime_quality_total,
                "immutable_eval_sha256": self.immutable_eval_sha256,
            },
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_hashes(root: Path) -> dict[str, str]:
    config = yaml.safe_load((root / "config" / "loop.yaml").read_text(encoding="utf-8"))
    configured = [*config["protected_paths"], *config["immutable_iteration_eval"]]
    hashes: dict[str, str] = {}
    for relative in configured:
        path = root / relative
        if path.is_dir():
            for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
                hashes[str(child.relative_to(root))] = _sha256(child)
        elif path.is_file():
            hashes[str(path.relative_to(root))] = _sha256(path)
        elif relative != "eval/golden/":
            raise FileNotFoundError(f"protected path missing: {relative}")
    return hashes


def _run_command(root: Path, command: Sequence[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    started_ns = monotonic_ns()
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_ms = (monotonic_ns() - started_ns) / 1_000_000.0
    output_lines = completed.stdout.splitlines()
    return {
        "argv": list(command),
        "returncode": completed.returncode,
        "elapsed_ms": round(elapsed_ms, 3),
        "output_tail": output_lines[-20:],
    }


def evaluate_worktree(root: Path) -> EvaluationSnapshot:
    root = root.resolve()
    stop_file = root / "STOP_LOOP"
    if stop_file.exists():
        raise RuntimeError("STOP_LOOP exists")
    loop_config = yaml.safe_load((root / "config" / "loop.yaml").read_text(encoding="utf-8"))
    if not loop_config["enabled"]:
        raise RuntimeError("quality loop is not enabled")
    before = _protected_hashes(root)
    commands = tuple(_run_command(root, command) for command in EVALUATION_COMMANDS)
    after = _protected_hashes(root)
    if before != after:
        raise RuntimeError("protected or immutable evaluation artifacts changed during evaluation")
    runtime_report = json.loads(
        (root / "reports" / "runtime-quality-eval.json").read_text(encoding="utf-8")
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return EvaluationSnapshot(
        root=str(root),
        git_commit=commit,
        generated_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        protected_hashes=after,
        commands=commands,
        runtime_quality_score=float(runtime_report["summary"]["score"]),
        runtime_quality_passed=int(runtime_report["summary"]["passed"]),
        runtime_quality_total=int(runtime_report["summary"]["total"]),
        immutable_eval_sha256=str(runtime_report["fixture"]["sha256"]),
    )


def compare_candidates(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline["protected_hashes"] != candidate["protected_hashes"]:
        reasons.append("protected_artifact_hash_mismatch")
    baseline_quality = baseline["runtime_quality"]
    candidate_quality = candidate["runtime_quality"]
    if baseline_quality["immutable_eval_sha256"] != candidate_quality["immutable_eval_sha256"]:
        reasons.append("immutable_eval_hash_mismatch")
    if not candidate["commands_passed"]:
        reasons.append("candidate_evaluation_command_failed")
    if candidate_quality["score"] <= baseline_quality["score"]:
        reasons.append("quality_score_did_not_improve")
    decision = "KEEP" if not reasons else "REVERT"
    return {
        "schema_version": 1,
        "decision": decision,
        "reasons": reasons or ["immutable_eval_improved_with_full_regression_pass"],
        "baseline_commit": baseline["git_commit"],
        "candidate_commit": candidate["git_commit"],
        "baseline_score": baseline_quality["score"],
        "candidate_score": candidate_quality["score"],
        "protected_hashes_equal": baseline["protected_hashes"] == candidate["protected_hashes"],
        "immutable_eval_hash_equal": (
            baseline_quality["immutable_eval_sha256"]
            == candidate_quality["immutable_eval_sha256"]
        ),
        "candidate_commands_passed": candidate["commands_passed"],
    }
