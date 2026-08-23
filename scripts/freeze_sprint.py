#!/usr/bin/env python3
"""Freeze the validated 72-hour build and preserve final local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parents[1]
TAG = "everlast-72h-build"


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def _run(command: list[str], *, timeout: int = 600) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.returncode:
        _fail(f"command failed ({' '.join(command)}):\n{completed.stdout}")
    return completed.stdout.rstrip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-freeze", action="store_true")
    args = parser.parse_args()
    if not args.confirm_freeze:
        parser.error("--confirm-freeze is required")
    if (ROOT / "STOP_LOOP").exists():
        _fail("STOP_LOOP exists")
    if (ROOT / "sprint" / "freeze.json").exists():
        _fail("sprint is already frozen")
    if _run(["git", "tag", "--list", TAG]):
        _fail(f"tag already exists: {TAG}")
    if _run(["git", "status", "--porcelain"]):
        _fail("worktree must be clean before freeze")
    manual_path = ROOT / "sprint" / "manual-validation.json"
    if not manual_path.is_file():
        _fail("manual browser/audio validation is missing")
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    if not manual.get("approved") or manual.get("representative_calls", 0) < 3:
        _fail("manual validation is incomplete")

    start = json.loads((ROOT / "sprint" / "start.json").read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    deadline = datetime.fromisoformat(start["sprint"]["deadline_utc"].replace("Z", "+00:00"))
    if now > deadline:
        _fail("72-hour deadline has passed")
    source_commit = _run(["git", "rev-parse", "HEAD"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", manual["validated_commit"], source_commit],
        cwd=ROOT,
        check=False,
    ).returncode:
        _fail("manual validation commit is not an ancestor of the freeze build")

    _run(["python3", "-m", "pytest", "-q"])
    _run(["ruff", "check", "."])
    _run(["make", "benchmark"])
    _run(["make", "torture-test"])
    _run(["make", "realtime-benchmark"])
    _run(["make", "recovery-benchmark"])
    _run(["make", "replay"])
    _run(["make", "scorecard"])
    _run(["make", "demo-benchmark"])
    _run(["make", "demo-package"])

    scorecard = json.loads((ROOT / "reports" / "scorecard.json").read_text(encoding="utf-8"))
    demo = json.loads(
        (ROOT / "reports" / "everlast-demo-manifest.json").read_text(encoding="utf-8")
    )
    if scorecard["summary"]["gates_failed"] or demo["automated_failures"]:
        _fail("final quality evidence failed")
    report_names = (
        "scorecard.json",
        "turn-benchmark.json",
        "realtime-core-benchmark.json",
        "recovery-benchmark.json",
        "torture-run.json",
        "timeline-replay.json",
        "everlast-demo-manifest.json",
        "dashboard-capture.json",
        "dashboard.png",
    )
    frozen_at = datetime.now(UTC)
    started_at = datetime.fromisoformat(start["sprint"]["start_utc"].replace("Z", "+00:00"))
    freeze = {
        "schema_version": 1,
        "event": "SPRINT_FROZEN",
        "frozen_at_utc": frozen_at.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": (frozen_at - started_at).total_seconds(),
        "final_build_commit": source_commit,
        "tag": TAG,
        "manual_validation_sha256": _sha(manual_path),
        "demo_build_identifier": _sha(ROOT / "reports" / "everlast-demo-manifest.json")[:16],
        "artifacts": {name: _sha(ROOT / "reports" / name) for name in report_names},
        "remote_evidence": "NOT_CONFIGURED",
    }
    freeze_path = ROOT / "sprint" / "freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "sprint" / "provenance.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(freeze, sort_keys=True) + "\n")
    _run(["git", "add", "reports", "sprint/manual-validation.json", "sprint/freeze.json", "sprint/provenance.jsonl"])
    _run(["git", "commit", "-m", "evidence: freeze 72-hour Everlast build"])
    _run(["git", "tag", "-a", TAG, source_commit, "-m", "HumanFlow 72-hour Everlast build"])
    print("HUMANFLOW_72H_BUILD_FROZEN")
    print(f"final_build_commit={source_commit}")
    print(f"tag={TAG}")
    print("remote_push=NOT_PERFORMED")


if __name__ == "__main__":
    main()
