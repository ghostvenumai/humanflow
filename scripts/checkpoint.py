#!/usr/bin/env python3
"""Create a tested, append-only local sprint checkpoint without pushing."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
START_FILE = ROOT / "sprint" / "start.json"
PROVENANCE_FILE = ROOT / "sprint" / "provenance.jsonl"
REPORT = ROOT / "reports" / "turn-benchmark.json"


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def run(command: list[str], *, timeout: int = 300) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.returncode != 0:
        fail(f"command failed ({' '.join(command)}):\n{completed.stdout}")
    # Git porcelain status uses a significant leading column. Keep it while
    # removing only trailing line terminators and command-output whitespace.
    return completed.stdout.rstrip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if not START_FILE.is_file():
        fail("sprint has not been started")
    dirty = run(["git", "status", "--porcelain"])
    unexpected = []
    for line in dirty.splitlines():
        path = line[3:]
        if not path.startswith("reports/"):
            unexpected.append(line)
    if unexpected:
        fail("commit source changes before checkpoint:\n" + "\n".join(unexpected))

    source_commit = run(["git", "rev-parse", "HEAD"])
    tests_output = run(["python3", "-m", "pytest", "-q"])
    lint_output = run(["ruff", "check", "."])
    run(["make", "benchmark"])
    benchmark = json.loads(REPORT.read_text(encoding="utf-8"))
    if benchmark.get("git_commit") != source_commit:
        fail("benchmark commit does not match checkpoint source commit")

    match = re.search(r"(\d+) passed", tests_output)
    if not match:
        fail("could not determine passing test count")
    timestamp = datetime.now(UTC)
    stamp = timestamp.strftime("%Y%m%d_%H%M%SZ")
    checkpoint_dir = ROOT / "sprint" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{stamp}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "event": "CHECKPOINT_CREATED",
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "tests": {"passed": int(match.group(1)), "command": "python3 -m pytest -q"},
        "lint": {"status": "PASS", "command": "ruff check .", "output": lint_output},
        "benchmark": {
            "path": str(REPORT.relative_to(ROOT)),
            "sha256": sha256(REPORT),
            "dataset": benchmark["dataset"],
            "winner": benchmark["winner"],
            "evaluations": [
                {
                    "policy_name": item["policy_name"],
                    "total": item["total"],
                    "correct": item["correct"],
                    "accuracy": item["accuracy"],
                }
                for item in benchmark["evaluations"]
            ],
        },
    }
    atomic_json(checkpoint_path, payload)
    with PROVENANCE_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "CHECKPOINT_CREATED",
                    "timestamp_utc": payload["timestamp_utc"],
                    "source_commit": source_commit,
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "benchmark_sha256": payload["benchmark"]["sha256"],
                    "tests_passed": int(match.group(1)),
                },
                sort_keys=True,
            )
            + "\n"
        )

    run(["git", "add", "reports", str(checkpoint_path.relative_to(ROOT)), "sprint/provenance.jsonl"])
    run(["git", "commit", "-m", f"evidence: checkpoint {stamp}"])
    checkpoint_commit = run(["git", "rev-parse", "HEAD"])
    print("HUMANFLOW_CHECKPOINT_CREATED")
    print(f"source_commit={source_commit}")
    print(f"checkpoint_commit={checkpoint_commit}")
    print(f"tests_passed={match.group(1)}")
    print(f"benchmark_sha256={payload['benchmark']['sha256']}")


if __name__ == "__main__":
    main()
