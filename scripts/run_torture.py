#!/usr/bin/env python3
"""Execute protected-catalog T01-T20 scenarios and preserve evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from humanflow.evaluation.torture import TortureRunner


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "golden" / "torture_scenarios.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-latency-ms", type=float, default=4_000.0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "torture-run.json")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    expected_ids = [entry["id"] for entry in catalog]
    results = asyncio.run(TortureRunner(tool_latency_ms=args.tool_latency_ms).run())
    actual_ids = [result.scenario_id for result in results]
    if actual_ids != expected_ids:
        raise RuntimeError("executable scenarios do not match protected catalog order")
    passed = sum(result.passed for result in results)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "catalog": {
            "path": str(CATALOG.relative_to(ROOT)),
            "sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            "scenarios": len(catalog),
            "modified": False,
        },
        "configured_tool_latency_ms": args.tool_latency_ms,
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "results": [result.to_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report={args.output}")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
