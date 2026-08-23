#!/usr/bin/env python3
"""Benchmark fixed-silence and hybrid policies on the protected corpus."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from humanflow.evaluation import evaluate_policy, load_turn_cases
from humanflow.turns import FixedSilencePolicy, HybridTurnPolicy


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "golden" / "turn_cases.jsonl"
OUTPUT = ROOT / "reports" / "turn-benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cases = load_turn_cases(CORPUS)
    fixed = evaluate_policy(FixedSilencePolicy(), cases)
    hybrid = evaluate_policy(HybridTurnPolicy(), cases)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "dataset": {
            "path": str(CORPUS.relative_to(ROOT)),
            "sha256": sha256(CORPUS),
            "cases": len(cases),
        },
        "evaluations": [fixed.to_dict(), hybrid.to_dict()],
        "winner": "HybridTurnPolicy" if hybrid.accuracy > fixed.accuracy else "TIE_OR_BASELINE",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
