#!/usr/bin/env python3
"""Produce an explicit KEEP/REVERT decision from two evaluation snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanflow.quality.loop import compare_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    decision = compare_candidates(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if decision["decision"] != "KEEP":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
