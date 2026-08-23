#!/usr/bin/env python3
"""Evaluate a baseline or candidate worktree with immutable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanflow.quality.loop import evaluate_worktree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = evaluate_worktree(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(snapshot.to_dict()["runtime_quality"], indent=2, sort_keys=True))
    print(f"commands_passed={snapshot.commands_passed}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
