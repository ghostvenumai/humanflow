#!/usr/bin/env python3
"""Build the evidence-linked HumanFlow scorecard."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from humanflow.evaluation.scorecard import build_scorecard


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "scorecard.json"


def main() -> None:
    scorecard = build_scorecard(ROOT)
    scorecard["generated_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    OUTPUT.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(scorecard["summary"], indent=2, sort_keys=True))
    print(f"report={OUTPUT}")
    if scorecard["summary"]["gates_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
