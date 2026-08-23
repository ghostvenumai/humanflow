#!/usr/bin/env python3
"""Record explainable router decisions and non-spending adapter readiness."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from humanflow.development import (
    ClaudeCliAdapter,
    CodexCliAdapter,
    DevelopmentModelRouter,
    EngineeringTask,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "router" / "tasks.json"
OUTPUT = ROOT / "reports" / "development-router.json"


def main() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    router = DevelopmentModelRouter()
    decisions = []
    correct = 0
    for raw in fixture["tasks"]:
        expected = raw.pop("expected_tier")
        decision = router.route(EngineeringTask(**raw))
        matches = decision.tier.value == expected
        correct += int(matches)
        decisions.append({**decision.to_dict(), "expected_tier": expected, "matches": matches})
    adapters = [CodexCliAdapter().status(), ClaudeCliAdapter().status()]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "fixture": {
            "path": str(FIXTURE.relative_to(ROOT)),
            "sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "tasks": len(fixture["tasks"]),
        },
        "summary": {"correct": correct, "total": len(decisions), "accuracy": correct / len(decisions)},
        "decisions": decisions,
        "adapters": [adapter.to_dict() for adapter in adapters],
        "external_agent_calls_made": 0,
        "external_spend_usd": 0.0,
        "authentication_secrets_probed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report={OUTPUT}")
    if correct != len(decisions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
