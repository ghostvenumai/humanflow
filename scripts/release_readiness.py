#!/usr/bin/env python3
"""Report automated, human, and freeze readiness without claiming production evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "release-readiness.json"


def main() -> None:
    demo = json.loads((REPORTS / "everlast-demo-manifest.json").read_text(encoding="utf-8"))
    scorecard = json.loads((REPORTS / "scorecard.json").read_text(encoding="utf-8"))
    manual_path = ROOT / "sprint" / "manual-validation.json"
    freeze_path = ROOT / "sprint" / "freeze.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.is_file() else None
    automated_pass = demo["automated_failures"] == 0 and scorecard["summary"]["gates_failed"] == 0
    manual_pass = bool(manual and manual.get("approved") and manual.get("representative_calls", 0) >= 3)
    frozen = freeze_path.is_file()
    if not automated_pass:
        status = "BLOCKED_AUTOMATED_FAILURE"
    elif not manual_pass:
        status = "BLOCKED_MANUAL_BROWSER_AUDIO_VALIDATION"
    elif not frozen:
        status = "READY_TO_FREEZE"
    else:
        status = "FROZEN"
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "automated": {
            "passed": automated_pass,
            "demo_manifest_status": demo["status"],
            "local_gates_failed": scorecard["summary"]["gates_failed"],
            "production_release_claim": scorecard["summary"]["production_release_claim"],
        },
        "manual_browser_audio_validation": {
            "passed": manual_pass,
            "path": "sprint/manual-validation.json",
            "exists": manual_path.is_file(),
        },
        "freeze": {
            "complete": frozen,
            "path": "sprint/freeze.json",
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
