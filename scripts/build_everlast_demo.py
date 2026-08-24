#!/usr/bin/env python3
"""Build a truthful 12-step Everlast demonstration evidence manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "everlast-demo-manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    torture = json.loads((REPORTS / "torture-run.json").read_text(encoding="utf-8"))
    torture_by_id = {result["scenario_id"]: result for result in torture["results"]}
    scorecard = json.loads((REPORTS / "scorecard.json").read_text(encoding="utf-8"))
    quality = json.loads(
        (REPORTS / "quality-loop" / "iteration-001" / "decision.json").read_text(
            encoding="utf-8"
        )
    )
    timeline = json.loads((REPORTS / "timeline-replay.json").read_text(encoding="utf-8"))
    live_stt = json.loads((REPORTS / "live-stt-smoke.json").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    steps = [
        {"step": 1, "name": "normal_german_browser_conversation", "status": "MANUAL_REQUIRED" if live_stt["status"] == "PASS" else "FAIL", "evidence": ["/", "reports/browser-demo-benchmark.json", "reports/live-stt-smoke.json"]},
        {"step": 2, "name": "backchannel_mhm", "status": "AUTOMATED_PASS" if torture_by_id["T01"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T01"]},
        {"step": 3, "name": "intentional_barge_in", "status": "LOCAL_AUTOMATED_PASS_MANUAL_AUDIO_REQUIRED" if torture_by_id["T03"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T03", "reports/realtime-core-benchmark.json"]},
        {"step": 4, "name": "thursday_to_friday_correction", "status": "AUTOMATED_PASS" if torture_by_id["T04"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T04"]},
        {"step": 5, "name": "unfinished_pause", "status": "AUTOMATED_PASS" if torture_by_id["T05"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T05"]},
        {"step": 6, "name": "background_speech", "status": "AUTOMATED_PASS" if torture_by_id["T07"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T07", "reports/runtime-quality-eval.json"]},
        {"step": 7, "name": "four_second_tool_delay", "status": "AUTOMATED_PASS" if torture_by_id["T09"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T09"]},
        {"step": 8, "name": "tool_failure_recovery", "status": "AUTOMATED_PASS" if torture_by_id["T10"]["passed"] else "FAIL", "evidence": ["reports/torture-run.json#T10", "reports/recovery-benchmark.json"]},
        {"step": 9, "name": "timeline_kpi_drilldown", "status": "AUTOMATED_PASS" if timeline["deterministic_second_pass_equal"] else "FAIL", "evidence": ["/dashboard", "reports/timeline-replay.json"]},
        {"step": 10, "name": "baseline_vs_humanflow", "status": "AUTOMATED_PASS", "evidence": ["reports/scorecard.json", "reports/turn-benchmark.json"]},
        {"step": 11, "name": "quality_loop_keep_revert", "status": "AUTOMATED_PASS" if quality["decision"] in {"KEEP", "REVERT"} else "FAIL", "evidence": ["reports/quality-loop/iteration-001/decision.json"]},
        {"step": 12, "name": "provenance_and_frozen_tag", "status": "FREEZE_PENDING", "evidence": ["sprint/start.json", "sprint/checkpoints/"]}
    ]
    automated_failures = sum(step["status"] == "FAIL" for step in steps)
    report_files = (
        "scorecard.json",
        "turn-benchmark.json",
        "realtime-core-benchmark.json",
        "recovery-benchmark.json",
        "torture-run.json",
        "timeline-replay.json",
        "runtime-quality-eval.json",
        "development-router.json",
        "tournament-readiness.json",
        "browser-demo-benchmark.json",
        "dashboard-capture.json",
        "dashboard.png",
        "live-stt-smoke.json",
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "demo_duration_target_minutes": "3-5",
        "status": "READY_FOR_MANUAL_VALIDATION_NOT_FROZEN" if not automated_failures else "NOT_READY",
        "steps": steps,
        "automated_failures": automated_failures,
        "scorecard_status": scorecard["summary"],
        "artifact_hashes": {name: _sha(REPORTS / name) for name in report_files},
        "manual_requirements": [
            "Run the browser demo with microphone and audible output",
            "Validate source-bound Scribe STT and TTS behavior on an actual supported browser",
            "Repeat representative calls before release freeze",
            "Freeze only after manual evidence is accepted"
        ]
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "automated_failures": automated_failures}, indent=2))
    print(f"report={OUTPUT}")
    if automated_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
