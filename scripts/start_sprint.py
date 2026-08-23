#!/usr/bin/env python3
"""Start the HumanFlow 72-hour sprint exactly once and preserve evidence."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
PRESTART = ROOT.parent
SPRINT_DIR = ROOT / "sprint"
START_FILE = SPRINT_DIR / "start.json"
PROVENANCE_FILE = SPRINT_DIR / "provenance.jsonl"
START_TAG = "everlast-sprint-start"


def fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def run(command: list[str], *, check: bool = True, timeout: int = 30) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latest_technical_go() -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (PRESTART / "supervisor_runs").glob("*/status.json"),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "TECHNICAL_GO" and payload.get("fail") == 0:
            return candidate.resolve(), payload
    fail("no complete TECHNICAL_GO preflight status found")


def version(command: list[str]) -> str:
    try:
        output = run(command, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return output.splitlines()[0] if output else "unavailable"


def main() -> None:
    if START_FILE.exists():
        fail(f"sprint already started; evidence exists at {START_FILE}")
    if run(["git", "tag", "--list", START_TAG], check=False):
        fail(f"sprint already started; Git tag {START_TAG} exists")
    if run(["git", "status", "--porcelain"]):
        fail("working tree must be clean before GO")

    status_path, status = latest_technical_go()
    report_path = Path(str(status.get("report", ""))).resolve()
    try:
        report_path.relative_to(PRESTART.resolve())
        status_path.relative_to(PRESTART.resolve())
    except ValueError:
        fail("preflight evidence resolves outside the PreStart workspace")
    if not report_path.is_file():
        fail(f"preflight report not found: {report_path}")

    master_briefing = PRESTART / "HumanFlow_Master_Briefing_V1_1.pdf"
    sprint_briefing = PRESTART / "HumanFlow_72H_Everlast_Application_Sprint_Briefing.pdf"
    for briefing in (master_briefing, sprint_briefing):
        if not briefing.is_file():
            fail(f"required briefing not found: {briefing}")

    local_start = datetime.now().astimezone()
    utc_start = local_start.astimezone(UTC)
    utc_deadline = utc_start + timedelta(hours=72)
    local_deadline = utc_deadline.astimezone(local_start.tzinfo)
    baseline_commit = run(["git", "rev-parse", "HEAD"])

    evidence_dir = SPRINT_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    copied_report = evidence_dir / "preflight.md"
    copied_status = evidence_dir / "preflight_status.json"
    shutil.copy2(report_path, copied_report)
    shutil.copy2(status_path, copied_status)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "sprint": {
            "name": "HumanFlow 72-Hour Everlast Application Sprint",
            "status": "RUNNING",
            "duration_hours": 72,
            "start_local": local_start.isoformat(),
            "start_utc": utc_start.isoformat().replace("+00:00", "Z"),
            "deadline_local": local_deadline.isoformat(),
            "deadline_utc": utc_deadline.isoformat().replace("+00:00", "Z"),
            "start_tag": START_TAG,
            "baseline_commit": baseline_commit,
        },
        "preflight": {
            "run_id": status.get("run_id"),
            "status": status.get("status"),
            "pass": status.get("pass"),
            "warn": status.get("warn"),
            "fail": status.get("fail"),
            "status_sha256": sha256(status_path),
            "report_sha256": sha256(report_path),
            "evidence_status": str(copied_status.relative_to(ROOT)),
            "evidence_report": str(copied_report.relative_to(ROOT)),
        },
        "briefings": {
            "master": {
                "filename": master_briefing.name,
                "sha256": sha256(master_briefing),
            },
            "sprint": {
                "filename": sprint_briefing.name,
                "sha256": sha256(sprint_briefing),
            },
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "tools": {
            "git": version(["git", "--version"]),
            "python": version(["python3", "--version"]),
            "node": version(["node", "--version"]),
            "npm": version(["npm", "--version"]),
            "docker": version(["docker", "--version"]),
            "docker_compose": version(["docker", "compose", "version"]),
            "codex": version(["codex", "--version"]),
            "claude": version(["claude", "--version"]),
            "ffmpeg": version(["ffmpeg", "-version"]),
        },
    }

    SPRINT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(START_FILE, payload)
    provenance_event = {
        "event": "SPRINT_STARTED",
        "timestamp_local": local_start.isoformat(),
        "timestamp_utc": utc_start.isoformat().replace("+00:00", "Z"),
        "baseline_commit": baseline_commit,
        "preflight_report_sha256": payload["preflight"]["report_sha256"],
        "sprint_briefing_sha256": payload["briefings"]["sprint"]["sha256"],
    }
    PROVENANCE_FILE.write_text(
        json.dumps(provenance_event, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run(["git", "add", "sprint/start.json", "sprint/provenance.jsonl", "sprint/evidence"])
    run(["git", "commit", "-m", "sprint: start authorized 72-hour build"])
    start_commit = run(["git", "rev-parse", "HEAD"])
    tag_message = (
        f"HumanFlow sprint start {payload['sprint']['start_utc']}\n"
        f"preflight={payload['preflight']['report_sha256']}\n"
        f"briefing={payload['briefings']['sprint']['sha256']}"
    )
    run(["git", "tag", "-a", START_TAG, "-m", tag_message, start_commit])

    print("HUMANFLOW_SPRINT_STARTED")
    print(f"start_local={payload['sprint']['start_local']}")
    print(f"start_utc={payload['sprint']['start_utc']}")
    print(f"deadline_local={payload['sprint']['deadline_local']}")
    print(f"deadline_utc={payload['sprint']['deadline_utc']}")
    print(f"baseline_commit={baseline_commit}")
    print(f"start_commit={start_commit}")
    print(f"tag={START_TAG}")


if __name__ == "__main__":
    main()
