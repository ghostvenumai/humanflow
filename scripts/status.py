#!/usr/bin/env python3
"""Show the authoritative HumanFlow sprint clock."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_FILE = ROOT / "sprint" / "start.json"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    if not START_FILE.is_file():
        raise SystemExit("Sprint not started. Run ./go only after explicit human GO.")
    payload = json.loads(START_FILE.read_text(encoding="utf-8"))
    sprint = payload["sprint"]
    started = datetime.fromisoformat(sprint["start_utc"].replace("Z", "+00:00"))
    deadline = datetime.fromisoformat(sprint["deadline_utc"].replace("Z", "+00:00"))
    now = datetime.now(UTC)
    elapsed = (now - started).total_seconds()
    remaining = (deadline - now).total_seconds()
    state = "RUNNING" if remaining > 0 else "WINDOW_ELAPSED"
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    print(f"status={state}")
    print(f"start_local={sprint['start_local']}")
    print(f"deadline_local={sprint['deadline_local']}")
    print(f"elapsed={format_duration(elapsed)}")
    print(f"remaining={format_duration(remaining)}")
    print(f"commit={commit}")
    print(f"tag={sprint['start_tag']}")


if __name__ == "__main__":
    main()

