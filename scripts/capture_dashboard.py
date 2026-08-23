#!/usr/bin/env python3
"""Render the evidence dashboard with real headless Chrome and hash the result."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT = ROOT / "reports" / "dashboard.png"
REPORT = ROOT / "reports" / "dashboard-capture.json"


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def main() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Chrome/Chromium is not installed")
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "humanflow.web.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}/dashboard"
    try:
        deadline = time.monotonic() + 8
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except OSError:
                if server.poll() is not None:
                    stdout, stderr = server.communicate()
                    raise RuntimeError(f"dashboard server failed: {stdout} {stderr}".strip())
                if time.monotonic() >= deadline:
                    raise TimeoutError("dashboard server readiness timeout") from None
                time.sleep(0.01)
        with tempfile.TemporaryDirectory(prefix="humanflow-chrome-") as profile:
            started_ns = time.monotonic_ns()
            completed = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--hide-scrollbars",
                    f"--user-data-dir={profile}",
                    "--window-size=1440,1400",
                    f"--screenshot={SCREENSHOT}",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=3000",
                    url,
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            render_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
            if completed.returncode != 0 or not SCREENSHOT.is_file():
                raise RuntimeError(f"Chrome dashboard render failed: {completed.stdout}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=3)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "url": "/dashboard",
        "browser": chrome,
        "headless": True,
        "viewport": {"width": 1440, "height": 1400},
        "render_ms": round(render_ms, 3),
        "screenshot": {
            "path": str(SCREENSHOT.relative_to(ROOT)),
            "sha256": hashlib.sha256(SCREENSHOT.read_bytes()).hexdigest(),
            "bytes": SCREENSHOT.stat().st_size,
        },
        "claim_limit": "Visual dashboard render; microphone and audible output are not exercised.",
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
