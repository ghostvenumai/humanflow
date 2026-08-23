#!/usr/bin/env python3
"""Start the real ASGI server and measure loopback HTTP readiness/health."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean


def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "mean": round(fmean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p90": round(_percentile(values, 90), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


def _request(url: str) -> tuple[float, int, bytes]:
    started_ns = time.monotonic_ns()
    with urllib.request.urlopen(url, timeout=2) as response:
        body = response.read()
        status = response.status
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
    return elapsed_ms, status, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/browser-demo-benchmark.json"))
    args = parser.parse_args()
    if args.requests < 20:
        parser.error("--requests must be at least 20")

    port = _free_loopback_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
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
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 8
        while True:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"demo server exited early: {stdout} {stderr}".strip())
            try:
                _, status, body = _request(base_url + "/health")
                if status == 200:
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("demo server did not become ready") from None
                time.sleep(0.01)
        startup_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        health_latencies: list[float] = []
        for _ in range(args.requests):
            latency_ms, status, body = _request(base_url + "/health")
            payload = json.loads(body)
            if status != 200 or payload.get("production_claim") is not False:
                raise RuntimeError("health contract failed")
            health_latencies.append(latency_ms)
        index_latency_ms, index_status, index_body = _request(base_url + "/")
        if index_status != 200 or b"HumanFlow Realtime Lab" not in index_body:
            raise RuntimeError("demo index contract failed")
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "measurement_scope": {
            "kind": "real_uvicorn_loopback_http_benchmark",
            "host": "127.0.0.1",
            "websocket_route_present": True,
            "browser_executed": False,
            "audio_device_executed": False,
            "claim_limit": (
                "Measures process readiness and HTTP loopback only. Browser microphone, "
                "WebAudio scheduling and websocket playback require the manual demo."
            ),
        },
        "metrics": {
            "server_startup_to_health_ms": round(startup_ms, 3),
            "health_request_ms": _summary(health_latencies),
            "index_request_ms": round(index_latency_ms, 3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
