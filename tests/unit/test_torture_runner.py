from __future__ import annotations

import asyncio

from humanflow.evaluation.torture import TortureRunner


def test_executable_torture_runner_covers_and_passes_t01_through_t20() -> None:
    results = asyncio.run(TortureRunner(tool_latency_ms=20).run())
    assert [result.scenario_id for result in results] == [
        f"T{number:02d}" for number in range(1, 21)
    ]
    assert all(result.passed for result in results), [
        result.to_dict() for result in results if not result.passed
    ]
