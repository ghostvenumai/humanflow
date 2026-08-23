from __future__ import annotations

import json
from pathlib import Path


CATALOG = Path(__file__).with_name("torture_scenarios.json")


def test_torture_catalog_preserves_required_t01_through_t20() -> None:
    scenarios = json.loads(CATALOG.read_text(encoding="utf-8"))
    expected_ids = [f"T{number:02d}" for number in range(1, 21)]

    assert [scenario["id"] for scenario in scenarios] == expected_ids
    assert all(scenario["category"] for scenario in scenarios)
    assert all(scenario["expected_evidence"] for scenario in scenarios)

