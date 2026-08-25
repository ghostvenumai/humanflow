#!/usr/bin/env python3
"""Build the Cost Ledger export without invoking any paid provider."""

from __future__ import annotations

import argparse
from pathlib import Path

from humanflow.cost import (
    CostLedger,
    PricingCatalog,
    build_cost_summary_report,
    write_cost_summary_report,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=Path, default=ROOT / "var" / "humanflow-demo.sqlite3"
    )
    parser.add_argument(
        "--pricing", type=Path, default=ROOT / "pricing" / "provider_pricing.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "cost-summary.json"
    )
    parser.add_argument("--session-id")
    arguments = parser.parse_args()

    ledger = CostLedger(arguments.database)
    pricing = PricingCatalog.load(arguments.pricing)
    ledger.store_pricing_rules(pricing.rules)
    report = build_cost_summary_report(
        ledger, pricing, session_id=arguments.session_id
    )
    write_cost_summary_report(arguments.output, report)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
