#!/usr/bin/env python3
"""Development-only reset for HumanFlow demo appointment bookings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from humanflow.tools.sqlite_appointments import SQLiteAppointmentToolProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DEMO RESET: entfernt nur Buchungen auf Demo-Ressourcen; "
            "Schema, Provider, Verfügbarkeiten und Cost Ledger bleiben erhalten."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "var" / "humanflow-demo.sqlite3",
        help="Pfad zur lokalen HumanFlow-Demo-SQLite-Datei",
    )
    arguments = parser.parse_args()
    provider = SQLiteAppointmentToolProvider(arguments.database)
    result = provider.reset_demo_appointments()
    print(
        "DEMO RESET abgeschlossen: "
        f"{result['deleted_demo_appointments']} Demo-Buchung(en) entfernt; "
        f"{result['preserved_demo_providers']} Demo-Provider und "
        f"{result['preserved_demo_availability']} Verfügbarkeiten erhalten."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
