"""Persist compact, evidence-labelled economic summaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .aggregation import aggregate_cost_rows
from .ledger import CostLedger
from .pricing import PricingCatalog


def build_cost_summary_report(
    ledger: CostLedger,
    catalog: PricingCatalog,
    *,
    session_id: str | None = None,
    active_duration_seconds: Decimal | None = None,
) -> dict[str, Any]:
    selected_session = session_id or ledger.latest_session_id()
    session_summary = (
        None
        if selected_session is None
        else aggregate_cost_rows(
            ledger.rows(session_id=selected_session),
            session_id=selected_session,
            active_duration_seconds=active_duration_seconds,
        )
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scope": {
            "session_id": selected_session,
            "sample_count": 0 if session_summary is None else session_summary["event_count"],
            "voice_scope": "REAL_BROWSER_SESSION" if session_summary else "NO_SESSION_DATA",
        },
        "pricing_catalog": {
            "rules": [rule.to_dict() for rule in catalog.rules],
            "active_rule_count": sum(rule.active for rule in catalog.rules),
            "production_provider_pricing_verified": False,
        },
        "session": session_summary,
        "evidence_rules": {
            "provider_usage": "ACTUAL_USAGE when returned or exactly observed",
            "money": "PROVIDER_REPORTED_COST only for provider monetary billing data",
            "estimates": "ESTIMATED_COST only with active verified pricing rule",
            "unheard_audio": "estimated allocation; generated provider usage is never reduced",
            "local_sqlite": "LOCAL_TOOL_COST with zero direct external provider cost",
        },
        "known_limitations": [
            "Production Anthropic and ElevenLabs pricing is unverified and inactive.",
            "Heard/unheard monetary allocation is estimated from Played Audio Ledger samples.",
            "Browser Web Speech does not expose provider billing metadata.",
        ],
    }


def write_cost_summary_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
