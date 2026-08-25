"""Recomputable session economics with explicit evidence completeness."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterable

from .models import CostSource, micros_to_decimal


def aggregate_cost_rows(
    rows: Iterable[dict[str, Any]],
    *,
    session_id: str,
    active_duration_seconds: Decimal | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    selected = [row for row in materialized if row["session_id"] == session_id]
    provider_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    service_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        provider_groups[(row["provider"], row["model"], row["service_type"])].append(row)
        service_groups[row["service_type"]].append(row)

    unavailable_sources = {
        CostSource.COST_UNAVAILABLE.value,
        CostSource.UNVERIFIED_PRICING.value,
    }
    monetary_provider_rows = [
        row
        for row in selected
        if row["service_type"] != "TOOL" or row["provider"] != "local-sqlite"
    ]
    estimate_complete = bool(monetary_provider_rows) and not any(
        row["cost_source"] in unavailable_sources for row in monetary_provider_rows
    )
    estimated_micros_known = sum(
        int(row["estimated_cost_micros"] or 0) for row in selected
    )
    actual_micros = sum(int(row["actual_cost_micros"] or 0) for row in selected)
    actual_cost_samples = sum(
        row["cost_source"] == CostSource.PROVIDER_REPORTED_COST.value for row in selected
    )
    turns = {row["turn_id"] for row in selected if row["turn_id"]}
    tool_rows = service_groups.get("TOOL", [])
    successful_actions = sum(row["tool_success"] == 1 for row in tool_rows)
    failed_actions = sum(row["tool_success"] == 0 for row in tool_rows)
    generated_audio_seconds = sum(
        Decimal(row["audio_output_seconds"] or "0")
        for row in selected
        if row["service_type"] in {"TTS", "FALLBACK_TTS"}
    )
    heard_audio_seconds = sum(
        Decimal(row["heard_units"] or "0")
        for row in selected
        if row["unit_type"] == "audio_seconds"
    )
    unheard_audio_seconds = sum(
        Decimal(row["unheard_units"] or "0")
        for row in selected
        if row["unit_type"] == "audio_seconds"
    )
    wasted_micros = sum(
        int(row["wasted_cost_estimate_micros"] or 0) for row in selected
    )
    estimated_total = (
        micros_to_decimal(estimated_micros_known) if estimate_complete else None
    )
    duration = None if active_duration_seconds is None else Decimal(active_duration_seconds)
    cost_per_minute = (
        estimated_total / (duration / Decimal("60"))
        if estimated_total is not None and duration is not None and duration > 0
        else None
    )
    cost_per_turn = (
        estimated_total / len(turns)
        if estimated_total is not None and turns
        else None
    )
    provider_breakdown = [
        _group_summary(provider, model, service, group)
        for (provider, model, service), group in sorted(provider_groups.items())
    ]
    return {
        "schema_version": 1,
        "session_id": session_id,
        "event_count": len(selected),
        "turn_count": len(turns),
        "active_duration_seconds": None if duration is None else str(duration),
        "active_duration_scope": (
            "ACTIVE_VOICE_SESSION_DURATION"
            if duration is not None
            else "ACTIVE_DURATION_NOT_RECORDED"
        ),
        "provider_reported_cost": {
            "value": str(micros_to_decimal(actual_micros)),
            "currency": _single_currency(selected, actual=True),
            "sample_count": actual_cost_samples,
            "evidence": CostSource.PROVIDER_REPORTED_COST.value,
        },
        "estimated_cost": {
            "value": None if estimated_total is None else str(estimated_total),
            "known_partial_value": str(micros_to_decimal(estimated_micros_known)),
            "currency": _single_currency(selected, actual=False),
            "complete": estimate_complete,
            "evidence": (
                CostSource.ESTIMATED_COST.value
                if estimate_complete
                else CostSource.COST_UNAVAILABLE.value
            ),
        },
        "cost_per_conversation_minute": {
            "value": None if cost_per_minute is None else str(cost_per_minute),
            "sample_size": len(selected),
            "scope": "ACTIVE_VOICE_SESSION_DURATION",
            "evidence": (
                CostSource.ESTIMATED_COST.value
                if cost_per_minute is not None
                else CostSource.COST_UNAVAILABLE.value
            ),
        },
        "cost_per_turn": {
            "value": None if cost_per_turn is None else str(cost_per_turn),
            "sample_size": len(turns),
            "evidence": (
                CostSource.ESTIMATED_COST.value
                if cost_per_turn is not None
                else CostSource.COST_UNAVAILABLE.value
            ),
        },
        "providers": provider_breakdown,
        "services": {
            service: _service_summary(group)
            for service, group in sorted(service_groups.items())
        },
        "tools": {
            "call_count": len(tool_rows),
            "successful_actions": successful_actions,
            "failed_actions": failed_actions,
            "direct_external_cost": "0"
            if tool_rows and all(row["provider"] == "local-sqlite" for row in tool_rows)
            else None,
            "evidence": CostSource.LOCAL_TOOL_COST.value if tool_rows else "NO_DATA",
        },
        "played_audio_economics": {
            "generated_audio_seconds": str(generated_audio_seconds),
            "heard_audio_seconds": str(heard_audio_seconds),
            "unheard_audio_seconds": str(unheard_audio_seconds),
            "wasted_cost_estimate": str(micros_to_decimal(wasted_micros)),
            "evidence": CostSource.ESTIMATED_COST.value,
        },
        "evidence_labels": sorted(
            {
                *(row["usage_source"] for row in selected),
                *(row["cost_source"] for row in selected),
            }
        ),
    }


def _group_summary(
    provider: str,
    model: str,
    service: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "service": service,
        "event_count": len(rows),
        "usage_basis": sorted({row["unit_type"] for row in rows if row["unit_type"]}),
        "actual_cost": str(
            micros_to_decimal(sum(int(row["actual_cost_micros"] or 0) for row in rows))
        ),
        "estimated_cost_known": str(
            micros_to_decimal(
                sum(int(row["estimated_cost_micros"] or 0) for row in rows)
            )
        ),
        "evidence": sorted({row["cost_source"] for row in rows}),
    }


def _service_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(rows),
        "input_tokens": sum(int(row["tokens_input"] or 0) for row in rows),
        "output_tokens": sum(int(row["tokens_output"] or 0) for row in rows),
        "characters": sum(int(row["characters"] or 0) for row in rows),
        "audio_input_seconds": str(
            sum(Decimal(row["audio_input_seconds"] or "0") for row in rows)
        ),
        "audio_output_seconds": str(
            sum(Decimal(row["audio_output_seconds"] or "0") for row in rows)
        ),
        "success_count": sum(row["tool_success"] == 1 for row in rows),
        "failure_count": sum(row["tool_success"] == 0 for row in rows),
    }


def _single_currency(rows: list[dict[str, Any]], *, actual: bool) -> str | None:
    field = "actual_cost_micros" if actual else "estimated_cost_micros"
    currencies = {row["currency"] for row in rows if row[field] is not None}
    return next(iter(currencies)) if len(currencies) == 1 else None


def format_adaptive_money(value: Decimal | None, currency: str | None) -> str:
    if value is None or currency is None:
        return "Kosten nicht verfügbar"
    absolute = abs(value)
    places = 2 if absolute >= Decimal("1") else 4 if absolute >= Decimal("0.01") else 6
    return f"{currency} {value:.{places}f}"
