from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from humanflow.cost import (
    AsyncCostRecorder,
    CostEvent,
    CostBudgetPolicy,
    CostLedger,
    CostSource,
    PricingCatalog,
    PricingRule,
    ServiceType,
    UsageSource,
    aggregate_cost_rows,
    build_cost_summary_report,
    decimal_to_micros,
    micros_to_decimal,
    write_cost_summary_report,
)


def _event(
    *,
    service: ServiceType = ServiceType.LLM,
    operation_id: str = "operation-1",
    provider: str = "anthropic-messages-api",
    model: str = "claude-test",
    **changes: object,
) -> CostEvent:
    event = CostEvent(
        session_id="session-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        response_id="response-1",
        provider=provider,
        model=model,
        service_type=service,
        operation="reasoning" if service is ServiceType.LLM else "provider-operation",
        operation_id=operation_id,
        usage_source=UsageSource.ACTUAL_USAGE,
        cost_source=CostSource.COST_UNAVAILABLE,
        actual_usage=True,
    )
    return replace(event, **changes)


def test_micro_currency_precision_round_trip() -> None:
    assert decimal_to_micros(Decimal("0.004701")) == 4701
    assert micros_to_decimal(4701) == Decimal("0.004701")
    assert decimal_to_micros(Decimal("0.0000005")) == 1


def test_cost_event_rejects_invalid_evidence_and_negative_usage() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _event(tokens_input=-1)
    with pytest.raises(ValueError, match="estimated cost requires"):
        _event(estimated_cost_micros=5, currency="EUR")
    with pytest.raises(ValueError, match="actual cost requires"):
        _event(actual_cost_micros=5, currency="EUR")


def test_verified_test_pricing_calculates_estimate_with_rule_provenance() -> None:
    catalog = PricingCatalog(
        [
            PricingRule(
                pricing_rule_id="test-llm-v1",
                pricing_version="v1",
                provider="anthropic-messages-api",
                model="claude-test",
                service=ServiceType.LLM,
                unit="per_1m_tokens",
                input_rate=Decimal("1.25"),
                output_rate=Decimal("5.00"),
                currency="USD",
                effective_date="2026-08-25",
                verified_at="deterministic-test-fixture",
                source_note="TEST ONLY",
                active=True,
            )
        ]
    )
    priced = catalog.price(_event(tokens_input=2_000, tokens_output=500))

    assert priced.cost_source is CostSource.ESTIMATED_COST
    assert priced.estimated_cost_micros == 5000
    assert priced.currency == "USD"
    assert priced.pricing_rule_id == "test-llm-v1"
    assert priced.pricing_version == "v1"


def test_production_catalog_never_prices_unverified_provider_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = PricingCatalog.load(root / "pricing" / "provider_pricing.json")

    llm = catalog.price(
        _event(
            model="claude-haiku-4-5-20251001",
            tokens_input=1_000,
            tokens_output=100,
        )
    )
    tts = catalog.price(
        _event(
            service=ServiceType.TTS,
            provider="elevenlabs-text-to-speech-stream",
            model="eleven_flash_v2_5",
            characters=120,
        )
    )

    assert llm.cost_source is CostSource.UNVERIFIED_PRICING
    assert tts.cost_source is CostSource.UNVERIFIED_PRICING
    assert llm.estimated_cost_micros is None
    assert tts.estimated_cost_micros is None


def test_local_sqlite_has_verified_zero_direct_external_cost() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = PricingCatalog.load(root / "pricing" / "provider_pricing.json")
    event = _event(
        service=ServiceType.TOOL,
        provider="local-sqlite",
        model="humanflow-appointments-v1",
        tool_success=True,
    )
    priced = catalog.price(event)

    assert priced.cost_source is CostSource.LOCAL_TOOL_COST
    assert priced.actual_cost_micros == 0
    assert priced.pricing_rule_id == "humanflow-local-sqlite-zero-direct-cost"


@pytest.mark.parametrize(
    "service",
    (
        ServiceType.STT,
        ServiceType.LLM,
        ServiceType.TTS,
        ServiceType.FALLBACK_TTS,
        ServiceType.TOOL,
    ),
)
def test_append_only_ledger_rejects_duplicate_provider_operation(
    tmp_path: Path,
    service: ServiceType,
) -> None:
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    event = _event(service=service, operation_id=f"{service.value}-operation")

    assert ledger.append(event) is True
    assert ledger.append(replace(event, cost_event_id=f"new-{service.value}")) is False
    assert len(ledger.rows(session_id="session-1")) == 1

    with sqlite3.connect(ledger.database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append_only"):
            connection.execute(
                "UPDATE cost_events SET operation = 'changed' WHERE cost_event_id = ?",
                (event.cost_event_id,),
            )


def test_pricing_rules_and_summary_persist_separately_from_raw_events(
    tmp_path: Path,
) -> None:
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    catalog = PricingCatalog.load(
        Path(__file__).resolve().parents[2] / "pricing" / "provider_pricing.json"
    )
    ledger.store_pricing_rules(catalog.rules)
    ledger.save_session_summary("session-1", "2026-08-25T10:00:00Z", {"total": "0"})

    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM pricing_rules").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM session_cost_summary").fetchone()[0] == 1
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list('cost_events')").fetchall()
        }
    assert {
        "cost_events_session_idx",
        "cost_events_response_idx",
        "cost_events_operation_idx",
        "cost_events_provider_idx",
        "cost_events_service_idx",
    } <= indexes
    assert ledger.latest_session_id() is None
    assert ledger.load_session_summary("session-1") == {"total": "0"}


def test_cost_report_export_has_pricing_provenance_and_known_limitations(
    tmp_path: Path,
) -> None:
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    catalog = PricingCatalog.load(
        Path(__file__).resolve().parents[2] / "pricing" / "provider_pricing.json"
    )
    ledger.append(_event(tokens_input=10, tokens_output=2))
    report = build_cost_summary_report(ledger, catalog)
    output = tmp_path / "cost-summary.json"
    write_cost_summary_report(output, report)

    assert report["scope"]["session_id"] == "session-1"
    assert report["scope"]["sample_count"] == 1
    assert report["pricing_catalog"]["production_provider_pricing_verified"] is False
    assert report["session"]["estimated_cost"]["value"] is None
    assert "unverified" in report["known_limitations"][0].lower()
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_session_aggregation_keeps_actual_estimated_and_local_tool_cost_separate(
    tmp_path: Path,
) -> None:
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    llm = replace(
        _event(tokens_input=100, tokens_output=20),
        cost_source=CostSource.ESTIMATED_COST,
        estimated_cost_micros=4701,
        currency="EUR",
        pricing_rule_id="test",
        pricing_version="v1",
        pricing_effective_date="2026-08-25",
    )
    tool = replace(
        _event(
            service=ServiceType.TOOL,
            operation_id="tool-1",
            provider="local-sqlite",
            model="humanflow-appointments-v1",
            tool_success=True,
        ),
        cost_source=CostSource.LOCAL_TOOL_COST,
        actual_cost_micros=0,
        currency="EUR",
    )
    ledger.append(llm)
    ledger.append(tool)

    summary = aggregate_cost_rows(
        ledger.rows(),
        session_id="session-1",
        active_duration_seconds=Decimal("120"),
    )

    assert summary["estimated_cost"]["value"] == "0.004701"
    assert summary["provider_reported_cost"]["value"] == "0"
    assert summary["cost_per_conversation_minute"]["value"] == "0.0023505"
    assert summary["cost_per_turn"]["value"] == "0.004701"
    assert summary["tools"]["call_count"] == 1
    assert summary["tools"]["successful_actions"] == 1
    assert summary["tools"]["failed_actions"] == 0
    assert summary["tools"]["direct_external_cost"] == "0"
    assert summary["tools"]["evidence"] == "LOCAL_TOOL_COST"
    assert summary["tools"]["by_operation"]["provider-operation"]["call_count"] == 1
    assert summary["turns"]["turn-1"]["event_count"] == 2


def test_cost_budget_hooks_are_disabled_by_default_and_require_complete_estimate() -> None:
    disabled = CostBudgetPolicy()
    configured = CostBudgetPolicy(
        warning_eur=Decimal("0.01"), hard_limit_eur=Decimal("0.02")
    )

    assert disabled.enabled is False
    assert configured.evaluate(
        estimated_eur=Decimal("0.05"), estimate_complete=False
    ) is None
    assert configured.evaluate(
        estimated_eur=Decimal("0.015"), estimate_complete=True
    ) == "WARNING_THRESHOLD_REACHED"
    assert configured.evaluate(
        estimated_eur=Decimal("0.02"), estimate_complete=True
    ) == "HARD_LIMIT_REACHED"


def test_provider_reported_money_is_not_mislabeled_as_estimated(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.sqlite3")
    actual = replace(
        _event(),
        cost_source=CostSource.PROVIDER_REPORTED_COST,
        actual_cost_micros=1234,
        currency="EUR",
    )
    ledger.append(actual)
    summary = aggregate_cost_rows(ledger.rows(), session_id="session-1")

    assert summary["provider_reported_cost"]["value"] == "0.001234"
    assert summary["provider_reported_cost"]["sample_count"] == 1
    assert summary["estimated_cost"]["value"] is None
    assert summary["estimated_cost"]["complete"] is False


def test_async_recorder_isolates_writer_failure_from_caller() -> None:
    class BrokenWriter:
        def append(self, event: CostEvent) -> bool:
            del event
            raise sqlite3.OperationalError("simulated unavailable database")

    async def scenario() -> None:
        failures: list[tuple[str, dict[str, object]]] = []
        recorder = AsyncCostRecorder(
            BrokenWriter(), on_failure=lambda kind, payload: failures.append((kind, payload))
        )
        assert recorder.record_nowait(_event()) is True
        await recorder.flush()
        await recorder.close()
        assert failures[0][0] == "COST_LEDGER_WRITE_FAILED"
        assert failures[0][1]["reason"] == "OperationalError"

    asyncio.run(scenario())


def test_async_recorder_emits_duplicate_cost_anomaly(tmp_path: Path) -> None:
    async def scenario() -> None:
        anomalies: list[str] = []
        ledger = CostLedger(tmp_path / "costs.sqlite3")
        recorder = AsyncCostRecorder(
            ledger, on_failure=lambda kind, _: anomalies.append(kind)
        )
        event = _event()
        recorder.record_nowait(event)
        recorder.record_nowait(replace(event, cost_event_id="duplicate-delivery"))
        await recorder.close()
        assert anomalies == ["COST_ANOMALY_DETECTED"]
        assert len(ledger.rows()) == 1

    asyncio.run(scenario())
