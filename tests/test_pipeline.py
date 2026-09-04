"""End-to-end tests for pipeline.py (Task 12.1 — Phase 1 deterministic driver).

Standard pytest only (NO Hypothesis, no new dependencies). These are essential
end-to-end tests of the Phase 1 driver :func:`run_pipeline`, which wires the
deterministic stages and assembles the immutable ``DeterministicResults``
snapshot (Req 11, 13.2):

- Normal multi-vendor flow over synthetic data.
- Vendors flow coherently through the stages (metrics -> scores/risk ->
  recommendations).
- Deterministic repeated execution (byte-identical snapshot).
- Empty input yields an empty snapshot and an ok result.
- A vendor with too few periods for a trend is excluded (with recorded errors)
  while healthy vendors still produce recommendations.
"""

from vendor_forecasting_agent.config import load_config
from vendor_forecasting_agent.pipeline import run_pipeline
from vendor_forecasting_agent.schema import (
    Config,
    DeterministicResults,
    HistoricalDataRecord,
    canonical_json,
)
from vendor_forecasting_agent.synthetic import (
    DEFAULT_VENDOR_IDS,
    generate_historical_data,
)


# ---- Small builders ---------------------------------------------------------


def _record(vendor_id: str, period_index: int, lead_time_days: float = 7.0) -> HistoricalDataRecord:
    """Build a schema-valid record with healthy, non-triggering metric values."""
    return HistoricalDataRecord(
        vendor_id=vendor_id,
        period_index=period_index,
        demand=1000.0,
        lead_time_days=lead_time_days,
        defect_rate=0.01,
        on_time_rate=0.98,
        quantity_fulfillment_rate=0.99,
        capacity_utilization=0.80,
        allocation_ratio=0.97,
    )


# ---- 1. Normal multi-vendor flow --------------------------------------------


def test_normal_multi_vendor_flow_populates_snapshot():
    records = generate_historical_data(seed=42, record_count=250)
    config = load_config(None)

    result = run_pipeline(records, config)
    snapshot = result.data

    assert isinstance(snapshot, DeterministicResults)

    # All five default vendors have enough data to appear across the core stages.
    expected_vendors = set(DEFAULT_VENDOR_IDS)
    assert set(snapshot.metrics) == expected_vendors
    assert set(snapshot.trends) == expected_vendors
    assert set(snapshot.forecasts) == expected_vendors
    assert set(snapshot.scores) == expected_vendors
    assert set(snapshot.risk) == expected_vendors
    assert set(snapshot.outcomes) == expected_vendors
    assert set(snapshot.inventory) == expected_vendors

    # Exactly one recommendation per vendor that made it through risk.
    rec_vendors = [rec.vendor_id for rec in snapshot.recommendations]
    assert set(rec_vendors) == expected_vendors
    assert len(rec_vendors) == len(expected_vendors)

    # impacts (decision/change) is not part of the Phase 1 flow.
    assert snapshot.impacts == {}

    # Every stage produced its vendors without recorded errors on healthy data.
    assert result.ok


# ---- 2. Vendors flow through stages -----------------------------------------


def test_vendors_flow_coherently_through_stages():
    records = generate_historical_data(seed=7, record_count=200)
    config = Config(random_seed=0)

    result = run_pipeline(records, config)
    snapshot = result.data

    metric_vendors = set(snapshot.metrics)
    # Vendors present in metrics (and eligible) also appear in scores and risk.
    assert set(snapshot.scores) <= metric_vendors
    assert set(snapshot.risk) <= metric_vendors
    assert set(snapshot.scores) == set(snapshot.risk)

    # Every recommendation targets one of the input vendors present in risk.
    for rec in snapshot.recommendations:
        assert rec.vendor_id in snapshot.risk
        assert rec.vendor_id in metric_vendors
        # score is carried through unchanged from the risk result.
        assert rec.score == snapshot.risk[rec.vendor_id].risk_score


# ---- 3. Deterministic repeated execution ------------------------------------


def test_repeated_execution_is_byte_identical():
    records = generate_historical_data(seed=123, record_count=180)
    config = load_config(None)

    first = run_pipeline(records, config)
    second = run_pipeline(records, config)

    assert canonical_json(first.data) == canonical_json(second.data)


# ---- 4. Empty input ---------------------------------------------------------


def test_empty_input_yields_empty_snapshot():
    config = load_config(None)

    result = run_pipeline([], config)
    snapshot = result.data

    assert isinstance(snapshot, DeterministicResults)
    assert snapshot.metrics == {}
    assert snapshot.trends == {}
    assert snapshot.forecasts == {}
    assert snapshot.scores == {}
    assert snapshot.risk == {}
    assert snapshot.outcomes == {}
    assert snapshot.impacts == {}
    assert snapshot.inventory == {}
    assert snapshot.recommendations == []
    assert result.ok


# ---- 5. Under-data vendor excluded, healthy vendors still recommended -------


def test_vendor_with_too_few_periods_excluded_but_others_flow():
    # vendor-short has only 2 periods (< 3 required for a trend); the other two
    # have >= 3 periods and should flow through to recommendations.
    records = [
        _record("vendor-short", 0),
        _record("vendor-short", 1),
        _record("vendor-a", 0),
        _record("vendor-a", 1),
        _record("vendor-a", 2),
        _record("vendor-b", 0),
        _record("vendor-b", 1),
        _record("vendor-b", 2),
    ]
    config = load_config(None)

    result = run_pipeline(records, config)
    snapshot = result.data

    # Metrics are computed for every vendor, including the under-data one.
    assert set(snapshot.metrics) == {"vendor-short", "vendor-a", "vendor-b"}

    # The under-data vendor has no trend/forecast/recommendation.
    assert "vendor-short" not in snapshot.trends
    assert "vendor-short" not in snapshot.forecasts
    rec_vendors = {rec.vendor_id for rec in snapshot.recommendations}
    assert "vendor-short" not in rec_vendors

    # Healthy vendors still produce trends and recommendations.
    assert {"vendor-a", "vendor-b"} <= set(snapshot.trends)
    assert {"vendor-a", "vendor-b"} == rec_vendors

    # The pipeline did not crash and recorded an error for the excluded vendor.
    assert not result.ok
    assert any(err.vendor_id == "vendor-short" for err in result.errors)
