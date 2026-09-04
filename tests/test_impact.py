"""Tests for impact.py (Task 9).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
the ``Impact_Module`` (Req 9, 16):

- ``analyze_inventory_impact`` (primary, Req 16): hand-calculated exposure,
  no-delay clamping, non-negativity, multi-vendor sorted output, missing/invalid
  input exclusion, empty input, and determinism.
- ``analyze_impact`` (Req 9): score change pre/post/absolute/percentage,
  unknown-vendor error leaving inputs unchanged, and determinism.

Forecast / VendorMetric / VendorScore / DecisionChange instances are built
directly. The set is intentionally minimal (essential examples + edge cases +
determinism), not a large matrix.
"""

import pytest

from vendor_forecasting_agent.impact import (
    analyze_impact,
    analyze_inventory_impact,
)
from vendor_forecasting_agent.schema import (
    DecisionChange,
    Forecast,
    VendorMetric,
    VendorScore,
    canonical_json,
)


# ---- Small builders ---------------------------------------------------------


def _forecast(vendor_id: str, expected_lead_time: float, horizon: int = 12) -> Forecast:
    """A flat MVP forecast whose first value is the expected lead time."""
    return Forecast(
        vendor_id=vendor_id,
        horizon=horizon,
        values=[expected_lead_time] * horizon,
        seed=0,
    )


def _metric(
    vendor_id: str,
    avg_demand,
    avg_lead_time_days,
    sample_count: int = 12,
) -> VendorMetric:
    return VendorMetric(
        vendor_id=vendor_id,
        avg_demand=avg_demand,
        avg_lead_time_days=avg_lead_time_days,
        sample_count=sample_count,
    )


# =============================================================================
# analyze_inventory_impact (Req 16) — primary focus
# =============================================================================


def test_inventory_hand_calculated_exposure():
    """expected 15, avg 10 -> delay 5; demand 100 -> exposure 500 (Req 16.1/16.2)."""
    forecasts = {"v1": _forecast("v1", 15.0, horizon=12)}
    metrics = {"v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0)}

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.ok
    impact = result.data["v1"]
    assert impact.projected_units == pytest.approx(500.0)
    assert impact.reorder_delta_units == pytest.approx(500.0)
    assert impact.horizon == 12


def test_inventory_no_delay_when_expected_at_or_below_avg():
    """expected 8 <= avg 10 -> delay clamped to 0 -> exposure 0 (Req 16.2)."""
    forecasts = {"v1": _forecast("v1", 8.0)}
    metrics = {"v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0)}

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.ok
    impact = result.data["v1"]
    assert impact.projected_units == pytest.approx(0.0)
    assert impact.reorder_delta_units == pytest.approx(0.0)


def test_inventory_non_negativity_holds_generally():
    """Exposure is always non-negative even for large negative delay (Req 16.2)."""
    forecasts = {"v1": _forecast("v1", 1.0)}
    metrics = {"v1": _metric("v1", avg_demand=500.0, avg_lead_time_days=100.0)}

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.ok
    assert result.data["v1"].projected_units >= 0.0


def test_inventory_multiple_vendors_sorted():
    """One InventoryImpact each, keyed and sorted by vendor_id (Req 16.3)."""
    forecasts = {
        "v2": _forecast("v2", 20.0),
        "v1": _forecast("v1", 15.0),
    }
    metrics = {
        "v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0),
        "v2": _metric("v2", avg_demand=50.0, avg_lead_time_days=5.0),
    }

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.ok
    assert list(result.data.keys()) == ["v1", "v2"]
    assert result.data["v1"].projected_units == pytest.approx(500.0)  # 100 * 5
    assert result.data["v2"].projected_units == pytest.approx(750.0)  # 50 * 15


def test_inventory_forecast_without_metric_excluded():
    """A vendor with a forecast but no VendorMetric is excluded (Req 16.4)."""
    forecasts = {"v1": _forecast("v1", 15.0)}
    metrics: dict[str, VendorMetric] = {}

    result = analyze_inventory_impact(forecasts, metrics)

    assert "v1" not in result.data
    assert result.data == {}
    assert len(result.errors) == 1
    assert result.errors[0].vendor_id == "v1"
    assert result.errors[0].code == "missing_input"
    assert result.errors[0].field == "metric"


def test_inventory_metric_without_forecast_excluded():
    """A vendor with a metric but no forecast never appears (Req 16.4)."""
    forecasts = {"v1": _forecast("v1", 15.0)}
    metrics = {
        "v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0),
        "v2": _metric("v2", avg_demand=50.0, avg_lead_time_days=5.0),
    }

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.ok
    assert list(result.data.keys()) == ["v1"]
    assert "v2" not in result.data


def test_inventory_none_lead_time_excluded():
    """avg_lead_time_days None -> excluded with validation_failure (Req 16.4)."""
    forecasts = {"v1": _forecast("v1", 15.0)}
    metrics = {"v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=None)}

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.data == {}
    assert len(result.errors) == 1
    assert result.errors[0].vendor_id == "v1"
    assert result.errors[0].code == "validation_failure"
    assert result.errors[0].field == "avg_lead_time_days"


def test_inventory_none_demand_excluded():
    """avg_demand None -> excluded with validation_failure (Req 16.4)."""
    forecasts = {"v1": _forecast("v1", 15.0)}
    metrics = {"v1": _metric("v1", avg_demand=None, avg_lead_time_days=10.0)}

    result = analyze_inventory_impact(forecasts, metrics)

    assert result.data == {}
    assert len(result.errors) == 1
    assert result.errors[0].vendor_id == "v1"
    assert result.errors[0].code == "validation_failure"
    assert result.errors[0].field == "avg_demand"


def test_inventory_empty_input_ok():
    """Empty input -> empty successful result (Req 16.4)."""
    result = analyze_inventory_impact({}, {})

    assert result.ok
    assert result.data == {}
    assert result.errors == []


def test_inventory_determinism_byte_identical():
    """Two calls with identical inputs -> byte-identical output (Req 16.3)."""
    forecasts = {
        "v1": _forecast("v1", 15.0),
        "v2": _forecast("v2", 20.0),
    }
    metrics = {
        "v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0),
        "v2": _metric("v2", avg_demand=50.0, avg_lead_time_days=5.0),
    }

    first = analyze_inventory_impact(forecasts, metrics)
    second = analyze_inventory_impact(forecasts, metrics)

    first_json = {vid: canonical_json(ii) for vid, ii in first.data.items()}
    second_json = {vid: canonical_json(ii) for vid, ii in second.data.items()}
    assert first_json == second_json
    assert list(first.data.keys()) == list(second.data.keys())


# =============================================================================
# analyze_impact (Req 9)
# =============================================================================


def test_impact_score_change():
    """metric='score', delta=+10 on score 50 -> pre 50, post 60, abs 10, pct 20 (Req 9.1)."""
    decision = DecisionChange(target_vendor_ids=["v1"], metric="score", delta=10.0)
    scores = {"v1": VendorScore(vendor_id="v1", score=50.0)}
    metrics = {"v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0)}
    forecasts = {"v1": _forecast("v1", 15.0)}

    result = analyze_impact(decision, scores, metrics, forecasts)

    assert result.ok
    change = result.data["v1"].score_change
    assert change is not None
    assert change.pre_value == pytest.approx(50.0)
    assert change.post_value == pytest.approx(60.0)
    assert change.absolute_change == pytest.approx(10.0)
    assert change.percentage_change == pytest.approx(20.0)


def test_impact_unknown_vendor_error_others_processed():
    """Unknown vendor -> unknown_vendor error; other targets still processed (Req 9.3)."""
    decision = DecisionChange(
        target_vendor_ids=["v1", "missing"], metric="score", delta=5.0
    )
    scores = {"v1": VendorScore(vendor_id="v1", score=40.0)}
    metrics = {"v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0)}
    forecasts = {"v1": _forecast("v1", 15.0)}

    result = analyze_impact(decision, scores, metrics, forecasts)

    # The unknown vendor is reported; the known vendor is still processed.
    assert "v1" in result.data
    assert "missing" not in result.data
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.vendor_id == "missing"
    assert err.code == "unknown_vendor"

    # Inputs unchanged.
    assert scores["v1"].score == pytest.approx(40.0)
    assert "missing" not in scores


def test_impact_determinism_byte_identical():
    """Two calls with identical inputs -> byte-identical output (Req 9.2)."""
    decision = DecisionChange(
        target_vendor_ids=["v2", "v1"], metric="score", delta=10.0
    )
    scores = {
        "v1": VendorScore(vendor_id="v1", score=50.0),
        "v2": VendorScore(vendor_id="v2", score=30.0),
    }
    metrics = {
        "v1": _metric("v1", avg_demand=100.0, avg_lead_time_days=10.0),
        "v2": _metric("v2", avg_demand=50.0, avg_lead_time_days=5.0),
    }
    forecasts = {
        "v1": _forecast("v1", 15.0),
        "v2": _forecast("v2", 20.0),
    }

    first = analyze_impact(decision, scores, metrics, forecasts)
    second = analyze_impact(decision, scores, metrics, forecasts)

    first_json = {vid: canonical_json(ir) for vid, ir in first.data.items()}
    second_json = {vid: canonical_json(ir) for vid, ir in second.data.items()}
    assert first_json == second_json
    assert list(first.data.keys()) == list(second.data.keys())
