"""Tests for forecast.py (sub-task 6.1).

Standard pytest only (NO Hypothesis, no new dependencies). These essential MVP
tests exercise ``forecast_vendors``:

- Hand-calculated expected values (base + trend_adjustment + variability_buffer)
- Exact horizon length (1, 12)
- Multiple vendors, sorted keying
- Trend direction and variability effects on the expected value
- Non-negativity clamping
- Empty / missing / insufficient inputs and invalid horizon rejection
- Determinism via canonical_json

VendorMetric / TrendResult instances are built directly. The set is intentionally
minimal (essential examples), not a large parametrized matrix.
"""

import math

import pytest

from vendor_forecasting_agent.forecast import forecast_vendors
from vendor_forecasting_agent.schema import (
    TrendResult,
    VendorMetric,
    canonical_json,
)


# ---- Small builders ---------------------------------------------------------


def _metric(vendor_id: str, avg_lead_time_days) -> VendorMetric:
    return VendorMetric(
        vendor_id=vendor_id,
        avg_lead_time_days=avg_lead_time_days,
        sample_count=1,
    )


def _trend(vendor_id: str, slope: float) -> TrendResult:
    # direction is not consumed by the forecast; only slope (trend_pct) is used.
    if slope < 0:
        direction = "improving"
    elif slope > 0:
        direction = "declining"
    else:
        direction = "stable"
    return TrendResult(
        vendor_id=vendor_id,
        direction=direction,
        slope=slope,
        point_count=3,
    )


# ---- 1. Hand-calculated expected values -------------------------------------


def test_stable_zero_variability_expected_equals_base() -> None:
    # base=10, slope=0 -> trend_adjustment=0; lead_times all 10 -> pstdev=0
    # expected = max(0, 10 + 0 + 0) = 10.0
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 0.0)},
        {"v1": [10.0, 10.0, 10.0]},
        horizon=3,
    )

    assert result.ok
    forecast = result.data["v1"]
    assert forecast.values == pytest.approx([10.0, 10.0, 10.0])
    assert all(v == pytest.approx(10.0) for v in forecast.values)
    assert forecast.seed == 0


def test_nonzero_trend_and_variability_hand_computed() -> None:
    # base=10, slope=20.0 -> trend_adjustment = 10 * (20/100) = 2.0
    # lead_times=[8,10,12] -> pstdev = sqrt(((-2)^2 + 0 + 2^2)/3) = sqrt(8/3)
    # expected = 10 + 2 + sqrt(8/3) = 12 + 1.6329931... = 13.6329931...
    expected = 12.0 + math.sqrt(8.0 / 3.0)

    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 20.0)},
        {"v1": [8.0, 10.0, 12.0]},
        horizon=4,
    )

    assert result.ok
    forecast = result.data["v1"]
    assert all(v == pytest.approx(expected) for v in forecast.values)
    assert forecast.values[0] == pytest.approx(13.632993161855452)


# ---- 2. Exact horizon length ------------------------------------------------


def test_horizon_length_typical_twelve() -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 0.0)},
        {"v1": [10.0, 10.0]},
        horizon=12,
    )

    assert result.ok
    assert len(result.data["v1"].values) == 12
    assert result.data["v1"].horizon == 12


def test_horizon_length_one() -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 0.0)},
        {"v1": [10.0, 10.0]},
        horizon=1,
    )

    assert result.ok
    assert len(result.data["v1"].values) == 1


# ---- 3. Multiple vendors, sorted keying -------------------------------------


def test_multiple_vendors_one_forecast_each_sorted() -> None:
    result = forecast_vendors(
        {
            "v2": _metric("v2", 5.0),
            "v1": _metric("v1", 10.0),
        },
        {
            "v2": _trend("v2", 0.0),
            "v1": _trend("v1", 0.0),
        },
        {
            "v2": [5.0, 5.0],
            "v1": [10.0, 10.0],
        },
        horizon=2,
    )

    assert result.ok
    assert set(result.data) == {"v1", "v2"}
    assert list(result.data) == ["v1", "v2"]  # sorted vendor-id order
    assert result.data["v1"].values[0] == pytest.approx(10.0)
    assert result.data["v2"].values[0] == pytest.approx(5.0)


# ---- 4. Trend affects direction ---------------------------------------------


def test_trend_sign_reflected_in_expected() -> None:
    # Same base and history; positive slope (declining lead time) yields a
    # higher expected than negative slope (improving).
    base = 10.0
    history = [10.0, 10.0, 10.0]  # pstdev == 0 so buffer is identical (0.0)

    result = forecast_vendors(
        {"up": _metric("up", base), "down": _metric("down", base)},
        {"up": _trend("up", 30.0), "down": _trend("down", -30.0)},
        {"up": list(history), "down": list(history)},
        horizon=1,
    )

    assert result.ok
    up_expected = result.data["up"].values[0]
    down_expected = result.data["down"].values[0]
    assert up_expected > base  # positive trend_adjustment
    assert down_expected < base  # negative trend_adjustment
    assert up_expected > down_expected


# ---- 5. Variability affects expected ----------------------------------------


def test_higher_variability_yields_strictly_higher_expected() -> None:
    # Same base and slope=0; the higher-spread history has a larger buffer.
    result = forecast_vendors(
        {"low": _metric("low", 10.0), "high": _metric("high", 10.0)},
        {"low": _trend("low", 0.0), "high": _trend("high", 0.0)},
        {"low": [10.0, 10.0, 10.0], "high": [2.0, 10.0, 18.0]},
        horizon=1,
    )

    assert result.ok
    assert result.data["high"].values[0] > result.data["low"].values[0]


# ---- 6. Non-negativity clamping ---------------------------------------------


def test_expected_clamped_to_non_negative() -> None:
    # base=10, slope=-200 -> trend_adjustment = 10 * (-200/100) = -20
    # history all equal -> buffer 0 -> raw = 10 - 20 + 0 = -10 -> clamped to 0.0
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", -200.0)},
        {"v1": [10.0, 10.0, 10.0]},
        horizon=2,
    )

    assert result.ok
    forecast = result.data["v1"]
    assert all(v == pytest.approx(0.0) for v in forecast.values)
    assert all(v >= 0.0 for v in forecast.values)


# ---- 7. Empty / missing / insufficient / invalid horizon --------------------


def test_empty_metrics_yields_empty_ok_result() -> None:
    result = forecast_vendors({}, {}, {}, horizon=12)
    assert result.ok
    assert result.data == {}


def test_vendor_missing_trend_is_excluded_with_error() -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {},  # no trend for v1
        {"v1": [10.0, 10.0]},
        horizon=3,
    )

    assert "v1" not in result.data
    assert not result.ok
    assert any(e.vendor_id == "v1" for e in result.errors)


def test_vendor_none_lead_time_metric_is_excluded_with_error() -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", None)},  # avg_lead_time_days None
        {"v1": _trend("v1", 0.0)},
        {"v1": [10.0, 10.0]},
        horizon=3,
    )

    assert "v1" not in result.data
    assert any(e.vendor_id == "v1" and e.field == "avg_lead_time_days" for e in result.errors)


def test_vendor_empty_history_is_excluded_with_error() -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 0.0)},
        {"v1": []},  # empty lead-time history
        horizon=3,
    )

    assert "v1" not in result.data
    assert any(e.vendor_id == "v1" for e in result.errors)


@pytest.mark.parametrize("bad_horizon", [0, 121, 5.0, True])
def test_invalid_horizon_rejects_request(bad_horizon) -> None:
    result = forecast_vendors(
        {"v1": _metric("v1", 10.0)},
        {"v1": _trend("v1", 0.0)},
        {"v1": [10.0, 10.0]},
        horizon=bad_horizon,
    )

    assert result.data == {}
    assert not result.ok
    assert len(result.errors) == 1
    assert result.errors[0].code == "invalid_horizon"
    assert result.errors[0].field == "horizon"
    assert result.errors[0].vendor_id is None


# ---- 8. Determinism ---------------------------------------------------------


def test_identical_inputs_produce_byte_identical_output() -> None:
    metrics = {"v1": _metric("v1", 10.0), "v2": _metric("v2", 7.5)}
    trends = {"v1": _trend("v1", 15.0), "v2": _trend("v2", -10.0)}
    history = {"v1": [8.0, 10.0, 12.0], "v2": [6.0, 7.0, 9.0]}

    first = forecast_vendors(metrics, trends, history, horizon=12)
    second = forecast_vendors(metrics, trends, history, horizon=12)

    assert list(first.data) == list(second.data)  # identical key order
    assert {vid: canonical_json(f) for vid, f in first.data.items()} == {
        vid: canonical_json(f) for vid, f in second.data.items()
    }
