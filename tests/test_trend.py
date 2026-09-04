"""Tests for trend.py (sub-task 5.1).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
``analyze_trend``:

- Unit tests: improving / stable / declining classification, the recent-third
  (``n // 3``) calculation for an ``n`` not divisible by 3, insufficient/empty
  data, and the zero historical-average path (Req 5.1-5.5).
- Property 8 flavor + Property 1 (determinism): identical input yields
  byte-identical output (Req 5.6).

The set is intentionally minimal (essential examples + one determinism check),
not a large parametrized matrix. A small ``stable_threshold_pct`` of 5.0 is used
unless a case needs otherwise.
"""

import pytest

from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.trend import analyze_trend

_THRESHOLD = 5.0


# ---- Classification: improving / stable / declining -------------------------


def test_improving_vendor_with_hand_calculated_trend_pct() -> None:
    # Feature: vendor-forecasting-agent, Property 8: Trend classification (lead-time)
    # n=9, k = max(1, 9//3) = 3, recent = last 3 = [4,4,4] -> recent_avg = 4.0
    # historical_avg = (10*6 + 4*3) / 9 = 72 / 9 = 8.0
    # trend_pct = ((4.0 - 8.0) / 8.0) * 100 = -50.0  -> improving (< -5.0)
    lead_times = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0]

    result = analyze_trend({"v1": lead_times}, stable_threshold_pct=_THRESHOLD)

    assert result.ok
    trend = result.data["v1"]
    assert trend.direction == "improving"
    assert trend.slope == pytest.approx(-50.0)
    assert trend.point_count == 9


def test_stable_vendor_within_threshold() -> None:
    # Feature: vendor-forecasting-agent, Property 8: Trend classification (lead-time)
    # All equal -> recent_avg == historical_avg -> trend_pct = 0.0 -> stable.
    lead_times = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    result = analyze_trend({"v1": lead_times}, stable_threshold_pct=_THRESHOLD)

    trend = result.data["v1"]
    assert trend.direction == "stable"
    assert trend.slope == pytest.approx(0.0)
    assert trend.point_count == 6


def test_declining_vendor_above_threshold() -> None:
    # Feature: vendor-forecasting-agent, Property 8: Trend classification (lead-time)
    # n=9, k=3, recent = [10,10,10] -> recent_avg = 10.0
    # historical_avg = (4*6 + 10*3) / 9 = 54 / 9 = 6.0
    # trend_pct = ((10.0 - 6.0) / 6.0) * 100 = 66.666... -> declining (> 5.0)
    lead_times = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 10.0, 10.0, 10.0]

    result = analyze_trend({"v1": lead_times}, stable_threshold_pct=_THRESHOLD)

    trend = result.data["v1"]
    assert trend.direction == "declining"
    assert trend.slope == pytest.approx((4.0 / 6.0) * 100.0)
    assert trend.point_count == 9


# ---- Recent-third calculation for n not divisible by 3 ----------------------


def test_recent_third_when_n_not_divisible_by_three() -> None:
    # n=7 -> k = max(1, 7//3) = 2, recent = last 2 = [2,2] -> recent_avg = 2.0
    # historical_avg = (10*5 + 2*2) / 7 = 54 / 7
    # trend_pct = ((2.0 - 54/7) / (54/7)) * 100 = (-40/54) * 100 = -74.074...
    #   -> improving (< -5.0)
    lead_times = [10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 2.0]

    result = analyze_trend({"v1": lead_times}, stable_threshold_pct=_THRESHOLD)

    trend = result.data["v1"]
    historical_avg = 54.0 / 7.0
    expected_pct = ((2.0 - historical_avg) / historical_avg) * 100.0
    assert trend.slope == pytest.approx(expected_pct)
    assert trend.direction == "improving"
    assert trend.point_count == 7


# ---- Insufficient / empty data (Req 5.4) ------------------------------------


def test_insufficient_data_produces_error_and_no_success_entry() -> None:
    lead_times_by_vendor = {"v1": [7.0, 7.0]}  # only 2 periods

    result = analyze_trend(lead_times_by_vendor, stable_threshold_pct=_THRESHOLD)

    assert "v1" not in result.data
    assert result.data == {}
    assert not result.ok
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.vendor_id == "v1"
    assert err.code == "insufficient_data"
    # Input preserved (not mutated).
    assert lead_times_by_vendor == {"v1": [7.0, 7.0]}


def test_empty_input_is_successful_and_empty() -> None:
    result = analyze_trend({}, stable_threshold_pct=_THRESHOLD)

    assert result.data == {}
    assert result.errors == []
    assert result.ok


def test_mixed_sufficient_and_insufficient_vendors() -> None:
    lead_times_by_vendor = {
        "good": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 10.0, 10.0, 10.0],
        "short": [5.0, 5.0],
    }

    result = analyze_trend(lead_times_by_vendor, stable_threshold_pct=_THRESHOLD)

    assert set(result.data) == {"good"}
    assert result.data["good"].direction == "declining"
    assert [e.vendor_id for e in result.errors] == ["short"]
    assert result.errors[0].code == "insufficient_data"


# ---- Zero historical average (Req 5.5) --------------------------------------


def test_zero_historical_average_is_stable_no_division_error() -> None:
    lead_times = [0.0, 0.0, 0.0]

    result = analyze_trend({"v1": lead_times}, stable_threshold_pct=_THRESHOLD)

    trend = result.data["v1"]
    assert trend.direction == "stable"
    assert trend.slope == 0.0
    assert trend.point_count == 3


# ---- Determinism (Req 5.6) --------------------------------------------------


def test_determinism_byte_identical_output() -> None:
    # Feature: vendor-forecasting-agent, Property 8: Trend classification (lead-time)
    lead_times_by_vendor = {
        "v2": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 4.0, 4.0, 4.0],
        "v1": [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 10.0, 10.0, 10.0],
        "v3": [7.0, 7.0],  # insufficient -> error
    }

    first = analyze_trend(lead_times_by_vendor, stable_threshold_pct=_THRESHOLD)
    second = analyze_trend(lead_times_by_vendor, stable_threshold_pct=_THRESHOLD)

    # Identical key order (sorted vendor-id order).
    assert list(first.data.keys()) == list(second.data.keys())
    assert list(first.data.keys()) == ["v1", "v2"]

    # Byte-identical serialized TrendResult payloads.
    first_json = {vid: canonical_json(tr) for vid, tr in first.data.items()}
    second_json = {vid: canonical_json(tr) for vid, tr in second.data.items()}
    assert first_json == second_json

    # Errors emitted in sorted vendor-id order, identical across runs.
    assert [e.vendor_id for e in first.errors] == ["v3"]
    assert [canonical_json(e) for e in first.errors] == [
        canonical_json(e) for e in second.errors
    ]
