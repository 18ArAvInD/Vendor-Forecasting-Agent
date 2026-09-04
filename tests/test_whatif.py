"""Tests for whatif.py (Layer 1: deterministic vendor-delay what-if).

Standard pytest only (NO Hypothesis, no streamlit, no boto3). These tests
exercise the deterministic what-if module, which REUSES the existing
inventory-exposure formula from ``impact.py`` without modifying it:

- ``analyze_what_if`` (low-level pure function): zero-day delay, positive delay
  with hand-verified values, exposure clamping edge, expected-lead-time
  arithmetic, invalid/negative/bool/non-number delay handling, and determinism.
- ``analyze_what_if_for_vendor`` (snapshot convenience): reuses the same base
  exposure the pipeline computed, and raises for an unknown vendor id.
"""

import pytest

from vendor_forecasting_agent.demo_scenarios import run_scenario_by_key
from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.whatif import (
    WhatIfResult,
    analyze_what_if,
    analyze_what_if_for_vendor,
)


def test_zero_day_delay_leaves_everything_unchanged():
    """delay_days=0 -> no change to expected lead time or exposure."""
    result = analyze_what_if(
        vendor_id="V1",
        delay_days=0.0,
        expected_lead_time=20.0,
        avg_lead_time_days=10.0,
        avg_demand=300.0,
    )
    assert result.new_expected_lead_time == pytest.approx(result.original_expected_lead_time)
    assert result.new_exposure == pytest.approx(result.original_exposure)
    assert result.exposure_change == pytest.approx(0.0)


def test_positive_delay_hand_verified_values():
    """expected=20, avg=10, demand=300, delay=10 -> exact hand-verified numbers."""
    result = analyze_what_if(
        vendor_id="V1",
        delay_days=10.0,
        expected_lead_time=20.0,
        avg_lead_time_days=10.0,
        avg_demand=300.0,
    )
    assert result.new_expected_lead_time == pytest.approx(30.0)
    assert result.original_exposure == pytest.approx(300.0 * (20.0 - 10.0))  # 3000
    assert result.new_exposure == pytest.approx(300.0 * (30.0 - 10.0))  # 6000
    assert result.exposure_change == pytest.approx(3000.0)


def test_exposure_clamps_to_zero_below_avg_then_delay_pushes_positive():
    """When expected <= avg, original_exposure clamps to 0.0; a delay that pushes
    new_expected above avg produces positive new_exposure = demand*(new - avg).

    Hand-verified: expected=8, avg=10, demand=50, delay=5 ->
      original_exposure = 50 * max(0, 8-10) = 0.0
      new_expected = 13
      new_exposure = 50 * (13-10) = 150
      exposure_change = 150 - 0 = 150
    """
    result = analyze_what_if(
        vendor_id="V1",
        delay_days=5.0,
        expected_lead_time=8.0,
        avg_lead_time_days=10.0,
        avg_demand=50.0,
    )
    assert result.original_exposure == pytest.approx(0.0)
    assert result.new_expected_lead_time == pytest.approx(13.0)
    assert result.new_exposure == pytest.approx(150.0)
    assert result.exposure_change == pytest.approx(150.0)


@pytest.mark.parametrize(
    "expected_lead_time,delay_days,expected_new",
    [
        (10.0, 0.0, 10.0),
        (10.0, 5.0, 15.0),
        (20.0, 12.5, 32.5),
        (0.0, 3.0, 3.0),
    ],
)
def test_new_expected_lead_time_is_expected_plus_delay(
    expected_lead_time, delay_days, expected_new
):
    """new_expected_lead_time == expected_lead_time + delay_days across values."""
    result = analyze_what_if(
        vendor_id="V1",
        delay_days=delay_days,
        expected_lead_time=expected_lead_time,
        avg_lead_time_days=5.0,
        avg_demand=100.0,
    )
    assert result.new_expected_lead_time == pytest.approx(expected_new)


def test_negative_delay_raises_value_error():
    with pytest.raises(ValueError):
        analyze_what_if(
            vendor_id="V1",
            delay_days=-1.0,
            expected_lead_time=20.0,
            avg_lead_time_days=10.0,
            avg_demand=300.0,
        )


def test_bool_delay_raises_value_error():
    with pytest.raises(ValueError):
        analyze_what_if(
            vendor_id="V1",
            delay_days=True,  # bool must be rejected, not treated as 1.0
            expected_lead_time=20.0,
            avg_lead_time_days=10.0,
            avg_demand=300.0,
        )


def test_non_number_delay_raises_value_error():
    with pytest.raises(ValueError):
        analyze_what_if(
            vendor_id="V1",
            delay_days="x",  # non-number -> ValueError (documented)
            expected_lead_time=20.0,
            avg_lead_time_days=10.0,
            avg_demand=300.0,
        )


def test_determinism_identical_args_produce_equal_result():
    """Two calls with identical args produce equal WhatIfResult."""
    kwargs = dict(
        vendor_id="V1",
        delay_days=7.0,
        expected_lead_time=18.0,
        avg_lead_time_days=10.0,
        avg_demand=250.0,
        inventory_horizon=12,
    )
    a = analyze_what_if(**kwargs)
    b = analyze_what_if(**kwargs)
    assert a == b
    assert canonical_json(a) == canonical_json(b)


def test_snapshot_convenience_reuses_pipeline_exposure_for_gamma():
    """analyze_what_if_for_vendor reuses the SAME base exposure the pipeline
    computed (proving no second engine), and a delay above avg adds exactly
    avg_demand * delay_days to exposure."""
    snapshot = run_scenario_by_key("acu-pmic450").snapshot

    vendor = "Gamma Micro"
    delay_days = 10.0
    result = analyze_what_if_for_vendor(snapshot, vendor, delay_days)

    assert isinstance(result, WhatIfResult)

    # original_exposure equals the pipeline-computed inventory exposure.
    pipeline_exposure = snapshot.inventory[vendor].projected_units
    assert result.original_exposure == pytest.approx(pipeline_exposure)

    # Gamma's expected lead time is above its avg lead time, so both original and
    # new exposure are on the linear (non-clamped) part of the formula; the delta
    # is therefore exactly avg_demand * delay_days.
    avg_demand = snapshot.metrics[vendor].avg_demand
    assert result.new_exposure - result.original_exposure == pytest.approx(
        avg_demand * delay_days
    )
    assert result.exposure_change == pytest.approx(avg_demand * delay_days)


def test_snapshot_convenience_unknown_vendor_raises_value_error():
    snapshot = run_scenario_by_key("acu-pmic450").snapshot
    with pytest.raises(ValueError):
        analyze_what_if_for_vendor(snapshot, "Nonexistent Vendor", 5.0)
