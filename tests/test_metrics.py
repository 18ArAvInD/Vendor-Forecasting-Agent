"""Tests for metrics.py (sub-tasks 4.2, 4.3, 4.4).

Standard pytest only (NO Hypothesis, no new dependencies). These three sub-tasks
share this single test file because they all exercise ``compute_metrics``:

- Sub-task 4.4 — Unit tests: correct per-dimension averages (hand-calculated),
  multiple vendors, single-record vendor, and empty-input null behavior
  (Req 4.1, 4.3, 4.6, 4.7).
- Sub-task 4.2 — Property 1: Per-stage determinism (metrics) (Req 4.2).
- Sub-task 4.3 — Property 9: Metrics exclusion accounting (Req 4.4).

The test set is intentionally minimal (essential examples + one determinism and
one exclusion property), not a large parametrized matrix.
"""

import pytest

from vendor_forecasting_agent.metrics import compute_metrics
from vendor_forecasting_agent.schema import (
    HistoricalDataRecord,
    VendorMetric,
    canonical_json,
)
from vendor_forecasting_agent.synthetic import generate_historical_data


def _record(vendor_id: str, period_index: int, **overrides: float) -> HistoricalDataRecord:
    """Build a schema-valid record with sensible in-bounds defaults.

    Callers override just the fields whose means they care about; the defaults
    keep every value inside the schema bounds so construction always succeeds.
    """
    fields: dict[str, float] = {
        "demand": 100.0,
        "lead_time_days": 7.0,
        "defect_rate": 0.02,
        "on_time_rate": 0.9,
        "quantity_fulfillment_rate": 0.95,
        "capacity_utilization": 0.8,
        "allocation_ratio": 0.7,
    }
    fields.update(overrides)
    return HistoricalDataRecord(vendor_id=vendor_id, period_index=period_index, **fields)


# ===========================================================================
# Sub-task 4.4 — Unit tests for correct averages, multi-vendor, single record,
# and empty-data null behavior
# Requirements: 4.1, 4.3, 4.6, 4.7
# ===========================================================================


def test_correct_averages_for_all_six_dimensions_hand_calculated() -> None:
    # (A) Hand-calculated means for ONE vendor with two records. Values are
    # chosen so every mean is trivially verifiable:
    #   on_time_rate:               (0.80 + 1.00) / 2 = 0.90
    #   lead_time_days:             (4.0  + 10.0) / 2 = 7.0
    #   quantity_fulfillment_rate:  (0.80 + 0.90) / 2 = 0.85
    #   capacity_utilization:       (0.60 + 1.00) / 2 = 0.80
    #   allocation_ratio:           (0.40 + 0.80) / 2 = 0.60
    #   defect_rate:                (0.02 + 0.06) / 2 = 0.04
    #   demand:                     (100.0 + 300.0)/ 2 = 200.0
    records = [
        _record(
            "vendor-A",
            0,
            on_time_rate=0.80,
            lead_time_days=4.0,
            quantity_fulfillment_rate=0.80,
            capacity_utilization=0.60,
            allocation_ratio=0.40,
            defect_rate=0.02,
            demand=100.0,
        ),
        _record(
            "vendor-A",
            1,
            on_time_rate=1.00,
            lead_time_days=10.0,
            quantity_fulfillment_rate=0.90,
            capacity_utilization=1.00,
            allocation_ratio=0.80,
            defect_rate=0.06,
            demand=300.0,
        ),
    ]

    result = compute_metrics(records)

    assert result.ok
    assert set(result.data.keys()) == {"vendor-A"}
    metric = result.data["vendor-A"]

    assert metric.avg_on_time_rate == pytest.approx(0.90)
    assert metric.avg_lead_time_days == pytest.approx(7.0)
    assert metric.avg_quantity_fulfillment_rate == pytest.approx(0.85)
    assert metric.avg_capacity_utilization == pytest.approx(0.80)
    assert metric.avg_allocation_ratio == pytest.approx(0.60)
    assert metric.avg_defect_rate == pytest.approx(0.04)
    assert metric.avg_demand == pytest.approx(200.0)

    assert metric.sample_count == 2
    assert metric.excluded_count == 0


def test_multiple_vendors_produce_separate_metrics_in_sorted_order() -> None:
    # (B) Three vendors, deliberately provided out of sorted order, with a
    # differing number of records each. Assert one VendorMetric per vendor,
    # keyed by exact vendor_id, and that result keys are in sorted order.
    records = [
        _record("vendor-c", 0),
        _record("vendor-a", 0),
        _record("vendor-a", 1),
        _record("vendor-b", 0),
        _record("vendor-b", 1),
        _record("vendor-b", 2),
    ]

    result = compute_metrics(records)

    assert result.ok
    # One metric per distinct vendor, keyed by vendor_id, ids preserved exactly.
    assert set(result.data.keys()) == {"vendor-a", "vendor-b", "vendor-c"}
    for vid, metric in result.data.items():
        assert isinstance(metric, VendorMetric)
        assert metric.vendor_id == vid

    # Keys are in sorted vendor-id order (Req 11.6 stable ordering).
    assert list(result.data.keys()) == sorted(result.data.keys())

    # Sample counts reflect each vendor's record count.
    assert result.data["vendor-a"].sample_count == 2
    assert result.data["vendor-b"].sample_count == 3
    assert result.data["vendor-c"].sample_count == 1


def test_single_record_vendor_returns_record_values_as_averages() -> None:
    # (C) A vendor with exactly one record: each avg_* equals that record's field
    # value and sample_count == 1.
    record = _record(
        "solo",
        0,
        on_time_rate=0.77,
        lead_time_days=12.5,
        quantity_fulfillment_rate=0.88,
        capacity_utilization=1.10,
        allocation_ratio=0.95,
        defect_rate=0.03,
        demand=555.0,
    )

    result = compute_metrics([record])

    assert result.ok
    metric = result.data["solo"]
    assert metric.avg_on_time_rate == pytest.approx(record.on_time_rate)
    assert metric.avg_lead_time_days == pytest.approx(record.lead_time_days)
    assert metric.avg_quantity_fulfillment_rate == pytest.approx(
        record.quantity_fulfillment_rate
    )
    assert metric.avg_capacity_utilization == pytest.approx(record.capacity_utilization)
    assert metric.avg_allocation_ratio == pytest.approx(record.allocation_ratio)
    assert metric.avg_defect_rate == pytest.approx(record.defect_rate)
    assert metric.avg_demand == pytest.approx(record.demand)
    assert metric.sample_count == 1
    assert metric.excluded_count == 0


def test_empty_input_returns_ok_result_with_empty_data() -> None:
    # (D) Empty input (Req 4.3 as implemented): ok StageResult, data == {}, no
    # vendors invented and no exception raised.
    result = compute_metrics([])

    assert result.ok
    assert result.data == {}


# ===========================================================================
# Sub-task 4.2
# Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (metrics)
# Validates: Requirements 4.2
# ===========================================================================


def test_property1_metrics_byte_identical_across_invocations() -> None:
    # Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (metrics)
    # Source realistic multi-vendor records from the synthetic generator, then
    # compute metrics twice and assert byte-identical canonical output per vendor
    # AND identical key ordering.
    records = generate_historical_data(seed=0, record_count=50)

    run1 = compute_metrics(records)
    run2 = compute_metrics(records)

    assert list(run1.data.keys()) == list(run2.data.keys())
    dump1 = {vid: canonical_json(vm) for vid, vm in run1.data.items()}
    dump2 = {vid: canonical_json(vm) for vid, vm in run2.data.items()}
    assert dump1 == dump2


# ===========================================================================
# Sub-task 4.3
# Feature: vendor-forecasting-agent, Property 9: Metrics exclusion accounting
# Validates: Requirements 4.4
# ===========================================================================


class _NotARecord:
    """Tiny stand-in exposing a matching ``.vendor_id`` but NOT a
    HistoricalDataRecord instance, so ``compute_metrics`` treats it as an invalid
    element attributable to that vendor (defensive exclusion path)."""

    def __init__(self, vendor_id: str) -> None:
        self.vendor_id = vendor_id


def test_exclusion_accounting_mixes_valid_and_invalid_for_same_vendor() -> None:
    # Feature: vendor-forecasting-agent, Property 9: Metrics exclusion accounting
    # Two valid records + two invalid elements attributed to the same vendor.
    # Averages derive from ONLY the valid records; excluded_count == 2.
    #   on_time_rate mean: (0.80 + 1.00) / 2 = 0.90
    #   defect_rate mean:  (0.02 + 0.06) / 2 = 0.04
    records = [
        _record("vendor-X", 0, on_time_rate=0.80, defect_rate=0.02),
        _NotARecord("vendor-X"),
        _record("vendor-X", 1, on_time_rate=1.00, defect_rate=0.06),
        _NotARecord("vendor-X"),
    ]

    result = compute_metrics(records)  # type: ignore[arg-type]

    assert result.ok
    metric = result.data["vendor-X"]
    assert metric.sample_count == 2
    assert metric.excluded_count == 2
    # Hand-checked means computed from the valid records only.
    assert metric.avg_on_time_rate == pytest.approx(0.90)
    assert metric.avg_defect_rate == pytest.approx(0.04)


def test_all_records_excluded_yields_null_equivalent_metric() -> None:
    # Feature: vendor-forecasting-agent, Property 9: Metrics exclusion accounting
    # When ALL of a vendor's elements are excluded, that vendor still appears
    # with sample_count 0, every avg_* None, and excluded_count == the count
    # (the implemented Req 4.3/4.4 null-equivalent path).
    records = [
        _NotARecord("vendor-Y"),
        _NotARecord("vendor-Y"),
        _NotARecord("vendor-Y"),
    ]

    result = compute_metrics(records)  # type: ignore[arg-type]

    assert result.ok
    metric = result.data["vendor-Y"]
    assert metric.sample_count == 0
    assert metric.excluded_count == 3
    assert metric.avg_on_time_rate is None
    assert metric.avg_lead_time_days is None
    assert metric.avg_quantity_fulfillment_rate is None
    assert metric.avg_capacity_utilization is None
    assert metric.avg_allocation_ratio is None
    assert metric.avg_defect_rate is None
    assert metric.avg_demand is None
