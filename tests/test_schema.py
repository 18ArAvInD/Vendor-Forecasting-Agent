"""Unit tests for schema.py (sub-task 1.4).

Covers, using standard pytest only (NO Hypothesis):
- Valid construction yields field values equal to the input for representative models.
- Missing required field raises pydantic.ValidationError naming the offending field
  and constructs no instance.
- Out-of-bounds / wrong-type values raise ValidationError naming the offending field.
- extra="forbid" rejects unknown fields; mutation of a frozen model raises.
- Canonical serialization stability (repeatability, kwarg-order independence,
  sorted keys, preserved list order).

_Requirements: 1.2, 1.3, 1.4, 1.5_
"""

import pytest
from pydantic import ValidationError

from vendor_forecasting_agent.schema import (
    Config,
    Forecast,
    HistoricalDataRecord,
    Recommendation,
    TrendResult,
    Vendor,
    VendorMetric,
    VendorRiskResult,
    VendorScore,
    canonical_json,
)


# ---------------------------------------------------------------------------
# Valid construction: field values equal the input (Req 1.2)
# ---------------------------------------------------------------------------


def test_config_valid_construction_preserves_values() -> None:
    cfg = Config(
        random_seed=42,
        forecast_horizon=24,
        record_count=5000,
        stable_threshold_pct=7.5,
    )
    assert cfg.random_seed == 42
    assert cfg.forecast_horizon == 24
    assert cfg.record_count == 5000
    assert cfg.stable_threshold_pct == 7.5


def test_config_defaults_applied() -> None:
    cfg = Config(random_seed=0)
    assert cfg.random_seed == 0
    assert cfg.forecast_horizon == 12
    assert cfg.record_count == 1000
    assert cfg.stable_threshold_pct == 5.0


def test_vendor_valid_construction_preserves_values() -> None:
    vendor = Vendor(vendor_id="v1", name="Acme")
    assert vendor.vendor_id == "v1"
    assert vendor.name == "Acme"


def test_historical_data_record_valid_construction_preserves_values() -> None:
    rec = HistoricalDataRecord(
        vendor_id="v1",
        period_index=3,
        demand=1234.5,
        lead_time_days=10.0,
        defect_rate=0.02,
        on_time_rate=0.98,
        quantity_fulfillment_rate=0.95,
        capacity_utilization=1.2,
        allocation_ratio=0.8,
    )
    assert rec.vendor_id == "v1"
    assert rec.period_index == 3
    assert rec.demand == 1234.5
    assert rec.lead_time_days == 10.0
    assert rec.defect_rate == 0.02
    assert rec.on_time_rate == 0.98
    assert rec.quantity_fulfillment_rate == 0.95
    assert rec.capacity_utilization == 1.2
    assert rec.allocation_ratio == 0.8


def test_vendor_metric_valid_construction_preserves_values() -> None:
    metric = VendorMetric(
        vendor_id="v1",
        avg_demand=100.0,
        avg_lead_time_days=5.0,
        avg_defect_rate=0.1,
        avg_on_time_rate=0.9,
        sample_count=42,
        excluded_count=3,
    )
    assert metric.vendor_id == "v1"
    assert metric.avg_demand == 100.0
    assert metric.avg_lead_time_days == 5.0
    assert metric.avg_defect_rate == 0.1
    assert metric.avg_on_time_rate == 0.9
    assert metric.sample_count == 42
    assert metric.excluded_count == 3


def test_vendor_metric_null_equivalent_empty_data() -> None:
    metric = VendorMetric(vendor_id="v1", sample_count=0)
    assert metric.avg_demand is None
    assert metric.avg_lead_time_days is None
    assert metric.avg_defect_rate is None
    assert metric.avg_on_time_rate is None
    assert metric.sample_count == 0
    assert metric.excluded_count == 0


def test_forecast_valid_construction_preserves_values() -> None:
    values = [1.0, 2.0, 3.0]
    fc = Forecast(vendor_id="v1", horizon=3, values=values, seed=7)
    assert fc.vendor_id == "v1"
    assert fc.horizon == 3
    assert fc.values == values
    assert fc.seed == 7


def test_vendor_score_valid_construction_preserves_values() -> None:
    score = VendorScore(vendor_id="v1", score=87.5)
    assert score.vendor_id == "v1"
    assert score.score == 87.5


def test_trend_result_valid_construction_preserves_values() -> None:
    trend = TrendResult(
        vendor_id="v1", direction="improving", slope=1.25, point_count=5
    )
    assert trend.vendor_id == "v1"
    assert trend.direction == "improving"
    assert trend.slope == 1.25
    assert trend.point_count == 5


def test_recommendation_valid_construction_preserves_values() -> None:
    rec = Recommendation(
        vendor_id="v1",
        action="prefer",
        score=91.0,
        rationale_keys=["high_score", "improving_trend"],
    )
    assert rec.vendor_id == "v1"
    assert rec.action == "prefer"
    assert rec.score == 91.0
    assert rec.rationale_keys == ["high_score", "improving_trend"]


# ---------------------------------------------------------------------------
# Missing required field: ValidationError naming the field, no instance (Req 1.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "kwargs", "missing_field"),
    [
        (Config, {}, "random_seed"),
        (Vendor, {"name": "Acme"}, "vendor_id"),
        (Vendor, {"vendor_id": "v1"}, "name"),
        (
            HistoricalDataRecord,
            {
                "period_index": 1,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "vendor_id",
        ),
        (VendorMetric, {"vendor_id": "v1"}, "sample_count"),
        (
            Forecast,
            {"vendor_id": "v1", "horizon": 1, "values": [1.0]},
            "seed",
        ),
        (VendorScore, {"vendor_id": "v1"}, "score"),
        (
            TrendResult,
            {"vendor_id": "v1", "slope": 0.0, "point_count": 3},
            "direction",
        ),
        (
            Recommendation,
            {"vendor_id": "v1", "score": 50.0},
            "action",
        ),
    ],
)
def test_missing_required_field_raises_naming_field(model, kwargs, missing_field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model(**kwargs)
    errors = exc_info.value.errors()
    assert any(err["loc"] == (missing_field,) for err in errors), (
        f"expected error naming {missing_field!r}, got {[e['loc'] for e in errors]}"
    )
    assert any(err["type"] == "missing" for err in errors)


# ---------------------------------------------------------------------------
# Out-of-bounds / wrong-type values: ValidationError naming the field (Req 1.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "kwargs", "bad_field"),
    [
        # score out of range
        (VendorScore, {"vendor_id": "v1", "score": 100.1}, "score"),
        (VendorScore, {"vendor_id": "v1", "score": -0.1}, "score"),
        (
            Recommendation,
            {"vendor_id": "v1", "action": "prefer", "score": 101.0},
            "score",
        ),
        (
            Recommendation,
            {"vendor_id": "v1", "action": "prefer", "score": -1.0},
            "score",
        ),
        # random_seed out of range
        (Config, {"random_seed": -1}, "random_seed"),
        (Config, {"random_seed": 4_294_967_296}, "random_seed"),
        # defect_rate / on_time_rate outside [0, 1]
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 1.5,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "defect_rate",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": -0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "defect_rate",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 1.5,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "on_time_rate",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": -0.1,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "on_time_rate",
        ),
        # new dimension fields out of bounds
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 1.5,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "quantity_fulfillment_rate",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": -0.1,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "quantity_fulfillment_rate",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": -0.1,
                "allocation_ratio": 0.8,
            },
            "capacity_utilization",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": 1.0,
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": -0.1,
            },
            "allocation_ratio",
        ),
        # forecast_horizon boundaries (0 and 121)
        (Config, {"random_seed": 0, "forecast_horizon": 0}, "forecast_horizon"),
        (Config, {"random_seed": 0, "forecast_horizon": 121}, "forecast_horizon"),
        # Forecast.horizon boundaries
        (
            Forecast,
            {"vendor_id": "v1", "horizon": 0, "values": [1.0], "seed": 0},
            "horizon",
        ),
        (
            Forecast,
            {"vendor_id": "v1", "horizon": 121, "values": [1.0], "seed": 0},
            "horizon",
        ),
        # record_count boundaries (0 and > 1_000_000)
        (Config, {"random_seed": 0, "record_count": 0}, "record_count"),
        (
            Config,
            {"random_seed": 0, "record_count": 1_000_001},
            "record_count",
        ),
    ],
)
def test_out_of_bounds_value_raises_naming_field(model, kwargs, bad_field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model(**kwargs)
    errors = exc_info.value.errors()
    assert any(err["loc"] == (bad_field,) for err in errors), (
        f"expected error naming {bad_field!r}, got {[e['loc'] for e in errors]}"
    )


@pytest.mark.parametrize(
    ("model", "kwargs", "bad_field"),
    [
        # wrong-type: string that cannot coerce to a number
        (Config, {"random_seed": "not-an-int"}, "random_seed"),
        (
            VendorScore,
            {"vendor_id": "v1", "score": "not-a-float"},
            "score",
        ),
        (
            HistoricalDataRecord,
            {
                "vendor_id": "v1",
                "period_index": 0,
                "demand": "nope",
                "lead_time_days": 1.0,
                "defect_rate": 0.1,
                "on_time_rate": 0.9,
                "quantity_fulfillment_rate": 0.95,
                "capacity_utilization": 1.0,
                "allocation_ratio": 0.8,
            },
            "demand",
        ),
    ],
)
def test_wrong_type_value_raises_naming_field(model, kwargs, bad_field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model(**kwargs)
    errors = exc_info.value.errors()
    assert any(err["loc"] == (bad_field,) for err in errors), (
        f"expected error naming {bad_field!r}, got {[e['loc'] for e in errors]}"
    )


def test_invalid_trend_direction_literal_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TrendResult(
            vendor_id="v1", direction="sideways", slope=0.0, point_count=3
        )
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("direction",) for err in errors)


def test_invalid_recommendation_action_literal_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Recommendation(vendor_id="v1", action="promote", score=50.0)
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("action",) for err in errors)


# ---------------------------------------------------------------------------
# extra="forbid" and frozen behaviour (Req 1.4)
# ---------------------------------------------------------------------------


def test_extra_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Vendor(vendor_id="v1", name="Acme", unexpected="x")
    errors = exc_info.value.errors()
    assert any(err["type"] == "extra_forbidden" for err in errors)
    assert any(err["loc"] == ("unexpected",) for err in errors)


def test_mutation_of_frozen_model_raises() -> None:
    vendor = Vendor(vendor_id="v1", name="Acme")
    with pytest.raises(ValidationError) as exc_info:
        vendor.name = "Changed"
    errors = exc_info.value.errors()
    assert any(err["type"] == "frozen_instance" for err in errors)


# ---------------------------------------------------------------------------
# Canonical serialization stability (Req 1.5)
# ---------------------------------------------------------------------------


def test_canonical_json_repeatable_across_calls() -> None:
    fc = Forecast(vendor_id="v1", horizon=3, values=[1.0, 2.0, 3.0], seed=7)
    first = canonical_json(fc)
    second = canonical_json(fc)
    assert first == second


def test_canonical_json_identical_for_different_kwarg_ordering() -> None:
    a = Recommendation(
        vendor_id="v1",
        action="prefer",
        score=91.0,
        rationale_keys=["k1", "k2"],
    )
    b = Recommendation(
        rationale_keys=["k1", "k2"],
        score=91.0,
        action="prefer",
        vendor_id="v1",
    )
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_keys_are_sorted() -> None:
    metric = VendorMetric(
        vendor_id="v1",
        avg_demand=100.0,
        avg_lead_time_days=5.0,
        avg_defect_rate=0.1,
        avg_on_time_rate=0.9,
        sample_count=42,
        excluded_count=3,
    )
    text = canonical_json(metric)
    # Extract the top-level key order from the serialized JSON.
    import json

    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)


def test_canonical_json_preserves_list_order() -> None:
    values = [3.0, 1.0, 2.0]
    fc = Forecast(vendor_id="v1", horizon=3, values=values, seed=7)
    reversed_fc = Forecast(
        vendor_id="v1", horizon=3, values=list(reversed(values)), seed=7
    )
    import json

    assert json.loads(canonical_json(fc))["values"] == [3.0, 1.0, 2.0]
    # Different list order produces different canonical output (order preserved).
    assert canonical_json(fc) != canonical_json(reversed_fc)


# ---------------------------------------------------------------------------
# VendorRiskResult (Req 8.7)
# ---------------------------------------------------------------------------


def test_vendor_risk_result_valid_construction_preserves_values() -> None:
    risk = VendorRiskResult(
        vendor_id="v1",
        risk_score=72.5,
        risk_level="high",
        confidence=0.85,
        main_risk_drivers=["defect_rate", "on_time_rate"],
    )
    assert risk.vendor_id == "v1"
    assert risk.risk_score == 72.5
    assert risk.risk_level == "high"
    assert risk.confidence == 0.85
    assert risk.main_risk_drivers == ["defect_rate", "on_time_rate"]


def test_vendor_risk_result_main_risk_drivers_defaults_empty() -> None:
    risk = VendorRiskResult(
        vendor_id="v1",
        risk_score=10.0,
        risk_level="low",
        confidence=0.5,
    )
    assert risk.main_risk_drivers == []


@pytest.mark.parametrize(
    ("kwargs", "missing_field"),
    [
        (
            {"vendor_id": "v1", "risk_score": 50.0, "confidence": 0.5},
            "risk_level",
        ),
        (
            {"vendor_id": "v1", "risk_level": "low", "confidence": 0.5},
            "risk_score",
        ),
        (
            {"vendor_id": "v1", "risk_score": 50.0, "risk_level": "low"},
            "confidence",
        ),
    ],
)
def test_vendor_risk_result_missing_required_field_raises(kwargs, missing_field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        VendorRiskResult(**kwargs)
    errors = exc_info.value.errors()
    assert any(err["loc"] == (missing_field,) for err in errors), (
        f"expected error naming {missing_field!r}, got {[e['loc'] for e in errors]}"
    )
    assert any(err["type"] == "missing" for err in errors)


@pytest.mark.parametrize(
    ("kwargs", "bad_field"),
    [
        # risk_score out of range
        (
            {"vendor_id": "v1", "risk_score": 100.1, "risk_level": "low", "confidence": 0.5},
            "risk_score",
        ),
        (
            {"vendor_id": "v1", "risk_score": -0.1, "risk_level": "low", "confidence": 0.5},
            "risk_score",
        ),
        # confidence out of range
        (
            {"vendor_id": "v1", "risk_score": 50.0, "risk_level": "low", "confidence": 1.1},
            "confidence",
        ),
        (
            {"vendor_id": "v1", "risk_score": 50.0, "risk_level": "low", "confidence": -0.1},
            "confidence",
        ),
        # invalid risk_level literal
        (
            {"vendor_id": "v1", "risk_score": 50.0, "risk_level": "severe", "confidence": 0.5},
            "risk_level",
        ),
    ],
)
def test_vendor_risk_result_out_of_bounds_raises_naming_field(kwargs, bad_field) -> None:
    with pytest.raises(ValidationError) as exc_info:
        VendorRiskResult(**kwargs)
    errors = exc_info.value.errors()
    assert any(err["loc"] == (bad_field,) for err in errors), (
        f"expected error naming {bad_field!r}, got {[e['loc'] for e in errors]}"
    )
