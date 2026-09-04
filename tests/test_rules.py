"""Tests for rules.py (Task 7).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
``evaluate_rules`` — the fixed six-rule ``Rules_Engine`` (Req 7):

- One concern test per rule (on-time delivery, lead-time declining, quantity
  fulfillment, capacity utilization, allocation, quality/defect).
- Boundary behavior: values exactly at each threshold -> triggered False (strict
  comparison, Req 7.4).
- Multiple vendors: each keyed + sorted with its own six-outcome list; a clean
  vendor has all six triggered False.
- Empty/missing input: {} -> empty ok; avg_* None -> excluded with StageError;
  missing TrendResult -> excluded with StageError (Req 7.5).
- Determinism: identical inputs -> byte-identical outcomes (Req 7.6).

The set is intentionally minimal (essential examples + boundary + determinism),
not a large parametrized matrix. VendorMetric / TrendResult are built directly.
"""

from vendor_forecasting_agent.rules import evaluate_rules
from vendor_forecasting_agent.schema import (
    RuleOutcome,
    TrendResult,
    VendorMetric,
    canonical_json,
)

# Fixed rule_id order the engine emits (Req 7.2/7.3).
_RULE_ORDER = [
    "on_time_delivery_low",
    "lead_time_declining",
    "quantity_fulfillment_low",
    "capacity_utilization_high",
    "allocation_low",
    "defect_rate_high",
]


# ---- Small builders ---------------------------------------------------------


def _clean_metric(vendor_id: str = "V1", **overrides: object) -> VendorMetric:
    """An all-good VendorMetric: every rule condition False by default.

    on_time 0.99, quantity_fulfillment 0.99, allocation 0.99, capacity 0.80,
    defect 0.01. Override one field per concern test.
    """
    fields = {
        "vendor_id": vendor_id,
        "avg_on_time_rate": 0.99,
        "avg_quantity_fulfillment_rate": 0.99,
        "avg_allocation_ratio": 0.99,
        "avg_capacity_utilization": 0.80,
        "avg_defect_rate": 0.01,
        "sample_count": 12,
    }
    fields.update(overrides)
    return VendorMetric(**fields)


def _stable_trend(vendor_id: str = "V1", direction: str = "stable") -> TrendResult:
    """A stable (non-declining) TrendResult by default; override direction."""
    return TrendResult(
        vendor_id=vendor_id,
        direction=direction,
        slope=0.0,
        point_count=12,
    )


def _outcomes(result_data: dict, vendor_id: str) -> dict[str, bool]:
    """Map rule_id -> triggered for the vendor's outcome list, asserting shape."""
    outs = result_data[vendor_id]
    assert [o.rule_id for o in outs] == _RULE_ORDER
    assert len(outs) == 6
    assert all(o.vendor_id == vendor_id for o in outs)
    return {o.rule_id: o.triggered for o in outs}


# ---- One concern per rule ---------------------------------------------------


def test_on_time_delivery_low_concern() -> None:
    # Req 7.2: avg_on_time_rate < 0.90 -> on_time_delivery_low triggered.
    result = evaluate_rules(
        {"V1": _clean_metric(avg_on_time_rate=0.80)},
        {"V1": _stable_trend()},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["on_time_delivery_low"] is True
    # Others unaffected.
    assert triggered["lead_time_declining"] is False
    assert triggered["quantity_fulfillment_low"] is False
    assert triggered["capacity_utilization_high"] is False
    assert triggered["allocation_low"] is False
    assert triggered["defect_rate_high"] is False


def test_lead_time_declining_concern() -> None:
    # Req 7.2: TrendResult.direction == "declining" -> lead_time_declining triggered.
    result = evaluate_rules(
        {"V1": _clean_metric()},
        {"V1": _stable_trend(direction="declining")},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["lead_time_declining"] is True
    assert triggered["on_time_delivery_low"] is False


def test_quantity_fulfillment_low_concern() -> None:
    # Req 7.2: avg_quantity_fulfillment_rate < 0.90 -> quantity_fulfillment_low triggered.
    result = evaluate_rules(
        {"V1": _clean_metric(avg_quantity_fulfillment_rate=0.85)},
        {"V1": _stable_trend()},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["quantity_fulfillment_low"] is True


def test_capacity_utilization_high_concern() -> None:
    # Req 7.2: avg_capacity_utilization > 0.95 -> capacity_utilization_high triggered.
    result = evaluate_rules(
        {"V1": _clean_metric(avg_capacity_utilization=0.98)},
        {"V1": _stable_trend()},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["capacity_utilization_high"] is True


def test_allocation_low_concern() -> None:
    # Req 7.2: avg_allocation_ratio < 0.90 -> allocation_low triggered.
    result = evaluate_rules(
        {"V1": _clean_metric(avg_allocation_ratio=0.70)},
        {"V1": _stable_trend()},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["allocation_low"] is True


def test_defect_rate_high_concern() -> None:
    # Req 7.2: avg_defect_rate > 0.05 -> defect_rate_high triggered.
    result = evaluate_rules(
        {"V1": _clean_metric(avg_defect_rate=0.10)},
        {"V1": _stable_trend()},
    )
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["defect_rate_high"] is True


# ---- Boundary: values exactly at each threshold -> triggered False ----------


def test_boundary_values_exactly_at_thresholds_not_triggered() -> None:
    # Req 7.4: strict comparisons; a value exactly at the threshold is NOT a concern.
    metric = _clean_metric(
        avg_on_time_rate=0.90,
        avg_quantity_fulfillment_rate=0.90,
        avg_allocation_ratio=0.90,
        avg_capacity_utilization=0.95,
        avg_defect_rate=0.05,
    )
    result = evaluate_rules({"V1": metric}, {"V1": _stable_trend()})
    assert result.ok
    triggered = _outcomes(result.data, "V1")
    assert triggered["on_time_delivery_low"] is False
    assert triggered["quantity_fulfillment_low"] is False
    assert triggered["allocation_low"] is False
    assert triggered["capacity_utilization_high"] is False
    assert triggered["defect_rate_high"] is False


# ---- Multiple vendors -------------------------------------------------------


def test_multiple_vendors_each_get_own_six_outcomes_sorted() -> None:
    # Req 7.3/7.6: each vendor keyed with its own six-outcome list, sorted keys.
    metrics = {
        "V2": _clean_metric("V2", avg_defect_rate=0.20),  # defect concern
        "V1": _clean_metric("V1"),  # clean vendor
    }
    trends = {"V1": _stable_trend("V1"), "V2": _stable_trend("V2")}
    result = evaluate_rules(metrics, trends)

    assert result.ok
    assert list(result.data.keys()) == ["V1", "V2"]  # sorted order

    clean = _outcomes(result.data, "V1")
    assert all(v is False for v in clean.values())  # clean vendor all False

    v2 = _outcomes(result.data, "V2")
    assert v2["defect_rate_high"] is True
    assert v2["on_time_delivery_low"] is False


# ---- Empty / missing input --------------------------------------------------


def test_empty_input_returns_empty_ok_result() -> None:
    # Req 7.5: no vendors -> empty successful result, no vendor invented.
    result = evaluate_rules({}, {})
    assert result.ok
    assert result.data == {}
    assert result.errors == []


def test_vendor_with_none_metric_field_is_excluded_with_error() -> None:
    # Req 7.5: a required avg_* is None -> exclude vendor, record StageError.
    metric = _clean_metric("V1")
    bad = metric.model_copy(update={"avg_on_time_rate": None})
    result = evaluate_rules({"V1": bad}, {"V1": _stable_trend()})

    assert not result.ok
    assert "V1" not in result.data
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.vendor_id == "V1"
    assert err.field == "avg_on_time_rate"
    assert err.code == "validation_failure"


def test_vendor_missing_trend_is_excluded_with_error() -> None:
    # Req 7.5: metric present but no TrendResult -> exclude vendor, record StageError.
    result = evaluate_rules({"V1": _clean_metric("V1")}, {})

    assert not result.ok
    assert "V1" not in result.data
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.vendor_id == "V1"
    assert err.field == "trend"
    assert err.code == "missing_input"


def test_missing_input_does_not_affect_other_vendors() -> None:
    # Req 7.5: excluding one vendor leaves others intact.
    metrics = {
        "V1": _clean_metric("V1"),
        "V2": _clean_metric("V2"),  # no trend for V2
    }
    result = evaluate_rules(metrics, {"V1": _stable_trend("V1")})

    assert "V1" in result.data
    assert "V2" not in result.data
    assert [e.vendor_id for e in result.errors] == ["V2"]
    clean = _outcomes(result.data, "V1")
    assert all(v is False for v in clean.values())


# ---- Determinism ------------------------------------------------------------


def test_repeated_execution_is_byte_identical() -> None:
    # Req 7.6: identical inputs -> byte-identical outcomes and identical key order.
    metrics = {
        "V2": _clean_metric("V2", avg_on_time_rate=0.50),
        "V1": _clean_metric("V1"),
    }
    trends = {
        "V1": _stable_trend("V1"),
        "V2": _stable_trend("V2", direction="declining"),
    }

    first = evaluate_rules(metrics, trends)
    second = evaluate_rules(metrics, trends)

    def serialize(result) -> dict[str, list[str]]:
        return {vid: [canonical_json(o) for o in outs] for vid, outs in result.data.items()}

    assert serialize(first) == serialize(second)
    assert list(first.data.keys()) == list(second.data.keys())
