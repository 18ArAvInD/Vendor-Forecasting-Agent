"""Tests for recommend.py (Task 10).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
the ``Recommendation_Module`` (Req 10):

- Fixed action mapping across risk levels + inventory exposure.
- ``score`` carried through unchanged from ``risk_score``.
- ``rationale_keys`` = ``main_risk_drivers`` (order preserved) plus
  ``"inventory_exposure"`` appended only when ``projected_units > 0``.
- One Recommendation per vendor, output ordered by ``vendor_id``.
- Inventory impact optional; empty input -> empty successful result.
- Determinism (byte-identical output across invocations).

``VendorRiskResult`` / ``InventoryImpact`` instances are built directly. The set
is intentionally minimal (essential examples + edge cases + determinism), not a
large matrix.
"""

import pytest

from vendor_forecasting_agent.recommend import recommend_vendors
from vendor_forecasting_agent.schema import (
    InventoryImpact,
    VendorRiskResult,
    canonical_json,
)


# ---- Small builders ---------------------------------------------------------


def _risk(
    vendor_id: str,
    risk_score: float,
    risk_level: str,
    confidence: float = 0.9,
    main_risk_drivers=None,
) -> VendorRiskResult:
    return VendorRiskResult(
        vendor_id=vendor_id,
        risk_score=risk_score,
        risk_level=risk_level,
        confidence=confidence,
        main_risk_drivers=list(main_risk_drivers or []),
    )


def _impact(
    vendor_id: str,
    projected_units: float,
    reorder_delta_units: float = 0.0,
    horizon: int = 12,
) -> InventoryImpact:
    return InventoryImpact(
        vendor_id=vendor_id,
        projected_units=projected_units,
        reorder_delta_units=reorder_delta_units,
        horizon=horizon,
    )


# ---- Action mapping ---------------------------------------------------------


def test_critical_with_impact_is_replace():
    """critical + projected_units > 0 -> "replace" (Req 10.2)."""
    risk = _risk("v1", 92.0, "critical", main_risk_drivers=["defect_rate", "on_time_rate"])
    result = recommend_vendors({"v1": risk}, {"v1": _impact("v1", 10.0)})

    assert result.ok
    rec = result.data[0]
    assert rec.action == "replace"
    assert rec.score == pytest.approx(92.0)
    assert "inventory_exposure" in rec.rationale_keys
    # main_risk_drivers preserved (in order) ahead of the appended exposure code.
    assert rec.rationale_keys == ["defect_rate", "on_time_rate", "inventory_exposure"]


def test_critical_no_impact_is_review():
    """critical with projected_units == 0 or no InventoryImpact -> "review" (Req 10.2)."""
    risk = _risk("v1", 88.0, "critical", main_risk_drivers=["lead_time"])

    # projected_units == 0
    r_zero = recommend_vendors({"v1": risk}, {"v1": _impact("v1", 0.0)})
    assert r_zero.data[0].action == "review"
    assert "inventory_exposure" not in r_zero.data[0].rationale_keys

    # no InventoryImpact at all
    r_none = recommend_vendors({"v1": risk}, {})
    assert r_none.data[0].action == "review"
    assert "inventory_exposure" not in r_none.data[0].rationale_keys


def test_high_any_impact_is_review():
    """high -> "review" regardless of inventory exposure (Req 10.2)."""
    risk = _risk("v1", 70.0, "high", main_risk_drivers=["capacity"])

    with_impact = recommend_vendors({"v1": risk}, {"v1": _impact("v1", 5.0)})
    without_impact = recommend_vendors({"v1": risk}, {})

    assert with_impact.data[0].action == "review"
    assert without_impact.data[0].action == "review"


def test_moderate_is_review():
    """moderate -> "review" (Req 10.2)."""
    risk = _risk("v1", 50.0, "moderate", main_risk_drivers=["allocation"])
    result = recommend_vendors({"v1": risk}, {"v1": _impact("v1", 3.0)})
    assert result.data[0].action == "review"


def test_low_with_exposure_is_maintain():
    """low + projected_units > 0 -> "maintain" (Req 10.2)."""
    risk = _risk("v1", 10.0, "low", main_risk_drivers=[])
    result = recommend_vendors({"v1": risk}, {"v1": _impact("v1", 4.0)})
    rec = result.data[0]
    assert rec.action == "maintain"
    assert "inventory_exposure" in rec.rationale_keys


def test_low_no_exposure_is_prefer():
    """low + no InventoryImpact -> "prefer"; no "inventory_exposure" (Req 10.2, 10.3)."""
    risk = _risk("v1", 8.0, "low", main_risk_drivers=[])
    result = recommend_vendors({"v1": risk}, {})
    rec = result.data[0]
    assert rec.action == "prefer"
    assert "inventory_exposure" not in rec.rationale_keys


# ---- rationale_keys derivation ----------------------------------------------


def test_rationale_keys_preserve_order_and_append_exposure():
    """main_risk_drivers order preserved; exposure appended only when > 0 (Req 10.3)."""
    drivers = ["defect_rate", "lead_time", "on_time_rate"]

    with_exposure = recommend_vendors(
        {"v1": _risk("v1", 60.0, "high", main_risk_drivers=drivers)},
        {"v1": _impact("v1", 2.0)},
    ).data[0]
    assert with_exposure.rationale_keys == drivers + ["inventory_exposure"]

    without_exposure = recommend_vendors(
        {"v1": _risk("v1", 60.0, "high", main_risk_drivers=drivers)},
        {"v1": _impact("v1", 0.0)},
    ).data[0]
    assert without_exposure.rationale_keys == drivers


# ---- Ordering / one-per-vendor ----------------------------------------------


def test_multiple_vendors_ordered_by_vendor_id():
    """Output is a list ordered by vendor_id, one Recommendation per vendor (Req 10.1, 10.5)."""
    risk_by_vendor = {
        "v3": _risk("v3", 90.0, "critical", main_risk_drivers=["defect_rate"]),
        "v1": _risk("v1", 10.0, "low"),
        "v2": _risk("v2", 55.0, "moderate"),
    }
    inventory_by_vendor = {
        "v3": _impact("v3", 8.0),
        "v1": _impact("v1", 1.0),
    }
    result = recommend_vendors(risk_by_vendor, inventory_by_vendor)

    assert result.ok
    assert [rec.vendor_id for rec in result.data] == ["v1", "v2", "v3"]
    assert len(result.data) == 3
    # v3 critical + exposure -> replace; v1 low + exposure -> maintain; v2 moderate -> review
    by_id = {rec.vendor_id: rec for rec in result.data}
    assert by_id["v3"].action == "replace"
    assert by_id["v1"].action == "maintain"
    assert by_id["v2"].action == "review"


def test_inventory_optional_still_recommends():
    """A vendor with risk but absent from inventory still gets a Recommendation (Req 10.4)."""
    risk_by_vendor = {
        "v1": _risk("v1", 30.0, "moderate"),
        "v2": _risk("v2", 12.0, "low"),
    }
    # Only v1 has inventory; v2 is absent -> treated as no exposure.
    result = recommend_vendors(risk_by_vendor, {"v1": _impact("v1", 5.0)})

    assert [rec.vendor_id for rec in result.data] == ["v1", "v2"]
    by_id = {rec.vendor_id: rec for rec in result.data}
    assert by_id["v2"].action == "prefer"  # low, no exposure
    assert "inventory_exposure" not in by_id["v2"].rationale_keys


# ---- Empty input ------------------------------------------------------------


def test_empty_input_is_empty_successful_result():
    """Empty input -> data == [] and no errors (Req 10.4)."""
    result = recommend_vendors({}, {})
    assert result.data == []
    assert result.errors == []
    assert result.ok


# ---- Determinism ------------------------------------------------------------


def test_determinism_byte_identical_and_same_order():
    """Two calls with identical inputs -> byte-identical output + same order (Req 10.5)."""
    risk_by_vendor = {
        "v2": _risk("v2", 90.0, "critical", main_risk_drivers=["defect_rate", "lead_time"]),
        "v1": _risk("v1", 45.0, "moderate", main_risk_drivers=["capacity"]),
    }
    inventory_by_vendor = {"v2": _impact("v2", 7.0)}

    first = recommend_vendors(risk_by_vendor, inventory_by_vendor)
    second = recommend_vendors(risk_by_vendor, inventory_by_vendor)

    assert [rec.vendor_id for rec in first.data] == [rec.vendor_id for rec in second.data]
    assert [canonical_json(rec) for rec in first.data] == [
        canonical_json(rec) for rec in second.data
    ]
