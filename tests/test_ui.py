"""Tests for the presentation-only UI data-shaping helpers (view_model.py).

Standard pytest only (NO Hypothesis, NO streamlit import, NO streamlit server).
These tests import ONLY from ``vendor_forecasting_agent.ui.view_model`` plus the
existing ``demo`` / ``config`` / ``schema`` modules — never ``app.py`` and never
``streamlit`` — so the suite passes whether or not streamlit is installed.

The helpers under test read values straight off an already-computed
``run_demo`` snapshot/explanation; they recompute no domain values. The tests
assert that the shaped output equals the snapshot values exactly (proving reads,
not recomputation) and that shaping is deterministic.
"""

from vendor_forecasting_agent.config import load_config
from vendor_forecasting_agent.demo import run_demo
from vendor_forecasting_agent.ui.view_model import (
    ELEVATED_RISK_LEVELS,
    risk_table_rows,
    summary_metrics,
    vendor_detail,
)


# ---- 1. Summary metrics match manual counts from the snapshot ---------------


def test_summary_metrics_match_snapshot_counts():
    output = run_demo()
    snapshot = output.snapshot

    metrics = summary_metrics(snapshot)

    expected_analyzed = len(snapshot.recommendations)
    expected_high_critical = sum(
        1
        for risk in snapshot.risk.values()
        if risk.risk_level in ELEVATED_RISK_LEVELS
    )
    expected_exposure = sum(
        1
        for inventory in snapshot.inventory.values()
        if inventory.projected_units > 0
    )

    assert metrics["vendors_analyzed"] == expected_analyzed
    assert metrics["high_critical"] == expected_high_critical
    assert metrics["with_inventory_exposure"] == expected_exposure


# ---- 2. Risk table: one row per recommendation, sorted, values are reads ----


def test_risk_table_rows_one_per_recommendation_sorted_and_reads_values():
    output = run_demo()
    snapshot = output.snapshot

    rows = risk_table_rows(snapshot)

    # One row per recommendation.
    assert len(rows) == len(snapshot.recommendations)

    # Sorted by vendor id.
    vendor_ids = [row["Vendor ID"] for row in rows]
    assert vendor_ids == sorted(vendor_ids)

    # Every value is read straight from the snapshot (no recomputation).
    for row in rows:
        vendor_id = row["Vendor ID"]
        risk = snapshot.risk.get(vendor_id)
        forecast = snapshot.forecasts.get(vendor_id)
        inventory = snapshot.inventory.get(vendor_id)

        assert row["Risk score"] == (risk.risk_score if risk else None)
        assert row["Risk level"] == (risk.risk_level if risk else None)

        expected_lead = (
            forecast.values[0] if forecast and forecast.values else None
        )
        assert row["Expected lead time"] == expected_lead

        expected_units = inventory.projected_units if inventory else 0.0
        assert row["Projected units"] == expected_units

        # Recommended action matches the recommendation for that vendor.
        rec = next(
            r for r in snapshot.recommendations if r.vendor_id == vendor_id
        )
        assert row["Recommended action"] == rec.action


# ---- 3. Vendor detail reads risk fields + includes explanation text ---------


def test_vendor_detail_reads_snapshot_fields_and_explanation():
    output = run_demo()
    snapshot = output.snapshot
    explanation = output.explanation

    # Pick a known recommended vendor.
    known_vendor = sorted(rec.vendor_id for rec in snapshot.recommendations)[0]

    detail = vendor_detail(snapshot, explanation, known_vendor)
    assert detail is not None

    risk = snapshot.risk.get(known_vendor)
    assert detail["vendor_id"] == known_vendor
    assert detail["risk_score"] == (risk.risk_score if risk else None)
    assert detail["risk_level"] == (risk.risk_level if risk else None)
    assert detail["main_risk_drivers"] == (
        list(risk.main_risk_drivers) if risk else []
    )

    rec = next(
        r for r in snapshot.recommendations if r.vendor_id == known_vendor
    )
    assert detail["recommended_action"] == rec.action

    # The explanation text is included verbatim (not recomputed).
    assert detail["explanation_text"] == explanation.text


# ---- 4. Empty snapshot handled gracefully -----------------------------------


def test_empty_snapshot_handled_gracefully():
    output = run_demo(config=load_config(None), records=[])
    snapshot = output.snapshot
    explanation = output.explanation

    metrics = summary_metrics(snapshot)
    assert metrics == {
        "vendors_analyzed": 0,
        "high_critical": 0,
        "with_inventory_exposure": 0,
    }

    assert risk_table_rows(snapshot) == []

    # A missing vendor returns None rather than crashing.
    assert vendor_detail(snapshot, explanation, "vendor-001") is None


# ---- 5. Determinism: shaping the same snapshot twice is equal ---------------


def test_risk_table_rows_deterministic():
    output = run_demo(config=load_config(None))
    snapshot = output.snapshot

    assert risk_table_rows(snapshot) == risk_table_rows(snapshot)
