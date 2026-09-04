"""Tests for the curated demo scenario (demo_scenario.py).

Standard pytest only (NO Hypothesis, NO streamlit). These tests import only the
scenario helpers, the streamlit-free view_model, the schema, and the demo output
container. They assert that:

* the curated records are schema-valid, cover exactly the three named vendors,
  and have >= 3 periods each;
* ``run_scenario`` flows into the EXISTING pipeline and yields the pinned risk
  levels (Alpha=low, Beta=moderate, Gamma=critical);
* inventory ``projected_units`` are READ from the snapshot (view_model surfaces
  the same value, not a recomputation);
* the vendor comparison and recommendation views read snapshot values verbatim;
* the run is fully deterministic (identical canonical JSON across two runs).
"""

from vendor_forecasting_agent.demo import DemoOutput
from vendor_forecasting_agent.demo_scenario import (
    ALPHA,
    BETA,
    GAMMA,
    UPSTREAM_REQUEST,
    build_scenario_records,
    run_scenario,
)
from vendor_forecasting_agent.schema import HistoricalDataRecord, canonical_json
from vendor_forecasting_agent.ui import view_model

NAMED_VENDORS = {ALPHA, BETA, GAMMA}


# ---- 2. Curated records are schema-valid, three vendors, >= 3 periods -------


def test_build_scenario_records_shape():
    records = build_scenario_records()

    assert all(isinstance(r, HistoricalDataRecord) for r in records)

    periods_by_vendor: dict[str, int] = {}
    for record in records:
        periods_by_vendor[record.vendor_id] = periods_by_vendor.get(
            record.vendor_id, 0
        ) + 1

    assert set(periods_by_vendor) == NAMED_VENDORS
    for vendor_id, count in periods_by_vendor.items():
        assert count >= 3, f"{vendor_id} has only {count} periods"


# ---- 3. run_scenario flows into the existing pipeline; pinned risk levels ---


def test_run_scenario_produces_pinned_risk_levels():
    output = run_scenario()

    assert isinstance(output, DemoOutput)

    recommended = {rec.vendor_id for rec in output.snapshot.recommendations}
    assert recommended == NAMED_VENDORS

    assert output.snapshot.risk[ALPHA].risk_level == "low"
    assert output.snapshot.risk[BETA].risk_level == "moderate"
    assert output.snapshot.risk[GAMMA].risk_level == "critical"


# ---- 4. Inventory context is READ from the snapshot, not recomputed ---------


def test_inventory_projected_units_read_from_snapshot():
    output = run_scenario()
    snapshot = output.snapshot

    assert set(snapshot.inventory) == NAMED_VENDORS

    for vendor_id in NAMED_VENDORS:
        view = view_model.selected_vendor_view(
            snapshot, output.explanation, vendor_id, UPSTREAM_REQUEST
        )
        assert view is not None
        # The surfaced exposure equals the snapshot value exactly (a read).
        assert (
            view["potential_exposure_units"]
            == snapshot.inventory[vendor_id].projected_units
        )
        assert view["projected_units"] == snapshot.inventory[vendor_id].projected_units


# ---- 5. Vendor comparison rows read snapshot risk values, sorted ------------


def test_vendor_risk_overview_reads_snapshot_values_sorted():
    output = run_scenario()
    snapshot = output.snapshot

    rows = view_model.vendor_risk_overview(snapshot)

    assert [row["Vendor"] for row in rows] == sorted(NAMED_VENDORS)
    for row in rows:
        vendor_id = row["Vendor"]
        assert row["Risk Level"] == snapshot.risk[vendor_id].risk_level
        assert row["Risk Score"] == snapshot.risk[vendor_id].risk_score


# ---- 6. Recommendation shown equals the snapshot recommendation action ------


def test_selected_vendor_recommendation_reads_snapshot():
    output = run_scenario()
    snapshot = output.snapshot

    rec_by_vendor = {rec.vendor_id: rec for rec in snapshot.recommendations}

    for vendor_id in NAMED_VENDORS:
        view = view_model.selected_vendor_view(
            snapshot, output.explanation, vendor_id, UPSTREAM_REQUEST
        )
        assert view is not None
        assert view["recommended_action"] == rec_by_vendor[vendor_id].action


# ---- 7. Determinism: two runs produce identical canonical JSON --------------


def test_run_scenario_is_deterministic():
    first = run_scenario()
    second = run_scenario()
    assert canonical_json(first.snapshot) == canonical_json(second.snapshot)


# ---- Product context presentation-only coverage ----------------------------


def test_product_context_coverage_days_is_presentation_ratio():
    ctx = view_model.product_context(UPSTREAM_REQUEST)
    assert ctx["product_id"] == "ACU-100"
    assert ctx["component_id"] == "PMIC-450"
    # 6500 / 300 == presentation-only coverage days ratio.
    assert ctx["inventory_coverage_days"] == 6500.0 / 300.0
