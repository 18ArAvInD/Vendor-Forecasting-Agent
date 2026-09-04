"""Tests for the multi-scenario registry (demo_scenarios.py).

Standard pytest only (NO Hypothesis, NO streamlit). These tests import only the
scenario registry helpers, the original single-scenario module (for the reuse
guarantee), the input contract, the schema, the streamlit-free view_model, and
the demo output container. They assert that:

* every registered scenario loads, with the default first and present in the
  registry, and ``get_scenario`` round-trips each key;
* every scenario yields a valid ``UpstreamRequest`` whose vendor set matches the
  vendor set of its curated (schema-valid, >= 3 periods) records;
* the default scenario reuses scenario-1 data verbatim and preserves the pinned
  ACU / PMIC-450 results (risk levels, scores, and recommended actions);
* selecting another scenario runs successfully, covers exactly that scenario's
  vendors, and shows a differentiated spread of risk levels;
* the view_model only READS snapshot values (no independent risk calculation);
* each scenario run is fully deterministic (identical canonical JSON).
"""

from vendor_forecasting_agent import demo_scenario
from vendor_forecasting_agent.demo import DemoOutput
from vendor_forecasting_agent.demo_scenarios import (
    DEFAULT_SCENARIO_KEY,
    SCENARIOS,
    Scenario,
    get_scenario,
    list_scenarios,
    run_scenario_by_key,
)
from vendor_forecasting_agent.schema import HistoricalDataRecord, canonical_json
from vendor_forecasting_agent.ui import view_model
from vendor_forecasting_agent.upstream import UpstreamRequest

SCENARIO_KEYS = ["acu-pmic450", "imc-gd88", "eac-is12"]
VALID_RISK_LEVELS = {"low", "moderate", "high", "critical"}


# ---- 1. All scenarios load --------------------------------------------------


def test_all_scenarios_load():
    scenarios = list_scenarios()

    assert all(isinstance(s, Scenario) for s in scenarios)
    assert len(scenarios) >= 3

    # The default scenario is registered and listed first.
    assert scenarios[0].key == DEFAULT_SCENARIO_KEY
    assert DEFAULT_SCENARIO_KEY == "acu-pmic450"
    assert DEFAULT_SCENARIO_KEY in SCENARIOS

    # get_scenario round-trips each expected key.
    for key in SCENARIO_KEYS:
        scenario = get_scenario(key)
        assert isinstance(scenario, Scenario)
        assert scenario.key == key


# ---- 2. Every scenario produces a valid UpstreamRequest ---------------------


def test_every_scenario_request_matches_records():
    for scenario in list_scenarios():
        request = scenario.to_upstream_request()
        assert isinstance(request, UpstreamRequest)
        assert request.relevant_vendors  # non-empty

        records = scenario.build_records()
        assert all(isinstance(r, HistoricalDataRecord) for r in records)

        # The request vendor list matches the record vendor set exactly.
        record_vendors = {r.vendor_id for r in records}
        assert set(request.relevant_vendors) == record_vendors

        # Each vendor has >= 3 periods of schema-valid history.
        periods_by_vendor: dict[str, int] = {}
        for record in records:
            periods_by_vendor[record.vendor_id] = (
                periods_by_vendor.get(record.vendor_id, 0) + 1
            )
        for vendor_id, count in periods_by_vendor.items():
            assert count >= 3, f"{scenario.key}/{vendor_id} has only {count} periods"


# ---- 3. Default scenario reuses scenario-1 data and preserves results -------


def test_default_scenario_reuses_scenario_one_verbatim():
    scenario = get_scenario(DEFAULT_SCENARIO_KEY)

    # The request and records are the identical curated inputs used today.
    assert scenario.request == demo_scenario.UPSTREAM_REQUEST
    assert scenario.build_records() == demo_scenario.build_scenario_records()


def test_default_scenario_preserves_pinned_results():
    output = run_scenario_by_key(DEFAULT_SCENARIO_KEY)
    assert isinstance(output, DemoOutput)
    snapshot = output.snapshot

    risk = snapshot.risk
    assert risk["Alpha Semiconductors"].risk_level == "low"
    assert risk["Alpha Semiconductors"].risk_score == 0.0
    assert risk["Beta Electronics"].risk_level == "moderate"
    assert risk["Beta Electronics"].risk_score == 33.0
    assert risk["Gamma Micro"].risk_level == "critical"
    assert risk["Gamma Micro"].risk_score == 100.0

    action_by_vendor = {rec.vendor_id: rec.action for rec in snapshot.recommendations}
    assert action_by_vendor["Alpha Semiconductors"] == "prefer"
    assert action_by_vendor["Beta Electronics"] == "review"
    assert action_by_vendor["Gamma Micro"] == "replace"


# ---- 4. Selecting another scenario runs successfully ------------------------


def test_other_scenarios_run_and_show_differentiation():
    for key in ("imc-gd88", "eac-is12"):
        scenario = get_scenario(key)
        expected_vendors = set(scenario.to_upstream_request().relevant_vendors)

        output = run_scenario_by_key(key)
        assert isinstance(output, DemoOutput)
        snapshot = output.snapshot

        # Recommendations cover EXACTLY that scenario's relevant vendors.
        recommended = {rec.vendor_id for rec in snapshot.recommendations}
        assert recommended == expected_vendors

        # Every vendor's risk level is a valid literal value.
        levels = [snapshot.risk[v].risk_level for v in expected_vendors]
        assert all(level in VALID_RISK_LEVELS for level in levels)

        # The scenario shows differentiation (more than one distinct level).
        assert len(set(levels)) >= 2, f"{key} levels not differentiated: {levels}"


# ---- 5. view_model reads only (no independent risk calculation) -------------


def test_view_model_reads_snapshot_values_for_non_default_scenario():
    key = "eac-is12"
    scenario = get_scenario(key)
    request = scenario.to_upstream_request()

    output = run_scenario_by_key(key)
    snapshot = output.snapshot

    # Overview rows mirror snapshot risk values exactly.
    rows = view_model.vendor_risk_overview(snapshot)
    for row in rows:
        vendor_id = row["Vendor"]
        assert row["Risk Level"] == snapshot.risk[vendor_id].risk_level
        assert row["Risk Score"] == snapshot.risk[vendor_id].risk_score

    # Selected vendor view surfaces snapshot values, not recomputations.
    action_by_vendor = {rec.vendor_id: rec.action for rec in snapshot.recommendations}
    for vendor_id in request.relevant_vendors:
        view = view_model.selected_vendor_view(
            snapshot, output.explanation, vendor_id, request
        )
        assert view is not None
        assert (
            view["potential_exposure_units"]
            == snapshot.inventory[vendor_id].projected_units
        )
        assert view["recommended_action"] == action_by_vendor[vendor_id]


# ---- 6. Determinism: two runs yield identical canonical JSON ----------------


def test_scenario_runs_are_deterministic():
    for key in (DEFAULT_SCENARIO_KEY, "eac-is12"):
        first = run_scenario_by_key(key)
        second = run_scenario_by_key(key)
        assert canonical_json(first.snapshot) == canonical_json(second.snapshot)
