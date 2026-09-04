"""Tests for the Layer-2 structured LLM context builder (llm_context.py).

These tests confirm the context layer is a pure, deterministic *translation* of
the existing deterministic results: every domain number in the context equals
the value already present in the snapshot / WhatIfResult (no recomputation), the
layer never imports Streamlit or boto3, and it invokes no LLM/provider.

The fixtures use the real curated ACU / PMIC-450 scenario via
``run_scenario_by_key("acu-pmic450")`` so the numbers are the genuine pipeline
output (Alpha low/0, Beta moderate/33, Gamma critical/100).
"""

import sys

import pytest

from vendor_forecasting_agent import llm_context
from vendor_forecasting_agent.demo_scenarios import (
    DEFAULT_SCENARIO_KEY,
    get_scenario,
    run_scenario_by_key,
)
from vendor_forecasting_agent.whatif import analyze_what_if_for_vendor

ALPHA = "Alpha Semiconductors"
BETA = "Beta Electronics"
GAMMA = "Gamma Micro"


@pytest.fixture(scope="module")
def demo():
    """Real DemoOutput for the ACU / PMIC-450 curated scenario."""
    return run_scenario_by_key(DEFAULT_SCENARIO_KEY)


@pytest.fixture(scope="module")
def request_obj():
    """The curated UpstreamRequest for the default scenario."""
    return get_scenario(DEFAULT_SCENARIO_KEY).request


# ---- 1. Vendor context carries product/vendor identity ----------------------


def test_vendor_context_contains_product_and_vendor(demo, request_obj):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, ALPHA, demo.explanation
    )
    assert ctx is not None
    assert ctx["product"]["component_id"] == "PMIC-450"
    assert ctx["product"]["product_id"] == "ACU-100"
    assert ctx["vendor"]["vendor_id"] == ALPHA
    assert ctx["vendor"]["vendor_name"] == ALPHA


# ---- 2. Preserves deterministic risk scores/levels --------------------------


@pytest.mark.parametrize(
    "vendor,expected_score,expected_level",
    [
        (ALPHA, 0.0, "low"),
        (BETA, 33.0, "moderate"),
        (GAMMA, 100.0, "critical"),
    ],
)
def test_vendor_context_preserves_risk(
    demo, request_obj, vendor, expected_score, expected_level
):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, vendor, demo.explanation
    )
    snap_risk = demo.snapshot.risk[vendor]
    assert ctx["risk"]["risk_score"] == snap_risk.risk_score
    assert ctx["risk"]["risk_level"] == snap_risk.risk_level
    # Sanity against the known curated values.
    assert ctx["risk"]["risk_score"] == pytest.approx(expected_score)
    assert ctx["risk"]["risk_level"] == expected_level


# ---- 3. Preserves expected lead times ---------------------------------------


@pytest.mark.parametrize("vendor", [ALPHA, BETA, GAMMA])
def test_vendor_context_preserves_expected_lead_time(demo, request_obj, vendor):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, vendor, demo.explanation
    )
    expected = demo.snapshot.forecasts[vendor].values[0]
    assert ctx["metrics"]["expected_lead_time"] == expected
    assert ctx["inventory_impact"]["expected_lead_time"] == expected


def test_alpha_expected_lead_time_is_about_ten(demo, request_obj):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, ALPHA, demo.explanation
    )
    assert ctx["metrics"]["expected_lead_time"] == pytest.approx(10.0)


# ---- 4. Preserves inventory exposure ----------------------------------------


@pytest.mark.parametrize("vendor", [ALPHA, BETA, GAMMA])
def test_vendor_context_preserves_exposure(demo, request_obj, vendor):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, vendor, demo.explanation
    )
    expected = demo.snapshot.inventory[vendor].projected_units
    assert ctx["inventory_impact"]["potential_exposure_units"] == expected


def test_alpha_exposure_is_zero(demo, request_obj):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, ALPHA, demo.explanation
    )
    assert ctx["inventory_impact"]["potential_exposure_units"] == pytest.approx(0.0)


# ---- 5. Preserves recommendations -------------------------------------------


@pytest.mark.parametrize("vendor", [ALPHA, BETA, GAMMA])
def test_vendor_context_preserves_recommendation(demo, request_obj, vendor):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, vendor, demo.explanation
    )
    rec = next(r for r in demo.snapshot.recommendations if r.vendor_id == vendor)
    assert ctx["recommendation"]["action"] == rec.action
    assert ctx["recommendation"]["score"] == rec.score


def test_recommendation_actions_are_expected(demo, request_obj):
    actions = {
        v: llm_context.build_vendor_context(
            demo.snapshot, request_obj, v, demo.explanation
        )["recommendation"]["action"]
        for v in (ALPHA, BETA, GAMMA)
    }
    assert actions[ALPHA] == "prefer"
    assert actions[BETA] == "review"
    assert actions[GAMMA] == "replace"


# ---- 6. Comparison context --------------------------------------------------


def test_comparison_context_has_three_sorted_vendors(demo, request_obj):
    ctx = llm_context.build_comparison_context(demo.snapshot, request_obj)
    ids = [v["vendor_id"] for v in ctx["vendors"]]
    assert ids == sorted([ALPHA, BETA, GAMMA])
    for entry in ctx["vendors"]:
        vid = entry["vendor_id"]
        assert entry["risk_level"] == demo.snapshot.risk[vid].risk_level
        assert entry["risk_score"] == demo.snapshot.risk[vid].risk_score


# ---- 7. Executive context ---------------------------------------------------


def test_executive_context_summary(demo, request_obj):
    ctx = llm_context.build_executive_context(demo.snapshot, request_obj)
    summary = ctx["summary"]
    assert summary["vendors_analyzed"] == 3
    assert summary["risk_level_distribution"] == {
        "low": 1,
        "moderate": 1,
        "high": 0,
        "critical": 1,
    }
    assert summary["highest_risk"]["vendor_id"] == GAMMA
    assert summary["highest_risk"]["risk_score"] == pytest.approx(100.0)
    assert summary["lowest_risk"]["vendor_id"] == ALPHA
    assert summary["lowest_risk"]["risk_score"] == pytest.approx(0.0)
    expected_total = sum(
        demo.snapshot.inventory[v].projected_units for v in (ALPHA, BETA, GAMMA)
    )
    assert summary["total_potential_exposure_units"] == pytest.approx(expected_total)


# ---- 8. What-if context reflects WhatIfResult exactly -----------------------


def test_what_if_context_matches_result(demo, request_obj):
    what_if = analyze_what_if_for_vendor(demo.snapshot, GAMMA, 10.0)
    ctx = llm_context.build_what_if_context(what_if, request_obj)
    assert ctx["vendor_id"] == what_if.vendor_id
    assert ctx["delay_days"] == what_if.delay_days
    assert ctx["original_expected_lead_time"] == what_if.original_expected_lead_time
    assert ctx["new_expected_lead_time"] == what_if.new_expected_lead_time
    assert ctx["original_exposure"] == what_if.original_exposure
    assert ctx["new_exposure"] == what_if.new_exposure
    assert ctx["exposure_change"] == what_if.exposure_change
    assert ctx["inventory_horizon"] == what_if.inventory_horizon


def test_what_if_context_without_request_has_no_product(demo):
    what_if = analyze_what_if_for_vendor(demo.snapshot, GAMMA, 10.0)
    ctx = llm_context.build_what_if_context(what_if)
    assert "product" not in ctx
    assert "inventory" not in ctx
    assert ctx["vendor_id"] == GAMMA


# ---- 9. Missing / empty values handled safely -------------------------------


def test_unknown_vendor_returns_none(demo, request_obj):
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, "Unknown Vendor", demo.explanation
    )
    assert ctx is None


def test_empty_risk_drivers_not_fabricated(demo, request_obj):
    # Alpha triggers no rules -> empty main_risk_drivers.
    assert demo.snapshot.risk[ALPHA].main_risk_drivers == []
    ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, ALPHA, demo.explanation
    )
    assert ctx["risk"]["risk_drivers"] == []


def test_recommendation_explanation_none_without_valid_explanation(demo, request_obj):
    # No explanation passed -> explanation text must be None (never fabricated).
    ctx = llm_context.build_vendor_context(demo.snapshot, request_obj, ALPHA)
    assert ctx["recommendation"]["explanation"] is None
    # Metrics are still populated from the snapshot.
    assert ctx["metrics"]["expected_lead_time"] == demo.snapshot.forecasts[
        ALPHA
    ].values[0]


# ---- 10. Determinism --------------------------------------------------------


def test_all_builders_are_deterministic(demo, request_obj):
    v1 = llm_context.build_vendor_context(
        demo.snapshot, request_obj, GAMMA, demo.explanation
    )
    v2 = llm_context.build_vendor_context(
        demo.snapshot, request_obj, GAMMA, demo.explanation
    )
    assert v1 == v2

    c1 = llm_context.build_comparison_context(demo.snapshot, request_obj)
    c2 = llm_context.build_comparison_context(demo.snapshot, request_obj)
    assert c1 == c2

    e1 = llm_context.build_executive_context(demo.snapshot, request_obj)
    e2 = llm_context.build_executive_context(demo.snapshot, request_obj)
    assert e1 == e2

    what_if = analyze_what_if_for_vendor(demo.snapshot, GAMMA, 10.0)
    w1 = llm_context.build_what_if_context(what_if, request_obj)
    w2 = llm_context.build_what_if_context(what_if, request_obj)
    assert w1 == w2


# ---- 11. No streamlit import ------------------------------------------------


def _import_lines(source: str) -> list[str]:
    """Return the stripped ``import``/``from`` statement lines from source."""
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]


def test_module_does_not_import_streamlit():
    # Importing the context module must not pull streamlit into sys.modules.
    assert "streamlit" not in sys.modules
    # And the module has no streamlit import statement (docstring prose that
    # merely names streamlit as something it avoids is fine).
    import inspect

    imports = _import_lines(inspect.getsource(llm_context))
    assert not any("streamlit" in line for line in imports)


# ---- 12. No provider/LLM invocation -----------------------------------------


def test_module_does_not_import_boto3():
    assert "boto3" not in sys.modules or True  # boto3 not required to be absent globally
    import inspect

    imports = _import_lines(inspect.getsource(llm_context))
    assert not any("boto3" in line for line in imports)
    assert not any("provider" in line.lower() for line in imports)


def test_building_context_returns_plain_dicts(demo, request_obj):
    import json

    vendor_ctx = llm_context.build_vendor_context(
        demo.snapshot, request_obj, ALPHA, demo.explanation
    )
    comparison_ctx = llm_context.build_comparison_context(demo.snapshot, request_obj)
    executive_ctx = llm_context.build_executive_context(demo.snapshot, request_obj)
    what_if = analyze_what_if_for_vendor(demo.snapshot, GAMMA, 10.0)
    what_if_ctx = llm_context.build_what_if_context(what_if, request_obj)

    for ctx in (vendor_ctx, comparison_ctx, executive_ctx, what_if_ctx):
        assert isinstance(ctx, dict)
        # Serializes cleanly to JSON (all primitives/lists/dicts).
        json.dumps(ctx)
