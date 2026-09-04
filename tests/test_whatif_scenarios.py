"""Tests for the deterministic multi-scenario what-if engine (whatif_scenarios).

Standard pytest only — NO Hypothesis, no boto3, no AWS, no network. Uses the
curated ACU / PMIC-450 scenario for real deterministic fixtures.
"""

import json
import math

import pytest

from vendor_forecasting_agent.decision_support import (
    FALLBACK_TEXT,
    explain_scenario,
)
from vendor_forecasting_agent.demo_scenarios import get_scenario, run_scenario_by_key
from vendor_forecasting_agent.llm_prompts import SYSTEM_PROMPT
from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.ui import view_model
from vendor_forecasting_agent.whatif import analyze_what_if_for_vendor
from vendor_forecasting_agent.whatif_scenarios import (
    ScenarioResult,
    analyze_demand_increase,
    analyze_fulfillment_reduction,
    analyze_supplier_delay,
    build_scenario_context,
)

_KEY = "acu-pmic450"


class FakeProvider:
    """Records the (user_prompt, system_prompt) and returns a canned reply."""

    def __init__(self, reply: str = "canned scenario explanation") -> None:
        self.reply = reply
        self.last_user = None
        self.last_system = None

    def generate(self, user_prompt: str, system_prompt=None) -> str:
        self.last_user = user_prompt
        self.last_system = system_prompt
        return self.reply


class FailingProvider:
    """Always raises to exercise the graceful-fallback path."""

    def generate(self, user_prompt: str, system_prompt=None) -> str:
        raise RuntimeError("boom")


class NullLLMProvider:
    """Offline provider that echoes the raw prompt (mirrors the real null provider)."""

    def generate(self, user_prompt: str, system_prompt=None) -> str:
        return user_prompt


def _fixture():
    output = run_scenario_by_key(_KEY)
    request = get_scenario(_KEY).request
    vendor = view_model.vendor_ids(output.snapshot)[0]
    return output.snapshot, request, vendor


# ---- 1. Supplier delay: byte-identical reuse of the delay math --------------


def test_supplier_delay_increases_lead_time_and_matches_whatif_module():
    snapshot, request, vendor = _fixture()
    delay = 10.0

    result = analyze_supplier_delay(snapshot, request, vendor, delay)
    reference = analyze_what_if_for_vendor(snapshot, vendor, delay)

    # Lead time increases by exactly the delay.
    assert result.whatif_expected_lead_time == result.baseline_expected_lead_time + delay
    assert result.whatif_expected_lead_time > result.baseline_expected_lead_time
    # Exposure matches the existing module byte-for-byte (reuse, no re-derivation).
    assert result.baseline_potential_exposure == reference.original_exposure
    assert result.whatif_potential_exposure == reference.new_exposure
    assert result.exposure_change == reference.exposure_change
    assert result.exposure_change >= 0.0
    assert result.scenario == "supplier_delay"
    assert result.parameter_value == delay


# ---- 2. Demand increase: exposure scales, coverage shrinks ------------------


def test_demand_increase_scales_exposure_and_shrinks_coverage():
    snapshot, request, vendor = _fixture()

    result = analyze_demand_increase(snapshot, request, vendor, 20.0)

    # Whatif exposure is 1.2x the baseline exposure basis (when there is exposure).
    assert result.whatif_daily_demand == pytest.approx(
        result.baseline_daily_demand * 1.2
    )
    assert result.whatif_potential_exposure == pytest.approx(
        result.baseline_potential_exposure * 1.2
    )
    # Coverage shrinks under higher demand when demand > 0.
    assert result.baseline_daily_demand > 0
    assert result.whatif_inventory_coverage_days is not None
    assert result.baseline_inventory_coverage_days is not None
    assert result.whatif_inventory_coverage_days < result.baseline_inventory_coverage_days
    assert result.coverage_change_days is not None
    assert result.coverage_change_days < 0.0
    assert result.scenario == "demand_increase"


# ---- 3. Fulfillment reduction: replenishment + shortfall --------------------


def test_fulfillment_reduction_replenishment_and_shortage():
    snapshot, request, vendor = _fixture()
    required = float(request.required_quantity)

    result = analyze_fulfillment_reduction(snapshot, request, vendor, 80.0)

    assert result.whatif_expected_replenishment_units == pytest.approx(required * 0.8)
    assert result.shortage_gap_units == pytest.approx(required * 0.2)
    assert result.shortage_gap_units > 0.0
    # Potential exposure grows by the undelivered committed quantity.
    assert result.exposure_change == pytest.approx(required * 0.2)
    assert result.whatif_potential_exposure == pytest.approx(
        result.baseline_potential_exposure + required * 0.2
    )
    assert result.scenario == "fulfillment_reduction"


# ---- 4. Baseline / inputs never mutated -------------------------------------


def test_scenarios_do_not_mutate_inputs():
    snapshot, request, vendor = _fixture()
    snapshot_before = canonical_json(snapshot)
    request_before = request.model_dump()

    analyze_supplier_delay(snapshot, request, vendor, 7.0)
    analyze_demand_increase(snapshot, request, vendor, 30.0)
    analyze_fulfillment_reduction(snapshot, request, vendor, 60.0)

    assert canonical_json(snapshot) == snapshot_before
    assert request.model_dump() == request_before


# ---- 5. Scenario independence / no compounding ------------------------------


def test_scenarios_are_independent_no_compounding():
    snapshot, request, vendor = _fixture()

    # Baseline captured from a demand-increase run with no prior scenario.
    fresh = analyze_demand_increase(snapshot, request, vendor, 20.0)

    # Run a supplier delay first, then demand increase again.
    analyze_supplier_delay(snapshot, request, vendor, 25.0)
    after_delay = analyze_demand_increase(snapshot, request, vendor, 20.0)

    # The demand-increase baseline is read fresh and is unaffected by the delay.
    assert after_delay.baseline_expected_lead_time == fresh.baseline_expected_lead_time
    assert after_delay.baseline_daily_demand == fresh.baseline_daily_demand
    assert after_delay.baseline_potential_exposure == fresh.baseline_potential_exposure
    assert after_delay.whatif_potential_exposure == fresh.whatif_potential_exposure


# ---- 6. Invalid parameters raise ValueError ---------------------------------


def test_invalid_parameters_raise_value_error():
    snapshot, request, vendor = _fixture()

    with pytest.raises(ValueError):
        analyze_supplier_delay(snapshot, request, vendor, -1.0)
    with pytest.raises(ValueError):
        analyze_demand_increase(snapshot, request, vendor, -5.0)
    with pytest.raises(ValueError):
        analyze_fulfillment_reduction(snapshot, request, vendor, -1.0)
    with pytest.raises(ValueError):
        analyze_fulfillment_reduction(snapshot, request, vendor, 150.0)

    # bool and non-number are rejected everywhere.
    with pytest.raises(ValueError):
        analyze_supplier_delay(snapshot, request, vendor, True)
    with pytest.raises(ValueError):
        analyze_demand_increase(snapshot, request, vendor, "20")
    with pytest.raises(ValueError):
        analyze_fulfillment_reduction(snapshot, request, vendor, False if False else "80")


# ---- 7. build_scenario_context: facts + JSON serializable -------------------


def test_build_scenario_context_contains_values_and_is_json_serializable():
    snapshot, request, vendor = _fixture()
    result = analyze_demand_increase(snapshot, request, vendor, 20.0)

    context = build_scenario_context(result, request)

    assert context["scenario"] == "demand_increase"
    assert context["vendor_id"] == vendor
    # Baseline + whatif + change values are present and equal the result verbatim.
    assert context["baseline"]["potential_exposure"] == result.baseline_potential_exposure
    assert context["whatif"]["potential_exposure"] == result.whatif_potential_exposure
    assert context["changes"]["exposure_change"] == result.exposure_change
    assert context["parameter"]["value"] == result.parameter_value
    # Fully JSON-serializable.
    encoded = json.dumps(context, sort_keys=True)
    assert "demand_increase" in encoded


# ---- 8. explain_scenario: embeds calculated values; failure fallback --------


def test_explain_scenario_embeds_calculated_values():
    snapshot, request, vendor = _fixture()
    result = analyze_demand_increase(snapshot, request, vendor, 20.0)
    provider = FakeProvider("here is the explanation")

    resp = explain_scenario(result, provider, request=request)

    assert resp.success is True
    assert resp.text == "here is the explanation"
    assert provider.last_system == SYSTEM_PROMPT
    # The prompt embeds the CALCULATED whatif exposure (not just a raw instruction).
    assert str(result.whatif_potential_exposure) in provider.last_user
    assert str(result.exposure_change) in provider.last_user
    # It is not merely a raw "increase demand by 20%" instruction as sole content.
    assert "ANALYSIS CONTEXT" in provider.last_user


def test_explain_scenario_failure_and_none_provider_fallback():
    snapshot, request, vendor = _fixture()
    result = analyze_supplier_delay(snapshot, request, vendor, 10.0)

    failing = explain_scenario(result, FailingProvider(), request=request)
    assert failing.success is False
    assert failing.text == FALLBACK_TEXT
    assert failing.error is not None

    none_resp = explain_scenario(result, None, request=request)
    assert none_resp.success is False
    assert none_resp.text == FALLBACK_TEXT
    assert none_resp.provider == "none"


# ---- 9. Offline null provider: graceful, deterministic numbers intact -------


def test_explain_scenario_offline_null_provider_non_crashing():
    snapshot, request, vendor = _fixture()
    result = analyze_fulfillment_reduction(snapshot, request, vendor, 80.0)

    # The offline null provider echoes the prompt; _run treats non-empty echo as
    # success (the app's raw-context guard hides it in the UI). The key guarantee
    # here is that it never raises and the deterministic numbers are unaffected.
    resp = explain_scenario(result, NullLLMProvider(), request=request)
    assert isinstance(resp.text, str)
    assert result.whatif_expected_replenishment_units == pytest.approx(
        float(request.required_quantity) * 0.8
    )

    # ScenarioResult is frozen / immutable.
    assert isinstance(result, ScenarioResult)
    with pytest.raises(Exception):
        result.whatif_potential_exposure = 0.0  # type: ignore[misc]


# ---- extra determinism guard -----------------------------------------------


def test_identical_inputs_produce_identical_results():
    snapshot, request, vendor = _fixture()
    a = analyze_demand_increase(snapshot, request, vendor, 15.0)
    b = analyze_demand_increase(snapshot, request, vendor, 15.0)
    assert a == b
    assert not math.isnan(a.whatif_potential_exposure)



# ---- 10. Audit traceability: baseline + scenario-specific whatif fields -----


def test_scenario_result_exposes_baseline_and_whatif_audit_fields():
    """The AI Audit renders baseline + what-if straight from ScenarioResult, so
    those authoritative fields must be present/populated for every scenario
    (only scenario-appropriate optional fields differ)."""
    snapshot, request, vendor = _fixture()

    delay = analyze_supplier_delay(snapshot, request, vendor, 10.0)
    demand = analyze_demand_increase(snapshot, request, vendor, 30.0)
    fulfil = analyze_fulfillment_reduction(snapshot, request, vendor, 80.0)

    for r in (delay, demand, fulfil):
        # Baseline block (always present for the audit).
        assert r.baseline_daily_demand is not None
        assert r.baseline_current_inventory is not None
        assert r.baseline_expected_lead_time is not None
        assert r.baseline_potential_exposure is not None
        # What-If block (always present for the audit).
        assert r.whatif_daily_demand is not None
        assert r.whatif_expected_lead_time is not None
        assert r.whatif_potential_exposure is not None
        # exposure_change is the authoritative delta the audit shows.
        assert r.exposure_change == (
            r.whatif_potential_exposure - r.baseline_potential_exposure
        )

    # Demand increase shows a coverage delta + shortage gap in the audit.
    assert demand.whatif_inventory_coverage_days is not None
    assert demand.shortage_gap_units is not None
    # Fulfillment reduction shows expected replenishment + shortage gap.
    assert fulfil.whatif_expected_replenishment_units is not None
    assert fulfil.shortage_gap_units is not None
    # Supplier delay shows the lead-time change in the audit.
    assert delay.whatif_expected_lead_time > delay.baseline_expected_lead_time


def test_build_scenario_context_carries_baseline_and_whatif_for_all_scenarios():
    """The LLM context (and thus the audit's source) carries baseline + what-if +
    changes for each of the three scenarios."""
    snapshot, request, vendor = _fixture()
    for result in (
        analyze_supplier_delay(snapshot, request, vendor, 10.0),
        analyze_demand_increase(snapshot, request, vendor, 30.0),
        analyze_fulfillment_reduction(snapshot, request, vendor, 80.0),
    ):
        ctx = build_scenario_context(result, request)
        assert "baseline" in ctx and "whatif" in ctx and "changes" in ctx
        assert ctx["baseline"]["potential_exposure"] == result.baseline_potential_exposure
        assert ctx["whatif"]["potential_exposure"] == result.whatif_potential_exposure
        assert ctx["changes"]["exposure_change"] == result.exposure_change
