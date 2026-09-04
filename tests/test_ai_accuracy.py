"""Regression tests for AI accuracy / context-consistency fixes.

Standard pytest only — no live Mistral, no network, no boto3. These assert on the
STRUCTURED CONTEXT and PROMPTS supplied to the model (not on any generated
wording), which is where the authoritative values and priority instructions live.
"""

import json

import pytest

from vendor_forecasting_agent.decision_support import ask_vendor_agent
from vendor_forecasting_agent.demo_scenarios import get_scenario, run_scenario_by_key
from vendor_forecasting_agent.llm_context import build_executive_context
from vendor_forecasting_agent.llm_prompts import (
    build_ask_prompt,
    build_executive_prompt,
)
from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.ui import view_model
from vendor_forecasting_agent.whatif_scenarios import (
    analyze_supplier_delay,
    build_scenario_context,
)

_KEY = "acu-pmic450"


class _FakeProvider:
    """Records the last (user_prompt, system_prompt) and returns canned text."""

    def __init__(self, reply="ok"):
        self.reply = reply
        self.last_user = None
        self.last_system = None

    def generate(self, user_prompt, system_prompt=None):
        self.last_user = user_prompt
        self.last_system = system_prompt
        return self.reply


def _fixture():
    output = run_scenario_by_key(_KEY)
    request = get_scenario(_KEY).request
    ids = view_model.vendor_ids(output.snapshot)
    return output.snapshot, request, ids


# ---- Test 1: executive summary cannot contradict deterministic exposure ------


def test_executive_context_and_prompt_carry_authoritative_exposure():
    snapshot, request, ids = _fixture()
    ctx = build_executive_context(snapshot, request)

    # The total and each per-vendor exposure are present and internally consistent.
    total = ctx["summary"]["total_potential_exposure_units"]
    per_vendor = [v["potential_exposure_units"] for v in ctx["vendors"]]
    assert total == pytest.approx(sum(x for x in per_vendor if x is not None))

    prompt = build_executive_prompt(ctx)
    # The prompt embeds the authoritative context (so the exact values are supplied)
    # and explicitly forbids substituting one vendor's value for another.
    assert "ANALYSIS CONTEXT" in prompt
    assert "EXACTLY" in prompt
    assert "total_potential_exposure_units" in prompt
    assert "potential_exposure_units" in prompt
    # The authoritative total appears in the serialized context block.
    assert json.dumps(total) in prompt or str(total) in prompt


# ---- Test 2: latest what-if context is prioritized ---------------------------


def test_ask_prompt_prioritizes_latest_what_if_context():
    snapshot, request, ids = _fixture()
    vendor = ids[0]
    result = analyze_supplier_delay(snapshot, request, vendor, 10.0)
    wi_ctx = build_scenario_context(result, request)

    provider = _FakeProvider()
    ask_vendor_agent(
        snapshot, request, "Explain the latest What-If result", provider,
        vendor_id=vendor, what_if_context=wi_ctx,
    )
    prompt = provider.last_user
    # The what-if block is present and flagged as primary for scenario questions.
    assert "latest_what_if" in prompt
    assert "PRIMARY" in prompt
    # The what-if numbers are in the prompt (baseline + what-if exposure).
    assert str(result.whatif_potential_exposure) in prompt
    assert str(result.baseline_potential_exposure) in prompt


# ---- Test 3: follow-up keeps latest what-if available ------------------------


def test_followup_still_carries_latest_what_if():
    snapshot, request, ids = _fixture()
    vendor = ids[0]
    result = analyze_supplier_delay(snapshot, request, vendor, 10.0)
    wi_ctx = build_scenario_context(result, request)

    provider = _FakeProvider()
    # A follow-up like "Is that serious?" with the same latest what-if context.
    ask_vendor_agent(
        snapshot, request, "Is that serious?", provider,
        vendor_id=vendor,
        history=[
            {"role": "user", "text": "Explain the latest What-If result"},
            {"role": "assistant", "text": "The delay raises exposure."},
        ],
        what_if_context=wi_ctx,
    )
    prompt = provider.last_user
    assert "latest_what_if" in prompt
    assert "PRIMARY" in prompt
    assert "conversation_history" in prompt


# ---- Test 4: no baseline/scenario semantic confusion -------------------------


def test_ask_prompt_distinguishes_baseline_from_scenario():
    snapshot, request, ids = _fixture()
    vendor = ids[0]
    result = analyze_supplier_delay(snapshot, request, vendor, 10.0)
    wi_ctx = build_scenario_context(result, request)

    provider = _FakeProvider()
    ask_vendor_agent(
        snapshot, request, "What changed?", provider,
        vendor_id=vendor, what_if_context=wi_ctx,
    )
    prompt = provider.last_user.lower()
    # The guidance tells the model to separate baseline risk from scenario impact
    # and not to claim "no risk" when the scenario raises exposure.
    assert "baseline" in prompt and "scenario impact" in prompt
    assert "no risk" in prompt  # appears in the "do NOT say ... no risk ..." rule
    # Potential exposure vs shortage/gap distinction is reinforced.
    assert "planning signal" in prompt
    assert "shortage/gap" in prompt


# ---- Test 5: cross-vendor isolation ------------------------------------------


def test_scenario_context_is_single_vendor_isolated():
    snapshot, request, ids = _fixture()
    assert len(ids) >= 2
    vendor_a, vendor_b = ids[0], ids[1]

    ctx_a = build_scenario_context(
        analyze_supplier_delay(snapshot, request, vendor_a, 10.0), request
    )
    ctx_b = build_scenario_context(
        analyze_supplier_delay(snapshot, request, vendor_b, 10.0), request
    )
    # Each scenario context names exactly its own vendor.
    assert ctx_a["vendor_id"] == vendor_a
    assert ctx_b["vendor_id"] == vendor_b
    assert ctx_a["vendor_id"] != ctx_b["vendor_id"]
    # No other vendor id leaks into a scenario's serialized context.
    encoded_a = json.dumps(ctx_a)
    assert vendor_b not in encoded_a


# ---- Determinism guard: building context/prompts mutates nothing -------------


def test_context_building_does_not_mutate_snapshot():
    snapshot, request, ids = _fixture()
    before = canonical_json(snapshot)
    build_executive_context(snapshot, request)
    result = analyze_supplier_delay(snapshot, request, ids[0], 10.0)
    build_scenario_context(result, request)
    build_ask_prompt({"latest_what_if": {"scenario": "x"}}, "is that serious?")
    assert canonical_json(snapshot) == before



# ---- Test A: exposure/gap distinction (even when numerically equal) ----------


def test_prompt_keeps_exposure_and_gap_as_distinct_concepts():
    # Base ask guidance always separates the two concepts and forbids equating
    # them on equal values.
    ask = build_ask_prompt({}, "Explain the exposure and shortage.")
    low = ask.lower()
    assert "distinct concepts" in low
    assert "never say one is the other" in low
    assert "happen to have the same value" in low

    # The what-if explanation prompt reinforces the same separation.
    from vendor_forecasting_agent.llm_prompts import build_what_if_prompt
    wi = build_what_if_prompt({"scenario": "supplier_delay"}).lower()
    assert "distinct concepts" in wi
    assert "happen to have the same value" in wi
    # And still forbids recalculation (existing guardrail intact).
    assert "do not recalculate" in wi


# ---- Test B: next-step guidance = recommendation -> validation -> approval ---


def test_next_step_guidance_requires_human_approval_not_autonomous():
    ask = build_ask_prompt({}, "What should I do next?")
    low = ask.lower()
    # Starts from the deterministic recommendation and gives a planner next step.
    assert "deterministic recommendation" in low
    assert "next step" in low
    assert "planner" in low
    # Requires human review/approval.
    assert "human review/approval" in low
    # Never instructs autonomous execution.
    assert "automatically execute" in low  # appears in the "Never ... execute" rule
    assert "autonomous action" in low


# ---- Test C: latest What-If still prioritized (regression) -------------------


def test_latest_what_if_still_prioritized_after_wording_change():
    snapshot, request, ids = _fixture()
    vendor = ids[0]
    result = analyze_supplier_delay(snapshot, request, vendor, 10.0)
    wi_ctx = build_scenario_context(result, request)

    provider = _FakeProvider()
    ask_vendor_agent(
        snapshot, request, "Explain the latest What-If result", provider,
        vendor_id=vendor, what_if_context=wi_ctx,
    )
    prompt = provider.last_user
    assert "latest_what_if" in prompt
    assert "PRIMARY" in prompt
    assert str(result.whatif_potential_exposure) in prompt
