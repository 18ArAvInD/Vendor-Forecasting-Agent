"""Tests for the conversation-memory / what-if enrichment of ask_vendor_agent.

Focused on the NEW keyword-only enrichments (``history`` and
``what_if_context``) added to
:func:`vendor_forecasting_agent.decision_support.ask_vendor_agent`, the
recommendation-justification + comparison data that were ALREADY available via
the ask context (regression guards proving no new backend is needed), and the
guardrail wording. Standard pytest only — no boto3, no AWS, no network. Uses the
curated ACU / PMIC-450 scenario for real deterministic fixtures.
"""

from vendor_forecasting_agent.decision_support import ask_vendor_agent
from vendor_forecasting_agent.demo_scenarios import get_scenario, run_scenario_by_key
from vendor_forecasting_agent.llm_prompts import SYSTEM_PROMPT
from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.whatif_scenarios import (
    analyze_demand_increase,
    build_scenario_context,
)

_KEY = "acu-pmic450"
# Highest-risk vendor in the curated scenario (risk_score 100 -> "replace").
_HIGHEST_RISK_VENDOR = "Gamma Micro"


class FakeProvider:
    """Records the (user_prompt, system_prompt) and returns a canned reply."""

    def __init__(self, reply: str = "canned analysis") -> None:
        self.reply = reply
        self.last_user = None
        self.last_system = None

    def generate(self, user_prompt: str, system_prompt=None) -> str:
        self.last_user = user_prompt
        self.last_system = system_prompt
        return self.reply


def _fixture():
    output = run_scenario_by_key(_KEY)
    request = get_scenario(_KEY).request
    return output.snapshot, request


# ---- 1. Backward compatibility ----------------------------------------------


def test_ask_vendor_agent_old_signature_still_works():
    """Calling with no history / what_if_context is identical to before."""
    snapshot, request = _fixture()
    provider = FakeProvider("answer")
    resp = ask_vendor_agent(
        snapshot, request, "Why is it risky?", provider,
        vendor_id=_HIGHEST_RISK_VENDOR,
    )
    assert resp.success is True
    assert resp.text == "answer"
    assert provider.last_system == SYSTEM_PROMPT
    # selected_vendor + comparison context still embedded.
    assert "selected_vendor" in provider.last_user
    assert "comparison" in provider.last_user
    # No enrichment blocks when not supplied.
    assert "conversation_history" not in provider.last_user
    assert "latest_what_if" not in provider.last_user


# ---- 2. Conversation history pass-through -----------------------------------


def test_history_is_passed_through_to_prompt():
    snapshot, request = _fixture()
    provider = FakeProvider()
    history = [
        {"role": "user", "text": "Which vendor is riskiest?"},
        {"role": "assistant", "text": "Gamma Micro is highest risk"},
    ]
    ask_vendor_agent(
        snapshot, request, "Why?", provider,
        vendor_id=_HIGHEST_RISK_VENDOR, history=history,
    )
    assert "conversation_history" in provider.last_user
    assert "Gamma Micro is highest risk" in provider.last_user


# ---- 3. Latest what-if pass-through -----------------------------------------


def test_what_if_context_is_passed_through_to_prompt():
    snapshot, request = _fixture()
    result = analyze_demand_increase(snapshot, request, _HIGHEST_RISK_VENDOR, 30.0)
    context = build_scenario_context(result, request)
    provider = FakeProvider()
    ask_vendor_agent(
        snapshot, request, "Explain the latest What-If result", provider,
        vendor_id=_HIGHEST_RISK_VENDOR, what_if_context=context,
    )
    assert "latest_what_if" in provider.last_user
    # A calculated what-if value appears verbatim (not recomputed).
    assert str(result.whatif_potential_exposure) in provider.last_user


# ---- 4. Recommendation justification (reused existing context) --------------


def test_recommendation_and_drivers_present_for_justification():
    snapshot, request = _fixture()
    provider = FakeProvider()
    ask_vendor_agent(
        snapshot, request, "Why is the recommendation what it is?", provider,
        vendor_id=_HIGHEST_RISK_VENDOR,
    )
    # The deterministic recommended action + risk drivers are already in context,
    # so the AI can justify the recommendation with no new backend.
    assert "replace" in provider.last_user
    assert "on_time_delivery_low" in provider.last_user
    assert "defect_rate_high" in provider.last_user


# ---- 5. Vendor comparison (reused existing context) -------------------------


def test_comparison_data_present_for_all_vendors():
    snapshot, request = _fixture()
    provider = FakeProvider()
    ask_vendor_agent(
        snapshot, request, "Compare the top two vendors", provider,
        vendor_id=_HIGHEST_RISK_VENDOR,
    )
    user = provider.last_user
    assert "comparison" in user
    # Multiple vendors with their deterministic risk_score + recommended_action.
    assert "Alpha Semiconductors" in user
    assert "Beta Electronics" in user
    assert "Gamma Micro" in user
    assert "risk_score" in user
    assert "recommended_action" in user


# ---- 6. Guardrail wording ---------------------------------------------------


def test_guardrail_wording_preserved_and_extended():
    lowered = SYSTEM_PROMPT.lower()
    assert "authoritative" in lowered
    assert "deterministic" in lowered
    assert "never" in lowered
    # The appended conversation-history caveat is present.
    assert "conversation history" in lowered
    assert "resolve what the user is referring to" in lowered


# ---- 7. Deterministic values unchanged --------------------------------------


def test_enrichment_does_not_mutate_deterministic_values():
    snapshot, request = _fixture()
    before = canonical_json(snapshot)
    result = analyze_demand_increase(snapshot, request, _HIGHEST_RISK_VENDOR, 30.0)
    context = build_scenario_context(result, request)
    history = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "ok"}]
    provider = FakeProvider()
    ask_vendor_agent(
        snapshot, request, "q?", provider,
        vendor_id=_HIGHEST_RISK_VENDOR, history=history, what_if_context=context,
    )
    # Snapshot unchanged; ScenarioResult is frozen (pydantic frozen model).
    assert canonical_json(snapshot) == before
    import pytest
    with pytest.raises(Exception):
        result.whatif_potential_exposure = 0.0
