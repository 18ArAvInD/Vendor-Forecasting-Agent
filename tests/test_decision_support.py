"""Tests for the Layer 3 orchestration in ``decision_support.py``.

Standard pytest only. Providers are faked/injected — no boto3, no AWS, no
network. Uses the curated ACU / PMIC-450 scenario for real deterministic
fixtures.
"""

from vendor_forecasting_agent.decision_support import (
    FALLBACK_TEXT,
    ask_vendor_agent,
    compare_vendors,
    explain_what_if,
    generate_executive_summary,
)
from vendor_forecasting_agent.demo_scenarios import get_scenario, run_scenario_by_key
from vendor_forecasting_agent.llm_prompts import SYSTEM_PROMPT
from vendor_forecasting_agent.schema import canonical_json
from vendor_forecasting_agent.whatif import analyze_what_if_for_vendor

_KEY = "acu-pmic450"


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


class FailingProvider:
    """Always raises to exercise the graceful-fallback path."""

    def generate(self, user_prompt: str, system_prompt=None) -> str:
        raise RuntimeError("boom")


def _fixture():
    output = run_scenario_by_key(_KEY)
    request = get_scenario(_KEY).request
    return output.snapshot, request


def test_ask_vendor_agent_builds_context_and_calls_provider():
    snapshot, request = _fixture()
    provider = FakeProvider("here is the answer")
    resp = ask_vendor_agent(
        snapshot, request, "Why is Gamma risky?", provider, vendor_id="Gamma Micro"
    )
    assert resp.success is True
    assert resp.text == "here is the answer"
    assert provider.last_system == SYSTEM_PROMPT
    assert "Why is Gamma risky?" in provider.last_user
    assert "Gamma Micro" in provider.last_user


def test_compare_vendors_uses_comparison_context():
    snapshot, request = _fixture()
    provider = FakeProvider()
    resp = compare_vendors(snapshot, request, provider)
    assert resp.success is True
    assert "Gamma Micro" in provider.last_user
    assert "Alpha Semiconductors" in provider.last_user


def test_generate_executive_summary_uses_executive_context():
    snapshot, request = _fixture()
    provider = FakeProvider()
    resp = generate_executive_summary(snapshot, request, provider)
    assert resp.success is True
    assert "vendors_analyzed" in provider.last_user
    assert "100" in provider.last_user
    assert "critical" in provider.last_user


def test_explain_what_if_does_not_recompute():
    snapshot, request = _fixture()
    what_if = analyze_what_if_for_vendor(snapshot, "Gamma Micro", 10.0)
    provider = FakeProvider()
    resp = explain_what_if(what_if, provider, request=request)
    assert resp.success is True
    # The what-if numbers in the prompt equal the Layer-1 result (no recompute).
    assert str(what_if.new_exposure) in provider.last_user
    assert str(what_if.exposure_change) in provider.last_user


def test_provider_failure_returns_graceful_fallback():
    snapshot, request = _fixture()
    what_if = analyze_what_if_for_vendor(snapshot, "Gamma Micro", 10.0)
    failing = FailingProvider()

    for resp in (
        ask_vendor_agent(snapshot, request, "q?", failing, vendor_id="Gamma Micro"),
        compare_vendors(snapshot, request, failing),
        generate_executive_summary(snapshot, request, failing),
        explain_what_if(what_if, failing, request=request),
    ):
        assert resp.success is False
        assert resp.text == FALLBACK_TEXT
        assert resp.error is not None


def test_none_provider_returns_fallback():
    snapshot, request = _fixture()
    resp = compare_vendors(snapshot, request, None)
    assert resp.success is False
    assert resp.text == FALLBACK_TEXT
    assert resp.provider == "none"


def test_deterministic_values_not_modified():
    snapshot, request = _fixture()
    before = canonical_json(snapshot)
    provider = FakeProvider()
    what_if = analyze_what_if_for_vendor(snapshot, "Gamma Micro", 10.0)

    ask_vendor_agent(snapshot, request, "q?", provider, vendor_id="Gamma Micro")
    compare_vendors(snapshot, request, provider)
    generate_executive_summary(snapshot, request, provider)
    explain_what_if(what_if, provider, request=request)

    assert canonical_json(snapshot) == before



# ---- get_default_provider selection -----------------------------------------


def test_get_default_provider_prefers_mistral_api_when_key_set():
    from vendor_forecasting_agent.decision_support import get_default_provider
    from vendor_forecasting_agent.llm_provider import load_llm_config

    cfg = load_llm_config(
        {"MISTRAL_API_KEY": "test-key-not-real", "BEDROCK_MODEL_ID": "mistral.some"}
    )
    provider = get_default_provider(cfg)
    assert type(provider).__name__ == "MistralAPIProvider"


def test_get_default_provider_falls_back_to_bedrock_when_only_model_id_set():
    from vendor_forecasting_agent.decision_support import get_default_provider
    from vendor_forecasting_agent.llm_provider import load_llm_config

    cfg = load_llm_config({"BEDROCK_MODEL_ID": "mistral.some"})
    provider = get_default_provider(cfg)
    assert type(provider).__name__ == "MistralBedrockProvider"


def test_get_default_provider_offline_when_nothing_configured():
    from vendor_forecasting_agent.decision_support import get_default_provider
    from vendor_forecasting_agent.llm_provider import load_llm_config

    cfg = load_llm_config({})
    provider = get_default_provider(cfg)
    assert type(provider).__name__ == "NullLLMProvider"
