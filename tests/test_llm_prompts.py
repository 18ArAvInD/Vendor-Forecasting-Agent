"""Tests for the Layer 3 prompt builders in ``llm_prompts.py``.

Standard pytest only. No provider, no boto3, no network — prompts are pure
strings built from context dicts.
"""

from vendor_forecasting_agent.demo_scenarios import get_scenario, run_scenario_by_key
from vendor_forecasting_agent.llm_context import (
    build_comparison_context,
    build_executive_context,
    build_vendor_context,
    build_what_if_context,
)
from vendor_forecasting_agent.llm_prompts import (
    SYSTEM_PROMPT,
    build_ask_prompt,
    build_comparison_prompt,
    build_executive_prompt,
    build_what_if_prompt,
)
from vendor_forecasting_agent.whatif import analyze_what_if_for_vendor

_KEY = "acu-pmic450"


def _fixture():
    output = run_scenario_by_key(_KEY)
    request = get_scenario(_KEY).request
    return output.snapshot, request


def test_system_prompt_states_deterministic_source_of_truth():
    lowered = SYSTEM_PROMPT.lower()
    assert "authoritative" in lowered
    assert "never" in lowered
    assert "deterministic" in lowered


def test_ask_prompt_embeds_context_and_question():
    snapshot, request = _fixture()
    context = build_comparison_context(snapshot, request)
    combined = {"comparison": context}
    prompt = build_ask_prompt(combined, "Which vendor is riskiest?")
    assert "Gamma Micro" in prompt
    assert "Which vendor is riskiest?" in prompt
    assert "ANALYSIS CONTEXT" in prompt


def test_comparison_prompt_embeds_vendors():
    snapshot, request = _fixture()
    context = build_comparison_context(snapshot, request)
    prompt = build_comparison_prompt(context)
    assert "Gamma Micro" in prompt
    assert "Alpha Semiconductors" in prompt


def test_executive_prompt_embeds_known_values():
    snapshot, request = _fixture()
    context = build_executive_context(snapshot, request)
    prompt = build_executive_prompt(context)
    assert "Gamma Micro" in prompt
    # Gamma's critical risk_score of 100 appears in the exec context JSON.
    assert "100" in prompt
    assert "critical" in prompt


def test_what_if_prompt_embeds_numbers():
    snapshot, request = _fixture()
    what_if = analyze_what_if_for_vendor(snapshot, "Gamma Micro", 10.0)
    context = build_what_if_context(what_if, request)
    prompt = build_what_if_prompt(context)
    assert "new_exposure" in prompt
    assert "delay_days" in prompt


def test_prompts_contain_no_provider_specifics():
    snapshot, request = _fixture()
    prompts = [
        build_ask_prompt({"c": build_comparison_context(snapshot, request)}, "q?"),
        build_comparison_prompt(build_comparison_context(snapshot, request)),
        build_executive_prompt(build_executive_context(snapshot, request)),
        build_what_if_prompt(
            build_what_if_context(
                analyze_what_if_for_vendor(snapshot, "Gamma Micro", 5.0), request
            )
        ),
    ]
    for prompt in prompts:
        lowered = prompt.lower()
        assert "boto3" not in lowered
        assert "converse" not in lowered
        assert "modelid" not in lowered



# ---- lead-time terminology consistency + interpretive executive prompt ------


def test_system_prompt_prefers_deteriorating_terminology():
    from vendor_forecasting_agent.llm_prompts import SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    # Preferred wording is present.
    assert "deteriorating" in low
    assert "lead time is increasing" in low
    # The ambiguous phrasing is explicitly discouraged.
    assert "time declining" in low  # appears inside the "Do NOT say ..." guidance
    # Existing guardrail wording is still intact (not weakened).
    assert "authoritative" in low
    assert "deterministic" in low


def test_executive_prompt_is_interpretive_not_restatement():
    from vendor_forecasting_agent.llm_prompts import build_executive_prompt
    prompt = build_executive_prompt({"summary": {"vendors_analyzed": 3}})
    assert "INTERPRETIVE" in prompt
    assert "do not simply" in prompt.lower()
    # Still forbids inventing/recomputing numbers and changing recommendations.
    assert "recompute" in prompt.lower()
    assert "recommendation" in prompt.lower()
