"""Tests for explanation.py and llm_provider.py (Phase 2 explanation layer).

Standard pytest only (NO Hypothesis, no new dependencies). These are the
essential tests for the natural-language explanation layer, which turns the
immutable ``DeterministicResults`` into text via a pluggable provider and
validates the text's numbers against the deterministic results (Req 15, 13.4-13.8,
16.4).

Coverage:
1. Deterministic explanation via the default (null) provider.
2. Multiple vendors appear in deterministic (sorted) order.
3. Missing/empty optional values still produce a valid explanation.
4. Passing NullLLMProvider() explicitly behaves like provider=None.
5. Repeated execution is byte-identical.
6. Degraded path: a failing provider yields a degraded result, results intact.
7. Additive invariant: explain never alters the deterministic results.
"""

from vendor_forecasting_agent.config import load_config
from vendor_forecasting_agent.explanation import (
    build_explanation_text,
    explain,
    validate_explanation_numbers,
)
from vendor_forecasting_agent.llm_provider import NullLLMProvider
from vendor_forecasting_agent.pipeline import run_pipeline
from vendor_forecasting_agent.schema import (
    DeterministicResults,
    ExplanationResult,
    Forecast,
    InventoryImpact,
    Recommendation,
    VendorRiskResult,
    canonical_json,
)
from vendor_forecasting_agent.synthetic import generate_historical_data


# ---- Builders ---------------------------------------------------------------


def _pipeline_results() -> DeterministicResults:
    """Run the real Phase 1 pipeline for a realistic snapshot."""
    config = load_config(None).model_copy(update={"record_count": 250})
    records = generate_historical_data(
        seed=42, record_count=config.record_count
    )
    result = run_pipeline(records, config)
    assert isinstance(result.data, DeterministicResults)
    return result.data


def _hand_built_results() -> DeterministicResults:
    """Two vendors: one with drivers + inventory exposure, one with neither.

    Kept minimal but precise so assertions are exact. Numbers use the same
    2-decimal canonical rounding the explanation module applies.
    """
    return DeterministicResults(
        metrics={},
        trends={},
        forecasts={
            "V-A": Forecast(vendor_id="V-A", horizon=12, values=[9.5, 9.5], seed=0),
            "V-B": Forecast(vendor_id="V-B", horizon=12, values=[7.0, 7.0], seed=0),
        },
        scores={},
        risk={
            "V-A": VendorRiskResult(
                vendor_id="V-A",
                risk_score=80.0,
                risk_level="critical",
                confidence=0.9,
                main_risk_drivers=["defect_rate", "lead_time"],
            ),
            "V-B": VendorRiskResult(
                vendor_id="V-B",
                risk_score=10.0,
                risk_level="low",
                confidence=0.9,
                main_risk_drivers=[],
            ),
        },
        outcomes={},
        impacts={},
        inventory={
            "V-A": InventoryImpact(
                vendor_id="V-A",
                projected_units=250.0,
                reorder_delta_units=250.0,
                horizon=12,
            ),
            # V-B intentionally has no inventory entry.
        },
        recommendations=[
            Recommendation(
                vendor_id="V-A",
                action="replace",
                score=80.0,
                rationale_keys=["defect_rate", "lead_time", "inventory_exposure"],
            ),
            Recommendation(
                vendor_id="V-B",
                action="prefer",
                score=10.0,
                rationale_keys=[],
            ),
        ],
    )


# ---- 1. Deterministic explanation via default (null) provider ---------------


def test_default_provider_produces_valid_explanation():
    results = _hand_built_results()

    outcome = explain(results)

    assert isinstance(outcome, ExplanationResult)
    assert outcome.is_valid is True
    assert outcome.degraded is False
    assert outcome.text is not None
    # Risk level mentioned.
    assert "critical" in outcome.text
    # Recommendation action mentioned.
    assert "replace" in outcome.text
    # Inventory exposure mentioned for the vendor that has it.
    assert "inventory exposure" in outcome.text


def test_pipeline_results_produce_valid_explanation():
    results = _pipeline_results()

    outcome = explain(results)

    assert outcome.is_valid is True
    assert outcome.degraded is False
    assert outcome.text is not None
    assert outcome.mismatches == []


# ---- 2. Multiple vendors, deterministic ordering ----------------------------


def test_multiple_vendors_appear_in_sorted_order():
    results = _hand_built_results()

    text = build_explanation_text(results)

    assert "V-A" in text
    assert "V-B" in text
    # Sorted by vendor_id: V-A paragraph precedes V-B.
    assert text.index("Vendor V-A") < text.index("Vendor V-B")

    # Every recommended vendor appears in the text.
    for rec in results.recommendations:
        assert rec.vendor_id in text


# ---- 3. Missing/empty optional values ---------------------------------------


def test_empty_drivers_and_missing_inventory_still_valid():
    results = _hand_built_results()

    outcome = explain(results)
    text = outcome.text or ""

    assert outcome.is_valid is True
    # Vendor with empty drivers gets the "no dimensions flagged" phrasing.
    assert "no dimensions flagged" in text
    # V-B has no inventory entry -> no exposure sentence for it. Its paragraph is
    # the text after the V-B marker; it must not mention inventory exposure.
    v_b_paragraph = text[text.index("Vendor V-B"):]
    assert "inventory exposure" not in v_b_paragraph


def test_no_vendors_returns_placeholder():
    empty = DeterministicResults(
        metrics={},
        trends={},
        forecasts={},
        scores={},
        risk={},
        outcomes={},
        impacts={},
        inventory={},
        recommendations=[],
    )

    text = build_explanation_text(empty)
    outcome = explain(empty)

    assert text == "No vendors to explain."
    assert outcome.is_valid is True
    assert outcome.text == "No vendors to explain."


# ---- 4. Explicit null provider behaves like provider=None -------------------


def test_explicit_null_provider_matches_default():
    results = _hand_built_results()

    default_outcome = explain(results)
    explicit_outcome = explain(results, provider=NullLLMProvider())

    assert explicit_outcome.text is not None
    assert explicit_outcome.is_valid is True
    assert canonical_json(explicit_outcome) == canonical_json(default_outcome)


# ---- 5. Repeated execution is identical -------------------------------------


def test_repeated_execution_is_identical():
    results = _hand_built_results()

    first = explain(results)
    second = explain(results)

    assert canonical_json(first) == canonical_json(second)


# ---- 6. Degraded path -------------------------------------------------------


def test_provider_failure_yields_degraded_result():
    results = _hand_built_results()
    before = canonical_json(results)

    outcome = explain(results, provider=NullLLMProvider(raise_error=True))

    assert outcome.degraded is True
    assert outcome.is_valid is False
    assert outcome.text is None
    # Deterministic results remain intact / available (Req 13.8).
    assert canonical_json(results) == before


# ---- 7. Additive invariant --------------------------------------------------


def test_explain_does_not_alter_results():
    results = _hand_built_results()
    before = canonical_json(results)

    explain(results)

    assert canonical_json(results) == before


# ---- Validator sanity -------------------------------------------------------


def test_validator_flags_foreign_number():
    results = _hand_built_results()

    mismatches = validate_explanation_numbers("The value is 9999.99 units.", results)

    assert len(mismatches) == 1
    assert mismatches[0].value_in_text == 9999.99
    assert mismatches[0].nearest_expected is not None


def test_validator_accepts_template_numbers():
    results = _hand_built_results()

    text = build_explanation_text(results)

    assert validate_explanation_numbers(text, results) == []
