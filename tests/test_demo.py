"""Essential end-to-end tests for demo.py (local console demo).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
the thin demo orchestrator that wires config -> synthetic -> pipeline ->
explanation and formats a readable console report:

1. The demo runs end-to-end and returns a deterministic snapshot plus a valid
   offline explanation.
2. The formatted report lists the analyzed vendors and reflects the count.
3. Repeated execution is deterministic (identical report string + canonical
   snapshot JSON).
4. Empty input records do not crash; the report renders with zeroed summary.
5. Summary counts (elevated risk / inventory exposure) match manual counts from
   the snapshot.
"""

from vendor_forecasting_agent.config import load_config
from vendor_forecasting_agent.demo import format_report, run_demo
from vendor_forecasting_agent.schema import (
    DeterministicResults,
    ExplanationResult,
    canonical_json,
)
from vendor_forecasting_agent.synthetic import DEFAULT_VENDOR_IDS


# ---- 1. End-to-end execution ------------------------------------------------


def test_run_demo_executes_end_to_end():
    output = run_demo()  # default config, offline explanation path

    assert isinstance(output.snapshot, DeterministicResults)
    assert isinstance(output.explanation, ExplanationResult)
    # Offline NullLLMProvider path yields a valid, non-None explanation.
    assert output.explanation.is_valid is True
    assert output.explanation.text is not None
    # A real multi-vendor run produces recommendations.
    assert len(output.snapshot.recommendations) > 0


# ---- 2. Multiple vendors displayed ------------------------------------------


def test_format_report_lists_analyzed_vendors():
    output = run_demo()
    report = format_report(output.snapshot, output.explanation)

    recommended_vendor_ids = {
        rec.vendor_id for rec in output.snapshot.recommendations
    }

    # Every default vendor that made it into recommendations appears in the
    # report string.
    for vendor_id in DEFAULT_VENDOR_IDS:
        if vendor_id in recommended_vendor_ids:
            assert vendor_id in report

    # The "vendors analyzed" summary reflects the recommendation count.
    assert (
        f"Vendors analyzed              : {len(recommended_vendor_ids)}"
        in report
    )


# ---- 3. Deterministic repeated execution ------------------------------------


def test_run_demo_is_deterministic():
    config = load_config(None)

    first = run_demo(config=config)
    second = run_demo(config=config)

    report_first = format_report(first.snapshot, first.explanation)
    report_second = format_report(second.snapshot, second.explanation)

    assert report_first == report_second
    assert canonical_json(first.snapshot) == canonical_json(second.snapshot)


# ---- 4. Empty input edge case does not crash --------------------------------


def test_run_demo_with_empty_records_does_not_crash():
    config = load_config(None)

    output = run_demo(config=config, records=[])

    assert isinstance(output.snapshot, DeterministicResults)
    assert output.snapshot.recommendations == []
    # explain() returns the "no vendors" template for an empty snapshot.
    assert output.explanation.text == "No vendors to explain."

    report = format_report(output.snapshot, output.explanation)
    assert isinstance(report, str)
    assert "Vendors analyzed              : 0" in report
    assert "High/critical-risk vendors    : 0" in report
    assert "Vendors with inventory exposure: 0" in report


# ---- 5. Summary counts correctness ------------------------------------------


def test_format_report_summary_counts_match_snapshot():
    output = run_demo()
    snapshot = output.snapshot
    report = format_report(snapshot, output.explanation)

    expected_analyzed = len(snapshot.recommendations)
    expected_elevated = sum(
        1
        for risk in snapshot.risk.values()
        if risk.risk_level in {"high", "critical"}
    )
    expected_exposure = sum(
        1
        for inventory in snapshot.inventory.values()
        if inventory.projected_units > 0
    )

    assert f"Vendors analyzed              : {expected_analyzed}" in report
    assert f"High/critical-risk vendors    : {expected_elevated}" in report
    assert (
        f"Vendors with inventory exposure: {expected_exposure}" in report
    )
