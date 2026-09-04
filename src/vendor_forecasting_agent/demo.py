"""Minimal local end-to-end demo for the Vendor Forecasting Agent (demo.py).

This is a small, deterministic, console-only demonstration of the full MVP flow
as it exists today. It is intentionally a *thin orchestrator*: it wires together
the already-implemented modules and formats their output for a human reader. It
computes no business logic of its own and introduces no new dependencies.

Flow (reusing existing functions exactly):

1. ``load_config(None)``                       (config.py)   -> validated Config
2. ``generate_historical_data(...)``           (synthetic.py) -> historical records
3. ``run_pipeline(records, config)``           (pipeline.py) -> DeterministicResults
4. ``explain(snapshot, provider=None)``        (explanation.py) -> ExplanationResult
                                                (offline NullLLMProvider path)

Scope (hackathon MVP): plain deterministic Python, console output only. This
module deliberately does **not** build a web frontend/dashboard, auth, database,
API server, or deployment; it does **not** connect to AWS Bedrock or any external
LLM; and it does **not** implement LangGraph. It only orchestrates the existing
deterministic core plus the offline explanation step and prints a readable
report.

Determinism: given the same ``Config`` the produced records, snapshot, and
explanation are byte-for-byte reproducible, so :func:`format_report` returns an
identical string across runs. ``format_report`` reads only from the snapshot and
explanation — it never recomputes any value — and iterates vendors in sorted
``vendor_id`` order with fixed formatting.

Everything here is importable and side-effect-free except :func:`main`, which
prints the formatted report.
"""

from typing import NamedTuple, Optional, Sequence

from .config import load_config
from .explanation import explain
from .pipeline import run_pipeline
from .schema import (
    Config,
    DeterministicResults,
    ExplanationResult,
    HistoricalDataRecord,
    StageError,
)
from .synthetic import generate_historical_data

# Risk levels considered "elevated" for the summary section. Kept as a module
# constant so the demo's counting is explicit and matches the schema's
# ``VendorRiskResult.risk_level`` literal values.
_ELEVATED_RISK_LEVELS: frozenset[str] = frozenset({"high", "critical"})


class DemoOutput(NamedTuple):
    """Result container for a single demo run.

    A tiny stdlib ``NamedTuple`` (no new dependency) bundling the immutable
    deterministic ``snapshot``, the offline ``explanation``, and any aggregated
    per-stage ``errors`` from the pipeline. The deterministic ``snapshot`` is the
    source of truth; ``explanation`` is presentation-only natural language.
    """

    snapshot: DeterministicResults
    explanation: ExplanationResult
    errors: list[StageError]


def run_demo(
    config: Optional[Config] = None,
    records: Optional[Sequence[HistoricalDataRecord]] = None,
) -> DemoOutput:
    """Run the full local end-to-end demo flow and return its output.

    Orchestrates the existing modules only (no recomputation, no new logic):

    * ``config`` defaults to ``load_config(None)`` (all-defaults, offline).
    * ``records`` defaults to ``generate_historical_data(seed=config.random_seed,
      record_count=config.record_count)``. When the caller passes ``records``
      (including an empty list) that sequence is used verbatim, so edge cases such
      as ``records=[]`` are exercisable without generating data.
    * ``run_pipeline(records, config)`` produces the immutable
      :class:`~vendor_forecasting_agent.schema.DeterministicResults` snapshot.
    * ``explain(snapshot, provider=None)`` runs the offline
      :class:`~vendor_forecasting_agent.llm_provider.NullLLMProvider` path (no
      network, no external LLM) and returns an
      :class:`~vendor_forecasting_agent.schema.ExplanationResult`.

    Returns a :class:`DemoOutput` bundling the snapshot, explanation, and the
    pipeline's aggregated per-stage errors. Fully deterministic for a given
    ``config``; performs no network/LLM/I/O.
    """
    config = config if config is not None else load_config(None)

    if records is None:
        records = generate_historical_data(
            seed=config.random_seed,
            record_count=config.record_count,
        )

    result = run_pipeline(records, config)
    snapshot: DeterministicResults = result.data
    explanation = explain(snapshot, provider=None)

    return DemoOutput(
        snapshot=snapshot,
        explanation=explanation,
        errors=list(result.errors),
    )


def _fmt_float(value: float) -> str:
    """Format a float with fixed 2-decimal precision for stable console output."""
    return f"{value:.2f}"


def format_report(
    snapshot: DeterministicResults,
    explanation: ExplanationResult,
) -> str:
    """Build a readable, deterministic console report string.

    Presentation-only: reads exclusively from ``snapshot`` and ``explanation``
    and recomputes nothing. Vendors are iterated in sorted ``vendor_id`` order
    (driven by ``snapshot.recommendations``) with fixed formatting, so the same
    inputs always yield an identical string.

    Per vendor it shows: vendor id, risk score, risk level, main risk drivers
    ("none" when empty), expected lead time (first forecast value, else "n/a"),
    projected inventory units (0.00 when no exposure), and the recommended
    action. An "Explanation" section prints ``explanation.text`` once (covering
    all vendors); when the explanation is unavailable/flagged the report notes
    that while still showing the deterministic results. A final summary reports
    vendors analyzed, elevated-risk (high/critical) vendors, and vendors with
    inventory exposure (projected units > 0).

    Returns the report string; does not print (see :func:`main`).
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Vendor Forecasting Agent — End-to-End Demo")
    lines.append("=" * 70)

    recommendations = sorted(
        snapshot.recommendations, key=lambda rec: rec.vendor_id
    )

    # ---- Per-vendor structured blocks --------------------------------------
    if not recommendations:
        lines.append("")
        lines.append("No vendors were produced by the deterministic pipeline.")
    else:
        for rec in recommendations:
            vendor_id = rec.vendor_id
            risk = snapshot.risk.get(vendor_id)
            forecast = snapshot.forecasts.get(vendor_id)
            inventory = snapshot.inventory.get(vendor_id)

            lines.append("")
            lines.append(f"Vendor: {vendor_id}")
            lines.append("-" * 70)

            if risk is not None:
                lines.append(f"  Risk score        : {_fmt_float(risk.risk_score)}")
                lines.append(f"  Risk level        : {risk.risk_level}")
                drivers = (
                    ", ".join(risk.main_risk_drivers)
                    if risk.main_risk_drivers
                    else "none"
                )
                lines.append(f"  Main risk drivers : {drivers}")
            else:
                lines.append("  Risk score        : n/a")
                lines.append("  Risk level        : n/a")
                lines.append("  Main risk drivers : none")

            if forecast is not None and forecast.values:
                expected_lead_time = _fmt_float(forecast.values[0])
            else:
                expected_lead_time = "n/a"
            lines.append(f"  Expected lead time: {expected_lead_time}")

            projected_units = (
                inventory.projected_units if inventory is not None else 0.0
            )
            lines.append(
                f"  Projected units   : {_fmt_float(projected_units)}"
            )

            lines.append(f"  Recommended action: {rec.action}")

    # ---- Explanation section -----------------------------------------------
    lines.append("")
    lines.append("=" * 70)
    lines.append("Explanation")
    lines.append("=" * 70)
    if explanation.text and explanation.is_valid:
        lines.append(explanation.text)
    else:
        lines.append(
            "Explanation unavailable or flagged; the deterministic results "
            "above remain valid."
        )
        if explanation.text:
            lines.append("")
            lines.append(explanation.text)

    # ---- Overall summary ----------------------------------------------------
    vendors_analyzed = len(recommendations)
    elevated_risk_count = sum(
        1
        for risk in snapshot.risk.values()
        if risk.risk_level in _ELEVATED_RISK_LEVELS
    )
    inventory_exposure_count = sum(
        1
        for inventory in snapshot.inventory.values()
        if inventory.projected_units > 0
    )

    lines.append("")
    lines.append("=" * 70)
    lines.append("Summary")
    lines.append("=" * 70)
    lines.append(f"  Vendors analyzed              : {vendors_analyzed}")
    lines.append(f"  High/critical-risk vendors    : {elevated_risk_count}")
    lines.append(f"  Vendors with inventory exposure: {inventory_exposure_count}")

    return "\n".join(lines)


def main() -> None:
    """Build the demo report and print it to the console.

    The only side-effecting entry point in this module. Uses all-defaults
    ``load_config(None)`` and the offline explanation path via :func:`run_demo`,
    then prints the formatted report.
    """
    output = run_demo()
    report = format_report(output.snapshot, output.explanation)
    print(report)


if __name__ == "__main__":
    main()
