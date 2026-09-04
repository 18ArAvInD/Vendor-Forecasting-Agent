"""Phase 2 natural-language explanation module for the Vendor Forecasting Agent (explanation.py).

This module is the ``Explanation_Module`` described in the design. It turns the
already-computed, immutable
:class:`~vendor_forecasting_agent.schema.DeterministicResults` into a
human-readable explanation via a pluggable
:class:`~vendor_forecasting_agent.llm_provider.LLMProvider`, and validates that
the returned text's numbers match the deterministic results.

Hard boundary (Req 13.4-13.6, 15.2, 15.5, 16.4): every quantitative statement is
derived from ``results``; the module introduces no independently computed number,
never mutates ``results`` (it is frozen anyway), and never selects or changes any
risk/forecast/inventory/recommendation value — it only turns those values into
words. Graceful degradation (Req 13.8): if the provider is unavailable/errors or
returns empty text, the deterministic results remain intact and only the
explanation is flagged/omitted.

Scope: plain deterministic Python. The validator
(:func:`validate_explanation_numbers`) is a simple rule-based, non-LLM
consistency check — no model, no network, no second agent, no recomputation.
Determinism: vendors are iterated in sorted ``vendor_id`` order and every number
is formatted with a single fixed rounding (:data:`_DECIMALS`).
"""

import re
from typing import Optional

from .llm_provider import LLMProvider, NullLLMProvider
from .schema import (
    DeterministicResults,
    ExplanationResult,
    NumberMismatch,
)

# Canonical rounding used consistently by both the template builder and the
# validator, so a number written by the template always matches an allowed value.
_DECIMALS = 2

# Module-level default provider (offline, no-network pass-through). Reused so the
# provider=None path introduces nothing and stays fully deterministic/offline.
_DEFAULT_PROVIDER = NullLLMProvider()

# Numeric token pattern: integers or decimals (e.g. "7", "12.50"). Deliberately
# simple and robust; never raises for normal input. The surrounding negative
# look-around excludes digits embedded in identifier-like labels (e.g. the
# ``001`` in a vendor id ``vendor-001`` or digits inside a driver key), so only
# standalone quantitative tokens are validated — identifiers are labels, not
# quantitative statements.
_NUMBER_RE = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w.-])")


def _round(value: float) -> float:
    """Round ``value`` to the canonical number of decimals."""
    return round(float(value), _DECIMALS)


def _fmt(value: float) -> str:
    """Format ``value`` deterministically with the canonical rounding."""
    return f"{_round(value):.{_DECIMALS}f}"


def build_explanation_text(results: DeterministicResults) -> str:
    """Construct the complete deterministic, human-readable explanation template.

    Built ONLY from values already present in ``results`` (Req 15.2): no number is
    introduced that is not derived from the deterministic results. Vendors are
    driven by ``results.recommendations`` ordered by ``vendor_id`` (sorted), and
    each vendor's risk / forecast / inventory are pulled by ``vendor_id``. For
    each vendor a short paragraph covers risk level + score, main risk drivers,
    expected lead time, inventory exposure (only when present and > 0), the
    recommended action, and one connecting sentence.

    Returns a short ``"No vendors to explain."`` message when there are no
    recommendations/vendors. Fully deterministic (sorted order, fixed rounding);
    no randomness, network, LLM, or I/O.
    """
    recommendations = sorted(results.recommendations, key=lambda rec: rec.vendor_id)
    if not recommendations:
        return "No vendors to explain."

    paragraphs: list[str] = []
    for rec in recommendations:
        vendor_id = rec.vendor_id
        risk = results.risk.get(vendor_id)
        forecast = results.forecasts.get(vendor_id)
        inventory = results.inventory.get(vendor_id)

        sentences: list[str] = []

        # Risk level + score.
        if risk is not None:
            sentences.append(
                f"Vendor {vendor_id} has a {risk.risk_level} risk level "
                f"with a risk score of {_fmt(risk.risk_score)}."
            )
            # Main risk drivers.
            if risk.main_risk_drivers:
                drivers = ", ".join(risk.main_risk_drivers)
                sentences.append(f"The main risk drivers are {drivers}.")
            else:
                sentences.append("There are no dimensions flagged as risk drivers.")
        else:
            sentences.append(f"Vendor {vendor_id} has no computed risk result.")

        # Expected lead time (first forecast value).
        if forecast is not None and forecast.values:
            sentences.append(
                f"The expected lead time is {_fmt(forecast.values[0])} days."
            )

        # Inventory exposure — only when present and > 0.
        if inventory is not None and inventory.projected_units > 0:
            sentences.append(
                f"This vendor carries an inventory exposure of "
                f"{_fmt(inventory.projected_units)} units over a horizon of "
                f"{inventory.horizon} periods."
            )

        # Recommendation action.
        sentences.append(
            f"The recommended action is to {rec.action} this vendor "
            f"(recommendation score {_fmt(rec.score)})."
        )

        # Connecting sentence tying risk + recommendation together.
        if risk is not None:
            sentences.append(
                f"Because the assessed risk is {risk.risk_level}, "
                f"the recommendation is to {rec.action}."
            )

        paragraphs.append(" ".join(sentences))

    return "\n\n".join(paragraphs)


def _allowed_numbers(results: DeterministicResults) -> set[float]:
    """Build the set of numbers the explanation is allowed to contain.

    Uses the SAME canonical rounding as :func:`build_explanation_text`. Includes
    risk scores, forecast expected values, inventory projected_units /
    reorder_delta_units / horizon, and recommendation scores. Derived purely from
    ``results`` — no recomputation of any result.
    """
    allowed: set[float] = set()

    for risk in results.risk.values():
        allowed.add(_round(risk.risk_score))

    for forecast in results.forecasts.values():
        for value in forecast.values:
            allowed.add(_round(value))

    for inventory in results.inventory.values():
        allowed.add(_round(inventory.projected_units))
        allowed.add(_round(inventory.reorder_delta_units))
        allowed.add(_round(inventory.horizon))

    for rec in results.recommendations:
        allowed.add(_round(rec.score))

    return allowed


def validate_explanation_numbers(
    text: str,
    results: DeterministicResults,
) -> list[NumberMismatch]:
    """Pure, deterministic, rule-based numeric consistency check (Req 15.4).

    Extracts numeric tokens from ``text`` and confirms each corresponds to a value
    present in ``results`` (within canonical rounding). Any numeric token that
    does not match an allowed value is reported as a
    :class:`~vendor_forecasting_agent.schema.NumberMismatch` whose
    ``nearest_expected`` is the closest allowed value (or ``None`` when there is
    no allowed number to compare against).

    No LLM, no network, no second agent, no recomputation of any result. Never
    raises for normal input. For the Null/template path this always returns an
    empty list, because the template only uses allowed numbers.
    """
    allowed = _allowed_numbers(results)
    mismatches: list[NumberMismatch] = []

    for token in _NUMBER_RE.findall(text):
        try:
            value = _round(float(token))
        except ValueError:  # pragma: no cover - regex only matches parseable numbers
            continue

        if value in allowed:
            continue

        nearest: Optional[float] = None
        if allowed:
            nearest = min(allowed, key=lambda candidate: abs(candidate - value))
        mismatches.append(
            NumberMismatch(value_in_text=value, nearest_expected=nearest)
        )

    return mismatches


def explain(
    results: DeterministicResults,
    provider: LLMProvider | None = None,
) -> ExplanationResult:
    """Generate a natural-language explanation of ``results`` (Req 15.1).

    Builds the deterministic template strictly from ``results``, calls
    ``provider.generate(...)``, and validates the returned text's numbers against
    ``results``. When ``provider`` is ``None`` the offline, no-network default
    :class:`~vendor_forecasting_agent.llm_provider.NullLLMProvider` is used.

    Behavior:

    * Provider raises OR returns falsy text -> return
      ``ExplanationResult(text=None, is_valid=False, degraded=True)`` while the
      deterministic ``results`` remain intact (Req 13.8).
    * Success with mismatched numbers -> return
      ``ExplanationResult(text=..., is_valid=False, mismatches=...)`` (Req 15.4).
    * Success with all numbers matching -> return
      ``ExplanationResult(text=..., is_valid=True)``.

    Never mutates ``results`` (it is frozen), and never computes or changes any
    risk/forecast/inventory/recommendation value — it only turns them into words
    (Req 13.4-13.6, 15.2, 15.5, 16.4).
    """
    active_provider = provider if provider is not None else _DEFAULT_PROVIDER

    text = build_explanation_text(results)

    try:
        returned_text = active_provider.generate(text)
    except Exception:  # noqa: BLE001 - any provider failure -> degraded (Req 13.8)
        return ExplanationResult(text=None, is_valid=False, degraded=True)

    if not returned_text:
        return ExplanationResult(text=None, is_valid=False, degraded=True)

    mismatches = validate_explanation_numbers(returned_text, results)
    if mismatches:
        return ExplanationResult(
            text=returned_text, is_valid=False, mismatches=mismatches
        )

    return ExplanationResult(text=returned_text, is_valid=True)
