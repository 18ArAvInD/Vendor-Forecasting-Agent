"""Vendor scoring for the Vendor Forecasting Agent (scoring.py).

This module is the ``Scoring_Module`` described in the design. It turns the six
Task 7 :class:`~vendor_forecasting_agent.schema.RuleOutcome` signals per vendor
into the internal :class:`~vendor_forecasting_agent.schema.VendorScore` scalar
(:func:`score_vendors`) and the externally exposed
:class:`~vendor_forecasting_agent.schema.VendorRiskResult`
(:func:`assess_risk`). It reuses the Task 7 rule outcomes and Task 4
:class:`~vendor_forecasting_agent.schema.VendorMetric` sample counts; it does
**not** recompute metrics, trends, or rules.

Scope (Task 8 only): plain deterministic Python. This is a hackathon MVP — the
``risk_score`` is simply the proportion of the six rules triggered mapped to
0–100, **not** an ML/statistical/weighted/probability model, and not a
configurable/generic scoring engine. Inventory is **not** part of the risk
score. This module produces no impact, recommendation, LLM, or pipeline output,
and adds no new dependencies.

Design constraints honored here (Requirement 8):

* **8.1 / 8.4 / 8.6** — Each vendor's score is a float in the inclusive range
  ``0.0..100.0`` derived from all six vendor-risk dimensions: ``n`` triggered
  rules (of six) map to ``round(n / 6 * 100)`` giving ``0→0, 1→17, 2→33, 3→50,
  4→67, 5→83, 6→100``. The value is clamped to ``[0.0, 100.0]`` (it already lies
  in range).
* **8.2** — A vendor with missing/empty rule outcomes, or with no
  ``VendorMetric`` (needed for ``confidence``), is excluded from ``data`` and a
  per-vendor :class:`~vendor_forecasting_agent.schema.StageError` is recorded;
  other vendors are unaffected. Empty input yields an empty successful result.
* **8.3 / 8.8** — Deterministic: vendors iterated in sorted ``vendor_id`` order,
  ``data`` and ``errors`` built in sorted vendor-id order; byte-identical output
  for identical inputs.
* **8.7** — ``assess_risk`` produces a ``VendorRiskResult`` per vendor with
  ``risk_score`` equal to the ``VendorScore`` scalar, a ``risk_level`` band, a
  ``confidence`` in ``[0.0, 1.0]``, and the ordered ``main_risk_drivers``.
* **8.5 / 8.9** — Pure local computation: no network, LLM, or external API
  calls, and no I/O.
"""

from typing import Mapping, Sequence

from .schema import (
    RuleOutcome,
    StageError,
    StageResult,
    VendorMetric,
    VendorRiskResult,
    VendorScore,
)

# ---- Fixed MVP constants (not configurable) ---------------------------------

# The MVP score is the proportion of the six fixed Task 7 rules triggered,
# mapped to 0..100: score = round(n / 6 * 100). Six is the fixed number of
# vendor-risk dimensions (Req 8.6). In practice Task 7 always supplies six
# outcomes; scoring only counts triggered outcomes among whatever list is
# provided (a non-empty list is sufficient), keeping this stage decoupled.
_RULE_DIMENSION_COUNT = 6

# confidence = min(1.0, sample_count / N). N is a fixed, documented sample-count
# heuristic (NOT a statistical/ML confidence): 12 samples (e.g. ~monthly points
# over a year) is treated as "full" confidence. sample_count is >= 0 so the
# result always lies in [0.0, 1.0].
_CONFIDENCE_SAMPLE_TARGET = 12

# risk_level bands over the score (inclusive as written): 0–24 low, 25–49
# moderate, 50–74 high, 75–100 critical. Given the fixed score table this yields
# 0→low, 17→low, 33→moderate, 50→high, 67→high, 83→critical, 100→critical.
_LOW_MAX = 24.0
_MODERATE_MAX = 49.0
_HIGH_MAX = 74.0


# ---- Small private helpers --------------------------------------------------


def _score_from_outcomes(outcomes: Sequence[RuleOutcome]) -> float:
    """Map the number of triggered outcomes to the fixed 0..100 score.

    ``score = round(n / 6 * 100)`` clamped to ``[0.0, 100.0]`` (Req 8.4). Shared
    by :func:`score_vendors` and :func:`assess_risk` so their scores match
    exactly (Req 8.7).
    """
    triggered = sum(1 for outcome in outcomes if outcome.triggered)
    score = float(round(triggered / _RULE_DIMENSION_COUNT * 100))
    # Clamp to the range invariant (Req 8.4); already in range for n in 0..6.
    return min(100.0, max(0.0, score))


def _risk_level_for(score: float) -> str:
    """Map a score to its ``risk_level`` band (Req 8.7)."""
    if score <= _LOW_MAX:
        return "low"
    if score <= _MODERATE_MAX:
        return "moderate"
    if score <= _HIGH_MAX:
        return "high"
    return "critical"


def _confidence_for(sample_count: int) -> float:
    """``min(1.0, sample_count / N)`` bounded to ``[0.0, 1.0]`` (Req 8.7).

    ``N = _CONFIDENCE_SAMPLE_TARGET`` (12). ``sample_count`` is ``>= 0`` so the
    result is always in ``[0.0, 1.0]``.
    """
    confidence = sample_count / _CONFIDENCE_SAMPLE_TARGET
    return min(1.0, max(0.0, confidence))


def _main_risk_drivers(outcomes: Sequence[RuleOutcome]) -> list[str]:
    """The ``rule_id``s of the triggered outcomes, preserving the input order.

    The vendor's outcomes arrive in the fixed Task 7 rule order, so the drivers
    come out in that order (empty when none triggered) (Req 8.7).
    """
    return [outcome.rule_id for outcome in outcomes if outcome.triggered]


def score_vendors(
    outcomes_by_vendor: Mapping[str, Sequence[RuleOutcome]],
    metrics_by_vendor: Mapping[str, VendorMetric],
) -> StageResult:
    """Compute the internal ``VendorScore`` scalar per vendor (Req 8.1, 8.6).

    The design's ``StageResult[dict[str, VendorScore]]`` notation is conceptual:
    the concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On success ``data`` is a
    ``dict[str, VendorScore]`` keyed by ``vendor_id`` (sorted order).

    Scoring:

    * ``n`` = number of triggered rule outcomes for the vendor (of six).
    * ``score = round(n / 6 * 100)`` -> ``0, 17, 33, 50, 67, 83, 100`` for
      ``n = 0..6``, clamped to ``[0.0, 100.0]`` (Req 8.4).

    A vendor with missing/empty outcomes, or with no ``VendorMetric`` (required
    for the paired :func:`assess_risk` ``confidence``), is excluded from ``data``
    and a per-vendor ``StageError`` (code ``"missing_input"``) is recorded; other
    vendors are unaffected (Req 8.2). Empty input yields
    ``StageResult(data={}, errors=[])``.

    Deterministic (Req 8.3/8.8): vendors iterated in sorted ``vendor_id`` order;
    ``data`` and ``errors`` built in sorted order. No network/LLM/I/O (Req 8.5).
    """
    data: dict[str, VendorScore] = {}
    errors: list[StageError] = []

    for vendor_id in sorted(outcomes_by_vendor):
        error = _validate_vendor(vendor_id, outcomes_by_vendor, metrics_by_vendor)
        if error is not None:
            errors.append(error)
            continue

        outcomes = outcomes_by_vendor[vendor_id]
        data[vendor_id] = VendorScore(
            vendor_id=vendor_id,
            score=_score_from_outcomes(outcomes),
        )

    return StageResult(data=data, errors=errors)


def assess_risk(
    outcomes_by_vendor: Mapping[str, Sequence[RuleOutcome]],
    metrics_by_vendor: Mapping[str, VendorMetric],
) -> StageResult:
    """Produce the externally exposed ``VendorRiskResult`` per vendor (Req 8.7).

    The design's ``StageResult[dict[str, VendorRiskResult]]`` notation is
    conceptual (see :func:`score_vendors`); the runtime annotation is the plain
    ``StageResult``. On success ``data`` is a ``dict[str, VendorRiskResult]``
    keyed by ``vendor_id`` (sorted order). Each result carries:

    * ``risk_score`` — equal to the :func:`score_vendors` ``VendorScore`` scalar
      (proportion of six rules triggered mapped to 0..100).
    * ``risk_level`` — band from the score: ``0–24 low, 25–49 moderate, 50–74
      high, 75–100 critical``.
    * ``main_risk_drivers`` — the triggered ``rule_id``s in the fixed Task 7 rule
      order (empty when none triggered).
    * ``confidence`` — ``min(1.0, sample_count / N)`` with ``N = 12``, bounded to
      ``[0.0, 1.0]`` (a sample-count heuristic, NOT a statistical/ML confidence).

    A vendor with missing/empty outcomes, or with no ``VendorMetric`` (needed for
    ``confidence``), is excluded and a per-vendor ``StageError`` (code
    ``"missing_input"``) is recorded; other vendors are unaffected (Req 8.2).
    Empty input yields ``StageResult(data={}, errors=[])``.

    Deterministic (Req 8.3/8.8): sorted ``vendor_id`` order for ``data`` and
    ``errors``; byte-identical for identical inputs. No network/LLM/I/O (Req
    8.9).
    """
    data: dict[str, VendorRiskResult] = {}
    errors: list[StageError] = []

    for vendor_id in sorted(outcomes_by_vendor):
        error = _validate_vendor(vendor_id, outcomes_by_vendor, metrics_by_vendor)
        if error is not None:
            errors.append(error)
            continue

        outcomes = outcomes_by_vendor[vendor_id]
        score = _score_from_outcomes(outcomes)
        data[vendor_id] = VendorRiskResult(
            vendor_id=vendor_id,
            risk_score=score,
            risk_level=_risk_level_for(score),
            confidence=_confidence_for(metrics_by_vendor[vendor_id].sample_count),
            main_risk_drivers=_main_risk_drivers(outcomes),
        )

    return StageResult(data=data, errors=errors)


def _validate_vendor(
    vendor_id: str,
    outcomes_by_vendor: Mapping[str, Sequence[RuleOutcome]],
    metrics_by_vendor: Mapping[str, VendorMetric],
) -> StageError | None:
    """Return a ``StageError`` if the vendor lacks required scoring input, else
    ``None`` (Req 8.2).

    Required input: a non-empty rule-outcome list, and a ``VendorMetric`` (needed
    for ``confidence``). Shared by :func:`score_vendors` and :func:`assess_risk`
    so both stages exclude the same vendors identically.
    """
    outcomes = outcomes_by_vendor.get(vendor_id)
    if not outcomes:
        return StageError(
            vendor_id=vendor_id,
            field="outcomes",
            code="missing_input",
            message=(
                f"vendor {vendor_id!r} has no rule outcomes; cannot compute a score"
            ),
        )
    if vendor_id not in metrics_by_vendor:
        return StageError(
            vendor_id=vendor_id,
            field="metric",
            code="missing_input",
            message=(
                f"vendor {vendor_id!r} has no VendorMetric; cannot compute confidence"
            ),
        )
    return None
