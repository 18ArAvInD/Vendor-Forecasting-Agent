"""Phase 1 deterministic pipeline driver for the Vendor Forecasting Agent (pipeline.py).

This module is the plain, deterministic Phase-1 driver referenced in the design's
"Execution Flow": it wires the already-implemented deterministic stages together
in build order and assembles the immutable
:class:`~vendor_forecasting_agent.schema.DeterministicResults` snapshot.

Scope (Task 12.1 — Phase 1 only): plain deterministic Python. This is a hackathon
MVP driver — **no** LangGraph, LLM, provider, orchestration framework, DB, async,
network, or new dependencies. It reuses each stage's existing function exactly and
recomputes/duplicates no business logic; every number in the snapshot comes
straight from the stage that produced it.

Stage wiring (in order), each stage returning a
:class:`~vendor_forecasting_agent.schema.StageResult` whose ``.data`` is threaded
forward and whose ``.errors`` are aggregated:

1. ``compute_metrics(records)``                                      (metrics.py)
2. build ``lead_times_by_vendor`` from ``records`` (see below)       (this driver)
3. ``analyze_trend(lead_times_by_vendor, stable_threshold_pct=...)`` (trend.py)
4. ``forecast_vendors(metrics, trends, lead_times, horizon=...)``    (forecast.py)
5. ``evaluate_rules(metrics, trends)``                               (rules.py)
6. ``score_vendors(outcomes, metrics)``                              (scoring.py)
7. ``assess_risk(outcomes, metrics)``                                (scoring.py)
8. ``analyze_inventory_impact(forecasts, metrics)``                  (impact.py)
9. ``recommend_vendors(risk, inventory)``                            (recommend.py)

``impacts`` (decision/change impact via ``analyze_impact``) is intentionally NOT
part of this Phase 1 flow (there is no ``DecisionChange`` input), so the snapshot
carries an empty ``impacts`` dict.

Determinism (Req 11, 11.6): the only ordering this driver introduces is the
``lead_times_by_vendor`` grouping — records are grouped by ``vendor_id`` and each
vendor's lead-time history is ordered chronologically by ``period_index`` (stable
keys). Every stage already sorts its own output by ``vendor_id``. The driver
introduces no randomness and performs no network/LLM/I/O. Given identical
``records`` and ``config`` the assembled ``DeterministicResults`` is byte-for-byte
reproducible (Req 11.1).

Error handling (StageResult pattern): a per-vendor problem in any stage never
aborts the pipeline. Each stage records its own per-vendor
:class:`~vendor_forecasting_agent.schema.StageError`s and simply omits that
vendor from its ``.data``; downstream stages then process only the vendors that
made it through. The driver aggregates every stage's ``.errors`` (in the stage
order above; within a stage the errors are already sorted by ``vendor_id``) into
one list on the returned ``StageResult``. Empty input records -> every stage
yields empty data -> a ``DeterministicResults`` with empty dicts/list and an ok
``StageResult`` (no errors).
"""

from typing import Sequence

from .forecast import forecast_vendors
from .impact import analyze_inventory_impact
from .metrics import compute_metrics
from .recommend import recommend_vendors
from .rules import evaluate_rules
from .schema import (
    Config,
    DeterministicResults,
    HistoricalDataRecord,
    StageResult,
)
from .scoring import assess_risk, score_vendors
from .trend import analyze_trend


def _build_lead_times_by_vendor(
    records: Sequence[HistoricalDataRecord],
) -> dict[str, list[float]]:
    """Group ``records`` into per-vendor chronological lead-time histories.

    Returns a mapping ``vendor_id -> [lead_time_days, ...]`` where each vendor's
    list is ordered chronologically by ``period_index`` (Req 11.6 stable-key
    ordering). This is the input ``trend.py`` and ``forecast.py`` expect. The
    mapping is built deterministically: vendors keyed in sorted ``vendor_id``
    order and, within each vendor, records sorted by ``period_index``. Only valid
    ``HistoricalDataRecord`` instances contribute (mirrors the metrics stage's
    defensive handling); anything else is ignored here.
    """
    grouped: dict[str, list[HistoricalDataRecord]] = {}
    for record in records:
        if not isinstance(record, HistoricalDataRecord):
            continue
        grouped.setdefault(record.vendor_id, []).append(record)

    lead_times_by_vendor: dict[str, list[float]] = {}
    for vendor_id in sorted(grouped):
        ordered = sorted(grouped[vendor_id], key=lambda r: r.period_index)
        lead_times_by_vendor[vendor_id] = [r.lead_time_days for r in ordered]
    return lead_times_by_vendor


def run_pipeline(
    records: Sequence[HistoricalDataRecord],
    config: Config,
) -> StageResult:
    """Run the Phase 1 deterministic core and assemble the results snapshot (Req 13.2).

    The concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not a
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On return, ``data`` is a
    :class:`~vendor_forecasting_agent.schema.DeterministicResults` snapshot and
    ``errors`` is the aggregation of every stage's per-vendor/whole-request
    ``StageError``s in stage order (metrics, trends, forecasts, rules, scores,
    risk, inventory, recommendations).

    Wiring reuses each stage's existing function exactly and recomputes nothing:
    ``compute_metrics`` -> build ``lead_times_by_vendor`` -> ``analyze_trend``
    (with ``config.stable_threshold_pct``) -> ``forecast_vendors`` (with
    ``config.forecast_horizon``) -> ``evaluate_rules`` -> ``score_vendors`` /
    ``assess_risk`` -> ``analyze_inventory_impact`` -> ``recommend_vendors``.

    ``impacts`` is left empty: decision/change impact (``analyze_impact``) needs a
    ``DecisionChange`` input and is not part of this Phase 1 flow.

    Determinism (Req 11.1, 11.6): the driver only orders the ``lead_times``
    grouping (sorted vendor ids, records ordered by ``period_index``); every
    stage sorts internally. No randomness, network, LLM, or I/O. Per-vendor stage
    errors are threaded through (not raised): a bad vendor is dropped by its stage
    while healthy vendors continue downstream, and all errors are collected onto
    the returned ``StageResult``. Empty ``records`` -> empty snapshot, ok result.
    """
    errors = []

    # 1. Per-vendor performance metrics.
    metrics = compute_metrics(records)
    errors.extend(metrics.errors)

    # 2. Per-vendor chronological lead-time history (the only ordering the driver
    #    introduces; stable keys per Req 11.6).
    lead_times_by_vendor = _build_lead_times_by_vendor(records)

    # 3. Lead-time trend classification.
    trends = analyze_trend(
        lead_times_by_vendor,
        stable_threshold_pct=config.stable_threshold_pct,
    )
    errors.extend(trends.errors)

    # 4. Deterministic expected lead-time forecast over the configured horizon.
    forecasts = forecast_vendors(
        metrics.data or {},
        trends.data or {},
        lead_times_by_vendor,
        horizon=config.forecast_horizon,
    )
    errors.extend(forecasts.errors)

    # 5. Fixed six-rule risk-concern signals.
    rules = evaluate_rules(metrics.data or {}, trends.data or {})
    errors.extend(rules.errors)

    # 6. Internal risk score scalar.
    scores = score_vendors(rules.data or {}, metrics.data or {})
    errors.extend(scores.errors)

    # 7. Externally exposed risk result (level / confidence / drivers).
    risk = assess_risk(rules.data or {}, metrics.data or {})
    errors.extend(risk.errors)

    # 8. Inventory delay-exposure estimate.
    inventory = analyze_inventory_impact(forecasts.data or {}, metrics.data or {})
    errors.extend(inventory.errors)

    # 9. One recommendation per eligible vendor.
    recommendations = recommend_vendors(risk.data or {}, inventory.data or {})
    errors.extend(recommendations.errors)

    snapshot = DeterministicResults(
        metrics=metrics.data or {},
        trends=trends.data or {},
        forecasts=forecasts.data or {},
        scores=scores.data or {},
        risk=risk.data or {},
        outcomes=rules.data or {},
        impacts={},  # decision/change impact is not part of the Phase 1 flow
        inventory=inventory.data or {},
        recommendations=recommendations.data or [],
    )

    return StageResult(data=snapshot, errors=errors)
