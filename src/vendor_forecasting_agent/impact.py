"""Impact analysis for the Vendor Forecasting Agent (impact.py).

This module is the ``Impact_Module`` described in the design. It provides two
deterministic, plain-Python functions (Task 9):

* :func:`analyze_impact` — decision/change impact: pre, post, absolute and
  percentage change for each affected vendor value (Requirement 9).
* :func:`analyze_inventory_impact` — inventory delay-exposure estimate from a
  vendor's forecast (expected lead time, Task 6) and metric (normal lead time +
  demand, Task 4) (Requirement 16).

Scope (Task 9 only): simple deterministic Python. This is a hackathon MVP — the
inventory estimate is a single delay-exposure number, **not** inventory
forecasting, safety stock, demand prediction, Monte Carlo, or scenarios. Neither
function recomputes metrics, trends, forecasts, or risk; they only reuse the
outputs already produced upstream. There is no recommendation, pipeline, LLM, or
LangGraph logic here, no new dependencies, and no network/LLM/I/O.

Design constraints honored here (Requirements 9, 16):

* **9.1 / 16.1-16.2** — Deterministic pre/post/absolute/percentage change per
  affected value, and the inventory exposure ``avg_demand * max(0, expected -
  avg_lead_time)`` (never negative).
* **9.2 / 16.3** — Byte-identical output for identical inputs: vendors iterated
  in sorted ``vendor_id`` order, ``data`` and ``errors`` built in sorted order.
* **9.3 / 9.4 / 16.4** — Per-vendor error handling: unknown vendor id, missing/
  invalid required field, or missing inventory input excludes the vendor and
  records a :class:`~vendor_forecasting_agent.schema.StageError`; other vendors
  are unaffected and the provided inputs are left unchanged.
* **9.5 / 16.5** — Pure local computation: no network, LLM, or external API
  calls, and no I/O.
"""

from typing import Mapping, Optional

from .schema import (
    ChangeDelta,
    DecisionChange,
    Forecast,
    ImpactResult,
    InventoryImpact,
    StageError,
    StageResult,
    VendorMetric,
    VendorScore,
)

# The VendorMetric numeric fields a decision may target (besides "score").
# Only these plain float measures are meaningful change targets; identifiers,
# counts, and Optional gate fields are excluded on purpose.
_METRIC_FIELDS = frozenset(
    {
        "avg_demand",
        "avg_lead_time_days",
        "avg_defect_rate",
        "avg_on_time_rate",
        "avg_quantity_fulfillment_rate",
        "avg_capacity_utilization",
        "avg_allocation_ratio",
    }
)


# ---- Small private helpers --------------------------------------------------


def _make_change_delta(pre: float, delta: float) -> ChangeDelta:
    """Build a :class:`ChangeDelta` for applying ``delta`` to ``pre`` (Req 9.1).

    ``post = pre + delta``; ``absolute_change = post - pre``; and
    ``percentage_change = absolute_change / pre * 100`` when ``pre`` is non-zero,
    else ``0.0`` (avoids division by zero).
    """
    post = pre + delta
    absolute = post - pre
    percentage = (absolute / pre * 100.0) if pre != 0 else 0.0
    return ChangeDelta(
        pre_value=pre,
        post_value=post,
        absolute_change=absolute,
        percentage_change=percentage,
    )


def analyze_impact(
    decision: DecisionChange,
    scores_by_vendor: Mapping[str, VendorScore],
    metrics_by_vendor: Mapping[str, VendorMetric],
    forecasts_by_vendor: Mapping[str, Forecast],
) -> StageResult:
    """Compute decision/change impact per targeted vendor (Req 9).

    The design's ``StageResult[dict[str, ImpactResult]]`` notation is conceptual:
    the concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On success ``data`` is a
    ``dict[str, ImpactResult]`` keyed by ``vendor_id`` (sorted order).

    MVP interpretation (kept intentionally simple and deterministic): the
    ``decision`` applies its ``delta`` to the named ``decision.metric`` for each
    vendor in ``decision.target_vendor_ids``:

    * ``metric == "score"`` -> apply to the vendor's ``VendorScore.score``
      (``pre = score``, ``post = score + delta``) recorded as
      :attr:`ImpactResult.score_change`.
    * any other known ``VendorMetric`` numeric field -> apply to that field
      (``pre = getattr(metric, name)``, ``post = pre + delta``) recorded in
      :attr:`ImpactResult.metric_changes` under the field name.

    Per-vendor error handling (Req 9.3/9.4), leaving inputs unchanged:

    * A targeted vendor id absent from ``scores_by_vendor`` or
      ``metrics_by_vendor`` -> ``StageError`` (code ``"unknown_vendor"``) naming
      the missing id; other targeted vendors continue (Req 9.3).
    * An unknown/not-applicable ``decision.metric`` for the vendor, or a targeted
      metric field whose value is ``None`` -> ``StageError`` (code
      ``"validation_failure"``) naming the field; no impact for that vendor
      (Req 9.4).

    Deterministic (Req 9.2): ``target_vendor_ids`` iterated in sorted order;
    ``data`` and ``errors`` built in sorted vendor-id order; byte-identical for
    identical inputs. No network/LLM/I/O (Req 9.5).
    """
    data: dict[str, ImpactResult] = {}
    errors: list[StageError] = []

    metric_name = decision.metric
    delta = decision.delta

    for vendor_id in sorted(decision.target_vendor_ids):
        # Req 9.3: decision references a vendor absent from the provided data.
        if vendor_id not in scores_by_vendor or vendor_id not in metrics_by_vendor:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="target_vendor_ids",
                    code="unknown_vendor",
                    message=(
                        f"decision references unknown vendor {vendor_id!r} "
                        "(absent from provided vendor data)"
                    ),
                )
            )
            continue

        result = _impact_for_vendor(
            vendor_id,
            metric_name,
            delta,
            scores_by_vendor[vendor_id],
            metrics_by_vendor[vendor_id],
        )
        if isinstance(result, StageError):
            errors.append(result)
            continue

        data[vendor_id] = result

    return StageResult(data=data, errors=errors)


def _impact_for_vendor(
    vendor_id: str,
    metric_name: str,
    delta: float,
    score: VendorScore,
    metric: VendorMetric,
) -> ImpactResult | StageError:
    """Build the :class:`ImpactResult` for one targeted vendor, or a
    :class:`StageError` when the decision metric is not applicable (Req 9.4)."""
    if metric_name == "score":
        return ImpactResult(
            vendor_id=vendor_id,
            score_change=_make_change_delta(score.score, delta),
        )

    if metric_name in _METRIC_FIELDS:
        pre = getattr(metric, metric_name)
        if pre is None:
            return StageError(
                vendor_id=vendor_id,
                field=metric_name,
                code="validation_failure",
                message=(
                    f"vendor {vendor_id!r} metric field {metric_name!r} is None; "
                    "cannot compute impact"
                ),
            )
        return ImpactResult(
            vendor_id=vendor_id,
            metric_changes={metric_name: _make_change_delta(float(pre), delta)},
        )

    # Unknown / not-applicable metric name for this vendor (Req 9.4).
    return StageError(
        vendor_id=vendor_id,
        field="metric",
        code="validation_failure",
        message=(
            f"decision metric {metric_name!r} is not an applicable score/metric "
            f"field for vendor {vendor_id!r}"
        ),
    )


def analyze_inventory_impact(
    forecasts_by_vendor: Mapping[str, Forecast],
    metrics_by_vendor: Mapping[str, VendorMetric],
) -> StageResult:
    """Estimate deterministic inventory delay-exposure per vendor (Req 16).

    The design's ``StageResult[dict[str, InventoryImpact]]`` notation is
    conceptual (see :func:`analyze_impact`); the runtime annotation is the plain
    ``StageResult``. On success ``data`` is a ``dict[str, InventoryImpact]`` keyed
    by ``vendor_id`` (sorted order).

    MVP formula (reuses the Task 6 forecast + Task 4 metric; recomputes
    neither):

    * ``expected_lead_time = forecast.values[0]`` — flat expected lead time.
    * ``avg_lead_time = metric.avg_lead_time_days`` — normal lead time.
    * ``expected_delay_days = max(0.0, expected_lead_time - avg_lead_time)`` —
      never negative (Req 16.2).
    * ``exposure = metric.avg_demand * expected_delay_days`` — ``>= 0``.

    Produces ``InventoryImpact(vendor_id, projected_units=exposure,
    reorder_delta_units=exposure, horizon=forecast.horizon)``. This is a single
    simple delay-exposure estimate — not inventory forecasting, safety stock,
    demand prediction, Monte Carlo, or scenarios.

    Per-vendor error handling (Req 16.4), other vendors unaffected:

    * no ``Forecast`` for the vendor -> ``StageError`` (code ``"missing_input"``).
    * no ``VendorMetric`` for the vendor -> ``StageError`` (``"missing_input"``).
    * ``forecast.values`` empty -> ``StageError`` (``"validation_failure"``).
    * ``metric.avg_lead_time_days`` or ``metric.avg_demand`` is ``None`` ->
      ``StageError`` (``"validation_failure"``) naming the missing field.

    Empty input yields ``StageResult(data={}, errors=[])`` (no vendors invented).

    Deterministic (Req 16.3): vendors iterated in sorted ``vendor_id`` order;
    ``data`` and ``errors`` built in sorted order; byte-identical for identical
    inputs. No network/LLM/I/O (Req 16.5).
    """
    data: dict[str, InventoryImpact] = {}
    errors: list[StageError] = []

    for vendor_id in sorted(forecasts_by_vendor):
        forecast = forecasts_by_vendor[vendor_id]
        metric = metrics_by_vendor.get(vendor_id)

        error = _validate_inventory_inputs(vendor_id, forecast, metric)
        if error is not None:
            errors.append(error)
            continue

        # ``metric`` is not None and both fields are non-None past validation.
        expected_lead_time = forecast.values[0]
        expected_delay_days = max(0.0, expected_lead_time - metric.avg_lead_time_days)
        exposure = metric.avg_demand * expected_delay_days

        data[vendor_id] = InventoryImpact(
            vendor_id=vendor_id,
            projected_units=exposure,
            reorder_delta_units=exposure,
            horizon=forecast.horizon,
        )

    return StageResult(data=data, errors=errors)


def _validate_inventory_inputs(
    vendor_id: str,
    forecast: Forecast,
    metric: Optional[VendorMetric],
) -> StageError | None:
    """Return a ``StageError`` if inventory inputs are missing/invalid, else
    ``None`` (Req 16.4).

    ``forecast`` is always present (iteration is over ``forecasts_by_vendor``); a
    missing ``metric`` is passed as ``None``.
    """
    if metric is None:
        return StageError(
            vendor_id=vendor_id,
            field="metric",
            code="missing_input",
            message=(
                f"vendor {vendor_id!r} has no VendorMetric; "
                "cannot compute inventory impact"
            ),
        )
    if not forecast.values:
        return StageError(
            vendor_id=vendor_id,
            field="values",
            code="validation_failure",
            message=(
                f"vendor {vendor_id!r} forecast has empty values; "
                "cannot read expected lead time"
            ),
        )
    if metric.avg_lead_time_days is None:
        return StageError(
            vendor_id=vendor_id,
            field="avg_lead_time_days",
            code="validation_failure",
            message=(
                f"vendor {vendor_id!r} metric avg_lead_time_days is None; "
                "cannot compute expected delay"
            ),
        )
    if metric.avg_demand is None:
        return StageError(
            vendor_id=vendor_id,
            field="avg_demand",
            code="validation_failure",
            message=(
                f"vendor {vendor_id!r} metric avg_demand is None; "
                "cannot compute exposure"
            ),
        )
    return None
