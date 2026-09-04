"""Deterministic lead-time forecasting for the Vendor Forecasting Agent (forecast.py).

This module is the ``Forecast_Module`` described in the design. Its single public
entry point, :func:`forecast_vendors`, produces a simple, deterministic expected
**lead-time** :class:`~vendor_forecasting_agent.schema.Forecast` per vendor over
the configured horizon [1, 120], reusing ``avg_lead_time_days`` (metrics), the
trend percentage (``TrendResult.slope`` == ``trend_pct`` from Task 5), and a
simple historical variability buffer.

Scope (sub-task 6.1 only): plain-Python arithmetic plus ``statistics.pstdev``.
No pandas/numpy, no ML/regression/time-series/seasonality/smoothing/Monte
Carlo/confidence intervals/scenario generation, and no new dependencies. This
forecasts VENDOR LEAD TIME ONLY — it is not inventory/demand forecasting and does
not compute runway, stockout, shortage, or replenishment.

MVP formula (a flat expected value repeated across the horizon)::

    base = metrics.avg_lead_time_days                     # Task 4
    trend_adjustment = base * (trend.slope / 100.0)       # trend.slope is trend_pct (Task 5)
    variability_buffer = statistics.pstdev(lead_times) if len(lead_times) >= 2 else 0.0
    expected = max(0.0, base + trend_adjustment + variability_buffer)

Each produced ``Forecast`` carries exactly ``horizon`` values, all equal to
``expected`` — the flat forecast is intentional and acceptable for the MVP (no
artificial period-by-period variation is added). ``Forecast.seed`` is retained
only to satisfy the existing schema field and is fixed at ``0`` (the formula uses
no randomness).

Design constraints honored here (Requirement 6):

* **6.1 / 6.4** — One ``Forecast`` per eligible vendor covering exactly
  ``horizon`` periods (``values`` length == ``horizon``), well within the 5s /
  vendor budget (a handful of arithmetic operations per vendor).
* **6.2** — A vendor is forecast only when it has a ``VendorMetric`` with a
  non-``None`` ``avg_lead_time_days``, a ``TrendResult`` (its ``slope`` used as
  ``trend_pct``), and a non-empty lead-time history. Any missing/``None``/empty
  input excludes that vendor and records a per-vendor
  :class:`~vendor_forecasting_agent.schema.StageError` naming the vendor and the
  reason; no exception is raised.
* **6.3 / 11.6** — Deterministic: identical inputs yield byte-identical output;
  both the ``data`` dict and the ``errors`` list are built in sorted vendor-id
  order.
* **6.5** — Invalid horizon (non-``int``, ``bool``, ``< 1``, or ``> 120``)
  rejects the whole request: returns ``data == {}`` with a single
  ``"invalid_horizon"`` error and produces no forecasts. ``bool`` is rejected
  explicitly (it is a subclass of ``int``), mirroring the synthetic-stage
  ``record_count`` check.
* **6.6** — Pure local computation: no network, LLM, or external API calls, and
  no I/O.
"""

import statistics
from typing import Mapping, Optional, Sequence

from .schema import Forecast, StageError, StageResult, TrendResult, VendorMetric

# Horizon bounds mirror the ``Forecast.horizon`` / ``Config.forecast_horizon``
# schema bounds. Enforcing them here (before building any Forecast) guarantees
# the ``values`` length (== horizon) is always schema-valid.
_MIN_HORIZON: int = 1
_MAX_HORIZON: int = 120

# Fixed seed retained only to satisfy the ``Forecast.seed`` schema field; the
# formula is fully deterministic and uses no randomness.
_FIXED_SEED: int = 0


def _horizon_error(horizon: object) -> Optional[StageError]:
    """Validate ``horizon`` (Req 6.5), returning a whole-request error or ``None``.

    ``bool`` is explicitly rejected even though it is a subclass of ``int`` so a
    ``True``/``False`` argument cannot silently be treated as ``1``/``0``
    (mirrors the synthetic-stage ``record_count`` check).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        return StageError(
            vendor_id=None,
            field="horizon",
            code="invalid_horizon",
            message=(
                "horizon must be an integer in the inclusive range "
                f"[{_MIN_HORIZON}, {_MAX_HORIZON}]; got {horizon!r} of type "
                f"{type(horizon).__name__}"
            ),
        )
    if horizon < _MIN_HORIZON or horizon > _MAX_HORIZON:
        return StageError(
            vendor_id=None,
            field="horizon",
            code="invalid_horizon",
            message=(
                "horizon is out of the accepted range "
                f"[{_MIN_HORIZON}, {_MAX_HORIZON}]; got {horizon}"
            ),
        )
    return None


def forecast_vendors(
    metrics_by_vendor: Mapping[str, VendorMetric],
    trends_by_vendor: Mapping[str, TrendResult],
    lead_times_by_vendor: Mapping[str, Sequence[float]],
    *,
    horizon: int,
) -> StageResult:
    """Produce a simple, deterministic expected lead-time ``Forecast`` per vendor (Req 6).

    The design's ``StageResult[dict[str, Forecast]]`` notation is conceptual: the
    concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not a
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On the normal path ``data`` is a
    ``dict[str, Forecast]`` keyed by ``vendor_id`` (sorted order) and ``errors``
    holds one per-vendor entry for each excluded vendor.

    MVP formula (a flat expected value repeated across the horizon)::

        base = metrics.avg_lead_time_days
        trend_adjustment = base * (trend.slope / 100.0)   # trend.slope is trend_pct
        variability_buffer = pstdev(lead_times) if len(lead_times) >= 2 else 0.0
        expected = max(0.0, base + trend_adjustment + variability_buffer)

    Each produced ``Forecast`` has exactly ``horizon`` values, all equal to
    ``expected`` (flat forecast is intentional for the MVP), and a fixed
    ``seed == 0`` (Req 6.4).

    Horizon validation runs first (Req 6.5): an invalid ``horizon``
    (non-``int``, ``bool``, ``< 1``, or ``> 120``) rejects the whole request,
    returning ``data == {}`` and a single ``"invalid_horizon"`` error with no
    forecasts.

    A vendor is forecast only when it has a ``VendorMetric`` with a non-``None``
    ``avg_lead_time_days``, a ``TrendResult``, and a non-empty lead-time history;
    any missing/``None``/empty input excludes the vendor and records a per-vendor
    ``StageError`` (Req 6.2). No input is mutated and no exception is raised.

    Deterministic (Req 6.3, 11.6): both ``data`` and ``errors`` are built in
    sorted vendor-id order. Performs no network, LLM, or external calls and no
    I/O (Req 6.6).
    """
    # Req 6.5: validate the horizon FIRST; on failure produce no forecasts.
    horizon_error = _horizon_error(horizon)
    if horizon_error is not None:
        return StageResult(data={}, errors=[horizon_error])

    forecasts: dict[str, Forecast] = {}
    errors: list[StageError] = []

    # Iterate deterministically by sorted vendor id (Req 6.3, 11.6).
    for vendor_id in sorted(metrics_by_vendor):
        metrics = metrics_by_vendor[vendor_id]

        base = metrics.avg_lead_time_days
        if base is None:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="avg_lead_time_days",
                    code="missing_input",
                    message=(
                        f"vendor {vendor_id!r} has no avg_lead_time_days metric; "
                        "cannot forecast lead time"
                    ),
                )
            )
            continue

        trend = trends_by_vendor.get(vendor_id)
        if trend is None:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="trend",
                    code="missing_input",
                    message=(
                        f"vendor {vendor_id!r} has no trend result; "
                        "cannot forecast lead time"
                    ),
                )
            )
            continue

        lead_times = lead_times_by_vendor.get(vendor_id)
        if not lead_times:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="lead_time_days",
                    code="insufficient_data",
                    message=(
                        f"vendor {vendor_id!r} has no lead-time history; "
                        "cannot forecast lead time"
                    ),
                )
            )
            continue

        trend_adjustment = base * (trend.slope / 100.0)
        variability_buffer = (
            statistics.pstdev(lead_times) if len(lead_times) >= 2 else 0.0
        )
        expected = max(0.0, base + trend_adjustment + variability_buffer)

        forecasts[vendor_id] = Forecast(
            vendor_id=vendor_id,
            horizon=horizon,
            values=[expected] * horizon,
            seed=_FIXED_SEED,
        )

    return StageResult(data=forecasts, errors=errors)
