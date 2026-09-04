"""Deterministic vendor-delay what-if module (whatif.py).

This module provides a deterministic, LLM-free, side-effect-free "what-if"
analysis that projects a hypothetical *additional lead-time delay* onto a
vendor's already-computed expected lead time and inventory exposure.

Reuse of the existing exposure formula
--------------------------------------
This module **REUSES the existing inventory-exposure formula** defined in
:mod:`vendor_forecasting_agent.impact` (see ``analyze_inventory_impact``):

    expected_lead_time = forecast.values[0]
    avg_lead_time      = metric.avg_lead_time_days
    exposure           = metric.avg_demand * max(0.0, expected_lead_time - avg_lead_time)

It does **NOT** create a second inventory-impact engine and does **NOT** modify
``impact.py``. ``impact.py`` remains the authoritative implementation of the
exposure formula; this module merely *mirrors that single one-line formula* for a
hypothetical delay so it never has to run the whole impact stage. To avoid
running the entire stage, the ``impact.py`` functions are intentionally not
imported — only the one-line formula is mirrored here, clearly documented as the
same formula.

Everything here is pure: deterministic and free of network, LLM, or I/O side
effects. Given identical inputs, it always produces an identical
:class:`WhatIfResult`.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .schema import DeterministicResults, Forecast, VendorMetric

__all__ = ["WhatIfResult", "analyze_what_if", "analyze_what_if_for_vendor"]


class WhatIfResult(BaseModel):
    """Immutable result of a single deterministic vendor-delay what-if.

    Matches the project's frozen / ``extra="forbid"`` convention with explicit
    types and sensible bounds. ``exposure_change`` is deliberately unbounded so
    the field stays general (it is ``>= 0`` for a non-negative delay, but the
    model does not enforce that).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    delay_days: float = Field(ge=0.0)
    original_expected_lead_time: float = Field(ge=0.0)
    new_expected_lead_time: float = Field(ge=0.0)
    original_exposure: float = Field(ge=0.0)
    new_exposure: float = Field(ge=0.0)
    exposure_change: float  # new_exposure - original_exposure (unbounded on purpose)
    inventory_horizon: Optional[int] = None


# ---- Small private validation helpers ---------------------------------------


def _validate_delay_days(delay_days: float) -> float:
    """Validate ``delay_days`` is a real, non-negative, non-bool number.

    Project convention is to raise for whole-request invalid scalar inputs (like
    the synthetic ``RecordCountError`` / forecast invalid-horizon pattern). For a
    pure helper the simplest equivalent is a ``ValueError`` with a clear message,
    so this raises ``ValueError`` when ``delay_days`` is a bool, is not a number,
    or is negative.
    """
    # ``bool`` is a subclass of ``int``; reject it explicitly so True/False are
    # not silently treated as 1.0/0.0.
    if isinstance(delay_days, bool):
        raise ValueError(
            f"delay_days must be a real number, got bool {delay_days!r}"
        )
    if not isinstance(delay_days, (int, float)):
        raise ValueError(
            f"delay_days must be a real number, got {type(delay_days).__name__}"
        )
    value = float(delay_days)
    if value < 0.0:
        raise ValueError(f"delay_days must be >= 0.0, got {value}")
    return value


def analyze_what_if(
    *,
    vendor_id: str,
    delay_days: float,
    expected_lead_time: float,
    avg_lead_time_days: float,
    avg_demand: float,
    inventory_horizon: Optional[int] = None,
) -> WhatIfResult:
    """Project a hypothetical additional lead-time ``delay_days`` for one vendor.

    Low-level pure function operating on primitives / existing-model values. It
    reuses the EXISTING exposure formula from ``impact.py`` (mirrored here as a
    single line, not a new engine):

    * ``original_expected_lead_time = expected_lead_time``
    * ``new_expected_lead_time = expected_lead_time + delay_days``
    * ``original_exposure = avg_demand * max(0.0, expected_lead_time - avg_lead_time_days)``
    * ``new_exposure = avg_demand * max(0.0, new_expected_lead_time - avg_lead_time_days)``
    * ``exposure_change = new_exposure - original_exposure``

    ``delay_days`` must be a real number ``>= 0.0`` and not a bool; a negative
    value, a non-number, or a bool raises :class:`ValueError`.

    Deterministic and side-effect-free.
    """
    delay = _validate_delay_days(delay_days)

    original_expected_lead_time = expected_lead_time
    new_expected_lead_time = expected_lead_time + delay

    original_exposure = avg_demand * max(0.0, expected_lead_time - avg_lead_time_days)
    new_exposure = avg_demand * max(0.0, new_expected_lead_time - avg_lead_time_days)
    exposure_change = new_exposure - original_exposure

    return WhatIfResult(
        vendor_id=vendor_id,
        delay_days=delay,
        original_expected_lead_time=original_expected_lead_time,
        new_expected_lead_time=new_expected_lead_time,
        original_exposure=original_exposure,
        new_exposure=new_exposure,
        exposure_change=exposure_change,
        inventory_horizon=inventory_horizon,
    )


def analyze_what_if_for_vendor(
    results: DeterministicResults,
    vendor_id: str,
    delay_days: float,
) -> WhatIfResult:
    """Run a what-if for ``vendor_id`` using inputs read from an existing snapshot.

    Convenience entry point that pulls the vendor's already-computed inputs
    straight from a :class:`~vendor_forecasting_agent.schema.DeterministicResults`
    snapshot (reads only; recomputes nothing) and delegates to
    :func:`analyze_what_if`. Because it delegates, it uses the SAME exposure
    formula and never recomputes risk / forecast.

    It reads:

    * ``forecast = results.forecasts.get(vendor_id)`` -> ``expected_lead_time = forecast.values[0]``
    * ``metric = results.metrics.get(vendor_id)`` -> ``avg_lead_time_days`` and ``avg_demand``
    * ``inventory_horizon`` = ``results.inventory[vendor_id].horizon`` when present,
      else ``forecast.horizon`` when a forecast exists, else ``None``.

    Raises :class:`ValueError` naming the vendor and missing input when the vendor
    is absent from forecasts/metrics, when ``forecast.values`` is empty, or when
    ``avg_lead_time_days`` / ``avg_demand`` is ``None`` (nothing is fabricated).
    """
    forecast: Optional[Forecast] = results.forecasts.get(vendor_id)
    if forecast is None:
        raise ValueError(
            f"vendor {vendor_id!r} missing input: no Forecast in results.forecasts"
        )
    if not forecast.values:
        raise ValueError(
            f"vendor {vendor_id!r} missing input: forecast.values is empty; "
            "cannot read expected lead time"
        )

    metric: Optional[VendorMetric] = results.metrics.get(vendor_id)
    if metric is None:
        raise ValueError(
            f"vendor {vendor_id!r} missing input: no VendorMetric in results.metrics"
        )
    if metric.avg_lead_time_days is None:
        raise ValueError(
            f"vendor {vendor_id!r} missing input: metric.avg_lead_time_days is None"
        )
    if metric.avg_demand is None:
        raise ValueError(
            f"vendor {vendor_id!r} missing input: metric.avg_demand is None"
        )

    inventory = results.inventory.get(vendor_id)
    inventory_horizon = inventory.horizon if inventory is not None else forecast.horizon

    return analyze_what_if(
        vendor_id=vendor_id,
        delay_days=delay_days,
        expected_lead_time=forecast.values[0],
        avg_lead_time_days=metric.avg_lead_time_days,
        avg_demand=metric.avg_demand,
        inventory_horizon=inventory_horizon,
    )
