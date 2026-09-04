"""Deterministic multi-scenario what-if engine (whatif_scenarios.py).

This module extends the single-scenario supplier-delay what-if
(:mod:`vendor_forecasting_agent.whatif`) into a small, deterministic, LLM-free,
side-effect-free family of "what-if" scenarios:

* **Supplier delay** — a hypothetical additional lead-time delay (REUSES
  :func:`vendor_forecasting_agent.whatif.analyze_what_if_for_vendor` so the
  delay math stays byte-identical to the existing behaviour).
* **Demand increase** — a hypothetical percentage increase in daily demand.
* **Quantity fulfillment reduction** — the vendor delivers only a percentage of
  the committed / required quantity.

Boundaries / guarantees
-----------------------
* **No Streamlit, no boto3, no provider, no LLM, no network / I/O.** This module
  imports none of them.
* **Reads only.** Baseline inputs are read from the deterministic snapshot
  (:class:`~vendor_forecasting_agent.schema.DeterministicResults`) and the
  :class:`~vendor_forecasting_agent.upstream.UpstreamRequest`. Nothing is ever
  mutated — every scenario derives new local scalars and returns a frozen
  :class:`ScenarioResult`. Callers can run any scenario in any order without
  affecting later runs (each call re-reads the baseline fresh).
* **Reuses the EXISTING exposure formula** from
  :mod:`vendor_forecasting_agent.impact` (mirrored here as a single documented
  one-liner, exactly as :mod:`vendor_forecasting_agent.whatif` already does — the
  whole impact stage is never re-run):

      exposure = avg_demand * max(0.0, expected_lead_time - avg_lead_time_days)

  Consistent with ``impact.py`` / ``view_model``:

  * the **exposure** basis is ``metric.avg_demand`` (the deterministic exposure
    basis), and
  * the **inventory coverage / runway** basis is ``request.daily_demand``
    (``current_inventory / daily_demand``, matching
    :func:`ui.view_model.product_context`).

  These are two distinct, pre-existing demand definitions; this module does not
  invent a new one.

Determinism: identical inputs always produce an identical :class:`ScenarioResult`.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .schema import DeterministicResults, Forecast, VendorMetric
from .upstream import UpstreamRequest
from .whatif import analyze_what_if_for_vendor

__all__ = [
    "ScenarioResult",
    "analyze_supplier_delay",
    "analyze_demand_increase",
    "analyze_fulfillment_reduction",
    "build_scenario_context",
]

# Scenario keys (stable identifiers used in ScenarioResult.scenario and context).
SUPPLIER_DELAY = "supplier_delay"
DEMAND_INCREASE = "demand_increase"
FULFILLMENT_REDUCTION = "fulfillment_reduction"


class ScenarioResult(BaseModel):
    """Immutable result of a single deterministic what-if scenario.

    Frozen / ``extra="forbid"`` per project convention. Fields that do not apply
    to a given scenario are ``None`` (nothing is fabricated). All numbers are
    deterministic; the UI is responsible for display rounding/formatting.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    parameter_label: str = Field(min_length=1)
    parameter_value: float

    # baseline inputs (read from snapshot / request)
    baseline_daily_demand: float
    baseline_current_inventory: float
    baseline_expected_lead_time: float
    baseline_avg_lead_time_days: float
    baseline_inventory_coverage_days: Optional[float] = None
    baseline_potential_exposure: float

    # what-if outputs
    whatif_daily_demand: float
    whatif_expected_lead_time: float
    whatif_expected_replenishment_units: Optional[float] = None
    whatif_inventory_coverage_days: Optional[float] = None
    whatif_potential_exposure: float

    # deltas / derived
    coverage_change_days: Optional[float] = None
    exposure_change: float
    shortage_gap_units: Optional[float] = None
    inventory_horizon: Optional[int] = None


# ---- Baseline reader --------------------------------------------------------


class _Baseline:
    """Plain container of the baseline scalars needed by every scenario.

    Not a Pydantic model on purpose — it is an internal, read-only bundle of
    already-validated primitives (no external surface).
    """

    __slots__ = (
        "vendor_id",
        "expected_lead_time",
        "avg_lead_time_days",
        "avg_demand",
        "current_inventory",
        "daily_demand",
        "required_quantity",
        "inventory_horizon",
        "coverage_days",
        "potential_exposure",
    )

    def __init__(
        self,
        *,
        vendor_id: str,
        expected_lead_time: float,
        avg_lead_time_days: float,
        avg_demand: float,
        current_inventory: float,
        daily_demand: float,
        required_quantity: float,
        inventory_horizon: Optional[int],
    ) -> None:
        self.vendor_id = vendor_id
        self.expected_lead_time = expected_lead_time
        self.avg_lead_time_days = avg_lead_time_days
        self.avg_demand = avg_demand
        self.current_inventory = current_inventory
        self.daily_demand = daily_demand
        self.required_quantity = required_quantity
        self.inventory_horizon = inventory_horizon
        # Presentation-only runway ratio (matches view_model.product_context):
        # current_inventory / daily_demand, None when demand <= 0.
        self.coverage_days: Optional[float] = (
            current_inventory / daily_demand if daily_demand > 0 else None
        )
        # EXISTING exposure formula (mirrored from impact.py / whatif.py):
        # avg_demand * max(0, expected_lead_time - avg_lead_time_days).
        self.potential_exposure: float = avg_demand * max(
            0.0, expected_lead_time - avg_lead_time_days
        )


def _read_baseline(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_id: str,
) -> _Baseline:
    """Read the baseline scalars for ``vendor_id`` (reads only; recomputes nothing).

    Mirrors the validation of
    :func:`vendor_forecasting_agent.whatif.analyze_what_if_for_vendor`: raises
    :class:`ValueError` naming the vendor and the missing input when the
    forecast/metric is absent, ``forecast.values`` is empty, or
    ``avg_lead_time_days`` / ``avg_demand`` is ``None``. Nothing is fabricated.

    * ``expected_lead_time`` = ``forecast.values[0]``
    * ``avg_lead_time_days`` / ``avg_demand`` from ``metric`` (exposure basis)
    * ``current_inventory`` / ``daily_demand`` / ``required_quantity`` from the
      request (runway basis)
    * ``inventory_horizon`` = ``inventory[vendor].horizon`` if present, else
      ``forecast.horizon``.
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
    inventory_horizon = (
        inventory.horizon if inventory is not None else forecast.horizon
    )

    return _Baseline(
        vendor_id=vendor_id,
        expected_lead_time=forecast.values[0],
        avg_lead_time_days=metric.avg_lead_time_days,
        avg_demand=metric.avg_demand,
        current_inventory=request.current_inventory,
        daily_demand=request.daily_demand,
        required_quantity=float(request.required_quantity),
        inventory_horizon=inventory_horizon,
    )


# ---- Parameter validation ---------------------------------------------------


def _validate_non_negative_number(value: Any, name: str) -> float:
    """Validate ``value`` is a real, non-bool number ``>= 0`` -> float, else ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got bool {value!r}")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {number}")
    return number


def _validate_percent_0_100(value: Any, name: str) -> float:
    """Validate ``value`` is a real, non-bool number in ``[0, 100]`` -> float."""
    number = _validate_non_negative_number(value, name)
    if number > 100.0:
        raise ValueError(f"{name} must be <= 100.0, got {number}")
    return number


# ---- Scenario 1: supplier delay --------------------------------------------


def analyze_supplier_delay(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_id: str,
    delay_days: float,
) -> ScenarioResult:
    """Deterministic supplier-delay scenario (REUSES the existing delay math).

    ``delay_days`` must be a real number ``>= 0`` (bool / non-number / negative
    raise :class:`ValueError`). The lead-time and exposure math is delegated to
    :func:`vendor_forecasting_agent.whatif.analyze_what_if_for_vendor` so it stays
    byte-identical to the existing supplier-delay behaviour.

    A supply delay does not consume on-hand inventory faster, so the on-hand
    *runway* (``current_inventory / daily_demand``) is unchanged
    (``coverage_change_days = 0.0``). The delay does, however, widen the gap
    between when replenishment arrives and when on-hand runs out; that potential
    shortfall is expressed as ``shortage_gap_units`` =
    ``daily_demand * max(0, new_expected_lead_time - baseline_coverage_days)``
    when coverage is known.
    """
    delay = _validate_non_negative_number(delay_days, "delay_days")
    baseline = _read_baseline(results, request, vendor_id)

    # REUSE the existing delay engine for byte-identical delay math.
    wi = analyze_what_if_for_vendor(results, vendor_id, delay)

    coverage = baseline.coverage_days
    if coverage is not None:
        shortage_gap_units: Optional[float] = baseline.daily_demand * max(
            0.0, wi.new_expected_lead_time - coverage
        )
    else:
        shortage_gap_units = None

    return ScenarioResult(
        vendor_id=vendor_id,
        scenario=SUPPLIER_DELAY,
        parameter_label="Additional delay (days)",
        parameter_value=delay,
        baseline_daily_demand=baseline.daily_demand,
        baseline_current_inventory=baseline.current_inventory,
        baseline_expected_lead_time=baseline.expected_lead_time,
        baseline_avg_lead_time_days=baseline.avg_lead_time_days,
        baseline_inventory_coverage_days=coverage,
        baseline_potential_exposure=wi.original_exposure,
        whatif_daily_demand=baseline.daily_demand,  # unchanged by a supply delay
        whatif_expected_lead_time=wi.new_expected_lead_time,
        whatif_expected_replenishment_units=None,
        # On-hand runway is unchanged by a supply delay.
        whatif_inventory_coverage_days=coverage,
        whatif_potential_exposure=wi.new_exposure,
        coverage_change_days=(0.0 if coverage is not None else None),
        exposure_change=wi.exposure_change,
        shortage_gap_units=shortage_gap_units,
        inventory_horizon=wi.inventory_horizon,
    )


# ---- Scenario 2: demand increase -------------------------------------------


def analyze_demand_increase(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_id: str,
    pct_increase: float,
) -> ScenarioResult:
    """Deterministic demand-increase scenario.

    ``pct_increase`` must be a real number ``>= 0`` (bool / non-number / negative
    raise :class:`ValueError`).

    * ``new_daily_demand = daily_demand * (1 + pct/100)``.
    * Expected lead time is unchanged.
    * Inventory coverage/runway shrinks:
      ``whatif_coverage = current_inventory / new_daily_demand`` (``None`` when
      new demand ``<= 0``).
    * Exposure uses the SAME formula scaled by the exposure basis:
      ``baseline = avg_demand * max(0, expected - avg_lead)``;
      ``whatif = avg_demand * (1 + pct/100) * max(0, expected - avg_lead)``.
    * ``shortage_gap_units = new_daily_demand * max(0, expected_lead_time -
      whatif_coverage)`` when coverage is known, else ``None``.
    """
    pct = _validate_non_negative_number(pct_increase, "pct_increase")
    baseline = _read_baseline(results, request, vendor_id)

    factor = 1.0 + pct / 100.0
    new_daily_demand = baseline.daily_demand * factor

    if new_daily_demand > 0:
        whatif_coverage: Optional[float] = (
            baseline.current_inventory / new_daily_demand
        )
    else:
        whatif_coverage = None

    delay_gap = max(0.0, baseline.expected_lead_time - baseline.avg_lead_time_days)
    baseline_exposure = baseline.avg_demand * delay_gap
    whatif_exposure = baseline.avg_demand * factor * delay_gap
    exposure_change = whatif_exposure - baseline_exposure

    if (
        baseline.coverage_days is not None
        and whatif_coverage is not None
    ):
        coverage_change_days: Optional[float] = (
            whatif_coverage - baseline.coverage_days
        )
    else:
        coverage_change_days = None

    if whatif_coverage is not None:
        shortage_gap_units: Optional[float] = new_daily_demand * max(
            0.0, baseline.expected_lead_time - whatif_coverage
        )
    else:
        shortage_gap_units = None

    return ScenarioResult(
        vendor_id=vendor_id,
        scenario=DEMAND_INCREASE,
        parameter_label="Demand increase (%)",
        parameter_value=pct,
        baseline_daily_demand=baseline.daily_demand,
        baseline_current_inventory=baseline.current_inventory,
        baseline_expected_lead_time=baseline.expected_lead_time,
        baseline_avg_lead_time_days=baseline.avg_lead_time_days,
        baseline_inventory_coverage_days=baseline.coverage_days,
        baseline_potential_exposure=baseline_exposure,
        whatif_daily_demand=new_daily_demand,
        whatif_expected_lead_time=baseline.expected_lead_time,  # unchanged
        whatif_expected_replenishment_units=None,
        whatif_inventory_coverage_days=whatif_coverage,
        whatif_potential_exposure=whatif_exposure,
        coverage_change_days=coverage_change_days,
        exposure_change=exposure_change,
        shortage_gap_units=shortage_gap_units,
        inventory_horizon=baseline.inventory_horizon,
    )


# ---- Scenario 3: quantity fulfillment reduction ----------------------------


def analyze_fulfillment_reduction(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_id: str,
    fulfillment_pct: float,
) -> ScenarioResult:
    """Deterministic quantity-fulfillment-reduction scenario.

    ``fulfillment_pct`` must be a real number in ``[0, 100]`` (bool / non-number /
    ``< 0`` / ``> 100`` raise :class:`ValueError`). The vendor delivers only
    ``fulfillment_pct`` percent of the committed / required quantity:

    * ``whatif_expected_replenishment_units = required_quantity * (fulfillment_pct/100)``.
    * ``reduced_incoming = required_quantity * (1 - fulfillment_pct/100)`` — the
      committed units NOT delivered. This is the potential shortfall, exposed as
      ``shortage_gap_units = max(0, reduced_incoming)``.
    * Expected lead time and daily demand are unchanged.

    Exposure: to keep the exposure figure deterministic and comparable, the
    POTENTIAL exposure from the shortfall is the undelivered committed quantity
    at risk, so ``whatif_potential_exposure = baseline_exposure + reduced_incoming``
    and ``exposure_change = reduced_incoming``. This is a POTENTIAL planning
    signal (undelivered committed quantity), NOT a guaranteed loss.

    ``coverage_change_days`` is ``None``: the fulfillment of a future incoming
    order does not change the on-hand runway basis, so no runway delta is claimed
    (documented choice — kept simple and deterministic).
    """
    fulfillment = _validate_percent_0_100(fulfillment_pct, "fulfillment_pct")
    baseline = _read_baseline(results, request, vendor_id)

    fraction = fulfillment / 100.0
    expected_replenishment = baseline.required_quantity * fraction
    reduced_incoming = baseline.required_quantity * (1.0 - fraction)
    shortage_gap_units = max(0.0, reduced_incoming)

    whatif_exposure = baseline.potential_exposure + reduced_incoming
    exposure_change = reduced_incoming

    return ScenarioResult(
        vendor_id=vendor_id,
        scenario=FULFILLMENT_REDUCTION,
        parameter_label="Fulfillment (%)",
        parameter_value=fulfillment,
        baseline_daily_demand=baseline.daily_demand,
        baseline_current_inventory=baseline.current_inventory,
        baseline_expected_lead_time=baseline.expected_lead_time,
        baseline_avg_lead_time_days=baseline.avg_lead_time_days,
        baseline_inventory_coverage_days=baseline.coverage_days,
        baseline_potential_exposure=baseline.potential_exposure,
        whatif_daily_demand=baseline.daily_demand,  # unchanged
        whatif_expected_lead_time=baseline.expected_lead_time,  # unchanged
        whatif_expected_replenishment_units=expected_replenishment,
        whatif_inventory_coverage_days=baseline.coverage_days,  # runway unchanged
        whatif_potential_exposure=whatif_exposure,
        coverage_change_days=None,
        exposure_change=exposure_change,
        shortage_gap_units=shortage_gap_units,
        inventory_horizon=baseline.inventory_horizon,
    )


# ---- LLM context builder ----------------------------------------------------


def _product_block(request: UpstreamRequest) -> dict[str, Any]:
    """Read-only product / component identity block (straight from the request)."""
    return {
        "product_id": request.product_id,
        "product_name": request.product_name,
        "component_id": request.component_id,
        "component_name": request.component_name,
        "required_quantity": request.required_quantity,
    }


def _inventory_block(request: UpstreamRequest) -> dict[str, Any]:
    """Read-only inventory block (runway basis) straight from the request."""
    coverage_days: Optional[float] = (
        request.current_inventory / request.daily_demand
        if request.daily_demand > 0
        else None
    )
    return {
        "current_inventory": request.current_inventory,
        "daily_demand": request.daily_demand,
        "inventory_coverage_days": coverage_days,
    }


def build_scenario_context(
    result: ScenarioResult,
    request: Optional[UpstreamRequest] = None,
) -> dict[str, Any]:
    """Reflect a :class:`ScenarioResult` EXACTLY into a JSON-serializable dict.

    Every numeric field equals the corresponding ``ScenarioResult`` field
    verbatim (read directly, never recomputed). This is exactly what the LLM
    receives — it contains the already-computed facts (baseline, what-if, and the
    changes) plus the scenario label, and NEVER a raw prompt. When ``request`` is
    supplied, read-only ``product`` / ``inventory`` context blocks are added; they
    do not affect any what-if number. Deterministic and sorted-friendly.
    """
    context: dict[str, Any] = {
        "scenario": result.scenario,
        "vendor_id": result.vendor_id,
        "parameter": {
            "label": result.parameter_label,
            "value": result.parameter_value,
        },
        "baseline": {
            "daily_demand": result.baseline_daily_demand,
            "current_inventory": result.baseline_current_inventory,
            "expected_lead_time": result.baseline_expected_lead_time,
            "avg_lead_time_days": result.baseline_avg_lead_time_days,
            "inventory_coverage_days": result.baseline_inventory_coverage_days,
            "potential_exposure": result.baseline_potential_exposure,
        },
        "whatif": {
            "daily_demand": result.whatif_daily_demand,
            "expected_lead_time": result.whatif_expected_lead_time,
            "expected_replenishment_units": result.whatif_expected_replenishment_units,
            "inventory_coverage_days": result.whatif_inventory_coverage_days,
            "potential_exposure": result.whatif_potential_exposure,
        },
        "changes": {
            "coverage_change_days": result.coverage_change_days,
            "exposure_change": result.exposure_change,
            "shortage_gap_units": result.shortage_gap_units,
        },
        "inventory_horizon": result.inventory_horizon,
        "notes": (
            "Potential exposure is a planning signal (potential undelivered / "
            "at-risk units), not a guaranteed loss."
        ),
    }
    if request is not None:
        context["product"] = _product_block(request)
        context["inventory"] = _inventory_block(request)
    return context
