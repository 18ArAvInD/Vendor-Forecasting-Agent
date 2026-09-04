"""Presentation-only data shaping for the Streamlit UI (view_model.py).

This module intentionally has **no** ``streamlit`` import and **no** business
logic. It only *reads* fields off an already-computed
:class:`~vendor_forecasting_agent.schema.DeterministicResults` snapshot (and an
:class:`~vendor_forecasting_agent.schema.ExplanationResult`) and reshapes them
into plain Python structures (dicts / lists of dicts) for display.

Every number displayed by the UI comes from ``run_demo(...)`` (which internally
runs ``run_pipeline`` + ``explain(provider=None)``). Nothing here recomputes any
risk / forecast / inventory / recommendation value — the only arithmetic is
presentation-level counting (mirroring what ``demo.format_report`` already does):
counting recommendations, counting elevated-risk vendors, and counting vendors
with inventory exposure.

Keeping these helpers free of ``streamlit`` makes them importable and unit
testable without streamlit installed. ``app.py`` imports these helpers and only
handles the Streamlit widgets/layout.
"""

from typing import Any, Optional

from ..schema import DeterministicResults, ExplanationResult
from ..upstream import UpstreamRequest

# Friendly, human-readable labels for the fixed rule ids (see ``rules.py``). Used
# only to turn an EXISTING triggered ``main_risk_drivers`` rule id into a short
# business phrase for the "Main Risk Issue" column. This is presentation labeling
# of existing signals — it computes no new score and changes no risk value.
_RISK_ISSUE_LABELS: dict[str, str] = {
    "on_time_delivery_low": "Late deliveries",
    "lead_time_declining": "Lead-time deterioration",
    "quantity_fulfillment_low": "Under-fulfilment of orders",
    "capacity_utilization_high": "Capacity pressure",
    "allocation_low": "Low allocation commitment",
    "defect_rate_high": "Quality / defect issues",
}

# Shown when a vendor has no triggered risk drivers at all.
_NO_RISK_ISSUE_LABEL = "Stable lead time"

# Risk levels considered "elevated" for the summary section. Mirrors demo.py's
# ``_ELEVATED_RISK_LEVELS`` so the UI summary matches the console report.
ELEVATED_RISK_LEVELS: frozenset[str] = frozenset({"high", "critical"})


def summary_metrics(snapshot: DeterministicResults) -> dict[str, int]:
    """Return presentation-only summary counts read from ``snapshot``.

    Mirrors the summary section of ``demo.format_report`` (counting only, no
    domain recomputation):

    * ``vendors_analyzed`` — number of recommendations.
    * ``high_critical`` — number of risk results whose ``risk_level`` is in
      :data:`ELEVATED_RISK_LEVELS`.
    * ``with_inventory_exposure`` — number of inventory impacts whose
      ``projected_units`` is greater than 0.

    An empty snapshot yields all-zero counts.
    """
    vendors_analyzed = len(snapshot.recommendations)
    high_critical = sum(
        1
        for risk in snapshot.risk.values()
        if risk.risk_level in ELEVATED_RISK_LEVELS
    )
    with_inventory_exposure = sum(
        1
        for inventory in snapshot.inventory.values()
        if inventory.projected_units > 0
    )
    return {
        "vendors_analyzed": vendors_analyzed,
        "high_critical": high_critical,
        "with_inventory_exposure": with_inventory_exposure,
    }


def _expected_lead_time(
    snapshot: DeterministicResults, vendor_id: str
) -> Optional[float]:
    """Read a vendor's expected lead time = first forecast value, else ``None``.

    Reads straight from ``snapshot.forecasts`` (no recomputation), matching the
    console report's "Expected lead time" field.
    """
    forecast = snapshot.forecasts.get(vendor_id)
    if forecast is not None and forecast.values:
        return forecast.values[0]
    return None


def _projected_units(snapshot: DeterministicResults, vendor_id: str) -> float:
    """Read a vendor's projected units from inventory, else ``0.0``.

    Reads straight from ``snapshot.inventory`` (no recomputation), matching the
    console report's "Projected units" field.
    """
    inventory = snapshot.inventory.get(vendor_id)
    if inventory is not None:
        return inventory.projected_units
    return 0.0


def risk_table_rows(snapshot: DeterministicResults) -> list[dict[str, Any]]:
    """Build one presentation row per recommendation, sorted by ``vendor_id``.

    Each row is a plain dict with values read directly from the snapshot's
    ``recommendations`` / ``risk`` / ``forecasts`` / ``inventory`` — no
    computation, just reads. Missing per-vendor entries degrade gracefully
    (``None`` risk fields, ``None`` expected lead time, ``0.0`` projected units),
    mirroring ``demo.format_report``.
    """
    rows: list[dict[str, Any]] = []
    for rec in sorted(snapshot.recommendations, key=lambda r: r.vendor_id):
        vendor_id = rec.vendor_id
        risk = snapshot.risk.get(vendor_id)
        rows.append(
            {
                "Vendor ID": vendor_id,
                "Risk score": risk.risk_score if risk is not None else None,
                "Risk level": risk.risk_level if risk is not None else None,
                "Recommended action": rec.action,
                "Expected lead time": _expected_lead_time(snapshot, vendor_id),
                "Projected units": _projected_units(snapshot, vendor_id),
            }
        )
    return rows


def vendor_detail(
    snapshot: DeterministicResults,
    explanation: ExplanationResult,
    vendor_id: str,
) -> Optional[dict[str, Any]]:
    """Return the detail fields for one vendor, or ``None`` if not recommended.

    Reads a single vendor's risk / forecast / inventory / recommendation values
    straight from ``snapshot`` plus the shared ``explanation`` text. Recomputes
    nothing. Returns ``None`` when ``vendor_id`` has no recommendation (graceful
    "not found" handling for empty/unknown-vendor cases).

    The returned dict includes:

    * ``vendor_id``
    * ``risk_score`` / ``risk_level`` / ``confidence`` / ``main_risk_drivers``
      (``None`` / empty when the vendor has no risk result)
    * ``expected_lead_time`` (first forecast value or ``None``)
    * ``projected_units`` (inventory exposure, ``0.0`` when none)
    * ``recommended_action``
    * ``explanation_text`` — the full ``explanation.text`` (covers all vendors)
    * ``explanation_available`` — ``False`` when the explanation is degraded or
      not valid (the deterministic fields above are still populated).
    """
    rec = next(
        (r for r in snapshot.recommendations if r.vendor_id == vendor_id),
        None,
    )
    if rec is None:
        return None

    risk = snapshot.risk.get(vendor_id)
    explanation_available = bool(
        explanation.is_valid
        and not explanation.degraded
        and explanation.text
    )

    return {
        "vendor_id": vendor_id,
        "risk_score": risk.risk_score if risk is not None else None,
        "risk_level": risk.risk_level if risk is not None else None,
        "confidence": risk.confidence if risk is not None else None,
        "main_risk_drivers": (
            list(risk.main_risk_drivers) if risk is not None else []
        ),
        "expected_lead_time": _expected_lead_time(snapshot, vendor_id),
        "projected_units": _projected_units(snapshot, vendor_id),
        "recommended_action": rec.action,
        "explanation_text": explanation.text,
        "explanation_available": explanation_available,
    }


def vendor_ids(snapshot: DeterministicResults) -> list[str]:
    """Return the recommended vendor ids sorted for a stable selectbox order."""
    return sorted(rec.vendor_id for rec in snapshot.recommendations)


# ---- Business-oriented product/vendor views (presentation-only) -------------


def product_context(request: UpstreamRequest) -> dict[str, Any]:
    """Shape the product/component context for display from an ``UpstreamRequest``.

    Reads fields straight off the request. The only arithmetic is a
    presentation-only coverage ratio:

        ``inventory_coverage_days = current_inventory / daily_demand``

    (``None`` when ``daily_demand`` is 0 to avoid division by zero). This is a
    plain display figure showing how many days of demand the current on-hand
    inventory covers — it is NOT the inventory-impact exposure formula (that lives
    in ``impact.py`` and is read from the snapshot elsewhere).
    """
    coverage_days: Optional[float]
    if request.daily_demand > 0:
        coverage_days = request.current_inventory / request.daily_demand
    else:
        coverage_days = None

    return {
        "product_id": request.product_id,
        "product_name": request.product_name,
        "component_id": request.component_id,
        "component_name": request.component_name,
        "required_quantity": request.required_quantity,
        "current_inventory": request.current_inventory,
        "daily_demand": request.daily_demand,
        "inventory_coverage_days": coverage_days,
    }


def relevant_vendors(request: UpstreamRequest) -> list[str]:
    """Return the relevant vendor display names from the request (order preserved)."""
    return list(request.relevant_vendors)


def _main_risk_issue(drivers: list[str]) -> str:
    """Map a vendor's triggered risk drivers to a short human "Main Risk Issue".

    Presentation labeling only: takes the FIRST driver rule id from the EXISTING
    ``main_risk_drivers`` (already ordered by the fixed rule order) and returns a
    friendly phrase; returns a "stable" label when there are no triggered drivers.
    Computes no score and changes no risk value.
    """
    if not drivers:
        return _NO_RISK_ISSUE_LABEL
    first = drivers[0]
    return _RISK_ISSUE_LABELS.get(first, first)


def vendor_risk_overview(snapshot: DeterministicResults) -> list[dict[str, Any]]:
    """Build a per-vendor risk comparison table, sorted by vendor id (display name).

    One row per recommended vendor, every value read straight from the snapshot
    (no recomputation):

    * ``Vendor`` — the vendor id (which is the display name in the curated
      scenario).
    * ``Risk Level`` / ``Risk Score`` — from ``snapshot.risk`` (``None`` when a
      vendor has no risk result).
    * ``Expected Lead Time`` — first forecast value from ``snapshot.forecasts``
      (``None`` when absent).
    * ``Main Risk Issue`` — a friendly label derived from the vendor's EXISTING
      ``main_risk_drivers`` (see :func:`_main_risk_issue`).
    """
    rows: list[dict[str, Any]] = []
    for rec in sorted(snapshot.recommendations, key=lambda r: r.vendor_id):
        vendor_id = rec.vendor_id
        risk = snapshot.risk.get(vendor_id)
        drivers = list(risk.main_risk_drivers) if risk is not None else []
        rows.append(
            {
                "Vendor": vendor_id,
                "Risk Level": risk.risk_level if risk is not None else None,
                "Risk Score": risk.risk_score if risk is not None else None,
                "Expected Lead Time": _expected_lead_time(snapshot, vendor_id),
                "Main Risk Issue": _main_risk_issue(drivers),
            }
        )
    return rows


def selected_vendor_view(
    snapshot: DeterministicResults,
    explanation: ExplanationResult,
    vendor_id: str,
    request: UpstreamRequest,
) -> Optional[dict[str, Any]]:
    """Full business detail view for one selected vendor, or ``None`` if not found.

    Combines the existing :func:`vendor_detail` fields with the supporting
    per-vendor metrics (read from ``snapshot.metrics``), the forecast expected
    lead time (from ``snapshot.forecasts``), the inventory exposure
    ``projected_units`` (read from ``snapshot.inventory`` — NOT recomputed), and
    the product-level inventory fields carried in from the ``request``
    (``current_inventory`` / ``daily_demand`` / presentation-only coverage days).

    Recomputes no business value. Returns ``None`` when ``vendor_id`` has no
    recommendation (graceful "not found").
    """
    detail = vendor_detail(snapshot, explanation, vendor_id)
    if detail is None:
        return None

    metric = snapshot.metrics.get(vendor_id)
    expected_lead_time = detail["expected_lead_time"]
    avg_lead_time = metric.avg_lead_time_days if metric is not None else None

    # Presentation-only "potential delay" for display: expected minus normal lead
    # time, floored at 0. The exposure UNITS themselves come from the snapshot
    # (projected_units), not recomputed here.
    if expected_lead_time is not None and avg_lead_time is not None:
        potential_delay_days = max(0.0, expected_lead_time - avg_lead_time)
    else:
        potential_delay_days = None

    coverage_days: Optional[float]
    if request.daily_demand > 0:
        coverage_days = request.current_inventory / request.daily_demand
    else:
        coverage_days = None

    view = dict(detail)
    view.update(
        {
            # Supporting metrics read straight from snapshot.metrics.
            "on_time_rate": metric.avg_on_time_rate if metric is not None else None,
            "avg_lead_time_days": avg_lead_time,
            "quantity_fulfillment_rate": (
                metric.avg_quantity_fulfillment_rate if metric is not None else None
            ),
            "capacity_utilization": (
                metric.avg_capacity_utilization if metric is not None else None
            ),
            "allocation_ratio": (
                metric.avg_allocation_ratio if metric is not None else None
            ),
            "defect_rate": metric.avg_defect_rate if metric is not None else None,
            # Inventory context (read/derived for display only).
            "current_inventory": request.current_inventory,
            "daily_demand": request.daily_demand,
            "inventory_coverage_days": coverage_days,
            "potential_delay_days": potential_delay_days,
            # exposure units sourced from the snapshot (impact.py), not recomputed.
            "potential_exposure_units": detail["projected_units"],
            "main_risk_issue": _main_risk_issue(detail["main_risk_drivers"]),
        }
    )
    return view
