"""Structured LLM context builder for the Vendor Forecasting Agent (llm_context.py).

This is **Layer 2**: a deterministic, side-effect-free *translation* layer that
turns the already-computed deterministic results (Layer 1) into plain, nested,
JSON-serializable dictionaries of *facts* that an LLM prompt layer (Layer 3, not
implemented here) can consume.

Boundaries / guarantees
-----------------------
* **No Streamlit, no boto3, no provider, no LLM.** This module imports none of
  them and invokes nothing external. It performs no network / I/O.
* **Computes NO business values.** It never recomputes risk, forecast, exposure,
  recommendation, or what-if numbers. Every domain number is *read* from the
  existing :class:`~vendor_forecasting_agent.schema.DeterministicResults`
  snapshot (via the read-only ``ui.view_model`` helpers so values match the UI
  byte-for-byte) or from an existing
  :class:`~vendor_forecasting_agent.whatif.WhatIfResult`.
* The **only** arithmetic performed here is presentation-level:
  * the inventory *coverage-days* ratio (``current_inventory / daily_demand``),
    which already lives in :func:`ui.view_model.product_context` and is reused
    from there — not reimplemented;
  * presentation **counting** (risk-level distribution, vendors analyzed);
  * presentation **aggregation** (summing existing ``projected_units``);
  * presentation **selection** (choosing the highest/lowest-risk vendor by the
    EXISTING deterministic ``risk_score`` — this selects an existing result, it
    does not compute a new risk value).
* Lists are emitted in a **stable order** (sorted by ``vendor_id``).
* Missing / optional values are represented with ``None`` (or ``[]`` for empty
  driver lists) — nothing is fabricated.

Reuse of ``ui.view_model``
--------------------------
``ui.view_model`` is itself Streamlit-free (verified: it imports only ``typing``,
``schema`` and ``upstream``). Reusing :func:`ui.view_model.selected_vendor_view`,
:func:`ui.view_model.vendor_risk_overview` and :func:`ui.view_model.product_context`
here guarantees the context numbers are identical to what the UI shows.
"""

from typing import Any, Optional

from .schema import DeterministicResults, ExplanationResult
from .upstream import UpstreamRequest
from .ui import view_model
from .whatif import WhatIfResult

__all__ = [
    "build_vendor_context",
    "build_comparison_context",
    "build_executive_context",
    "build_what_if_context",
]

# A neutral, non-valid, degraded explanation used ONLY to source the
# deterministic metric/risk/inventory fields from ``selected_vendor_view`` when
# the caller passes no explanation. ``selected_vendor_view`` reads the snapshot
# metrics regardless of explanation validity, so this changes no number; it just
# lets us reuse that helper. The recommendation explanation text is set to
# ``None`` in this case (never fabricated).
_NEUTRAL_EXPLANATION = ExplanationResult(text=None, is_valid=False, degraded=True)


def _product_block(request: UpstreamRequest) -> dict[str, Any]:
    """Product / component identity block (reads straight from the request)."""
    return {
        "product_id": request.product_id,
        "product_name": request.product_name,
        "component_id": request.component_id,
        "component_name": request.component_name,
        "required_quantity": request.required_quantity,
    }


def _inventory_block(request: UpstreamRequest) -> dict[str, Any]:
    """Inventory block reusing the presentation coverage-days ratio.

    Reuses :func:`ui.view_model.product_context` (the single place the
    coverage-days ratio lives) so the value matches the UI exactly.
    """
    ctx = view_model.product_context(request)
    return {
        "current_inventory": ctx["current_inventory"],
        "daily_demand": ctx["daily_demand"],
        "inventory_coverage_days": ctx["inventory_coverage_days"],
    }


def build_vendor_context(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_id: str,
    explanation: Optional[ExplanationResult] = None,
) -> Optional[dict[str, Any]]:
    """Structured context for ONE vendor, reading existing results only.

    Returns ``None`` when ``vendor_id`` has no recommendation in ``results``
    (mirrors :func:`ui.view_model.selected_vendor_view` returning ``None`` for a
    missing vendor), so callers can detect an unknown vendor. Product/inventory
    facts remain available via :func:`build_comparison_context` /
    :func:`build_executive_context` (or the caller's own request).

    All domain numbers are sourced from
    :func:`ui.view_model.selected_vendor_view` so they are identical to the UI.
    When ``explanation`` is ``None`` a neutral degraded default is used to source
    metrics (the helper reads snapshot metrics regardless of explanation
    validity); the recommendation ``explanation`` text is then ``None``.

    The recommendation ``explanation`` text is included **only** when an
    :class:`ExplanationResult` is supplied and it is valid and not degraded;
    otherwise it is ``None`` (never fabricated).
    """
    source_explanation = explanation if explanation is not None else _NEUTRAL_EXPLANATION
    view = view_model.selected_vendor_view(
        results, source_explanation, vendor_id, request
    )
    if view is None:
        # No recommendation for this vendor -> unknown/missing vendor.
        return None

    explanation_text: Optional[str] = None
    if (
        explanation is not None
        and explanation.is_valid
        and not explanation.degraded
        and explanation.text
    ):
        explanation_text = explanation.text

    return {
        "product": _product_block(request),
        "inventory": {
            "current_inventory": view["current_inventory"],
            "daily_demand": view["daily_demand"],
            "inventory_coverage_days": view["inventory_coverage_days"],
        },
        "vendor": {
            # vendor_id IS the display name in the curated scenarios; if no nicer
            # name is available, fall back to vendor_id.
            "vendor_id": vendor_id,
            "vendor_name": vendor_id,
        },
        "risk": {
            "risk_level": view["risk_level"],
            "risk_score": view["risk_score"],
            "risk_drivers": list(view["main_risk_drivers"]),
        },
        "metrics": {
            "on_time_rate": view["on_time_rate"],
            "avg_lead_time_days": view["avg_lead_time_days"],
            "expected_lead_time": view["expected_lead_time"],
            "quantity_fulfillment_rate": view["quantity_fulfillment_rate"],
            "capacity_utilization": view["capacity_utilization"],
            "allocation_ratio": view["allocation_ratio"],
            "defect_rate": view["defect_rate"],
        },
        "inventory_impact": {
            "current_inventory": view["current_inventory"],
            "inventory_coverage_days": view["inventory_coverage_days"],
            "expected_lead_time": view["expected_lead_time"],
            "potential_delay_days": view["potential_delay_days"],
            "potential_exposure_units": view["potential_exposure_units"],
        },
        "recommendation": {
            "action": view["recommended_action"],
            "score": _recommendation_score(results, vendor_id),
            "explanation": explanation_text,
        },
    }


def _recommendation_score(
    results: DeterministicResults, vendor_id: str
) -> Optional[float]:
    """Read the recommendation ``score`` for ``vendor_id`` (``None`` if absent)."""
    rec = next(
        (r for r in results.recommendations if r.vendor_id == vendor_id),
        None,
    )
    return rec.score if rec is not None else None


def _recommended_action(
    results: DeterministicResults, vendor_id: str
) -> Optional[str]:
    """Read the recommended ``action`` for ``vendor_id`` (``None`` if absent)."""
    rec = next(
        (r for r in results.recommendations if r.vendor_id == vendor_id),
        None,
    )
    return rec.action if rec is not None else None


def _comparable_vendor_entry(
    results: DeterministicResults, vendor_id: str
) -> dict[str, Any]:
    """One comparable per-vendor entry, every value read from the snapshot.

    Sources risk fields + expected lead time from
    :func:`ui.view_model.vendor_risk_overview` (matched by vendor id), the
    exposure ``projected_units`` from ``results.inventory``, and the recommended
    action from ``results.recommendations``. Recomputes nothing.
    """
    overview_row = next(
        (
            row
            for row in view_model.vendor_risk_overview(results)
            if row["Vendor"] == vendor_id
        ),
        None,
    )
    risk = results.risk.get(vendor_id)
    inventory = results.inventory.get(vendor_id)

    if overview_row is not None:
        risk_level = overview_row["Risk Level"]
        risk_score = overview_row["Risk Score"]
        expected_lead_time = overview_row["Expected Lead Time"]
    else:
        risk_level = risk.risk_level if risk is not None else None
        risk_score = risk.risk_score if risk is not None else None
        forecast = results.forecasts.get(vendor_id)
        expected_lead_time = (
            forecast.values[0] if forecast is not None and forecast.values else None
        )

    return {
        "vendor_id": vendor_id,
        "vendor_name": vendor_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_drivers": (list(risk.main_risk_drivers) if risk is not None else []),
        "expected_lead_time": expected_lead_time,
        "potential_exposure_units": (
            inventory.projected_units if inventory is not None else None
        ),
        "recommended_action": _recommended_action(results, vendor_id),
    }


def _recommended_vendor_ids(results: DeterministicResults) -> list[str]:
    """Recommended vendor ids present in the snapshot, sorted by vendor_id."""
    return sorted(rec.vendor_id for rec in results.recommendations)


def build_comparison_context(
    results: DeterministicResults,
    request: UpstreamRequest,
    vendor_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Comparable structured info across vendors for LLM comparison questions.

    Does **not** compute any ranking — the deterministic recommendation/risk
    results are authoritative. Vendors are emitted sorted by ``vendor_id`` purely
    as stable presentation ordering (not a risk ranking).

    When ``vendor_ids`` is ``None`` all vendors that have a recommendation in
    ``results`` are used (sorted). Only vendors present in the snapshot are
    included.
    """
    if vendor_ids is None:
        selected = _recommended_vendor_ids(results)
    else:
        present = set(_recommended_vendor_ids(results))
        selected = sorted(v for v in vendor_ids if v in present)

    return {
        "product": _product_block(request),
        "inventory": _inventory_block(request),
        "vendors": [
            _comparable_vendor_entry(results, vendor_id) for vendor_id in selected
        ],
    }


def build_executive_context(
    results: DeterministicResults,
    request: UpstreamRequest,
) -> dict[str, Any]:
    """Overall structured representation for an executive summary.

    Derived ONLY from deterministic output — no competing assessment. The
    ``summary`` block contains presentation-level counting / aggregation /
    selection over EXISTING values:

    * ``vendors_analyzed`` — number of recommendations.
    * ``risk_level_distribution`` — counts of existing ``risk_level`` values.
    * ``highest_risk`` / ``lowest_risk`` — the vendor **selected** by the max /
      min EXISTING deterministic ``risk_score`` (ties broken by ``vendor_id``
      ascending). This selects an existing result; it does NOT recompute risk.
    * ``total_potential_exposure_units`` — sum of existing ``projected_units``.

    ``vendors`` reuses the same comparable per-vendor entries as
    :func:`build_comparison_context`.
    """
    selected = _recommended_vendor_ids(results)

    distribution = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    for risk in results.risk.values():
        if risk.risk_level in distribution:
            distribution[risk.risk_level] += 1

    # Select highest / lowest-risk vendors by the EXISTING risk_score, ties
    # broken by vendor_id ascending. Selection over existing values only.
    scored = [
        (results.risk[v].risk_score, v)
        for v in selected
        if v in results.risk
    ]

    highest_risk: Optional[dict[str, Any]] = None
    lowest_risk: Optional[dict[str, Any]] = None
    if scored:
        # max risk_score, tie -> smallest vendor_id.
        _, highest_vendor = max(scored, key=lambda pair: (pair[0], _neg_key(pair[1])))
        # min risk_score, tie -> smallest vendor_id.
        _, lowest_vendor = min(scored, key=lambda pair: (pair[0], pair[1]))
        highest_risk = _risk_ref(results, highest_vendor)
        lowest_risk = _risk_ref(results, lowest_vendor)

    total_exposure = sum(
        results.inventory[v].projected_units
        for v in selected
        if v in results.inventory
    )

    return {
        "product": _product_block(request),
        "inventory": _inventory_block(request),
        "summary": {
            "vendors_analyzed": len(results.recommendations),
            "risk_level_distribution": distribution,
            "highest_risk": highest_risk,
            "lowest_risk": lowest_risk,
            "total_potential_exposure_units": total_exposure,
        },
        "vendors": [
            _comparable_vendor_entry(results, vendor_id) for vendor_id in selected
        ],
    }


class _NegStr:
    """Helper wrapper so ``max`` breaks ties toward the smallest ``vendor_id``.

    For ``max`` we want the highest risk_score but, on a tie, the *smallest*
    vendor_id. Wrapping the vendor_id so that a lexicographically smaller string
    compares as *greater* lets a single ``max`` express both.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: "_NegStr") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NegStr) and self.value == other.value


def _neg_key(value: str) -> _NegStr:
    return _NegStr(value)


def _risk_ref(results: DeterministicResults, vendor_id: str) -> dict[str, Any]:
    """A small ``{vendor_id, risk_level, risk_score}`` reference for a vendor."""
    risk = results.risk.get(vendor_id)
    return {
        "vendor_id": vendor_id,
        "risk_level": risk.risk_level if risk is not None else None,
        "risk_score": risk.risk_score if risk is not None else None,
    }


def build_what_if_context(
    what_if: WhatIfResult,
    request: Optional[UpstreamRequest] = None,
) -> dict[str, Any]:
    """Reflect a Layer-1 :class:`WhatIfResult` EXACTLY (no additional arithmetic).

    Every numeric field equals the corresponding ``WhatIfResult`` field verbatim
    (read directly, never recomputed). When ``request`` is supplied, read-only
    ``product`` / ``inventory`` blocks are added for context; they do not affect
    the what-if numbers.
    """
    context: dict[str, Any] = {
        "vendor_id": what_if.vendor_id,
        "delay_days": what_if.delay_days,
        "original_expected_lead_time": what_if.original_expected_lead_time,
        "new_expected_lead_time": what_if.new_expected_lead_time,
        "original_exposure": what_if.original_exposure,
        "new_exposure": what_if.new_exposure,
        "exposure_change": what_if.exposure_change,
        "inventory_horizon": what_if.inventory_horizon,
    }
    if request is not None:
        context["product"] = _product_block(request)
        context["inventory"] = _inventory_block(request)
    return context
