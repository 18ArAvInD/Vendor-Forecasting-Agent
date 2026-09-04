"""Vendor recommendations for the Vendor Forecasting Agent (recommend.py).

This module is the ``Recommendation_Module`` described in the design. It provides
one deterministic, plain-Python function (Task 10):

* :func:`recommend_vendors` — produce exactly one :class:`Recommendation` per
  input vendor from the vendor's Task 8 ``VendorRiskResult`` and its optional
  Task 9 ``InventoryImpact`` (Requirement 10).

Scope (Task 10 only): simple deterministic Python. This is a hackathon MVP — a
fixed rule mapping risk level + inventory exposure onto the ``Recommendation``
action. It **reuses** the upstream Task 8 (risk) and Task 9 (inventory impact)
outputs and recomputes nothing (no risk, metrics, forecasts, or inventory
recomputation). There is no ML/optimization/LLM/scenario/complex rule engine,
no pipeline or LangGraph logic here, no new dependencies, and no network/LLM/I/O.

Fixed action mapping (MVP), where "meaningful inventory impact" means the
vendor's ``InventoryImpact.projected_units > 0`` (a vendor with no
``InventoryImpact`` is treated as no exposure / ``projected_units`` 0):

    - risk_level == "critical" AND projected_units > 0            -> "replace"
    - risk_level == "critical" AND not meaningful                 -> "review"
    - risk_level == "high"                                        -> "review"
    - risk_level == "moderate"                                    -> "review"
    - risk_level == "low" AND projected_units > 0                 -> "maintain"
    - risk_level == "low" AND (projected_units == 0 / no impact)  -> "prefer"

``Recommendation.score`` is the vendor's ``risk_score`` carried through
unchanged. ``rationale_keys`` are deterministic codes: the vendor's
``main_risk_drivers`` (order preserved) plus ``"inventory_exposure"`` appended
last when ``projected_units > 0``. No free prose.

Design constraints honored here (Requirement 10):

* **10.1 / 10.3** — Exactly one Recommendation per vendor in ``risk_by_vendor``;
  score carried through unchanged; rationale keys derived from the inputs.
* **10.2 / 10.4** — Inventory impact is OPTIONAL: a vendor with risk but no
  ``InventoryImpact`` still yields a Recommendation (treated as no exposure);
  empty input yields an empty successful result.
* **10.5** — Deterministic: vendors iterated in sorted ``vendor_id`` order,
  ``data`` is a list ordered by ``vendor_id``, ``errors`` in sorted order;
  byte-identical for identical inputs.
* **10.6** — Every Recommendation conforms to the Schema model; a candidate
  failing schema validation is rejected with a descriptive validation error.
* **10.7** — Pure local computation: no network, LLM, or external API calls.
"""

from typing import Mapping

from pydantic import ValidationError

from .schema import (
    InventoryImpact,
    Recommendation,
    StageError,
    StageResult,
    VendorRiskResult,
)


# ---- Small private helpers --------------------------------------------------


def _action_for(risk_level: str, projected_units: float) -> str:
    """Map ``risk_level`` + inventory exposure onto the fixed action (Req 10.2).

    "Meaningful inventory impact" means ``projected_units > 0``.
    """
    if risk_level == "critical":
        return "replace" if projected_units > 0 else "review"
    if risk_level == "high":
        return "review"
    if risk_level == "moderate":
        return "review"
    # risk_level == "low"
    return "maintain" if projected_units > 0 else "prefer"


def recommend_vendors(
    risk_by_vendor: Mapping[str, VendorRiskResult],
    inventory_by_vendor: Mapping[str, InventoryImpact],
) -> StageResult:
    """Produce one Recommendation per vendor from its risk + inventory impact.

    The design's ``StageResult[list[Recommendation]]`` notation is conceptual:
    the concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On success ``data`` is a
    ``list[Recommendation]`` ordered by ``vendor_id``.

    MVP fixed rules (deterministic), where "meaningful inventory impact" means
    the vendor's ``InventoryImpact.projected_units > 0`` (a delay exposure
    exists); a vendor with no ``InventoryImpact`` is treated as
    ``projected_units`` 0 (no exposure):

      - risk_level == "critical" AND projected_units > 0 -> action "replace"
      - risk_level == "critical" (no/low impact) OR "high" -> action "review"
      - risk_level == "moderate" -> action "review"
      - risk_level == "low" AND projected_units > 0 -> action "maintain"
      - risk_level == "low" AND (projected_units == 0 or no InventoryImpact)
        -> action "prefer"

    ``Recommendation.score`` is the vendor's ``risk_score`` carried through
    unchanged. ``rationale_keys`` = ``list(main_risk_drivers)`` (order preserved)
    plus ``"inventory_exposure"`` appended last when ``projected_units > 0``.
    No free prose.

    - One Recommendation per vendor present in ``risk_by_vendor``, ordered
      deterministically by ``vendor_id`` (Req 10.1, 10.3, 10.5).
    - Inventory impact is OPTIONAL: a vendor with risk but no ``InventoryImpact``
      still produces a Recommendation, treated as having no inventory exposure
      (Req 10.4).
    - Empty input -> empty successful result (no vendors invented) (Req 10.4).
    - Every Recommendation conforms to the schema; a candidate failing schema
      validation is rejected with a descriptive ``StageError`` (code
      ``"validation_failure"``) for that vendor instead of raising (Req 10.6).
    - Deterministic; byte-identical for identical inputs. No network/LLM/I/O
      (Req 10.5, 10.7).
    """
    data: list[Recommendation] = []
    errors: list[StageError] = []

    for vendor_id in sorted(risk_by_vendor):
        risk = risk_by_vendor[vendor_id]

        inventory = inventory_by_vendor.get(vendor_id)
        projected_units = inventory.projected_units if inventory is not None else 0.0

        action = _action_for(risk.risk_level, projected_units)

        rationale_keys = list(risk.main_risk_drivers)
        if projected_units > 0:
            rationale_keys.append("inventory_exposure")

        # Constructing Recommendation validates against the Schema model. In
        # practice a mapped literal action + a valid carried-through risk_score
        # won't fail, but guard simply (Req 10.6) rather than raising.
        try:
            recommendation = Recommendation(
                vendor_id=vendor_id,
                action=action,
                score=risk.risk_score,
                rationale_keys=rationale_keys,
            )
        except ValidationError as exc:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    code="validation_failure",
                    message=(
                        f"recommendation candidate for vendor {vendor_id!r} "
                        f"failed schema validation: {exc}"
                    ),
                )
            )
            continue

        data.append(recommendation)

    return StageResult(data=data, errors=errors)
