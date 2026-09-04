"""Business rules engine for the Vendor Forecasting Agent (rules.py).

This module is the ``Rules_Engine`` described in the design. Its single public
entry point, :func:`evaluate_rules`, evaluates a fixed set of six
vendor-risk-dimension concern rules per vendor against that vendor's
:class:`~vendor_forecasting_agent.schema.VendorMetric` (Task 4) and lead-time
:class:`~vendor_forecasting_agent.schema.TrendResult` (Task 5), producing
deterministic :class:`~vendor_forecasting_agent.schema.RuleOutcome` signals that
say which vendor-risk dimensions are concerning. Those signals are consumed by
the downstream scoring stage.

Scope (sub-task 7 only): plain deterministic Python threshold comparisons. This
is a hackathon MVP — the six thresholds are fixed module-level constants, not a
configurable/generic rule engine, plugin registry, expression parser, or
strategy factory. This module does **not** compute any risk score/level,
inventory impact, or recommendations, and adds no new dependencies. The schema's
``BusinessRule`` model is intentionally not used here.

Design constraints honored here (Requirement 7):

* **7.1 / 7.2 / 7.3** — For each eligible vendor the six fixed rules are
  evaluated (on-time delivery, lead-time trend, quantity fulfillment, capacity
  utilization, allocation/commitment, quality/defect), and all six
  ``RuleOutcome`` entries are recorded (``vendor_id``, ``rule_id``,
  ``triggered``) with ``triggered`` reflecting whether the concern condition
  holds. No score/level, inventory, or recommendation is produced.
* **7.4** — Comparisons are strict as written: a value exactly equal to a
  threshold is NOT a concern (yields ``triggered`` False).
* **7.5** — Missing input for a vendor (no ``VendorMetric``, a required ``avg_*``
  value is ``None``, or no ``TrendResult`` for the ``lead_time_declining`` rule)
  excludes that vendor from evaluation and records a per-vendor
  :class:`~vendor_forecasting_agent.schema.StageError`; other vendors are
  unaffected. Empty input yields an empty successful result — no vendor invented.
* **7.6** — Deterministic: vendors iterated in sorted ``vendor_id`` order, rules
  emitted in a fixed ``rule_id`` order, ``data`` and ``errors`` built in sorted
  vendor-id order; byte-identical output for identical inputs.
* **7.7** — Pure local computation: no network, LLM, or external API calls, and
  no I/O.
"""

from typing import Mapping, Optional

from .schema import RuleOutcome, StageError, StageResult, TrendResult, VendorMetric

# ---- Fixed MVP threshold constants (not configurable) -----------------------
# These encode the exact MVP thresholds from Requirement 7.2 / the design. They
# are module-level constants (not a configurable rule engine) for the hackathon
# MVP. Comparisons using them are strict (Req 7.4): a value exactly at the
# threshold is NOT a concern.

_ON_TIME_MIN = 0.90  # on_time_delivery_low: avg_on_time_rate < 0.90
_QUANTITY_FULFILLMENT_MIN = 0.90  # quantity_fulfillment_low: avg_quantity_fulfillment_rate < 0.90
_ALLOCATION_MIN = 0.90  # allocation_low: avg_allocation_ratio < 0.90
_CAPACITY_MAX = 0.95  # capacity_utilization_high: avg_capacity_utilization > 0.95
_DEFECT_MAX = 0.05  # defect_rate_high: avg_defect_rate > 0.05
# lead_time_declining uses trend.direction == "declining" (no numeric threshold).

# The five VendorMetric avg_* fields required to evaluate the metric-based rules.
# A vendor missing any of these (field is None) is excluded (Req 7.5). Fixed
# order keeps the missing-input error deterministic.
_REQUIRED_METRIC_FIELDS: tuple[str, ...] = (
    "avg_on_time_rate",
    "avg_quantity_fulfillment_rate",
    "avg_allocation_ratio",
    "avg_capacity_utilization",
    "avg_defect_rate",
)


def _first_missing_metric_field(metric: VendorMetric) -> Optional[str]:
    """Return the first required ``avg_*`` field that is ``None``, else ``None``.

    Iterates ``_REQUIRED_METRIC_FIELDS`` in fixed order so the reported missing
    field is deterministic (Req 7.5/7.6).
    """
    for field in _REQUIRED_METRIC_FIELDS:
        if getattr(metric, field) is None:
            return field
    return None


def evaluate_rules(
    metrics_by_vendor: Mapping[str, VendorMetric],
    trends_by_vendor: Mapping[str, TrendResult],
) -> StageResult:
    """Evaluate the six fixed vendor-risk concern rules per vendor (Req 7).

    The design's ``StageResult[dict[str, list[RuleOutcome]]]`` notation is
    conceptual: the concrete
    :class:`~vendor_forecasting_agent.schema.StageResult` is not ``Generic`` (its
    ``data`` field is typed ``Any``), so the runtime annotation is the plain
    ``StageResult``. On success ``data`` is a ``dict[str, list[RuleOutcome]]``
    keyed by ``vendor_id`` (sorted order), holding one six-element outcome list
    per eligible vendor; ineligible vendors are reported via ``errors`` instead.

    Fixed MVP thresholds (module constants; not configurable), evaluated and
    emitted in this fixed ``rule_id`` order:

      1. "on_time_delivery_low":      avg_on_time_rate < 0.90
      2. "lead_time_declining":       trend.direction == "declining"
      3. "quantity_fulfillment_low":  avg_quantity_fulfillment_rate < 0.90
      4. "capacity_utilization_high": avg_capacity_utilization > 0.95
      5. "allocation_low":            avg_allocation_ratio < 0.90
      6. "defect_rate_high":          avg_defect_rate > 0.05

    Comparisons are strict as written (Req 7.4): a value exactly at the threshold
    is NOT a concern (``triggered`` False). For each eligible vendor all six
    ``RuleOutcome(vendor_id, rule_id, triggered)`` entries are recorded in the
    fixed order above (Req 7.1/7.2/7.3).

    Missing input excludes a vendor and records a per-vendor ``StageError``
    without altering other vendors (Req 7.5): no ``VendorMetric`` (code
    ``"missing_input"``), a required ``avg_*`` is ``None`` (code
    ``"validation_failure"``), or no ``TrendResult`` for the
    ``lead_time_declining`` rule (code ``"missing_input"``). Empty input yields
    ``StageResult(data={}, errors=[])`` — no vendor invented (Req 7.5).

    Deterministic (Req 7.6): vendors iterated in sorted ``vendor_id`` order, rules
    in the fixed order above, and both ``data`` and ``errors`` built in sorted
    vendor-id order. Performs no network, LLM, or external calls and no I/O
    (Req 7.7).
    """
    data: dict[str, list[RuleOutcome]] = {}
    errors: list[StageError] = []

    for vendor_id in sorted(metrics_by_vendor):
        metric = metrics_by_vendor[vendor_id]

        # A required avg_* value is None -> exclude (Req 7.5).
        missing_field = _first_missing_metric_field(metric)
        if missing_field is not None:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field=missing_field,
                    code="validation_failure",
                    message=(
                        f"vendor {vendor_id!r} is missing required metric "
                        f"{missing_field!r} (None); cannot evaluate rules"
                    ),
                )
            )
            continue

        # No lead-time TrendResult for the lead_time_declining rule -> exclude
        # (Req 7.5).
        trend = trends_by_vendor.get(vendor_id)
        if trend is None:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="trend",
                    code="missing_input",
                    message=(
                        f"vendor {vendor_id!r} has no lead-time TrendResult; "
                        f"cannot evaluate the 'lead_time_declining' rule"
                    ),
                )
            )
            continue

        outcomes = [
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="on_time_delivery_low",
                triggered=metric.avg_on_time_rate < _ON_TIME_MIN,
            ),
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="lead_time_declining",
                triggered=trend.direction == "declining",
            ),
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="quantity_fulfillment_low",
                triggered=metric.avg_quantity_fulfillment_rate < _QUANTITY_FULFILLMENT_MIN,
            ),
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="capacity_utilization_high",
                triggered=metric.avg_capacity_utilization > _CAPACITY_MAX,
            ),
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="allocation_low",
                triggered=metric.avg_allocation_ratio < _ALLOCATION_MIN,
            ),
            RuleOutcome(
                vendor_id=vendor_id,
                rule_id="defect_rate_high",
                triggered=metric.avg_defect_rate > _DEFECT_MAX,
            ),
        ]

        data[vendor_id] = outcomes

    return StageResult(data=data, errors=errors)
