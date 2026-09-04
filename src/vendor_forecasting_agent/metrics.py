"""Vendor performance metrics computation for the Vendor Forecasting Agent (metrics.py).

This module is the ``Metrics_Module`` described in the design. Its single public
entry point, :func:`compute_metrics`, groups ``HistoricalDataRecord``s by vendor
and computes a :class:`~vendor_forecasting_agent.schema.VendorMetric` per vendor.
Each ``VendorMetric`` carries a measure for every one of the six
``Vendor_Risk_Dimension``s (on-time delivery, lead-time, quantity fulfillment,
capacity utilization, allocation/commitment, and quality/defect), plus a
per-vendor mean demand for completeness, a ``sample_count``, and an
``excluded_count``.

Scope (sub-task 4.1 only): this module computes plain arithmetic means. It does
**not** compute trends, forecasts, scores, risk, inventory, recommendations,
normalization, or weighting — those live in later pipeline stages.

Design constraints honored here (Requirement 4):

* **4.1** — Metrics are computed per vendor using a single linear pass over the
  records, well within the ≤5s / 100k-record budget. No pandas/numpy; a plain
  ``sum``/``len`` mean keeps it readable and deterministic.
* **4.2 / 4.7** — Bit-for-bit identical results across invocations for identical
  input. Determinism is achieved by (a) iterating records in their given order
  when accumulating each vendor's running sum (a fixed, stable summation order),
  and (b) building the returned ``dict`` in vendor-id **sorted** order (the
  stable-key ordering the design mandates in Req 11.6). Given the same input the
  grouping, summation order, and output ordering are all fixed, so the six
  dimension measures are reproduced exactly.
* **4.3** — Empty-data handling. A vendor only appears in the input when it has
  at least one record, so within this function a "vendor with zero records"
  cannot be fabricated. The natural, schema-consistent reading is: if the entire
  input is empty, return a successful ``StageResult`` whose ``data`` is an empty
  dict ``{}`` (no vendors). No vendors or values are invented, and no exception
  is raised. (The ``VendorMetric`` ``avg_*`` fields are ``Optional`` precisely so
  they can serve as the defined null-equivalent value; that null-equivalent path
  is used for a vendor whose only records were all excluded — see 4.4 below.)
* **4.4** — Invalid-record handling. The input is typed as a sequence of already
  schema-valid ``HistoricalDataRecord``s, so the common path excludes nothing
  (``excluded_count == 0``). Defensively and simply, any element that is not a
  ``HistoricalDataRecord`` (e.g. ``None`` or a wrong type) is excluded from
  computation. Excluded elements that expose a usable string ``vendor_id`` are
  attributed to that vendor's ``excluded_count``; otherwise they are dropped as
  unattributable (they belong to no vendor and cannot create one). If every
  record for an attributable vendor was excluded, that vendor is emitted with the
  null-equivalent ``VendorMetric`` (all ``avg_*`` ``None``, ``sample_count == 0``)
  and its ``excluded_count`` — no exception is raised.
* **4.5 / 4.8** — Pure local computation: no network, LLM, or external API calls,
  and no I/O.
"""

from typing import Optional, Sequence

from .schema import HistoricalDataRecord, StageResult, VendorMetric

# The six Vendor_Risk_Dimension record attributes mapped to their VendorMetric
# average fields. avg_demand is carried too (not a risk dimension) for
# completeness, since the VendorMetric model declares it. Iteration over this
# tuple is in a fixed order, which keeps accumulation deterministic.
#
# Each entry is (record_attribute, vendor_metric_field).
_MEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("on_time_rate", "avg_on_time_rate"),
    ("lead_time_days", "avg_lead_time_days"),
    ("quantity_fulfillment_rate", "avg_quantity_fulfillment_rate"),
    ("capacity_utilization", "avg_capacity_utilization"),
    ("allocation_ratio", "avg_allocation_ratio"),
    ("defect_rate", "avg_defect_rate"),
    ("demand", "avg_demand"),
)


class _VendorAccumulator:
    """Mutable per-vendor accumulator used only during computation.

    Holds a running sum for each averaged dimension plus the count of valid
    records and the count of excluded (invalid) records attributed to the
    vendor. Sums are accumulated in the input record order (a stable summation
    order) so the resulting means are byte-identical across invocations.
    """

    __slots__ = ("sums", "sample_count", "excluded_count")

    def __init__(self) -> None:
        self.sums: dict[str, float] = {attr: 0.0 for attr, _ in _MEAN_FIELDS}
        self.sample_count: int = 0
        self.excluded_count: int = 0

    def add_valid(self, record: HistoricalDataRecord) -> None:
        """Fold one valid record's values into the running sums."""
        for attr, _field in _MEAN_FIELDS:
            self.sums[attr] += float(getattr(record, attr))
        self.sample_count += 1

    def to_metric(self, vendor_id: str) -> VendorMetric:
        """Build the immutable ``VendorMetric`` for this vendor.

        When ``sample_count`` is zero (every record was excluded), all ``avg_*``
        fields are left as their ``None`` null-equivalent default (Req 4.3/4.4).
        Otherwise each average is the running sum divided by ``sample_count``.
        """
        if self.sample_count == 0:
            return VendorMetric(
                vendor_id=vendor_id,
                sample_count=0,
                excluded_count=self.excluded_count,
            )

        averages: dict[str, Optional[float]] = {
            field: self.sums[attr] / self.sample_count
            for attr, field in _MEAN_FIELDS
        }
        return VendorMetric(
            vendor_id=vendor_id,
            sample_count=self.sample_count,
            excluded_count=self.excluded_count,
            **averages,
        )


def _is_valid_record(candidate: object) -> bool:
    """Return ``True`` when ``candidate`` is a usable ``HistoricalDataRecord``.

    Inputs are typed as already schema-valid records, so this is a light,
    defensive guard (Req 4.4). ``HistoricalDataRecord`` instances are immutable
    and bounds-checked at construction, so being an instance is sufficient.
    """
    return isinstance(candidate, HistoricalDataRecord)


def _excluded_vendor_id(candidate: object) -> Optional[str]:
    """Best-effort vendor attribution for an excluded (invalid) element.

    Returns the element's ``vendor_id`` when it is a non-empty ``str`` so the
    exclusion can be counted against that vendor; otherwise ``None`` (the
    exclusion is unattributable and is dropped without creating a vendor).
    """
    vendor_id = getattr(candidate, "vendor_id", None)
    if isinstance(vendor_id, str) and vendor_id:
        return vendor_id
    return None


def compute_metrics(
    records: Sequence[HistoricalDataRecord],
) -> StageResult:
    """Compute a :class:`VendorMetric` per vendor from historical records.

    The design's ``StageResult[dict[str, VendorMetric]]`` notation is conceptual:
    the concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not a
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On the normal path ``data`` is a
    ``dict[str, VendorMetric]`` keyed by ``vendor_id`` (sorted order) and
    ``errors`` is empty.

    Groups ``records`` by ``vendor_id`` in a stable manner and computes, per
    vendor, the arithmetic mean of each of the six ``Vendor_Risk_Dimension``
    measures (on-time delivery, lead-time, quantity fulfillment, capacity
    utilization, allocation/commitment, quality/defect) plus mean demand, along
    with ``sample_count`` (valid records used) and ``excluded_count`` (invalid
    records attributed to the vendor) (Req 4.1, 4.6).

    Determinism (Req 4.2, 4.7): each vendor's sums are accumulated in the input
    record order (a fixed, stable summation order) and the returned ``dict`` is
    built in vendor-id sorted order, so identical input yields byte-identical
    output.

    Empty input (Req 4.3): returns a successful ``StageResult`` whose ``data`` is
    an empty ``dict`` — no vendors are invented. A vendor whose records were all
    excluded is emitted with the null-equivalent ``VendorMetric`` (all ``avg_*``
    ``None``, ``sample_count == 0``) rather than raising.

    Invalid records (Req 4.4): elements that are not ``HistoricalDataRecord``
    instances are excluded from computation; each is counted in its attributable
    vendor's ``excluded_count`` (when it exposes a usable ``vendor_id``) and the
    remaining valid records drive the averages.

    Performs no network, LLM, or external calls and no I/O (Req 4.5, 4.8).
    """
    # Accumulators keyed by vendor_id. Insertion order does not affect output
    # because the returned dict is rebuilt in sorted vendor-id order below.
    accumulators: dict[str, _VendorAccumulator] = {}

    def accumulator_for(vendor_id: str) -> _VendorAccumulator:
        acc = accumulators.get(vendor_id)
        if acc is None:
            acc = _VendorAccumulator()
            accumulators[vendor_id] = acc
        return acc

    # Single stable pass over the input, preserving given order for summation.
    for candidate in records:
        if _is_valid_record(candidate):
            accumulator_for(candidate.vendor_id).add_valid(candidate)
        else:
            vendor_id = _excluded_vendor_id(candidate)
            if vendor_id is not None:
                accumulator_for(vendor_id).excluded_count += 1
            # Unattributable invalid elements belong to no vendor and are dropped.

    # Build the result dict in deterministic vendor-id sorted order (Req 11.6).
    metrics: dict[str, VendorMetric] = {
        vendor_id: accumulators[vendor_id].to_metric(vendor_id)
        for vendor_id in sorted(accumulators)
    }

    return StageResult(data=metrics)
