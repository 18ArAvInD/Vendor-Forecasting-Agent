"""Lead-time trend analysis for the Vendor Forecasting Agent (trend.py).

This module is the ``Trend_Module`` described in the design. Its single public
entry point, :func:`analyze_trend`, classifies each vendor's chronologically
ordered historical lead time (``lead_time_days``) as ``improving | stable |
declining`` by comparing the vendor's historical-average lead time against the
average over its most recent third of periods, using the configured
``stable_threshold_pct``.

The caller supplies per-vendor lead-time history as chronologically ordered
sequences (e.g., derived from ``HistoricalDataRecord.lead_time_days`` ordered by
``period_index``). This module does no grouping/ordering of raw records itself.

Scope (sub-task 5.1 only): plain-Python arithmetic. No pandas/numpy, no ML,
regression, time-series, anomaly detection, smoothing, or generic trend
framework, and no new dependencies.

Design constraints honored here (Requirement 5):

* **5.1 / 5.2 / 5.3** — For a vendor with ``n >= 3`` periods, ``historical_avg``
  is the arithmetic mean of all ``lead_time_days`` and ``recent_avg`` is the mean
  over the last ``max(1, n // 3)`` periods. ``trend_pct = ((recent_avg -
  historical_avg) / historical_avg) * 100`` and the vendor is classified against
  ``stable_threshold_pct``: ``trend_pct < -stable_threshold_pct`` -> ``improving``
  (lead time falling = better), ``abs(trend_pct) <= stable_threshold_pct`` ->
  ``stable``, ``trend_pct > stable_threshold_pct`` -> ``declining`` (lead time
  rising = worse). ``TrendResult.slope`` carries ``trend_pct`` and
  ``TrendResult.point_count`` carries ``n``.
* **5.4** — Insufficient data. A vendor with fewer than 3 periods is NOT given a
  fabricated classification. Following the metrics-stage style, such a vendor is
  omitted from the success ``data`` dict and is instead recorded as a per-vendor
  :class:`~vendor_forecasting_agent.schema.StageError` with code
  ``"insufficient_data"`` (``vendor_id`` set, message naming the period count).
  The provided input is never mutated, so the caller's values are preserved.
  Empty input (no vendors) yields a successful ``StageResult`` with ``data == {}``
  and no errors — no vendor is invented.
* **5.5** — When ``historical_avg == 0`` the vendor is classified ``stable`` with
  ``trend_pct = 0.0``, avoiding any division-by-zero.
* **5.6** — Deterministic: identical input yields byte-identical output. Averages
  use a fixed summation order (the given chronological order) and both the
  success ``data`` dict and the ``errors`` list are built in sorted vendor-id
  order (the stable-key ordering the design mandates in Req 11.6).
* **5.7** — Pure local computation: no network, LLM, or external API calls, and
  no I/O.
"""

from typing import Literal, Mapping, Sequence

from .schema import StageError, StageResult, TrendResult

TrendDirection = Literal["improving", "stable", "declining"]

_MIN_PERIODS = 3


def _classify(trend_pct: float, stable_threshold_pct: float) -> TrendDirection:
    """Classify ``trend_pct`` against the stable band (Req 5.3).

    Lead time falling (negative ``trend_pct``) beyond the band is ``improving``;
    within the band is ``stable``; rising beyond the band is ``declining``.
    """
    if trend_pct < -stable_threshold_pct:
        return "improving"
    if trend_pct > stable_threshold_pct:
        return "declining"
    return "stable"


def analyze_trend(
    lead_times_by_vendor: Mapping[str, Sequence[float]],
    *,
    stable_threshold_pct: float,
) -> StageResult:
    """Classify each vendor's lead-time trend (Req 5).

    The design's ``StageResult[dict[str, TrendResult]]`` notation is conceptual:
    the concrete :class:`~vendor_forecasting_agent.schema.StageResult` is not a
    ``Generic`` (its ``data`` field is typed ``Any``), so the runtime annotation
    is the plain ``StageResult``. On success ``data`` is a
    ``dict[str, TrendResult]`` keyed by ``vendor_id`` (sorted order), holding one
    entry per vendor that had at least 3 periods; vendors with fewer than 3
    periods are reported via ``errors`` (code ``"insufficient_data"``) instead.

    For each vendor with ``n = len(lead_times) >= 3``:

    * ``historical_avg = sum(lead_times) / n``
    * ``k = max(1, n // 3)``; ``recent = lead_times[-k:]``;
      ``recent_avg = sum(recent) / len(recent)``
    * ``historical_avg == 0`` -> ``stable`` with ``trend_pct = 0.0`` (Req 5.5);
      otherwise ``trend_pct = ((recent_avg - historical_avg) / historical_avg) *
      100`` and the direction follows :func:`_classify` (Req 5.2, 5.3).
    * Emits ``TrendResult(vendor_id, direction, slope=trend_pct, point_count=n)``.

    Insufficient data (``n < 3``): omitted from ``data`` and recorded as a
    per-vendor ``StageError`` (code ``"insufficient_data"``); the input is not
    mutated (Req 5.4). Empty input -> ``StageResult(data={}, errors=[])`` (Req
    5.4).

    Deterministic (Req 5.6): ``data`` and ``errors`` are both built in sorted
    vendor-id order. Performs no network, LLM, or external calls and no I/O
    (Req 5.7).
    """
    trends: dict[str, TrendResult] = {}
    errors: list[StageError] = []

    for vendor_id in sorted(lead_times_by_vendor):
        lead_times = lead_times_by_vendor[vendor_id]
        n = len(lead_times)

        if n < _MIN_PERIODS:
            errors.append(
                StageError(
                    vendor_id=vendor_id,
                    field="lead_time_days",
                    code="insufficient_data",
                    message=(
                        f"vendor {vendor_id!r} has {n} lead-time period(s); "
                        f"at least {_MIN_PERIODS} are required to compute a trend"
                    ),
                )
            )
            continue

        historical_avg = sum(lead_times) / n
        k = max(1, n // 3)
        recent = lead_times[-k:]
        recent_avg = sum(recent) / len(recent)

        if historical_avg == 0:
            trend_pct = 0.0
        else:
            trend_pct = ((recent_avg - historical_avg) / historical_avg) * 100.0

        trends[vendor_id] = TrendResult(
            vendor_id=vendor_id,
            direction=_classify(trend_pct, stable_threshold_pct),
            slope=trend_pct,
            point_count=n,
        )

    return StageResult(data=trends, errors=errors)
