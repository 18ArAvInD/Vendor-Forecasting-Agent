"""Synthetic historical data generation for the Vendor Forecasting Agent (synthetic.py).

This module is the ``Synthetic_Data_Generator`` described in the design. It
produces validated :class:`~vendor_forecasting_agent.schema.HistoricalDataRecord`
instances deterministically from a single ``Random_Seed`` so the rest of the
deterministic pipeline can be exercised without any external data source.

Design constraints honored here (Requirement 3):

* **3.1** — Every produced record passes ``Schema_Module`` validation. Records
  are constructed by instantiating ``HistoricalDataRecord`` directly, so any
  out-of-range draw would raise at construction time; the generation bounds
  below are chosen to always fall inside the schema field constraints.
* **3.2** — Byte-for-byte identical output for the same
  ``(seed, record_count, vendor_ids)``. The only source of randomness is a
  *local* ``random.Random(seed)`` instance. Field values are drawn in a fixed,
  documented order (see :func:`_draw_record`) so the RNG draw sequence is stable
  across invocations and independent of any global state.
* **3.3** — Exactly ``record_count`` records are produced.
* **3.4** — No network connection or external API call, and no ambient
  randomness: no global ``random`` module functions, no ``time``, no ``uuid``,
  no ``os.urandom``, and no I/O. The global ``random`` state is never read or
  mutated.
* **3.5** — :class:`RecordCountError` is raised (and no records produced) when
  ``record_count`` is not an ``int`` (``bool`` is rejected as well, since
  ``bool`` is a subclass of ``int``), is ``< 1``, or is ``> 1_000_000``. This
  check runs *before* any randomness is drawn or any record is built.

Determinism notes
-----------------
* The generator constructs its own ``random.Random(seed)`` and draws exclusively
  from it. It never touches ``random.random`` (the module-global generator), so
  calling this function does not perturb any global random stream.
* Records are distributed across vendors round-robin by index. Each vendor
  carries its own increasing chronological ``period_index`` starting at ``0``.
* Each vendor's behavior is shaped by a fixed, explicit per-vendor *profile*
  (see :data:`_VENDOR_PROFILES`) that applies modest deterministic biases and
  linear per-period trends to the drawn values. Profiles only shift
  already-drawn values before clamping, so they never change the RNG draw
  sequence and determinism is preserved.
* The returned list is ordered by ``(vendor_id, period_index)`` for a stable,
  deterministic ordering regardless of assignment order.

Rate/ratio semantics follow the pinned data dictionary in ``design.md``
("Vendor-risk dimension field semantics", Option A): each rate/ratio is stored
as a pre-computed value on the record; there are no raw
numerator/denominator fields.
"""

import random
from typing import Sequence

from .schema import HistoricalDataRecord

# Documented default vendor set used when ``vendor_ids`` is not supplied. A small
# fixed set keeps records distributed across several vendors so downstream
# grouping/metrics/trend stages have signal, while remaining fully deterministic.
DEFAULT_VENDOR_IDS: tuple[str, ...] = (
    "vendor-001",
    "vendor-002",
    "vendor-003",
    "vendor-004",
    "vendor-005",
)

# ---- Per-vendor behavior profiles -------------------------------------------
#
# Each profile is plain data (a flat dict of numeric adjustment parameters) that
# is applied deterministically to the *base* random draws inside
# :func:`_draw_record`. This lets the five default vendors show visibly
# different-but-realistic behavior (reliable / stable / deteriorating /
# capacity-pressure / quality-risk) so downstream scoring, rules, and
# recommendation stages can meaningfully differentiate suppliers.
#
# IMPORTANT: profiles never change *how many* values are drawn from the RNG or
# the order they are drawn in. They only add fixed arithmetic biases and linear
# per-period slopes to the already-drawn base values (before clamping). The RNG
# draw sequence is therefore unchanged and output stays deterministic for a
# given (seed, record_count, vendor_ids). No hashing is used — the mapping is
# keyed explicitly by the default vendor ids.
#
# Profile keys (all optional; missing keys default to 0.0 = no adjustment):
#   *_bias   -> constant offset added to that dimension every period.
#   *_slope  -> linear per-period term (multiplied by period_index) for trends.
_ProfileT = dict[str, float]

# NEUTRAL profile: no bias, no trend. Used for any vendor id not in the fixed
# default mapping (e.g. caller-supplied custom vendor_ids) so custom ids still
# work and remain deterministic.
_NEUTRAL_PROFILE: _ProfileT = {}

_VENDOR_PROFILES: dict[str, _ProfileT] = {
    # vendor-001 = RELIABLE: strong delivery/fulfillment/quality, no worsening.
    "vendor-001": {
        "on_time_bias": 0.03,
        "quantity_fulfillment_bias": 0.02,
        "defect_bias": -0.02,
        "lead_time_bias": -1.0,
    },
    # vendor-002 = STABLE: near-neutral biases and (intentionally) flat trend.
    # The empty mapping means every dimension stays roughly flat across periods.
    "vendor-002": {},
    # vendor-003 = DETERIORATING: clear worsening trend over periods.
    "vendor-003": {
        "on_time_slope": -0.004,
        "quantity_fulfillment_slope": -0.003,
        "lead_time_slope": 0.15,
    },
    # vendor-004 = CAPACITY-PRESSURE: higher utilization (often exceeds 1.0) and
    # a mild upward lead-time bias; other dimensions roughly normal.
    "vendor-004": {
        "capacity_utilization_bias": 0.25,
        "lead_time_bias": 1.0,
    },
    # vendor-005 = QUALITY-RISK: higher defect rate and slightly lower
    # fulfillment; delivery roughly normal.
    "vendor-005": {
        "defect_bias": 0.06,
        "quantity_fulfillment_bias": -0.03,
    },
}


def _resolve_profile(vendor_id: str) -> _ProfileT:
    """Return the fixed adjustment profile for ``vendor_id``.

    Vendor ids in the default mapping get their designed profile; any other id
    (custom caller-supplied vendor) gets the explicit :data:`_NEUTRAL_PROFILE`
    (no bias, no trend), keeping custom vendor generation deterministic.
    """
    return _VENDOR_PROFILES.get(vendor_id, _NEUTRAL_PROFILE)

# Bounds for record_count (mirrors the ``Config.record_count`` / schema bounds).
_MIN_RECORD_COUNT: int = 1
_MAX_RECORD_COUNT: int = 1_000_000


class RecordCountError(Exception):
    """Raised when ``record_count`` is non-integer, ``< 1``, or ``> 1_000_000``.

    When raised, no ``Historical_Data`` records are produced (Req 3.5).
    """


def _validate_record_count(record_count: int) -> None:
    """Validate ``record_count`` before any randomness is drawn (Req 3.5).

    ``bool`` is explicitly rejected even though it is a subclass of ``int`` so a
    ``True``/``False`` argument cannot silently be treated as ``1``/``0``.
    """
    if isinstance(record_count, bool) or not isinstance(record_count, int):
        raise RecordCountError(
            "record_count must be an integer in the inclusive range "
            f"[{_MIN_RECORD_COUNT}, {_MAX_RECORD_COUNT}]; got "
            f"{record_count!r} of type {type(record_count).__name__}"
        )
    if record_count < _MIN_RECORD_COUNT or record_count > _MAX_RECORD_COUNT:
        raise RecordCountError(
            "record_count is out of the accepted range "
            f"[{_MIN_RECORD_COUNT}, {_MAX_RECORD_COUNT}]; got {record_count}"
        )


def _resolve_vendor_ids(vendor_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Return the vendor ids to distribute records across.

    When ``vendor_ids`` is ``None`` the documented :data:`DEFAULT_VENDOR_IDS` are
    used. When provided, the caller-supplied order is preserved exactly (it feeds
    the round-robin assignment) but the final record list is still ordered by
    ``(vendor_id, period_index)``.
    """
    if vendor_ids is None:
        return DEFAULT_VENDOR_IDS
    resolved = tuple(vendor_ids)
    if not resolved:
        # An empty caller-supplied set has no vendor to attach records to; fall
        # back to the documented default set so exactly record_count records can
        # still be produced deterministically.
        return DEFAULT_VENDOR_IDS
    return resolved


def _draw_record(
    rng: random.Random, vendor_id: str, period_index: int, profile: _ProfileT
) -> HistoricalDataRecord:
    """Draw one validated record for ``vendor_id`` at ``period_index``.

    Field values are drawn from ``rng`` in a **fixed order** so the draw sequence
    is stable and byte-identical across invocations (Req 3.2). The vendor's
    ``profile`` (a flat dict of numeric ``*_bias`` / ``*_slope`` parameters) is
    applied as deterministic arithmetic *after* each field's base random value is
    drawn and *before* clamping, so trend/bias direction is profile-driven rather
    than uniformly positive for every vendor. Only linear per-period terms are
    used — no statistical/ML machinery. All values are clamped into the schema's
    field bounds so every record validates (Req 3.1).

    Because the profile only shifts already-drawn values (it never adds, removes,
    or reorders draws), the RNG draw sequence is identical regardless of profile,
    preserving byte-identical determinism for a given (seed, record_count,
    vendor_ids).

    Draw order (do not reorder — this defines the deterministic stream):
      1. on_time_rate
      2. lead_time_days
      3. defect_rate
      4. quantity_fulfillment_rate
      5. capacity_utilization
      6. allocation_ratio
      7. demand
    """
    # Profile adjustment = constant bias + linear per-period slope. Missing keys
    # default to 0.0 (no adjustment), which is exactly the neutral profile.
    def adjust(dimension: str) -> float:
        bias = profile.get(f"{dimension}_bias", 0.0)
        slope = profile.get(f"{dimension}_slope", 0.0)
        return bias + slope * period_index

    # 1. on_time_rate in [0, 1], skewed toward higher (better) values.
    on_time_rate = _clamp(
        rng.triangular(0.70, 1.0, 0.95) + adjust("on_time"), 0.0, 1.0
    )

    # 2. lead_time_days: realistic small-ish values (a few days to a few weeks).
    lead_time_days = _clamp(
        rng.triangular(2.0, 21.0, 7.0) + adjust("lead_time"), 0.0, 3650.0
    )

    # 3. defect_rate in [0, 1], skewed toward low (better) values.
    defect_rate = _clamp(
        rng.triangular(0.0, 0.15, 0.02) + adjust("defect"), 0.0, 1.0
    )

    # 4. quantity_fulfillment_rate = delivered/ordered in [0, 1], skewed high.
    quantity_fulfillment_rate = _clamp(
        rng.triangular(0.80, 1.0, 0.97) + adjust("quantity_fulfillment"), 0.0, 1.0
    )

    # 5. capacity_utilization = used/available, >= 0 and MAY exceed 1.0 so
    #    capacity pressure is exercisable (drawn roughly in 0.3..1.3).
    capacity_utilization = max(
        0.0, rng.triangular(0.30, 1.30, 0.85) + adjust("capacity_utilization")
    )

    # 6. allocation_ratio = allocated/committed, >= 0 and may slightly exceed 1.
    allocation_ratio = max(
        0.0, rng.triangular(0.0, 1.20, 0.95) + adjust("allocation_ratio")
    )

    # 7. demand: realistic non-negative demand values within schema bounds.
    demand = _clamp(
        rng.uniform(50.0, 5_000.0) + adjust("demand"), 0.0, 1_000_000_000.0
    )

    return HistoricalDataRecord(
        vendor_id=vendor_id,
        period_index=period_index,
        demand=demand,
        lead_time_days=lead_time_days,
        defect_rate=defect_rate,
        on_time_rate=on_time_rate,
        quantity_fulfillment_rate=quantity_fulfillment_rate,
        capacity_utilization=capacity_utilization,
        allocation_ratio=allocation_ratio,
    )


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def generate_historical_data(
    *,
    seed: int,
    record_count: int,
    vendor_ids: Sequence[str] | None = None,
) -> list[HistoricalDataRecord]:
    """Generate exactly ``record_count`` validated historical records.

    Deterministic given ``(seed, record_count, vendor_ids)``: the only source of
    randomness is a local ``random.Random(seed)`` instance, and field values are
    drawn in a fixed order, so output is byte-for-byte identical on every
    invocation with the same arguments (Req 3.2). Uses no global random, no
    ``time``/``uuid``/``os.urandom``, and performs no I/O or network calls
    (Req 3.4).

    Records are distributed round-robin across ``vendor_ids`` (or the documented
    default set when ``None``). Each vendor gets an increasing chronological
    ``period_index`` starting at ``0``. The returned list is ordered by
    ``(vendor_id, period_index)`` (Req 11.6 stable ordering).

    Raises :class:`RecordCountError` (producing no records) when ``record_count``
    is not an ``int``, is ``< 1``, or is ``> 1_000_000`` (Req 3.5). This check
    runs before any randomness is drawn.
    """
    # Validate the count BEFORE constructing the RNG or producing any records.
    _validate_record_count(record_count)

    resolved_vendor_ids = _resolve_vendor_ids(vendor_ids)
    vendor_count = len(resolved_vendor_ids)

    # Local, isolated RNG — never the global random module (Req 3.2, 3.4).
    rng = random.Random(seed)

    # Track the next chronological period_index per vendor so each vendor's series
    # increases from 0.
    next_period: dict[str, int] = {vid: 0 for vid in resolved_vendor_ids}

    # Resolve each vendor's fixed profile once (explicit mapping; neutral for any
    # non-default/custom vendor id). No hashing — deterministic dict lookup.
    profiles: dict[str, _ProfileT] = {
        vid: _resolve_profile(vid) for vid in resolved_vendor_ids
    }

    records: list[HistoricalDataRecord] = []
    for i in range(record_count):
        vendor_id = resolved_vendor_ids[i % vendor_count]
        period_index = next_period[vendor_id]
        next_period[vendor_id] = period_index + 1
        records.append(
            _draw_record(rng, vendor_id, period_index, profiles[vendor_id])
        )

    # Deterministic, stable ordering by (vendor_id, period_index).
    records.sort(key=lambda r: (r.vendor_id, r.period_index))
    return records
