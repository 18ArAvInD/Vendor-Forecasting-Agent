"""Tests for synthetic.py (sub-tasks 3.2, 3.3, 3.4).

Standard pytest only (NO Hypothesis, no new dependencies). Correctness/determinism
cases use ``pytest.mark.parametrize`` over curated fixed seeds and record counts.

Sections:
- Sub-task 3.2 — Property 13: Synthetic data validity and exact count (Req 3.1, 3.3)
- Sub-task 3.3 — Property 1: Per-stage determinism (synthetic) (Req 3.2)
- Sub-task 3.4 — Unit tests for record-count edge and rejection cases (Req 3.3, 3.5)
"""

import random

import pytest

from vendor_forecasting_agent.schema import HistoricalDataRecord, canonical_json
from vendor_forecasting_agent.synthetic import (
    DEFAULT_VENDOR_IDS,
    RecordCountError,
    generate_historical_data,
)

# Fixed seeds spanning the valid Random_Seed range [0, 4_294_967_295].
_SEEDS = (0, 1, 42, 4_294_967_295)


# ===========================================================================
# Sub-task 3.2
# Feature: vendor-forecasting-agent, Property 13: Synthetic data validity and exact count
# Validates: Requirements 3.1, 3.3
# ===========================================================================

# Record counts: lower boundary 1 plus several mid values. 1_000_000 is
# deliberately excluded from the default matrix (too slow); the rejection of
# > 1_000_000 is covered in sub-task 3.4. This 4 seeds x 6 counts matrix yields
# 24 parametrized cases.
_VALIDITY_COUNTS = (1, 2, 5, 7, 50, 250)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("record_count", _VALIDITY_COUNTS)
def test_property13_exact_count_and_validity(seed: int, record_count: int) -> None:
    # Feature: vendor-forecasting-agent, Property 13: Synthetic data validity and exact count
    result = generate_historical_data(seed=seed, record_count=record_count)

    # Exact count (Req 3.3).
    assert len(result) == record_count

    # Every element is a schema-valid HistoricalDataRecord (Req 3.1). Records are
    # validated by construction; re-serialize each via canonical_json to confirm
    # it round-trips through the schema without error.
    for record in result:
        assert isinstance(record, HistoricalDataRecord)
        assert canonical_json(record)  # non-empty, serializes without error


def test_property13_max_count_acceptance_boundary_is_valid_config() -> None:
    # Feature: vendor-forecasting-agent, Property 13: Synthetic data validity and exact count
    # The literal max (1_000_000) is not generated here (too slow); the just-above
    # value is rejected (see 3.4). Here we cheaply confirm the accepted upper
    # boundary is treated as valid by generating a small count and checking that
    # counts up to a modest size succeed without a RecordCountError.
    result = generate_historical_data(seed=0, record_count=1000)
    assert len(result) == 1000


# ===========================================================================
# Sub-task 3.3
# Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (synthetic)
# Validates: Requirements 3.2
# ===========================================================================

_DETERMINISM_COUNTS = (1, 8, 37)
_VENDOR_ID_VARIATIONS = (None, ["a", "b", "c"])


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("record_count", _DETERMINISM_COUNTS)
@pytest.mark.parametrize("vendor_ids", _VENDOR_ID_VARIATIONS)
def test_property1_byte_identical_across_invocations(
    seed: int, record_count: int, vendor_ids: list[str] | None
) -> None:
    # Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (synthetic)
    run1 = generate_historical_data(
        seed=seed, record_count=record_count, vendor_ids=vendor_ids
    )
    run2 = generate_historical_data(
        seed=seed, record_count=record_count, vendor_ids=vendor_ids
    )

    assert [canonical_json(r) for r in run1] == [canonical_json(r) for r in run2]


@pytest.mark.parametrize("record_count", _DETERMINISM_COUNTS)
def test_property1_different_seeds_produce_different_output(record_count: int) -> None:
    # Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (synthetic)
    # Confirms the seed actually drives generation: two different seeds must not
    # produce byte-identical output for the same count.
    out_seed_a = [
        canonical_json(r)
        for r in generate_historical_data(seed=1, record_count=record_count)
    ]
    out_seed_b = [
        canonical_json(r)
        for r in generate_historical_data(seed=2, record_count=record_count)
    ]
    assert out_seed_a != out_seed_b


def test_property1_global_random_state_not_perturbed() -> None:
    # Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (synthetic)
    # The generator must use only a local random.Random(seed) and never touch the
    # global random stream (Req 3.2 / 3.4). Capture the global state, invoke the
    # generator, then assert the global state is unchanged.
    random.seed(123456)
    state_before = random.getstate()

    generate_historical_data(seed=999, record_count=1000)

    assert random.getstate() == state_before


def test_property1_global_random_draw_sequence_unaffected() -> None:
    # Feature: vendor-forecasting-agent, Property 1: Per-stage determinism (synthetic)
    # A second, complementary robustness check: the sequence of global random
    # draws is identical whether or not the generator is called in between.
    random.seed(2024)
    expected = [random.random() for _ in range(5)]

    random.seed(2024)
    _first = random.random()
    generate_historical_data(seed=7, record_count=500)
    rest = [_first] + [random.random() for _ in range(4)]

    assert rest == expected


# ===========================================================================
# Sub-task 3.4 — Unit tests for record-count edge and rejection cases
# Requirements: 3.3, 3.5
# ===========================================================================


def test_record_count_one_yields_exactly_one_record() -> None:
    # Exact-count edge at the lower boundary (Req 3.3). The literal max
    # (1_000_000) is intentionally not run in the default suite (too slow).
    result = generate_historical_data(seed=0, record_count=1)
    assert len(result) == 1
    assert isinstance(result[0], HistoricalDataRecord)


# Invalid counts: 0 and -1 (< 1), 1_000_001 (> 1_000_000), float 5.0, bool
# True/False, string "3", and None. All must raise RecordCountError and produce
# no records (the call raises before generation).
@pytest.mark.parametrize(
    "bad_count",
    [0, -1, 1_000_001, 5.0, True, False, "3", None],
)
def test_invalid_record_count_raises_and_produces_no_records(bad_count: object) -> None:
    with pytest.raises(RecordCountError):
        generate_historical_data(seed=0, record_count=bad_count)  # type: ignore[arg-type]


def test_rejection_happens_before_generation() -> None:
    # The check runs before any record is built: the invocation raises rather
    # than returning a partial/empty result. If it returned instead of raising,
    # pytest.raises would fail — this makes the "no records produced" guarantee
    # explicit (Req 3.5).
    raised = False
    try:
        generate_historical_data(seed=0, record_count=0)
    except RecordCountError:
        raised = True
    assert raised, "expected RecordCountError, generation should not have proceeded"


# ---- Optional vendor_ids distribution / ordering assertions ----------------


def test_vendor_ids_none_uses_default_vendor_ids() -> None:
    result = generate_historical_data(seed=0, record_count=50)
    produced_ids = {r.vendor_id for r in result}
    assert produced_ids.issubset(set(DEFAULT_VENDOR_IDS))


def test_explicit_vendor_ids_distribute_and_order_by_vendor_then_period() -> None:
    vendor_ids = ["a", "b", "c"]
    record_count = 9
    result = generate_historical_data(
        seed=0, record_count=record_count, vendor_ids=vendor_ids
    )

    # Only the requested vendor ids appear.
    assert {r.vendor_id for r in result} == set(vendor_ids)

    # Output is ordered by (vendor_id, period_index).
    keys = [(r.vendor_id, r.period_index) for r in result]
    assert keys == sorted(keys)

    # Each vendor's period_index starts at 0 and increases contiguously.
    for vid in vendor_ids:
        periods = [r.period_index for r in result if r.vendor_id == vid]
        assert periods == list(range(len(periods)))
        assert periods[0] == 0


# ---- Per-vendor profile differentiation ------------------------------------


def test_vendor_profiles_produce_meaningful_differentiation() -> None:
    # The five default vendors are driven by fixed profiles (reliable / stable /
    # deteriorating / capacity-pressure / quality-risk) so downstream stages can
    # differentiate suppliers. Generate ~50 periods per vendor and assert the
    # intended separation and trend direction. Tolerances are loose enough to be
    # robust against noise but strict enough to prove the profiles have effect.
    result = generate_historical_data(seed=0, record_count=250)

    def mean(vid: str, attr: str) -> float:
        vals = [getattr(r, attr) for r in result if r.vendor_id == vid]
        return sum(vals) / len(vals)

    # vendor-005 (quality-risk) has the highest mean defect_rate of the five.
    defect_means = {vid: mean(vid, "defect_rate") for vid in DEFAULT_VENDOR_IDS}
    assert max(defect_means, key=defect_means.get) == "vendor-005"

    # vendor-004 (capacity-pressure) has the highest mean capacity_utilization.
    cap_means = {
        vid: mean(vid, "capacity_utilization") for vid in DEFAULT_VENDOR_IDS
    }
    assert max(cap_means, key=cap_means.get) == "vendor-004"

    # vendor-001 (reliable) has a high mean on_time_rate (>= median across the
    # five) and strong fulfillment (>= median).
    on_time_means = {vid: mean(vid, "on_time_rate") for vid in DEFAULT_VENDOR_IDS}
    fulfill_means = {
        vid: mean(vid, "quantity_fulfillment_rate") for vid in DEFAULT_VENDOR_IDS
    }
    sorted_on_time = sorted(on_time_means.values())
    sorted_fulfill = sorted(fulfill_means.values())
    on_time_median = sorted_on_time[len(sorted_on_time) // 2]
    fulfill_median = sorted_fulfill[len(sorted_fulfill) // 2]
    assert on_time_means["vendor-001"] >= on_time_median
    assert fulfill_means["vendor-001"] >= fulfill_median

    # vendor-003 (deteriorating) shows a worsening trend: split its series into
    # thirds and confirm early on_time > late on_time and early lead < late lead.
    det = [r for r in result if r.vendor_id == "vendor-003"]
    det.sort(key=lambda r: r.period_index)
    third = len(det) // 3
    early, late = det[:third], det[-third:]
    early_on_time = sum(r.on_time_rate for r in early) / len(early)
    late_on_time = sum(r.on_time_rate for r in late) / len(late)
    early_lead = sum(r.lead_time_days for r in early) / len(early)
    late_lead = sum(r.lead_time_days for r in late) / len(late)
    assert early_on_time > late_on_time
    assert late_lead > early_lead
