"""Tests for scoring.py (Task 8).

Standard pytest only (NO Hypothesis, no new dependencies). These tests exercise
``score_vendors`` and ``assess_risk`` — the ``Scoring_Module`` (Req 8):

- Score mapping across n = 0..6 -> 0, 17, 33, 50, 67, 83, 100 (VendorScore and
  the paired VendorRiskResult.risk_score agree).
- risk_level bands over the fixed score table.
- main_risk_drivers = triggered rule_ids in fixed Task 7 order (empty when none).
- confidence = min(1.0, sample_count / 12).
- Multiple vendors keyed + sorted by vendor_id.
- Empty/missing input: {} -> empty ok; empty outcomes -> excluded with
  StageError; no VendorMetric -> excluded with StageError (Req 8.2).
- Determinism: identical inputs -> byte-identical output (Req 8.3/8.8).

RuleOutcome lists and VendorMetric are built directly. The set is intentionally
minimal (essential examples + bands + determinism), not a large matrix.
"""

import pytest

from vendor_forecasting_agent.schema import (
    RuleOutcome,
    VendorMetric,
    canonical_json,
)
from vendor_forecasting_agent.scoring import assess_risk, score_vendors

# The six real rule_ids in the fixed Task 7 order.
_RULE_ORDER = [
    "on_time_delivery_low",
    "lead_time_declining",
    "quantity_fulfillment_low",
    "capacity_utilization_high",
    "allocation_low",
    "defect_rate_high",
]


# ---- Small builders ---------------------------------------------------------


def _outcomes(vendor_id: str, triggered_count: int) -> list[RuleOutcome]:
    """A six-rule outcome list (fixed order) with the first ``triggered_count``
    rules triggered."""
    return [
        RuleOutcome(vendor_id=vendor_id, rule_id=rule_id, triggered=i < triggered_count)
        for i, rule_id in enumerate(_RULE_ORDER)
    ]


def _outcomes_for(vendor_id: str, triggered_ids: list[str]) -> list[RuleOutcome]:
    """A six-rule outcome list (fixed order) triggering exactly ``triggered_ids``."""
    return [
        RuleOutcome(
            vendor_id=vendor_id, rule_id=rule_id, triggered=rule_id in triggered_ids
        )
        for rule_id in _RULE_ORDER
    ]


def _metric(vendor_id: str = "V1", sample_count: int = 12) -> VendorMetric:
    return VendorMetric(vendor_id=vendor_id, sample_count=sample_count)


# n -> (expected score, expected risk_level)
_SCORE_TABLE = [
    (0, 0.0, "low"),
    (1, 17.0, "low"),
    (2, 33.0, "moderate"),
    (3, 50.0, "high"),
    (4, 67.0, "high"),
    (5, 83.0, "critical"),
    (6, 100.0, "critical"),
]


# ---- 1 & 2: score mapping + risk_level bands --------------------------------


@pytest.mark.parametrize("n, expected_score, expected_level", _SCORE_TABLE)
def test_score_mapping_and_bands(n, expected_score, expected_level):
    outcomes = {"V1": _outcomes("V1", n)}
    metrics = {"V1": _metric("V1")}

    score_res = score_vendors(outcomes, metrics)
    assert score_res.ok
    vs = score_res.data["V1"]
    assert vs.score == expected_score
    assert isinstance(vs.score, float)
    assert 0.0 <= vs.score <= 100.0

    risk_res = assess_risk(outcomes, metrics)
    assert risk_res.ok
    vr = risk_res.data["V1"]
    # risk_score matches score_vendors exactly (Req 8.7).
    assert vr.risk_score == expected_score
    assert vr.risk_level == expected_level


# ---- 3: main_risk_drivers ---------------------------------------------------


def test_main_risk_drivers_in_fixed_order():
    # Trigger on_time (index 0) and defect (index 5); drivers keep fixed order.
    outcomes = {"V1": _outcomes_for("V1", ["defect_rate_high", "on_time_delivery_low"])}
    metrics = {"V1": _metric("V1")}

    vr = assess_risk(outcomes, metrics).data["V1"]
    assert vr.main_risk_drivers == ["on_time_delivery_low", "defect_rate_high"]


def test_main_risk_drivers_empty_when_none_triggered():
    outcomes = {"V1": _outcomes("V1", 0)}
    metrics = {"V1": _metric("V1")}

    vr = assess_risk(outcomes, metrics).data["V1"]
    assert vr.main_risk_drivers == []


# ---- 4: confidence from sample_count (N = 12) -------------------------------


@pytest.mark.parametrize(
    "sample_count, expected_confidence",
    [(0, 0.0), (6, 0.5), (12, 1.0), (24, 1.0)],
)
def test_confidence_from_sample_count(sample_count, expected_confidence):
    outcomes = {"V1": _outcomes("V1", 2)}
    metrics = {"V1": _metric("V1", sample_count=sample_count)}

    vr = assess_risk(outcomes, metrics).data["V1"]
    assert vr.confidence == expected_confidence
    assert 0.0 <= vr.confidence <= 1.0


# ---- 5: multiple vendors ----------------------------------------------------


def test_multiple_vendors_keyed_and_sorted():
    outcomes = {
        "V2": _outcomes("V2", 6),
        "V1": _outcomes("V1", 1),
    }
    metrics = {"V1": _metric("V1", 12), "V2": _metric("V2", 6)}

    score_res = score_vendors(outcomes, metrics)
    assert score_res.ok
    assert list(score_res.data.keys()) == ["V1", "V2"]
    assert score_res.data["V1"].score == 17.0
    assert score_res.data["V2"].score == 100.0

    risk_res = assess_risk(outcomes, metrics)
    assert risk_res.ok
    assert list(risk_res.data.keys()) == ["V1", "V2"]
    assert risk_res.data["V1"].risk_level == "low"
    assert risk_res.data["V2"].risk_level == "critical"
    assert risk_res.data["V2"].confidence == 0.5


# ---- 6: empty / missing input -----------------------------------------------


def test_empty_input_ok():
    for res in (score_vendors({}, {}), assess_risk({}, {})):
        assert res.ok
        assert res.data == {}
        assert res.errors == []


def test_empty_outcomes_excluded_with_error():
    outcomes = {"V1": [], "V2": _outcomes("V2", 3)}
    metrics = {"V1": _metric("V1"), "V2": _metric("V2")}

    for res in (score_vendors(outcomes, metrics), assess_risk(outcomes, metrics)):
        assert "V1" not in res.data
        assert "V2" in res.data  # other vendors unaffected
        assert len(res.errors) == 1
        assert res.errors[0].vendor_id == "V1"
        assert res.errors[0].code == "missing_input"


def test_missing_metric_excluded_with_error():
    outcomes = {"V1": _outcomes("V1", 2), "V2": _outcomes("V2", 4)}
    metrics = {"V2": _metric("V2")}  # no metric for V1

    for res in (score_vendors(outcomes, metrics), assess_risk(outcomes, metrics)):
        assert "V1" not in res.data
        assert "V2" in res.data
        assert len(res.errors) == 1
        assert res.errors[0].vendor_id == "V1"
        assert res.errors[0].code == "missing_input"


# ---- 7: determinism ---------------------------------------------------------


def test_determinism_byte_identical():
    outcomes = {
        "V1": _outcomes("V1", 3),
        "V2": _outcomes_for("V2", ["on_time_delivery_low", "defect_rate_high"]),
    }
    metrics = {"V1": _metric("V1", 12), "V2": _metric("V2", 6)}

    s1 = score_vendors(outcomes, metrics)
    s2 = score_vendors(outcomes, metrics)
    assert {vid: canonical_json(vs) for vid, vs in s1.data.items()} == {
        vid: canonical_json(vs) for vid, vs in s2.data.items()
    }
    assert list(s1.data.keys()) == list(s2.data.keys())

    r1 = assess_risk(outcomes, metrics)
    r2 = assess_risk(outcomes, metrics)
    assert {vid: canonical_json(vr) for vid, vr in r1.data.items()} == {
        vid: canonical_json(vr) for vid, vr in r2.data.items()
    }
    assert list(r1.data.keys()) == list(r2.data.keys())
