"""Tests for the upstream-agent input contract (upstream.py).

Standard pytest only (NO Hypothesis, NO streamlit). These tests import only the
``UpstreamRequest`` model and validate that a well-formed payload is accepted and
malformed payloads are rejected at construction time. This module is an input
contract only — there is no Itemate logic to exercise.
"""

import pytest
from pydantic import ValidationError

from vendor_forecasting_agent.upstream import UpstreamRequest


def _good_payload() -> dict:
    """The ACU-100 example payload used across the tests."""
    return {
        "product_id": "ACU-100",
        "product_name": "Automotive Control Unit",
        "component_id": "PMIC-450",
        "component_name": "Power Management IC",
        "required_quantity": 10000,
        "current_inventory": 6500.0,
        "daily_demand": 300.0,
        "relevant_vendors": ["Alpha Semiconductors", "Beta Electronics", "Gamma Micro"],
    }


def test_upstream_request_accepts_good_payload():
    request = UpstreamRequest(**_good_payload())

    assert request.product_id == "ACU-100"
    assert request.product_name == "Automotive Control Unit"
    assert request.component_id == "PMIC-450"
    assert request.component_name == "Power Management IC"
    assert request.required_quantity == 10000
    assert request.current_inventory == 6500.0
    assert request.daily_demand == 300.0
    assert request.relevant_vendors == [
        "Alpha Semiconductors",
        "Beta Electronics",
        "Gamma Micro",
    ]


def test_upstream_request_is_frozen():
    request = UpstreamRequest(**_good_payload())
    with pytest.raises(ValidationError):
        request.daily_demand = 1.0  # type: ignore[misc]


def test_upstream_request_rejects_empty_relevant_vendors():
    payload = _good_payload()
    payload["relevant_vendors"] = []
    with pytest.raises(ValidationError):
        UpstreamRequest(**payload)


def test_upstream_request_rejects_negative_daily_demand():
    payload = _good_payload()
    payload["daily_demand"] = -1.0
    with pytest.raises(ValidationError):
        UpstreamRequest(**payload)


def test_upstream_request_rejects_negative_required_quantity():
    payload = _good_payload()
    payload["required_quantity"] = -1
    with pytest.raises(ValidationError):
        UpstreamRequest(**payload)


def test_upstream_request_rejects_empty_string_ids():
    payload = _good_payload()
    payload["product_id"] = ""
    with pytest.raises(ValidationError):
        UpstreamRequest(**payload)


def test_upstream_request_rejects_unknown_field():
    payload = _good_payload()
    payload["unexpected"] = "nope"
    with pytest.raises(ValidationError):
        UpstreamRequest(**payload)
