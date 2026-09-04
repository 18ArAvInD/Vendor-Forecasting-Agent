"""Pydantic v2 data models for the Vendor Forecasting Agent (schema.py).

This module defines the domain and configuration data models used across the
deterministic core pipeline. Every model is immutable (``frozen=True``) and
rejects unknown fields (``extra="forbid"``). Numeric bounds are encoded directly
on the fields via ``Field(ge=..., le=...)`` so validation happens at
construction time.

Only the domain/config models are defined here (sub-task 1.2). The
``StageResult``/``StageError`` wrappers, canonical serialization helpers, and the
Phase 2 explanation/aggregate types are defined in later modules.
"""

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---- Configuration ----------------------------------------------------------


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: int = Field(ge=0, le=4_294_967_295)
    forecast_horizon: int = Field(default=12, ge=1, le=120)
    record_count: int = Field(default=1000, ge=1, le=1_000_000)
    stable_threshold_pct: float = Field(default=5.0, ge=0.0, le=100.0)


# ---- Core domain models -----------------------------------------------------


class Vendor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class HistoricalDataRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    period_index: int = Field(ge=0)
    demand: float = Field(ge=0.0, le=1_000_000_000.0)
    lead_time_days: float = Field(ge=0.0, le=3650.0)
    defect_rate: float = Field(ge=0.0, le=1.0)
    on_time_rate: float = Field(ge=0.0, le=1.0)
    quantity_fulfillment_rate: float = Field(ge=0.0, le=1.0)
    capacity_utilization: float = Field(ge=0.0)
    allocation_ratio: float = Field(ge=0.0)


class VendorMetric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    avg_demand: Optional[float] = Field(default=None, ge=0.0)
    avg_lead_time_days: Optional[float] = Field(default=None, ge=0.0)
    avg_defect_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_on_time_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_quantity_fulfillment_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_capacity_utilization: Optional[float] = Field(default=None, ge=0.0)
    avg_allocation_ratio: Optional[float] = Field(default=None, ge=0.0)
    sample_count: int = Field(ge=0)
    excluded_count: int = Field(default=0, ge=0)


class TrendPoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    period_index: int = Field(ge=0)
    value: float


class TrendResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    direction: Literal["improving", "declining", "stable"]
    slope: float
    point_count: int = Field(ge=0)


class Forecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    horizon: int = Field(ge=1, le=120)
    values: list[float] = Field(min_length=1, max_length=120)
    seed: int = Field(ge=0, le=4_294_967_295)


class BusinessRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rule_id: str = Field(min_length=1)
    description: str = Field(default="")


class RuleOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    triggered: bool


class VendorScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=100.0)


class VendorRiskResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    risk_score: float = Field(ge=0.0, le=100.0)
    risk_level: Literal["low", "moderate", "high", "critical"]
    confidence: float = Field(ge=0.0, le=1.0)
    main_risk_drivers: list[str] = Field(default_factory=list)


class DecisionChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    target_vendor_ids: list[str] = Field(min_length=1)
    metric: str = Field(min_length=1)
    delta: float


class ChangeDelta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pre_value: float
    post_value: float
    absolute_change: float
    percentage_change: float


class ImpactResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    score_change: Optional[ChangeDelta] = None
    metric_changes: dict[str, ChangeDelta] = Field(default_factory=dict)
    forecast_changes: dict[str, ChangeDelta] = Field(default_factory=dict)


class InventoryImpact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    projected_units: float = Field(ge=0.0)
    reorder_delta_units: float
    horizon: int = Field(ge=1, le=120)


class Recommendation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: str = Field(min_length=1)
    action: Literal["prefer", "maintain", "review", "replace"]
    score: float = Field(ge=0.0, le=100.0)
    rationale_keys: list[str] = Field(default_factory=list)


class DeterministicResults(BaseModel):
    """Immutable snapshot of the full Phase 1 deterministic-core output.

    Assembled by the Phase 1 pipeline driver (``pipeline.py``) from the outputs
    of every deterministic stage, keyed by ``vendor_id`` (except
    ``recommendations``, which is an ordered list). This is the single, read-only
    payload the Phase 2 explanation step consumes; it carries no LLM/explanation
    fields (those are added in Phase 2 via a separate aggregate type).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    metrics: dict[str, VendorMetric]
    trends: dict[str, TrendResult]
    forecasts: dict[str, Forecast]
    scores: dict[str, VendorScore]
    risk: dict[str, VendorRiskResult]
    outcomes: dict[str, list[RuleOutcome]]
    impacts: dict[str, ImpactResult]
    inventory: dict[str, InventoryImpact]
    recommendations: list[Recommendation]

# ---- Explanation result (Phase 2) -------------------------------------------


class NumberMismatch(BaseModel):
    """A single numeric token found in an explanation text that does not
    correspond to any value present in the deterministic results (within
    canonical rounding). ``nearest_expected`` is the closest allowed value, or
    ``None`` when there is no allowed number to compare against (Req 15.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    value_in_text: float
    nearest_expected: Optional[float] = None


class ExplanationResult(BaseModel):
    """Result of the Phase 2 explanation step: the natural-language ``text`` plus
    validation metadata. Carries no deterministic numbers of its own; the caller
    always retains the full ``DeterministicResults`` regardless of this result
    (Req 13.8, 15.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: Optional[str] = None
    is_valid: bool = True  # False if degraded or numbers mismatch
    degraded: bool = False  # True if provider unavailable/errored (Req 13.8)
    mismatches: list[NumberMismatch] = Field(default_factory=list)  # Req 15.4


# ---- Result / error wrapper types -------------------------------------------


class StageError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    vendor_id: Optional[str] = None  # None for whole-request errors (seed/horizon)
    field: Optional[str] = None
    code: str = Field(min_length=1)  # e.g. "insufficient_data", "unknown_vendor"
    message: str


class StageResult(BaseModel):
    """Generic wrapper: ``data`` present on success; ``errors`` carries structured,
    per-vendor or whole-request failures. Serialized with canonical ordering."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    data: Any = None
    errors: list[StageError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---- Canonical serialization ------------------------------------------------


def _sort_keys(value: Any) -> Any:
    """Recursively produce an order-stable structure by sorting dict keys.

    Lists preserve element order (order is meaningful for e.g. forecast values);
    only mapping key ordering is normalized so equivalent instances serialize to
    byte-identical output regardless of insertion order.
    """
    if isinstance(value, dict):
        return {key: _sort_keys(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sort_keys(item) for item in value]
    return value


def canonical_dump(model: BaseModel) -> dict[str, Any]:
    """Return a dict with deterministic key ordering (sorted keys), byte-stable
    across repeated serializations of equivalent instances (Req 1.5)."""
    return _sort_keys(model.model_dump(mode="json"))


def canonical_json(model: BaseModel) -> str:
    """json.dumps(canonical_dump(model), sort_keys=True, separators=(',', ':'))."""
    return json.dumps(canonical_dump(model), sort_keys=True, separators=(",", ":"))
