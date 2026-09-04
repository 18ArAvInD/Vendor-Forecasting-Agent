"""Upstream-agent input contract for the Vendor Forecasting Agent (upstream.py).

This module defines the *future integration boundary* between this Vendor
Forecasting Agent and an upstream agent (for example, an "Itemate"-style
product/component decomposition agent). It contains **only** an input data
contract: a single Pydantic v2 model, :class:`UpstreamRequest`, describing the
structured payload this agent expects to receive.

Scope: this task does **NOT** implement Itemate (or any upstream agent). There is
no vendor discovery, no product/component decomposition, no handoff protocol, no
network, and no orchestration here. The model is intentionally the entire public
surface of this module — it is the shape of the data an upstream agent would hand
in, nothing more. When that upstream agent is built later, it becomes the
producer of :class:`UpstreamRequest` instances; today the curated demo scenario
(``demo_scenario.py``) is the only producer.

The model is immutable (``frozen=True``) and rejects unknown fields
(``extra="forbid"``); every field carries an explicit type and validation bound
so a malformed payload is rejected at construction time.
"""

from pydantic import BaseModel, ConfigDict, Field


class UpstreamRequest(BaseModel):
    """Structured input this agent expects from an upstream agent.

    Represents a single "analyze these vendors for this product/component"
    request. The upstream agent is responsible for having already identified the
    product, the specific component of interest, the quantities involved, and the
    set of vendors relevant to that component; this agent then forecasts and
    risk-assesses those vendors using its existing deterministic core.

    Fields:

    * ``product_id`` — identifier of the finished product (non-empty).
    * ``product_name`` — human-readable product name (non-empty).
    * ``component_id`` — identifier of the specific component under analysis
      (non-empty).
    * ``component_name`` — human-readable component name (non-empty).
    * ``required_quantity`` — units of the component required (``>= 0``). Modeled
      as an ``int`` because it is a discrete count of units.
    * ``current_inventory`` — units of the component currently on hand
      (``>= 0.0``).
    * ``daily_demand`` — units of the component consumed per day (``>= 0.0``).
    * ``relevant_vendors`` — display names of the vendors to analyze; at least one
      must be provided.

    This is an input contract only: it holds no computed results and performs no
    business logic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    product_id: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    component_name: str = Field(min_length=1)
    required_quantity: int = Field(ge=0)
    current_inventory: float = Field(ge=0.0)
    daily_demand: float = Field(ge=0.0)
    relevant_vendors: list[str] = Field(min_length=1)
