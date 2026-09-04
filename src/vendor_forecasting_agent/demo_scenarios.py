"""Multi-scenario registry of curated DEMO inputs for the Vendor Forecasting Agent.

This module adds a small, clean **scenario abstraction** on top of the existing
demo flow so the Streamlit UI can offer more than one curated story. It changes
NO business logic: every scenario is nothing more than a bundle of curated INPUT
data — a :class:`~vendor_forecasting_agent.upstream.UpstreamRequest` plus a
deterministic builder of
:class:`~vendor_forecasting_agent.schema.HistoricalDataRecord` lists — that flows
through the EXISTING ``run_demo`` / ``run_pipeline`` path unchanged.

Key facts / boundaries
----------------------
* These are **DEMO INPUTS**, clearly not production data. The scenarios are the
  *input* side of the system; they are hand-authored fixtures.
* :class:`~vendor_forecasting_agent.upstream.UpstreamRequest` is the **input
  contract / future handoff boundary** — the shape an upstream agent (e.g. an
  "Itemate"-style decomposition agent) would eventually hand in. **Itemate is NOT
  implemented here** (no vendor discovery, no decomposition, no networking).
* The **pipeline** (metrics/trend/forecast/rules/scoring/impact/recommend/
  explanation) — NOT these scenarios — is the agent logic. This module reuses it
  verbatim via :func:`vendor_forecasting_agent.demo.run_demo` and introduces no
  second pipeline and no risk recomputation.

Scenario 1 reuse guarantee
--------------------------
The default scenario reuses the EXISTING
:data:`vendor_forecasting_agent.demo_scenario.UPSTREAM_REQUEST` and
:func:`vendor_forecasting_agent.demo_scenario.build_scenario_records` **verbatim**
(they are imported, not re-authored). Because the request and the curated records
are the identical objects/values used today, the Automotive Control Unit /
PMIC-450 story produces byte-identical results (Alpha low/0/prefer, Beta
moderate/33/review, Gamma critical/100/replace).

Scenarios 2 and 3 are NEW curated semiconductor stories with distinct products,
components, vendors, quantities, inventory and demand. Their records are
hand-authored so the EXISTING fixed rules/scoring naturally produce a meaningful,
differentiated spread of risk levels — no thresholds or rules are touched.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .config import load_config
from .demo import DemoOutput, run_demo
from .demo_scenario import (
    UPSTREAM_REQUEST as _ACU_REQUEST,
    build_scenario_records as _acu_build_records,
)
from .schema import Config, HistoricalDataRecord
from .upstream import UpstreamRequest

# Number of historical periods per vendor for the new scenarios. >= 3 is required
# by the trend stage; 12 mirrors the existing scenario's monthly-over-a-year shape
# and gives full scoring confidence.
_PERIODS = 12


@dataclass(frozen=True)
class Scenario:
    """A single curated DEMO scenario: an input request plus a record builder.

    A scenario is pure INPUT data — it holds no computed results and runs no
    business logic. It bundles:

    * ``key`` — short, stable id used as the dropdown option value
      (e.g. ``"acu-pmic450"``).
    * ``label`` — human-readable dropdown label
      (e.g. ``"Automotive Control Unit — PMIC-450"``).
    * ``request`` — the :class:`UpstreamRequest` that would be handed to this
      agent for this story (the input contract / future handoff boundary).
    * ``build_records`` — a deterministic, dependency-free callable returning the
      curated :class:`HistoricalDataRecord` list for this scenario's vendors.

    A stdlib frozen dataclass is used (rather than Pydantic) specifically to hold
    the ``build_records`` callable cleanly without Pydantic callable-field
    friction. The scenario already carries its ``request``; :meth:`to_upstream_request`
    simply exposes it under the API the UI calls.
    """

    key: str
    label: str
    request: UpstreamRequest
    build_records: Callable[[], list[HistoricalDataRecord]]

    def to_upstream_request(self) -> UpstreamRequest:
        """Return this scenario's :class:`UpstreamRequest` (the input contract)."""
        return self.request


# ---- Scenario 1 (default): reuse the EXISTING ACU / PMIC-450 story -----------
# Imported verbatim from demo_scenario.py so results are byte-identical. Do NOT
# re-author these records here.

DEFAULT_SCENARIO_KEY = "acu-pmic450"

_SCENARIO_ACU = Scenario(
    key=DEFAULT_SCENARIO_KEY,
    label="Automotive Control Unit — PMIC-450",
    request=_ACU_REQUEST,
    build_records=_acu_build_records,
)


# ---- Scenario 2: Industrial Motor Controller / Gate Driver (GD-88) ----------
# Two distinctly named semiconductor vendors. Engineered (via the EXISTING rules)
# to produce a clear spread: one strong low-risk vendor and one weaker vendor.

_NOVA = "Nova Power Devices"
_VERTEX = "Vertex Silicon"

_SCENARIO_IMC_REQUEST = UpstreamRequest(
    product_id="IMC-200",
    product_name="Industrial Motor Controller",
    component_id="GD-88",
    component_name="Gate Driver",
    required_quantity=4200,
    current_inventory=1800.0,
    daily_demand=90.0,
    relevant_vendors=[_NOVA, _VERTEX],
)


def _build_imc_records() -> list[HistoricalDataRecord]:
    """Curated records for the Industrial Motor Controller / Gate Driver story.

    Deterministic; shaped so the EXISTING rules yield a differentiated spread:

    * ``Nova Power Devices`` — strong on all six dimensions with a flat lead time
      (stable trend). Expect 0 rules triggered -> low risk.
    * ``Vertex Silicon`` — weak on-time and quantity fulfilment, elevated defect
      rate, plus a rising lead time in the final third (declining trend). Expect
      several rules triggered -> high/critical risk.
    """
    records: list[HistoricalDataRecord] = []
    for period in range(_PERIODS):
        # Nova Power Devices: strong, flat lead time -> stable -> low risk.
        records.append(
            HistoricalDataRecord(
                vendor_id=_NOVA,
                period_index=period,
                demand=90.0,
                lead_time_days=8.0,  # flat -> stable trend
                defect_rate=0.01,
                on_time_rate=0.97,
                quantity_fulfillment_rate=0.98,
                capacity_utilization=0.78,
                allocation_ratio=0.96,
            )
        )

        # Vertex Silicon: weak + rising lead time -> several rules -> elevated.
        vertex_lead = 14.0 if period < 8 else 19.0  # rising -> declining trend
        records.append(
            HistoricalDataRecord(
                vendor_id=_VERTEX,
                period_index=period,
                demand=90.0,
                lead_time_days=vertex_lead,
                defect_rate=0.08,  # > 0.05 -> defect_rate_high
                on_time_rate=0.84,  # < 0.90 -> on_time_delivery_low
                quantity_fulfillment_rate=0.86,  # < 0.90 -> quantity_fulfillment_low
                capacity_utilization=0.90,  # <= 0.95 ok
                allocation_ratio=0.92,  # >= 0.90 ok
            )
        )
    return records


_SCENARIO_IMC = Scenario(
    key="imc-gd88",
    label="Industrial Motor Controller — GD-88",
    request=_SCENARIO_IMC_REQUEST,
    build_records=_build_imc_records,
)


# ---- Scenario 3: Edge AI Camera / Image Sensor (IS-12) ----------------------
# Three distinctly named semiconductor vendors spanning low, moderate, and high
# risk under the EXISTING rules.

_ORION = "Orion Imaging"
_ZEPHYR = "Zephyr Sensors"
_LUMEN = "Lumen Fab"

_SCENARIO_EAC_REQUEST = UpstreamRequest(
    product_id="EAC-30",
    product_name="Edge AI Camera",
    component_id="IS-12",
    component_name="Image Sensor",
    required_quantity=15000,
    current_inventory=9200.0,
    daily_demand=250.0,
    relevant_vendors=[_ORION, _ZEPHYR, _LUMEN],
)


def _build_eac_records() -> list[HistoricalDataRecord]:
    """Curated records for the Edge AI Camera / Image Sensor story.

    Deterministic; shaped so the EXISTING rules yield a three-way spread:

    * ``Orion Imaging`` — strong on all six dimensions with a flat lead time
      (stable trend). Expect 0 rules triggered -> low risk.
    * ``Zephyr Sensors`` — healthy overall but allocation just under 0.90 and a
      rising lead time (declining trend). Expect 2 rules triggered -> moderate.
    * ``Lumen Fab`` — weak across the board with a rising lead time. Expect most
      rules triggered -> high/critical risk.
    """
    records: list[HistoricalDataRecord] = []
    for period in range(_PERIODS):
        # Orion Imaging: strong, flat lead time -> stable -> low risk.
        records.append(
            HistoricalDataRecord(
                vendor_id=_ORION,
                period_index=period,
                demand=250.0,
                lead_time_days=9.0,  # flat -> stable trend
                defect_rate=0.01,
                on_time_rate=0.98,
                quantity_fulfillment_rate=0.99,
                capacity_utilization=0.82,
                allocation_ratio=0.95,
            )
        )

        # Zephyr Sensors: allocation < 0.90 + rising lead time -> 2 rules -> moderate.
        zephyr_lead = 11.0 if period < 8 else 15.0  # rising -> declining trend
        records.append(
            HistoricalDataRecord(
                vendor_id=_ZEPHYR,
                period_index=period,
                demand=250.0,
                lead_time_days=zephyr_lead,
                defect_rate=0.03,  # ok
                on_time_rate=0.93,  # ok
                quantity_fulfillment_rate=0.94,  # ok
                capacity_utilization=0.90,  # ok
                allocation_ratio=0.86,  # < 0.90 -> allocation_low
            )
        )

        # Lumen Fab: weak on every dimension + rising lead time -> elevated.
        lumen_lead = 22.0 if period < 8 else 31.0  # rising -> declining trend
        records.append(
            HistoricalDataRecord(
                vendor_id=_LUMEN,
                period_index=period,
                demand=250.0,
                lead_time_days=lumen_lead,
                defect_rate=0.10,  # > 0.05 -> defect_rate_high
                on_time_rate=0.80,  # < 0.90 -> on_time_delivery_low
                quantity_fulfillment_rate=0.79,  # < 0.90 -> quantity_fulfillment_low
                capacity_utilization=0.99,  # > 0.95 -> capacity_utilization_high
                allocation_ratio=0.75,  # < 0.90 -> allocation_low
            )
        )
    return records


_SCENARIO_EAC = Scenario(
    key="eac-is12",
    label="Edge AI Camera — IS-12",
    request=_SCENARIO_EAC_REQUEST,
    build_records=_build_eac_records,
)


# ---- Registry ---------------------------------------------------------------
# Insertion-ordered dict with the default ACU / PMIC-450 scenario FIRST.

SCENARIOS: dict[str, Scenario] = {
    _SCENARIO_ACU.key: _SCENARIO_ACU,
    _SCENARIO_IMC.key: _SCENARIO_IMC,
    _SCENARIO_EAC.key: _SCENARIO_EAC,
}


def list_scenarios() -> list[Scenario]:
    """Return all scenarios in registry order (default ACU / PMIC-450 first)."""
    return list(SCENARIOS.values())


def get_scenario(key: str) -> Scenario:
    """Return the :class:`Scenario` registered under ``key``.

    Raises ``KeyError`` if no scenario is registered with that key.
    """
    return SCENARIOS[key]


# ---- Scenario-aware execution (reuses the EXISTING flow) --------------------


def run_scenario(scenario: Scenario, config: Optional[Config] = None) -> DemoOutput:
    """Run a scenario through the EXISTING demo/pipeline flow.

    Builds the scenario's curated records and hands them to the existing
    :func:`vendor_forecasting_agent.demo.run_demo` (which internally runs
    ``run_pipeline`` + the offline ``explain(provider=None)`` path). There is NO
    second pipeline and NO risk recomputation here — this only feeds curated
    input into the existing flow.

    ``config`` defaults to ``load_config(None)`` (all-defaults, offline), matching
    the settings the curated lead-time series are tuned against.

    NOTE: this is intentionally distinct from
    :func:`vendor_forecasting_agent.demo_scenario.run_scenario` (which takes only
    a ``config`` and always runs the ACU story). That function is left untouched.
    """
    config = config if config is not None else load_config(None)
    return run_demo(config=config, records=scenario.build_records())


def run_scenario_by_key(key: str, config: Optional[Config] = None) -> DemoOutput:
    """Look up a scenario by ``key`` and run it via :func:`run_scenario`."""
    return run_scenario(get_scenario(key), config=config)
