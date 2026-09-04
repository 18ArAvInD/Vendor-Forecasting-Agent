"""Curated, deterministic demo scenario for the Vendor Forecasting Agent (demo_scenario.py).

This module builds a small, hand-authored, **DEMO** dataset (clearly not
production data) that tells one concrete story: sourcing the ``PMIC-450`` Power
Management IC for an ``ACU-100`` Automotive Control Unit across three named
vendors — "Alpha Semiconductors", "Beta Electronics", and "Gamma Micro".

It reuses the EXISTING pipeline end to end. It does **not** modify or reimplement
any business logic (metrics, trend, forecast, rules, scoring, impact, recommend,
explanation). Instead it crafts :class:`~vendor_forecasting_agent.schema.HistoricalDataRecord`
values that NATURALLY produce the intended risk levels under the existing fixed
rules and scoring. Nothing here forces an outcome by touching thresholds.

How the curated data maps to the existing rules/scoring
------------------------------------------------------
Recall the fixed rule thresholds (see ``rules.py``): ``on_time < 0.90``,
``quantity_fulfillment < 0.90``, ``allocation < 0.90``, ``capacity > 0.95``,
``defect > 0.05``, and ``lead_time_declining`` when the lead-time trend direction
is ``"declining"``. Scoring maps ``n`` triggered rules (of six) to
``round(n/6*100)`` -> ``0,17,33,50,67,83,100`` with bands ``0-24 low``,
``25-49 moderate``, ``50-74 high``, ``75-100 critical``.

Trend (see ``trend.py``): with 12 periods, ``recent`` is the last 4 periods; the
direction is ``"declining"`` only when the recent-quarter average lead time is
more than ``stable_threshold_pct`` (default 5.0%) above the full-history average.

* "Alpha Semiconductors" -> LOW: strong on all six dimensions and a flat
  lead-time series (trend ``stable``) -> 0 rules trigger -> score 0 -> low.
* "Beta Electronics" -> MODERATE: healthy on-time/fulfillment/capacity/defect,
  but allocation just under 0.90 AND a clear lead-time deterioration over the
  later periods (trend ``declining``) -> exactly 2 rules trigger -> score 33 ->
  moderate.
* "Gamma Micro" -> HIGH/CRITICAL: weak on every dimension (on-time < 0.90,
  fulfillment < 0.90, allocation < 0.90, capacity > 0.95, defect > 0.05) plus a
  rising lead time (trend ``declining``) -> all 6 rules trigger -> score 100 ->
  critical.

Because ``vendor_id`` is just a string, the curated records use the vendor
DISPLAY NAME as ``vendor_id`` so the pipeline keys results by those names and the
UI can show them directly (no separate id->name mapping is required).

Everything is deterministic: values are plain literals shaped by small
period-indexed loops, with no randomness. :func:`run_scenario` reuses the
existing ``run_demo`` (``run_pipeline`` + offline ``explain``) path — it runs no
second pipeline and recomputes nothing.
"""

from typing import Optional

from .config import load_config
from .demo import DemoOutput, run_demo
from .schema import Config, HistoricalDataRecord
from .upstream import UpstreamRequest

# ---- Story constants (DEMO data — not production) ---------------------------

# Number of historical periods per vendor. >= 3 is required by the trend stage;
# 12 (e.g. monthly points over a year) also gives "full" scoring confidence.
_PERIODS = 12

# The three named vendors in this demo story (vendor_id == display name).
ALPHA = "Alpha Semiconductors"
BETA = "Beta Electronics"
GAMMA = "Gamma Micro"

# The upstream request for the story. This is the structured input an upstream
# agent (e.g. Itemate) would hand to this agent; here it is curated DEMO data.
UPSTREAM_REQUEST = UpstreamRequest(
    product_id="ACU-100",
    product_name="Automotive Control Unit",
    component_id="PMIC-450",
    component_name="Power Management IC",
    required_quantity=10000,
    current_inventory=6500.0,
    daily_demand=300.0,
    relevant_vendors=[ALPHA, BETA, GAMMA],
)

# Convenience mapping from vendor_id -> display name. Since vendor_id IS the
# display name in this curated scenario, this is an identity mapping; it is
# exposed so the UI has a single, explicit place to look up names.
VENDOR_DISPLAY_NAMES: dict[str, str] = {ALPHA: ALPHA, BETA: BETA, GAMMA: GAMMA}


def build_scenario_records() -> list[HistoricalDataRecord]:
    """Return the hand-authored DEMO records for the three named vendors.

    Deterministic and dependency-free: each vendor gets ``_PERIODS`` (12) records
    built from plain literals and simple period-indexed arithmetic. The values
    are engineered so the EXISTING pipeline yields Alpha=low, Beta=moderate,
    Gamma=high/critical (see the module docstring for the rule mapping). No
    thresholds or rules are changed to force these outcomes.
    """
    records: list[HistoricalDataRecord] = []

    for period in range(_PERIODS):
        # --- Alpha Semiconductors: LOW risk ---------------------------------
        # Strong across all six dimensions; lead time is FLAT (~10 days) so the
        # trend is "stable" and the lead_time_declining rule does NOT trigger.
        # Expected: 0 rules triggered -> score 0 -> "low".
        records.append(
            HistoricalDataRecord(
                vendor_id=ALPHA,
                period_index=period,
                demand=300.0,
                lead_time_days=10.0,  # flat -> stable trend
                defect_rate=0.01,  # <= 0.05 ok
                on_time_rate=0.98,  # >= 0.90 ok
                quantity_fulfillment_rate=0.99,  # >= 0.90 ok
                capacity_utilization=0.80,  # <= 0.95 ok
                allocation_ratio=0.97,  # >= 0.90 ok
            )
        )

        # --- Beta Electronics: MODERATE risk --------------------------------
        # Healthy on-time / fulfillment / capacity / defect, but:
        #   (1) allocation is just under 0.90 -> allocation_low triggers, and
        #   (2) lead time deteriorates in the later periods -> trend "declining"
        #       -> lead_time_declining triggers.
        # Exactly 2 rules -> score 33 -> "moderate".
        # Lead time: ~12 days for the first two-thirds, rising to ~16 days in the
        # final third so the recent-quarter average is clearly > 5% above the
        # full-history average.
        if period < 8:
            beta_lead = 12.0
        else:
            beta_lead = 16.0
        records.append(
            HistoricalDataRecord(
                vendor_id=BETA,
                period_index=period,
                demand=300.0,
                lead_time_days=beta_lead,  # rising later -> declining trend
                defect_rate=0.02,  # ok
                on_time_rate=0.94,  # ok (>= 0.90)
                quantity_fulfillment_rate=0.95,  # ok
                capacity_utilization=0.88,  # ok (<= 0.95)
                allocation_ratio=0.87,  # < 0.90 -> allocation_low triggers
            )
        )

        # --- Gamma Micro: HIGH/CRITICAL risk --------------------------------
        # Weak on every dimension AND a rising lead time. All six rules trigger
        # -> score 100 -> "critical".
        # Lead time: ~20 days early, rising to ~30 days late -> declining trend.
        if period < 8:
            gamma_lead = 20.0
        else:
            gamma_lead = 30.0
        records.append(
            HistoricalDataRecord(
                vendor_id=GAMMA,
                period_index=period,
                demand=300.0,
                lead_time_days=gamma_lead,  # rising -> declining trend
                defect_rate=0.09,  # > 0.05 -> defect_rate_high triggers
                on_time_rate=0.82,  # < 0.90 -> on_time_delivery_low triggers
                quantity_fulfillment_rate=0.80,  # < 0.90 -> quantity_fulfillment_low
                capacity_utilization=0.98,  # > 0.95 -> capacity_utilization_high
                allocation_ratio=0.78,  # < 0.90 -> allocation_low triggers
            )
        )

    return records


def run_scenario(config: Optional[Config] = None) -> DemoOutput:
    """Run the curated scenario through the EXISTING demo/pipeline flow.

    Builds the curated records and hands them to the existing
    :func:`vendor_forecasting_agent.demo.run_demo`, which internally runs
    ``run_pipeline`` and the offline ``explain(provider=None)`` path. There is NO
    second pipeline and NO recomputation here — this is purely a convenience
    entry point that feeds curated input into the existing flow.

    ``config`` defaults to ``load_config(None)`` (all-defaults, offline), whose
    ``stable_threshold_pct`` (5.0) and ``forecast_horizon`` (12) the curated
    lead-time series are tuned against.
    """
    config = config if config is not None else load_config(None)
    records = build_scenario_records()
    return run_demo(config=config, records=records)
