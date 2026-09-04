"""Decision-first Streamlit dashboard for the Vendor Forecasting Agent.

This is the interactive decision surface (Layer 4) on top of the deterministic
pipeline. It is organized top-to-bottom as **decision-first -> detail-on-demand
-> fully auditable**: a product header, a concise executive decision block, a
KPI band, prominent vendor risk cards, a concise selected-vendor summary that
opens into analyst detail on demand, an answer-first "Ask the Agent" panel and a
what-if delay tool, plus an interaction-aware "AI Audit & Data Traceability"
expander at the BOTTOM of the main page. Explicit, on-demand AI interactions
(Generate Executive Summary, Generate AI Explanation for the selected vendor,
Ask the Agent, and Explain What-If) each call Mistral only when the user clicks;
the audit then reflects the most recent real interaction.

The dashboard reads from the deterministic engine and never recomputes anything.
Every displayed number originates from the selected curated scenario's
``run_scenario()`` (which internally runs ``run_demo`` -> ``run_pipeline`` +
offline ``explain(provider=None)``) and is read straight from the returned
``DemoOutput.snapshot`` / ``DemoOutput.explanation`` via the streamlit-free
helpers in :mod:`vendor_forecasting_agent.ui.view_model`, from the
:class:`~vendor_forecasting_agent.whatif.WhatIfResult`, and from the structured
context builders in :mod:`vendor_forecasting_agent.llm_context`. The only
arithmetic in this file is presentation formatting (thousands separators, display
rounding) applied to *copies* — the authoritative data is never mutated.

AI decision support (free-form Q&A, what-if explanation) goes exclusively through
:mod:`vendor_forecasting_agent.decision_support`, which is both streamlit-free
and boto3-free. The provider is obtained once per run via
:func:`decision_support.get_default_provider`. With no LLM configured a
``NullLLMProvider`` is used and every call returns an ``LLMResponse`` (never
raises).

A robust raw-context guard (:func:`_is_showable_ai_text`) ensures the internal
LLM prompt / analysis context is NEVER shown as an answer. The offline provider
echoes the raw prompt (which embeds the authoritative context as JSON); that raw
text is detected and replaced with a clean deterministic-fallback message. The
structured AI context is used internally to prompt Mistral (via
:func:`llm_context.build_vendor_context`) but is never rendered as raw JSON in
the UI.

``streamlit`` is imported at module top level because this module is only ever
executed via ``streamlit run``. No core module and no test imports this file.

Run it with::

    pip install -e .[ui]
    streamlit run src/vendor_forecasting_agent/ui/app.py
"""

import streamlit as st

from vendor_forecasting_agent import decision_support
from vendor_forecasting_agent.conversation import (
    dedupe_ai_text,
    resolve_explicit_vendor,
    trim_history,
)
from vendor_forecasting_agent.demo_scenarios import (
    DEFAULT_SCENARIO_KEY,
    get_scenario,
    list_scenarios,
    run_scenario,
)
from vendor_forecasting_agent.llm_context import (
    build_executive_context,
    build_vendor_context,
)
from vendor_forecasting_agent.ui import view_model
from vendor_forecasting_agent.whatif_scenarios import (
    ScenarioResult,
    analyze_demand_increase,
    analyze_fulfillment_reduction,
    analyze_supplier_delay,
    build_scenario_context,
)

# Generic, scenario-agnostic UI suggestion strings. Plain, business-friendly
# interaction affordances (no vendor / product / component literals), so they
# stay data-driven and live here as presentation-only prompts.
_SUGGESTED_QUESTIONS = (
    "Which vendor is riskiest?",
    "Compare the top two vendors",
    "Why is the recommendation what it is?",
    "What should I do next?",
    "Explain the latest What-If result",
)

# ---- Raw-context guard ------------------------------------------------------
# Markers that indicate the text is (or contains) the internal system prompt /
# authoritative analysis context rather than a genuine AI answer. The offline
# NullLLMProvider echoes the prompt verbatim, so its LLMResponse.text will contain
# these markers. Such text must NEVER be shown to the user.
_FORBIDDEN_MARKERS = (
    "ANALYSIS CONTEXT",
    "authoritative, do not alter",
    "QUESTION:",
    "You are an AI decision-support assistant",
)

_OFFLINE_FALLBACK_TEXT = (
    "AI explanation is offline. The deterministic analysis on this page remains "
    "fully authoritative."
)

# Restrained, professional risk-level palette.
_RISK_COLORS = {
    "low": "#2e7d32",
    "moderate": "#b8860b",
    "high": "#d84315",
    "critical": "#c62828",
}
_NEUTRAL_COLOR = "#6b7280"

# Recommended-action pill palette (presentation only).
_ACTION_COLORS = {
    "prefer": "#2e7d32",
    "review": "#b8860b",
    "replace": "#c62828",
}


# ---- Presentation formatting helpers ----------------------------------------


def _fmt_ratio(value) -> str:
    """Format an optional rate / ratio for display, ``"n/a"`` when ``None``."""
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_days(value) -> str:
    """Format a lead time / delay in days, e.g. ``34.7140 -> "34.71 days"``."""
    if value is None:
        return "n/a"
    return f"{value:,.2f} days"


def _fmt_units(value, decimals: int = 0) -> str:
    """Format a unit count with thousands separators, ``"n/a"`` when ``None``.

    Use ``decimals=0`` for whole inventory counts (``6500 -> "6,500 units"``) and
    ``decimals=2`` for fractional computed exposure (``4779.90 -> "4,779.90 units"``).
    """
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f} units"


def _fmt_score(value) -> str:
    """Format a risk score for display, e.g. ``33 -> "33 / 100"``."""
    if value is None:
        return "n/a"
    return f"{int(round(value))} / 100"


def _fmt_coverage(days) -> str:
    """Format inventory coverage days, e.g. ``20.0 -> "20.0 days"``."""
    if days is None:
        return "n/a"
    return f"{days:,.1f} days"


def _score_percent(value) -> int:
    """Clamp a risk score to an integer 0..100 for a display-only progress bar."""
    if value is None:
        return 0
    return max(0, min(100, int(round(value))))


# ---- Risk-driver wording (must be faithful) ---------------------------------


def _friendly_drivers(drivers) -> list[str]:
    """Map each triggered rule id to its friendly business label (presentation only).

    Reuses ``view_model._RISK_ISSUE_LABELS`` and falls back to the raw rule id
    when a driver has no mapped label. Computes nothing and never changes a rule
    id or risk value — it only relabels EXISTING triggered drivers for display.
    """
    result: list[str] = []
    for driver in drivers or []:
        result.append(view_model._RISK_ISSUE_LABELS.get(driver, driver))
    return result


def _driver_phrasing(view) -> str:
    """Faithful one-line phrasing of a vendor's risk drivers (presentation only).

    Follows the required wording rule strictly:

    * 0 drivers -> "No risk drivers triggered."
    * 1 driver  -> the single driver's friendly label (``view['main_risk_issue']``).
    * >=2       -> "Multiple performance risks ({n} risk drivers)".

    Never collapses a multi-driver vendor to a single misleading label; the actual
    driver list is always inspectable in View Details.
    """
    drivers = view.get("main_risk_drivers") or []
    count = len(drivers)
    if count == 0:
        return "No risk drivers triggered."
    if count == 1:
        # ``main_risk_issue`` is already the friendly label for the single driver;
        # fall back to the friendly mapping of the driver id if it's missing.
        return str(view.get("main_risk_issue") or _friendly_drivers(drivers)[0])
    return f"Multiple performance risks ({count} risk drivers)"


# ---- Styling helpers --------------------------------------------------------


def _risk_color(level) -> str:
    """Map a deterministic risk level string to a restrained, professional color."""
    if not level:
        return _NEUTRAL_COLOR
    return _RISK_COLORS.get(str(level).strip().lower(), _NEUTRAL_COLOR)


def _action_color(action) -> str:
    """Map a recommended action to a restrained pill color (presentation only)."""
    if not action:
        return _NEUTRAL_COLOR
    return _ACTION_COLORS.get(str(action).strip().lower(), _NEUTRAL_COLOR)


def _badge(text: str, color: str) -> str:
    """Return inline HTML for a small colored pill badge (known strings only)."""
    label = str(text).upper()
    return (
        f"<span style='background:{color};color:#ffffff;padding:2px 10px;"
        f"border-radius:12px;font-size:0.78rem;font-weight:600;"
        f"letter-spacing:0.3px;white-space:nowrap;'>{label}</span>"
    )


def _score_bar(value, color: str) -> str:
    """Return inline HTML for a thin display-only risk-score bar."""
    pct = _score_percent(value)
    return (
        "<div style='background:#e5e7eb;border-radius:6px;height:8px;width:100%;"
        "overflow:hidden;margin:6px 0;'>"
        f"<div style='background:{color};height:8px;width:{pct}%;'></div>"
        "</div>"
    )


# ---- Provider / AI-response rendering ---------------------------------------


def _provider_is_offline(provider) -> bool:
    """Return ``True`` when ``provider`` is the offline null provider."""
    return type(provider).__name__ == "NullLLMProvider"


def _ai_status_label(provider) -> str:
    """Return the sidebar AI-status caption (class-name based, no secrets)."""
    name = type(provider).__name__
    if name == "MistralAPIProvider":
        return "AI assistant: Mistral Medium"
    if name == "MistralBedrockProvider":
        return "AI assistant: Active"
    return "AI assistant: Offline"


def _text_has_forbidden_marker(text) -> bool:
    """Return ``True`` when ``text`` contains any internal-context marker."""
    if not text:
        return False
    lowered = str(text).lower()
    return any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS)


def _is_showable_ai_text(resp, provider) -> bool:
    """Return ``True`` only for a genuine, safe-to-show AI answer.

    Returns ``False`` (so the UI shows a clean deterministic-fallback message)
    when any of the following hold:

    * ``resp`` is ``None`` or unsuccessful.
    * The provider is the offline ``NullLLMProvider`` (its text is the echoed
      raw prompt).
    * ``resp.text`` is empty / whitespace.
    * ``resp.text`` contains ANY forbidden marker (case-insensitive), starts with
      the context header, or otherwise looks like the echoed prompt (a JSON
      context block).
    """
    if resp is None or not getattr(resp, "success", False):
        return False
    if _provider_is_offline(provider):
        return False

    text = getattr(resp, "text", None)
    if not text or not str(text).strip():
        return False

    if _text_has_forbidden_marker(text):
        return False

    stripped = str(text).lstrip()
    if stripped.startswith("ANALYSIS CONTEXT"):
        return False
    # A JSON context block leaks as an object literal near the top of the echo.
    if stripped.startswith("{") and '"' in stripped:
        return False

    return True


def _normalize_heading(value) -> str:
    """Lowercase, strip markdown/punctuation for loose heading comparison."""
    s = str(value or "").strip().lstrip("#").strip()
    s = s.strip("*").strip().rstrip(":").strip().lower()
    return s


def _strip_redundant_title(text: str, heading=None) -> str:
    """Drop a redundant leading title line from ``text`` for display only.

    The UI already renders a section heading (e.g. "AI What-If Explanation"), and
    Mistral responses often begin with their OWN title line (e.g. "What-If
    Scenario Explanation (Demand Increase of 30%)"). When the response's first
    line is a markdown heading, or loosely restates the supplied ``heading`` /
    common section titles, it is removed so the title is not shown twice. This is
    presentation-only trimming: it never alters the substantive content
    (Key Takeaway / Key Impacts / Business Interpretation / Traceability, etc.).
    """
    if not text:
        return text
    lines = text.split("\n")
    first = lines[0].strip()
    if not first:
        return text

    norm_first = _normalize_heading(first)
    redundant = False
    # A markdown heading line (# / ##) as the very first line is redundant with
    # the UI-supplied section heading.
    if first.lstrip().startswith("#"):
        redundant = True
    # First line loosely restates the supplied section heading.
    elif heading and norm_first == _normalize_heading(heading):
        redundant = True
    # First line restates a generic self-title the model tends to emit.
    elif any(
        norm_first.startswith(prefix)
        for prefix in (
            "what-if scenario explanation",
            "what-if explanation",
            "ai what-if explanation",
            "executive summary",
            "ai executive summary",
            "vendor risk explanation",
            "ai explanation",
        )
    ):
        redundant = True

    if not redundant:
        return text
    # Remove the first line and any immediately-following blank line.
    rest = lines[1:]
    while rest and not rest[0].strip():
        rest = rest[1:]
    return "\n".join(rest).strip() or text


def _render_ai_answer(resp, provider, heading=None) -> None:
    """Render an ``LLMResponse`` behind the raw-context guard, answer-first.

    Only a genuine, showable AI answer is written. In every other case (offline
    provider, failure, empty text, or text that looks like the internal
    prompt/context) a clean deterministic-fallback message is shown instead. The
    raw prompt / analysis-context JSON is never rendered.

    The answer is rendered EXACTLY ONCE. When the response begins with its own
    redundant title line (duplicating the UI section ``heading``) that title is
    stripped for display. A one-line key takeaway is shown only when it is
    genuinely shorter than and distinct from the answer AND is not merely that
    stripped title. Substantive content is never removed.
    """
    if not _is_showable_ai_text(resp, provider):
        st.info(_OFFLINE_FALLBACK_TEXT)
        return

    text = _strip_redundant_title(str(resp.text).strip(), heading)
    # Collapse any model-emitted duplicated opening paragraph. The UI does NOT
    # synthesize its own "Key takeaway" (that previously duplicated the opening).
    text = dedupe_ai_text(text)

    # The answer is rendered EXACTLY ONCE.
    st.write(text)


# ---- AI interaction record (for the dynamic audit) --------------------------


def _interaction_key(state_prefix: str) -> str:
    """Session-state key holding the MOST RECENT AI interaction for the scenario."""
    return f"{state_prefix}:last_interaction"


def _record_interaction(
    state_prefix, kind, response, *, vendor=None, question=None, scenario_result=None
) -> None:
    """Store the most recent real AI interaction so the audit can reflect it.

    ``kind`` is a human label (e.g. "Executive Summary", "Vendor Explanation",
    "User Question", "What-If Explanation"). ``response`` is the ``LLMResponse``
    actually returned by the provider. ``scenario_result`` is the authoritative
    :class:`~vendor_forecasting_agent.whatif_scenarios.ScenarioResult` for a
    what-if interaction (``None`` otherwise) so the audit can show the exact
    deterministic baseline + what-if figures that were supplied to the model.
    This never parses the response into any authoritative field; it only records
    what happened for traceability.
    """
    st.session_state[_interaction_key(state_prefix)] = {
        "kind": kind,
        "vendor": vendor,
        "question": question,
        "response": response,
        "scenario_result": scenario_result,
    }


def _render_ai_interaction_result(kind_label, resp, provider) -> None:
    """Render a freshly generated AI interaction result under a clear label.

    Uses the same raw-context guard as everywhere else: genuine answers are shown
    under ``kind_label``; offline / failed / echoed-prompt responses fall back to
    a clean message. The raw prompt / context JSON is never rendered.
    """
    st.markdown(f"**{kind_label}**")
    _render_ai_answer(resp, provider, heading=kind_label)


# ---- Sidebar ----------------------------------------------------------------


def _select_scenario():
    """Render the scenario dropdown (default preselected) and return the choice."""
    scenarios = list_scenarios()
    keys = [s.key for s in scenarios]
    labels = {s.key: s.label for s in scenarios}
    default_index = keys.index(DEFAULT_SCENARIO_KEY) if DEFAULT_SCENARIO_KEY in keys else 0

    selected_key = st.sidebar.selectbox(
        "Scenario",
        options=keys,
        index=default_index,
        format_func=lambda key: labels.get(key, key),
        help="Pick a curated demo scenario. Changing it reruns the analysis.",
    )
    return get_scenario(selected_key)


def _render_sidebar_context(scenario, request, provider) -> None:
    """Render the compact sidebar Analysis Context and AI assistant status.

    A compact "Analysis Context" (scenario / product / component / vendors
    analyzed) plus the AI assistant status. The detailed AI Audit lives at the
    bottom of the MAIN page, not in the sidebar.
    """
    st.sidebar.markdown(f"**Scenario:** {scenario.label}")
    st.sidebar.markdown(
        f"**Product:** {request.product_name} ({request.product_id})"
    )
    st.sidebar.markdown(
        f"**Component:** {request.component_name} ({request.component_id})"
    )
    st.sidebar.markdown(
        "**Vendors analyzed:** " + ", ".join(view_model.relevant_vendors(request))
    )

    st.sidebar.divider()
    st.sidebar.subheader("AI Assistant")
    st.sidebar.caption(_ai_status_label(provider))
    st.sidebar.caption("Role: Explanation & Q&A")


# ---- Page header ------------------------------------------------------------


def _render_header(request) -> None:
    """Render the main page header, product/component line and demo badge."""
    title_col, badge_col = st.columns([5, 1])
    with title_col:
        st.title("Vendor Forecasting Agent")
        st.caption(
            "Supplier Risk \u2022 Lead-Time Forecast \u2022 Inventory Impact \u2022 "
            "Recommended Action"
        )
    with badge_col:
        st.markdown(
            "<div style='text-align:right;margin-top:8px;'>"
            + _badge("Demo Environment", _NEUTRAL_COLOR)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"Product: **{request.product_name} ({request.product_id})**")
    st.markdown(
        f"Component: **{request.component_name} ({request.component_id})**"
    )
    st.divider()


# ---- KPI band ---------------------------------------------------------------


def _render_summary_band(snapshot, errors) -> None:
    """Top-level KPI tiles + a tidy scenario status line."""
    metrics = view_model.summary_metrics(snapshot)
    col1, col2, col3 = st.columns(3)
    col1.metric("Vendors Analyzed", metrics["vendors_analyzed"])
    col2.metric("High / Critical", metrics["high_critical"])
    col3.metric("Vendors with Potential Exposure", metrics["with_inventory_exposure"])

    if not errors:
        st.caption("Scenario analysis completed.")
    else:
        st.caption(
            f"Scenario completed with {len(errors)} note(s); deterministic "
            "results below remain authoritative."
        )


# ---- Executive decision -----------------------------------------------------


def _issue_label(drivers):
    """Map the first driver id to a friendly label using view_model's mapping."""
    if not drivers:
        return view_model._NO_RISK_ISSUE_LABEL
    return view_model._RISK_ISSUE_LABELS.get(drivers[0], drivers[0])


def _deterministic_decision_statement(snapshot, request):
    """Compose a faithful one-line decision statement from deterministic data.

    Uses :func:`llm_context.build_executive_context` (highest-risk selection over
    EXISTING risk scores) plus the per-vendor recommended action and driver
    phrasing. Returns ``(statement, impact_line, detail)`` where ``detail`` is a
    dict of supporting figures for the executive block, or ``(None, None, None)``
    when there is nothing to summarize.
    """
    ctx = build_executive_context(snapshot, request)
    summary = ctx.get("summary", {})
    vendors = ctx.get("vendors", [])
    if not vendors:
        return None, None, None

    total_exposure = summary.get("total_potential_exposure_units")
    highest = summary.get("highest_risk") or {}
    highest_id = highest.get("vendor_id")
    highest_level = (highest.get("risk_level") or "").strip().lower()

    by_id = {v.get("vendor_id"): v for v in vendors}

    elevated = highest_level in {"high", "critical"}

    if elevated and highest_id in by_id:
        target = by_id[highest_id]
    else:
        # No elevated risk: highlight the safest / preferred vendor instead.
        preferred = next(
            (v for v in vendors if str(v.get("recommended_action") or "").lower() == "prefer"),
            None,
        )
        target = preferred or by_id.get(summary.get("lowest_risk", {}).get("vendor_id")) or vendors[0]

    vendor_name = target.get("vendor_id", "n/a")
    action = target.get("recommended_action")
    action_text = str(action).upper() if action else "REVIEW"
    target_drivers = target.get("risk_drivers") or []
    driver_phrase = _driver_phrasing(
        {
            "main_risk_drivers": target_drivers,
            "main_risk_issue": _issue_label(target_drivers),
        }
    ).rstrip(".")

    if elevated:
        statement = (
            f"**{vendor_name} is the highest-risk supplier and is recommended to "
            f"{action_text} due to {driver_phrase.lower()}.**"
        )
    else:
        statement = (
            f"**No supplier shows elevated risk; {vendor_name} is the preferred "
            f"choice and is recommended to {action_text}.**"
        )

    impact_line = None
    if total_exposure is not None:
        impact_line = (
            f"Potential inventory exposure: {_fmt_units(total_exposure, 2)}."
        )

    detail = {
        "vendor_name": vendor_name,
        "risk_score": target.get("risk_score"),
        "driver_count": len(target_drivers),
        "total_exposure": total_exposure,
        "recommendations": [
            (v.get("vendor_id", "n/a"), v.get("recommended_action"))
            for v in vendors
        ],
    }
    return statement, impact_line, detail


def _render_executive_summary(snapshot, request, provider, state_prefix) -> None:
    """One concise, deterministic Executive Decision block (no AI prose here).

    Shows the deterministic one-line decision, the headline figures (risk / driver
    count / potential exposure) for the highest-risk supplier, and a compact
    per-vendor recommended-action list. AI prose lives in "Ask the Agent"; it is
    intentionally NOT duplicated here.
    """
    st.subheader("Executive Decision")

    statement, impact_line, detail = _deterministic_decision_statement(
        snapshot, request
    )
    if not statement:
        st.info("No vendors to summarize for the selected scenario.")
        return

    st.markdown(statement)

    score_text = _fmt_score(detail.get("risk_score"))
    driver_count = detail.get("driver_count", 0)
    exposure_text = _fmt_units(detail.get("total_exposure"), 2)
    st.markdown(
        f"Risk: **{score_text}** \u00b7 Risk drivers: **{driver_count}** "
        f"\u00b7 Potential exposure: **{exposure_text}**"
    )

    st.markdown("**Recommended actions**")
    for vendor_name, action in detail.get("recommendations", []):
        action_text = str(action).upper() if action else "N/A"
        st.markdown(f"- {vendor_name} \u2192 **{action_text}**")

    st.caption(
        "Risk score is calculated by deterministic rules across 6 "
        "supplier-performance dimensions."
    )

    # --- AI Executive Summary interaction (explicit, on-demand) --------------
    # Distinct from the deterministic Executive Decision above: this calls Mistral
    # (via decision_support -> llm_context) ONLY when the user clicks, and only
    # summarizes the authoritative deterministic results.
    summary_state = f"{state_prefix}:exec_summary"
    if st.button("\u2728 Generate Executive Summary", key=f"{state_prefix}:exec_btn"):
        with st.spinner("Generating executive summary..."):
            resp = decision_support.generate_executive_summary(
                snapshot, request, provider
            )
        st.session_state[summary_state] = resp
        _record_interaction(state_prefix, "Executive Summary", resp)

    resp = st.session_state.get(summary_state)
    if resp is not None:
        _render_ai_interaction_result("AI Executive Summary", resp, provider)


# ---- Vendor risk cards ------------------------------------------------------


def _vendor_action(snapshot, explanation, request, vendor_id):
    """Read a vendor's recommended action (read-only; no recompute)."""
    view = view_model.selected_vendor_view(snapshot, explanation, vendor_id, request)
    if view is None:
        return None
    return view.get("recommended_action")


def _render_vendor_card(container, row, action) -> None:
    """Render one compact display-only vendor risk card into ``container``.

    Vendor cards are display-only. They all use the SAME neutral border; there is
    no selected/focused visual state (the selectbox below is the only vendor
    selector), so no card can look misleadingly "active".
    """
    vendor = row.get("Vendor", "n/a")
    level = row.get("Risk Level")
    score = row.get("Risk Score")
    lead = row.get("Expected Lead Time")
    color = _risk_color(level)

    with container:
        border = "1px solid #e5e7eb"
        st.markdown(
            f"<div style='border:{border};border-radius:10px;padding:12px 14px;"
            "margin-bottom:6px;'>"
            f"<div style='font-weight:700;font-size:1.02rem;margin-bottom:6px;'>"
            f"{vendor}</div>"
            + _badge(level or "unknown", color)
            + f"<div style='font-size:0.85rem;color:#374151;margin-top:8px;'>"
            f"Risk score: <b>{_fmt_score(score)}</b></div>"
            + _score_bar(score, color)
            + f"<div style='font-size:0.85rem;color:#374151;'>"
            f"Expected lead time: <b>{_fmt_days(lead)}</b></div>"
            + "<div style='font-size:0.85rem;color:#374151;margin-top:6px;'>"
            "Recommendation: "
            + (_badge(action, _action_color(action)) if action else "<b>n/a</b>")
            + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )


def _render_vendor_risk_cards(snapshot, explanation, request) -> None:
    """Primary visual: one display-only card per vendor, up to 3 per row.

    Cards are purely informational and identically styled; the ``st.selectbox``
    below is the sole vendor-selection mechanism.
    """
    st.subheader("Vendor Risk Overview")
    rows = view_model.vendor_risk_overview(snapshot)
    if not rows:
        st.info("No vendors to display.")
        return

    per_row = 3
    for start in range(0, len(rows), per_row):
        chunk = rows[start:start + per_row]
        cols = st.columns(per_row)
        for col, row in zip(cols, chunk):
            action = _vendor_action(snapshot, explanation, request, row.get("Vendor"))
            _render_vendor_card(col, row, action)

    # Secondary charts, tucked into an expander.
    with st.expander("Show comparison charts", expanded=False):
        risk_by_vendor = {
            row["Vendor"]: row["Risk Score"]
            for row in rows
            if row.get("Risk Score") is not None
        }
        lead_by_vendor = {
            row["Vendor"]: row["Expected Lead Time"]
            for row in rows
            if row.get("Expected Lead Time") is not None
        }
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if risk_by_vendor:
                st.caption("Risk score by vendor")
                st.bar_chart(risk_by_vendor)
            else:
                st.caption("Risk score by vendor: n/a")
        with chart_col2:
            if lead_by_vendor:
                st.caption("Expected lead time by vendor")
                st.bar_chart(lead_by_vendor)
            else:
                st.caption("Expected lead time by vendor: n/a")

    # Full comparison table, tucked into a separate expander.
    with st.expander("Full comparison table", expanded=False):
        st.dataframe(rows, width="stretch", hide_index=True)


# ---- Selected vendor: concise summary first, detail on demand ---------------


def _selected_vendor_explanation(view) -> str:
    """Compose a concise, SELECTED-VENDOR-ONLY explanation from ``view`` fields.

    Reads only this vendor's already-computed deterministic values (risk level /
    score, drivers, expected lead time, inventory exposure, recommended action)
    and renders one short paragraph. It NEVER uses the shared multi-vendor
    ``explanation_text`` (which covers every vendor) and recomputes nothing, so
    no other vendor's information can leak into this vendor's section. Triggered
    drivers are shown by their friendly business labels.
    """
    level = (view.get("risk_level") or "unknown")
    score = _fmt_score(view.get("risk_score"))
    action = (view.get("recommended_action") or "n/a")
    lead = _fmt_days(view.get("expected_lead_time"))
    exposure = _fmt_units(view.get("potential_exposure_units"), 2)
    driver_phrase = _driver_phrasing(view).rstrip(".")
    drivers = view.get("main_risk_drivers") or []

    parts = [
        f"Risk is assessed as {str(level).upper()} ({score}) driven by "
        f"{driver_phrase.lower()}."
    ]
    if drivers:
        parts.append(
            "Triggered risk drivers: " + ", ".join(_friendly_drivers(drivers)) + "."
        )
    parts.append(
        f"Expected lead time is {lead}, with potential inventory exposure of "
        f"{exposure}."
    )
    parts.append(
        f"Based on the deterministic analysis, the recommended action is "
        f"{str(action).upper()}."
    )
    return " ".join(parts)


def _render_detail_body(view, trend_label) -> None:
    """Analyst-level detail body for the selected vendor (inside View Details)."""
    st.markdown("### Risk Dimensions")
    st.caption(
        "Supporting evidence for each of the six deterministic supplier-risk "
        "dimensions, with the triggered status from the risk rules."
    )
    triggered = set(view.get("main_risk_drivers") or [])
    # (dimension label, triggered rule id, evidence value) -- all read-only from
    # the deterministic view; nothing is recomputed here.
    dimensions = [
        ("On-time delivery", "on_time_delivery_low", _fmt_ratio(view["on_time_rate"])),
        ("Lead-time trend", "lead_time_declining", trend_label),
        ("Quantity fulfillment", "quantity_fulfillment_low", _fmt_ratio(view["quantity_fulfillment_rate"])),
        ("Capacity utilization", "capacity_utilization_high", _fmt_ratio(view["capacity_utilization"])),
        ("Allocation / commitment", "allocation_low", _fmt_ratio(view["allocation_ratio"])),
        ("Quality / defect rate", "defect_rate_high", _fmt_ratio(view["defect_rate"])),
    ]
    st.table(
        [
            {
                "Dimension": label,
                "Evidence": evidence,
                "Risk status": ("At risk" if rule_id in triggered else "OK"),
            }
            for label, rule_id, evidence in dimensions
        ]
    )

    st.markdown("### Lead-Time Forecast")
    lt1, lt2, lt3 = st.columns(3)
    lt1.metric("Historical average", _fmt_days(view["avg_lead_time_days"]))
    lt2.metric("Expected lead time", _fmt_days(view["expected_lead_time"]))
    lt3.metric("Trend", trend_label)
    st.caption(
        "Lead-time forecast is derived from historical supplier performance "
        "using deterministic forecasting logic."
    )

    st.markdown("### Forecast / Supporting metrics")
    left, right = st.columns(2)
    with left:
        st.table(
            [
                {"Metric": "On-time rate", "Value": _fmt_ratio(view["on_time_rate"])},
                {"Metric": "Avg lead time", "Value": _fmt_days(view["avg_lead_time_days"])},
                {"Metric": "Expected lead time", "Value": _fmt_days(view["expected_lead_time"])},
                {"Metric": "Quantity fulfilment", "Value": _fmt_ratio(view["quantity_fulfillment_rate"])},
            ]
        )
    with right:
        st.table(
            [
                {"Metric": "Capacity utilization", "Value": _fmt_ratio(view["capacity_utilization"])},
                {"Metric": "Allocation ratio", "Value": _fmt_ratio(view["allocation_ratio"])},
                {"Metric": "Defect rate", "Value": _fmt_ratio(view["defect_rate"])},
            ]
        )

    st.markdown("### Inventory Impact")
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Current inventory", _fmt_units(view["current_inventory"], 0))
    ic2.metric("Inventory coverage", _fmt_coverage(view["inventory_coverage_days"]))
    ic3.metric("Expected lead time", _fmt_days(view["expected_lead_time"]))
    ic4, ic5 = st.columns(2)
    ic4.metric("Potential delay", _fmt_days(view["potential_delay_days"]))
    ic5.metric("Potential exposure", _fmt_units(view["potential_exposure_units"], 2))
    st.caption(
        "Potential exposure represents demand at risk from the lead-time gap."
    )

    st.markdown("### Recommendation")
    action = view.get("recommended_action")
    if action:
        st.markdown(
            "<div style='font-size:1.05rem;'>Recommended action: "
            + _badge(action, _action_color(action))
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("Recommended action: **n/a**")

    # Selected-vendor-only explanation. The shared ``explanation_text`` describes
    # EVERY vendor, so rendering it verbatim here leaked other vendors' details
    # into this vendor's section. Instead compose a concise explanation from THIS
    # vendor's own deterministic fields (read-only; no recompute, no cross-vendor
    # content).
    st.markdown("**Explanation**")
    st.write(_selected_vendor_explanation(view))


def _render_selected_vendor(view, selected, trend_label, snapshot, request, provider, state_prefix) -> None:
    """Concise selected-vendor summary first, analyst detail on demand."""
    st.subheader(f"Selected Vendor \u2014 {selected}")

    color = _risk_color(view["risk_level"])
    level_text = (view["risk_level"] or "unknown")
    st.markdown(
        f"<div style='font-size:1.05rem;'>"
        + _badge(level_text, color)
        + f"<span style='margin-left:10px;font-weight:600;'>"
        f"&nbsp;\u00b7 {_fmt_score(view['risk_score'])}</span>"
        + "</div>",
        unsafe_allow_html=True,
    )

    action = view.get("recommended_action")
    if action:
        st.markdown(
            "<div style='margin-top:8px;font-size:1.05rem;'>Recommendation: "
            + _badge(action, _action_color(action))
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("Recommendation: **n/a**")

    st.markdown(f"Main risk: {_driver_phrasing(view)}")

    drivers = view.get("main_risk_drivers") or []
    if drivers:
        st.markdown("**Why?**")
        for label in _friendly_drivers(drivers):
            st.markdown(f"- {label}")

    st.markdown(
        "Potential inventory exposure: "
        f"{_fmt_units(view['potential_exposure_units'], 2)}"
    )

    with st.expander("View Details", expanded=False):
        _render_detail_body(view, trend_label)

    # --- AI Explanation interaction for THIS selected vendor (on-demand) -----
    explain_state = f"{state_prefix}:vendor_explain:{selected}"
    if st.button("\u2728 Generate AI Explanation", key=f"{state_prefix}:vexp_btn:{selected}"):
        question = (
            f"Explain why {selected} received this risk level, lead-time forecast, "
            "inventory exposure and recommendation, based only on the deterministic "
            "analysis."
        )
        with st.spinner("Generating AI explanation..."):
            resp = decision_support.ask_vendor_agent(
                snapshot, request, question, provider, vendor_id=selected
            )
        st.session_state[explain_state] = resp
        _record_interaction(
            state_prefix, "Vendor Explanation", resp, vendor=selected, question=question
        )

    resp = st.session_state.get(explain_state)
    if resp is not None:
        _render_ai_interaction_result("AI Explanation", resp, provider)


# ---- Ask the agent ----------------------------------------------------------


def _latest_what_if_context(state_prefix, request):
    """Read the LATEST calculated What-If ScenarioResult and reflect it to context.

    Reads (never recomputes) the most recent scenario slug recorded under
    ``{state_prefix}:whatif_latest`` and the stored ``ScenarioResult`` under
    ``{state_prefix}:whatif_result:{slug}``. When a valid, current result exists
    it is reflected into the LLM context via
    :func:`whatif_scenarios.build_scenario_context`; otherwise ``None`` is
    returned (so no stale what-if is ever injected). A fresh Calculate Impact
    overwrites ``whatif_latest``, so only the latest valid scenario is used.
    """
    latest_slug = st.session_state.get(f"{state_prefix}:whatif_latest")
    if not latest_slug:
        return None
    result = st.session_state.get(f"{state_prefix}:whatif_result:{latest_slug}")
    if not isinstance(result, ScenarioResult):
        return None
    return build_scenario_context(result, request)


def _render_qa_transcript(history) -> None:
    """Render prior conversation turns as a compact plain-markdown transcript.

    Presentation only: reads the stored (already safe-to-show) turn text. The
    MOST RECENT answer is NOT rendered here; it is rendered once via
    :func:`_render_ai_answer` below so the raw-context guard / formatting apply
    to it exactly once (no double-rendering).
    """
    if not history:
        return
    # Render every turn EXCEPT the last assistant answer (shown via the guard).
    prior = history[:-1] if history and history[-1].get("role") == "assistant" else history
    if not prior:
        return
    st.markdown("**Recent conversation**")
    for turn in prior:
        role = turn.get("role")
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = "You" if role == "user" else "Agent"
        st.markdown(f"**{speaker}:** {text}")


def _render_qa(snapshot, request, provider, selected, state_prefix) -> None:
    """Answer-first AI Q&A about the vendors / comparison (guarded rendering).

    Adds short-term, session-only conversation memory (bounded to the most recent
    ~10 messages), lightweight context resolution (a vendor explicitly named in
    the question overrides the selected vendor; follow-ups fall back to the
    selected vendor + conversation history), and injection of the LATEST computed
    What-If scenario as authoritative context. No deterministic value is ever
    recomputed here; memory is session-only and never persisted.
    """
    st.subheader("Ask the Vendor Forecasting Agent")
    question_state = f"{state_prefix}:qa_question"
    answer_state = f"{state_prefix}:qa_answer"
    history_state = f"{state_prefix}:qa_history"

    st.caption("Suggested questions")
    button_cols = st.columns(len(_SUGGESTED_QUESTIONS))
    pending_question = None
    for col, suggestion in zip(button_cols, _SUGGESTED_QUESTIONS):
        if col.button(suggestion, key=f"{state_prefix}:suggest:{suggestion}"):
            pending_question = suggestion

    with st.form(key=f"{state_prefix}:qa_form"):
        typed = st.text_input(
            "Your question",
            value="",
            placeholder="Ask anything about these vendors or the comparison...",
        )
        submitted = st.form_submit_button("Ask")
    if submitted and typed.strip():
        pending_question = typed.strip()

    if pending_question:
        # --- Context resolution: an explicitly named vendor overrides `selected`;
        # follow-ups (no explicit vendor) fall back to `selected` and rely on the
        # conversation history to resolve the reference. Data-driven from ids.
        vendor_ids = view_model.relevant_vendors(request)
        resolved_vendor = (
            resolve_explicit_vendor(pending_question, vendor_ids) or selected
        )

        # --- Short-term conversation memory (bounded, session-only) ----------
        prior_history = trim_history(
            st.session_state.get(history_state) or [], 10
        )
        # --- Latest What-If injection (never stale) --------------------------
        what_if_context = _latest_what_if_context(state_prefix, request)

        with st.spinner("Asking the Vendor Forecasting Agent..."):
            resp = decision_support.ask_vendor_agent(
                snapshot,
                request,
                pending_question,
                provider,
                vendor_id=resolved_vendor,
                history=prior_history,
                what_if_context=what_if_context,
            )
        st.session_state[question_state] = pending_question
        st.session_state[answer_state] = resp
        # Record which vendor this answer is about (the RESOLVED vendor), so the
        # AI Audit only reuses it as the conclusion when the same vendor is
        # selected.
        st.session_state[f"{state_prefix}:qa_vendor"] = resolved_vendor
        _record_interaction(
            state_prefix, "User Question", resp,
            vendor=resolved_vendor, question=pending_question,
        )

        # --- Append this exchange to memory, then trim to the last ~10 -------
        answer_text = (
            str(resp.text).strip()
            if _is_showable_ai_text(resp, provider)
            else "[AI unavailable]"
        )
        updated = list(st.session_state.get(history_state) or [])
        updated.append({"role": "user", "text": pending_question})
        updated.append({"role": "assistant", "text": answer_text})
        st.session_state[history_state] = trim_history(updated, 10)

    history = st.session_state.get(history_state) or []
    last_question = st.session_state.get(question_state)
    last_answer = st.session_state.get(answer_state)
    if last_question and last_answer is not None:
        # Prior turns as a compact transcript (from stored text)...
        _render_qa_transcript(history)
        # ...and the CURRENT/last answer once, behind the raw-context guard.
        st.markdown(f"**Q:** {last_question}")
        _render_ai_answer(last_answer, provider)
        st.caption("Based on the deterministic analysis.")


# ---- What-if ----------------------------------------------------------------


# ---- What-If scenarios (deterministic; Mistral only explains) ---------------

# Presentation-only labels; the numeric parameter for each scenario is entered
# inside the form and never triggers Mistral (only Calculate Impact runs the
# deterministic calc, and only Explain What-If calls the LLM).
_WHATIF_SCENARIOS = (
    "Supplier Delay",
    "Demand Increase",
    "Quantity Fulfillment Reduction",
)


def _fmt_change_days(value) -> str:
    """Signed days change for a display table, ``"n/a"`` when ``None``."""
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):,.2f} days"


def _fmt_change_units(value) -> str:
    """Signed units change for a display table, ``"n/a"`` when ``None``."""
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{_fmt_units(abs(value), 2)}"


def _whatif_comparison_rows(result, request):
    """Build the scenario-appropriate BASELINE vs WHAT-IF rows from a ScenarioResult.

    Numbers come ONLY from ``result`` fields (plus the committed
    ``required_quantity`` read from ``request`` for the fulfillment baseline);
    formatting reuses the existing ``_fmt_*`` helpers. Rows are chosen per
    scenario so each scenario reads naturally, and potential exposure is ALWAYS
    labelled as potential.
    """
    rows = []
    scenario = result.scenario

    if scenario == "supplier_delay":
        rows.append(
            {
                "Metric": "Expected Lead Time",
                "Baseline": _fmt_days(result.baseline_expected_lead_time),
                "What-If": _fmt_days(result.whatif_expected_lead_time),
                "Change": _fmt_change_days(
                    result.whatif_expected_lead_time
                    - result.baseline_expected_lead_time
                ),
            }
        )
        rows.append(
            {
                "Metric": "Inventory Coverage",
                "Baseline": _fmt_coverage(result.baseline_inventory_coverage_days),
                "What-If": _fmt_coverage(result.whatif_inventory_coverage_days),
                "Change": _fmt_change_days(result.coverage_change_days),
            }
        )
    elif scenario == "demand_increase":
        rows.append(
            {
                "Metric": "Daily Demand",
                "Baseline": _fmt_units(result.baseline_daily_demand, 2),
                "What-If": _fmt_units(result.whatif_daily_demand, 2),
                "Change": _fmt_change_units(
                    result.whatif_daily_demand - result.baseline_daily_demand
                ),
            }
        )
        rows.append(
            {
                "Metric": "Inventory Coverage",
                "Baseline": _fmt_coverage(result.baseline_inventory_coverage_days),
                "What-If": _fmt_coverage(result.whatif_inventory_coverage_days),
                "Change": _fmt_change_days(result.coverage_change_days),
            }
        )
    elif scenario == "fulfillment_reduction":
        committed = float(request.required_quantity)
        rows.append(
            {
                "Metric": "Expected Replenishment",
                "Baseline": _fmt_units(committed, 2),
                "What-If": _fmt_units(
                    result.whatif_expected_replenishment_units, 2
                ),
                "Change": _fmt_change_units(
                    (result.whatif_expected_replenishment_units - committed)
                    if result.whatif_expected_replenishment_units is not None
                    else None
                ),
            }
        )

    rows.append(
        {
            "Metric": "Potential exposure (POTENTIAL)",
            "Baseline": _fmt_units(result.baseline_potential_exposure, 2),
            "What-If": _fmt_units(result.whatif_potential_exposure, 2),
            "Change": _fmt_change_units(result.exposure_change),
        }
    )
    if result.shortage_gap_units is not None:
        rows.append(
            {
                "Metric": "Shortage / Gap",
                "Baseline": "\u2014",
                "What-If": _fmt_units(result.shortage_gap_units, 2),
                "Change": "\u2014",
            }
        )
    return rows


def _whatif_interpretation(result) -> str:
    """One deterministic interpretation line derived only from ``result`` numbers."""
    scenario = result.scenario
    exposure = _fmt_units(result.whatif_potential_exposure, 2)
    if scenario == "supplier_delay":
        return (
            f"A {result.parameter_value:g}-day supplier delay extends expected "
            f"replenishment to {_fmt_days(result.whatif_expected_lead_time)}, "
            f"creating potential exposure of {exposure}."
        )
    if scenario == "demand_increase":
        return (
            f"A {result.parameter_value:g}% demand increase reduces inventory "
            f"coverage to {_fmt_coverage(result.whatif_inventory_coverage_days)} "
            f"and raises potential exposure to {exposure}."
        )
    if scenario == "fulfillment_reduction":
        return (
            f"At {result.parameter_value:g}% fulfillment, expected replenishment "
            f"falls to {_fmt_units(result.whatif_expected_replenishment_units, 2)}, "
            f"leaving a potential shortfall of "
            f"{_fmt_units(result.shortage_gap_units, 2)}."
        )
    return f"Potential exposure under this scenario is {exposure}."


def _run_whatif_scenario(scenario_label, snapshot, request, selected, param):
    """Dispatch to the matching deterministic scenario function (backend calc)."""
    if scenario_label == "Supplier Delay":
        return analyze_supplier_delay(snapshot, request, selected, float(param))
    if scenario_label == "Demand Increase":
        return analyze_demand_increase(snapshot, request, selected, float(param))
    return analyze_fulfillment_reduction(snapshot, request, selected, float(param))


def _render_what_if(snapshot, request, provider, selected, state_prefix) -> None:
    """Interactive multi-scenario what-if on the selected vendor.

    Three deterministic scenarios (Supplier Delay, Demand Increase, Quantity
    Fulfillment Reduction). Every number comes from a
    :class:`~vendor_forecasting_agent.whatif_scenarios.ScenarioResult`; Mistral
    only explains the computed result (via ``decision_support.explain_scenario``).

    The scenario selectbox drives which parameter input is shown; the numeric
    parameter lives inside an ``st.form`` so changing it alone never runs the
    calculation and never calls Mistral. Only Calculate Impact runs the
    deterministic calc; only the separate Explain What-If button calls the LLM.
    Each scenario's result and explanation are namespaced per-scenario so
    switching scenarios never compounds (every calc re-reads the baseline fresh).
    """
    st.subheader("What-If Scenario")
    st.markdown(f"**Vendor:** {selected}")

    scenario_label = st.selectbox(
        "Scenario",
        _WHATIF_SCENARIOS,
        key=f"{state_prefix}:whatif_scenario",
    )
    scenario_slug = (
        scenario_label.lower().replace(" ", "_").replace("/", "")
    )
    result_state = f"{state_prefix}:whatif_result:{scenario_slug}"
    explain_state = f"{state_prefix}:whatif_explain:{scenario_slug}"

    with st.form(key=f"{state_prefix}:whatif_form:{scenario_slug}"):
        if scenario_label == "Supplier Delay":
            param = st.number_input(
                "Supplier delay (days)",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )
        elif scenario_label == "Demand Increase":
            param = st.number_input(
                "Demand increase (%)",
                min_value=0.0,
                value=20.0,
                step=5.0,
            )
        else:  # Quantity Fulfillment Reduction
            param = st.number_input(
                "Fulfillment (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=5.0,
            )
        run_what_if = st.form_submit_button("Calculate Impact")

    if run_what_if:
        try:
            result = _run_whatif_scenario(
                scenario_label, snapshot, request, selected, param
            )
            st.session_state[result_state] = result
            # Record the latest calculated scenario so Ask-the-Agent injects the
            # LATEST (never stale) What-If as authoritative context. A fresh
            # Calculate Impact overwrites this, so only the newest is ever used.
            st.session_state[f"{state_prefix}:whatif_latest"] = scenario_slug
            # Clear any stale explanation for this scenario on a fresh calc.
            st.session_state.pop(explain_state, None)
        except ValueError as exc:
            st.session_state.pop(result_state, None)
            st.session_state.pop(explain_state, None)
            st.warning(f"Cannot run {scenario_label} for {selected!r}: {exc}")

    result = st.session_state.get(result_state)
    if result is not None:
        rows = _whatif_comparison_rows(result, request)
        st.table(rows)
        st.caption(
            "Potential exposure (POTENTIAL): demand at risk from the supplier "
            "lead-time gap — a planning signal, not a guaranteed loss."
        )
        if result.shortage_gap_units is not None:
            st.caption(
                "Shortage / Gap: inventory demand that may be uncovered before "
                "expected replenishment arrives. It is a distinct measure from "
                "Potential Exposure."
            )

        horizon = getattr(result, "inventory_horizon", None)
        if horizon is not None:
            st.caption(f"Inventory horizon: {horizon} periods.")

        st.markdown(_whatif_interpretation(result))

        if st.button(
            "\u2728 Explain What-If", key=f"{state_prefix}:whatif_explain_btn:{scenario_slug}"
        ):
            with st.spinner("Generating what-if explanation..."):
                resp = decision_support.explain_scenario(
                    result, provider, request=request
                )
            st.session_state[explain_state] = resp
            _record_interaction(
                state_prefix, "What-If Explanation", resp, vendor=selected,
                question=f"{scenario_label}: {result.parameter_value:g}",
                scenario_result=result,
            )

        explanation = st.session_state.get(explain_state)
        if explanation is not None:
            _render_ai_interaction_result(
                "AI What-If Explanation", explanation, provider
            )


# ---- AI Audit & Data Traceability (bottom of main page) ---------------------



def _latest_showable_ai_conclusion(snapshot, explanation, request, provider, selected, state_prefix):
    """Most recent showable AI answer for the scenario, else the deterministic line.

    The conclusion is SELECTED-VENDOR specific. It reuses the most recent Ask
    answer ONLY when that question was asked about this same vendor (so a
    conclusion about another vendor never appears here). Otherwise it falls back
    to a concise deterministic explanation of THIS vendor. Never returns raw
    context, and never the scenario-wide executive summary (which may refer to a
    different, highest-risk vendor).
    """
    qa_resp = st.session_state.get(f"{state_prefix}:qa_answer")
    qa_vendor = st.session_state.get(f"{state_prefix}:qa_vendor")
    if qa_vendor == selected and _is_showable_ai_text(qa_resp, provider):
        return str(qa_resp.text).strip()

    view = view_model.selected_vendor_view(snapshot, explanation, selected, request)
    if view is not None:
        return _selected_vendor_explanation(view)
    return "No AI conclusion available; deterministic analysis is authoritative."



_WHATIF_SCENARIO_LABELS = {
    "supplier_delay": "Supplier Delay",
    "demand_increase": "Demand Increase",
    "fulfillment_reduction": "Quantity Fulfillment Reduction",
}


def _render_audit_scenario_traceability(result) -> None:
    """Render Baseline vs What-If from a ScenarioResult for the AI Audit.

    Every value is read verbatim from the authoritative deterministic
    ``ScenarioResult`` (nothing is calculated in the UI or by the LLM). Only
    fields that are actually present (non-``None``) for the scenario are shown.
    """
    label = _WHATIF_SCENARIO_LABELS.get(result.scenario, result.scenario)
    st.markdown(f"- Scenario: **{label}**")
    st.markdown(
        f"- Parameter: **{result.parameter_value:g}** "
        f"({result.parameter_label})"
    )

    st.markdown("### Baseline")
    baseline_rows = [
        ("Daily demand", _fmt_units(result.baseline_daily_demand, 2)),
        ("Current inventory", _fmt_units(result.baseline_current_inventory, 0)),
        ("Expected lead time", _fmt_days(result.baseline_expected_lead_time)),
    ]
    if result.baseline_inventory_coverage_days is not None:
        baseline_rows.append(
            ("Inventory coverage", _fmt_coverage(result.baseline_inventory_coverage_days))
        )
    baseline_rows.append(
        ("Potential exposure", _fmt_units(result.baseline_potential_exposure, 2))
    )
    for field, value in baseline_rows:
        st.markdown(f"- {field}: **{value}**")

    st.markdown("### What-If")
    whatif_rows = [
        ("Daily demand", _fmt_units(result.whatif_daily_demand, 2)),
        ("Expected lead time", _fmt_days(result.whatif_expected_lead_time)),
    ]
    if result.whatif_expected_replenishment_units is not None:
        whatif_rows.append(
            ("Expected replenishment",
             _fmt_units(result.whatif_expected_replenishment_units, 2))
        )
    if result.whatif_inventory_coverage_days is not None:
        whatif_rows.append(
            ("Inventory coverage", _fmt_coverage(result.whatif_inventory_coverage_days))
        )
    whatif_rows.append(
        ("Potential exposure (POTENTIAL)",
         _fmt_units(result.whatif_potential_exposure, 2))
    )
    if result.shortage_gap_units is not None:
        whatif_rows.append(
            ("Shortage / Gap", _fmt_units(result.shortage_gap_units, 2))
        )
    for field, value in whatif_rows:
        st.markdown(f"- {field}: **{value}**")


def _render_ai_audit(snapshot, explanation, request, provider, selected, state_prefix) -> None:
    """Bottom-of-page, INTERACTION-AWARE AI Audit & Data Traceability.

    Before any real AI interaction the panel invites the user to trigger one.
    After a genuine Mistral call (executive summary, vendor explanation, user
    question or what-if explanation) it reflects the MOST RECENT interaction: the
    interaction type, the vendor/question where applicable, the model, the
    deterministic context actually supplied (read from the REAL
    :func:`llm_context.build_vendor_context`), the AI role, the actual generated
    response (behind the raw-context guard), and the traceability chain. NO raw
    JSON is ever shown, and the response is never parsed into authoritative fields.
    """
    with st.expander("AI Audit & Data Traceability", expanded=False):
        interaction = st.session_state.get(_interaction_key(state_prefix))

        if not interaction:
            st.info(
                "No AI interaction yet. Generate an executive summary, a vendor "
                "explanation, ask the agent, or explain a what-if to see the AI "
                "audit."
            )
            st.caption(
                "The deterministic analysis on this page is authoritative "
                "regardless of AI availability."
            )
            return

        kind = interaction.get("kind", "AI Interaction")
        audit_vendor = interaction.get("vendor") or selected
        question = interaction.get("question")
        resp = interaction.get("response")

        st.markdown("### AI Interaction")
        st.markdown(f"- Interaction type: **{kind}**")
        if interaction.get("vendor"):
            st.markdown(f"- Selected vendor: **{interaction['vendor']}**")
        if question:
            st.markdown(f"- Request: **{question}**")

        scenario_result = interaction.get("scenario_result")
        if scenario_result is not None:
            _render_audit_scenario_traceability(scenario_result)

        st.markdown("### AI Model")
        st.markdown(f"- **{_ai_status_label(provider).replace('AI assistant: ', '')}**")

        st.markdown("### Status")
        if _is_showable_ai_text(resp, provider):
            st.markdown("- **Success** (Mistral response shown below)")
        else:
            st.markdown(
                "- **Fallback** (AI unavailable; deterministic analysis remains "
                "authoritative)"
            )

        st.markdown("### Data / Context Referenced")
        ctx = build_vendor_context(snapshot, request, audit_vendor) or {}
        if ctx:
            metrics = ctx.get("metrics", {})
            risk = ctx.get("risk", {})
            impact = ctx.get("inventory_impact", {})
            rec = ctx.get("recommendation", {})
            referenced = [
                ("Vendor", audit_vendor),
                ("On-time rate", _fmt_ratio(metrics.get("on_time_rate"))),
                ("Avg lead time", _fmt_days(metrics.get("avg_lead_time_days"))),
                ("Expected lead time", _fmt_days(metrics.get("expected_lead_time"))),
                ("Quantity fulfilment", _fmt_ratio(metrics.get("quantity_fulfillment_rate"))),
                ("Capacity utilization", _fmt_ratio(metrics.get("capacity_utilization"))),
                ("Allocation ratio", _fmt_ratio(metrics.get("allocation_ratio"))),
                ("Defect rate", _fmt_ratio(metrics.get("defect_rate"))),
                ("Risk score", _fmt_score(risk.get("risk_score"))),
                ("Risk level", (risk.get("risk_level") or "n/a")),
                ("Inventory coverage", _fmt_coverage(impact.get("inventory_coverage_days"))),
                ("Potential exposure", _fmt_units(impact.get("potential_exposure_units"), 2)),
                ("Recommendation", (rec.get("action") or "n/a")),
            ]
            for field, value in referenced:
                st.markdown(f"- {field}: **{value}**")
        else:
            st.markdown("- Executive deterministic analysis (all vendors)")
        st.caption("Source: Vendor Forecasting Agent \u2014 Deterministic Analysis Engine")

        st.markdown("### AI Role")
        for line in (
            "Explains the deterministic analysis",
            "Summarizes results and answers user questions",
            "Does NOT calculate authoritative risk",
            "Does NOT calculate authoritative inventory impact",
            "Does NOT override deterministic recommendations",
        ):
            st.markdown(f"- {line}")

        st.markdown("### Generated AI Response")
        _render_ai_answer(resp, provider)

        st.markdown("### Traceability")
        st.markdown(
            "Deterministic Analysis \u2192 Structured LLM Context \u2192 Mistral "
            "\u2192 Generated AI Response"
        )

def _trend_label(snapshot, vendor_id) -> str:
    """Return a friendly lead-time trend label from the deterministic direction.

    Reads ``snapshot.trends[vendor_id].direction`` (read-only; no recompute) and
    maps it to business wording. Missing vendors / directions yield ``"n/a"``.
    """
    trends = getattr(snapshot, "trends", None) or {}
    trend = trends.get(vendor_id)
    direction = getattr(trend, "direction", None) if trend is not None else None
    return {
        "declining": "Deteriorating",
        "improving": "Improving",
        "stable": "Stable",
    }.get(direction, "n/a")


# ---- Selection helper -------------------------------------------------------



def main() -> None:
    """Render the structured vendor-risk analysis dashboard.

    Simple structured layout: header, executive decision, KPI band, vendor risk
    overview (display-only cards), a plain selectbox for vendor selection,
    selected-vendor detail (with View Details expander), Ask the Agent, What-If,
    and AI Audit at the bottom of the main page. Every number is sourced from the
    deterministic snapshot/explanation via ``view_model``; nothing is recomputed
    here.
    """
    st.set_page_config(page_title="Vendor Forecasting Agent", layout="wide")

    # Obtain the provider ONCE per run (lazy; never touches boto3/AWS eagerly).
    provider = decision_support.get_default_provider()

    st.sidebar.header("Analysis Context")
    selected_scenario = _select_scenario()
    request = selected_scenario.to_upstream_request()
    _render_sidebar_context(selected_scenario, request, provider)

    # Namespace session-state by scenario key so a scenario switch drops stale
    # AI answers, what-if results and vendor selection automatically.
    state_prefix = f"vfa:{selected_scenario.key}"

    output = run_scenario(selected_scenario)
    snapshot = output.snapshot
    explanation = output.explanation

    _render_header(request)
    _render_summary_band(snapshot, output.errors)
    st.divider()
    _render_executive_summary(snapshot, request, provider, state_prefix)
    st.divider()

    ids = view_model.vendor_ids(snapshot)
    if not ids:
        st.info("No vendors to display for the selected scenario.")
        return

    # Display-only vendor cards (no Select buttons, no custom selection state).
    _render_vendor_risk_cards(snapshot, explanation, request)

    st.divider()

    # A single plain selectbox is the ONLY vendor-selection control. Streamlit
    # manages its own widget state; there is no pending/selected_key logic.
    selected = st.selectbox("Select a vendor", ids)

    view = view_model.selected_vendor_view(snapshot, explanation, selected, request)
    if view is None:
        st.warning(f"Vendor {selected!r} not found.")
        return

    trend_label = _trend_label(snapshot, selected)


    _render_selected_vendor(view, selected, trend_label, snapshot, request, provider, state_prefix)
    st.divider()
    _render_qa(snapshot, request, provider, selected, state_prefix)
    st.divider()
    _render_what_if(snapshot, request, provider, selected, state_prefix)
    st.divider()
    _render_ai_audit(snapshot, explanation, request, provider, selected, state_prefix)


main()
