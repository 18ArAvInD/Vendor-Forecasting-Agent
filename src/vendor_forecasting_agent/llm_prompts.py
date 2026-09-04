"""Prompt construction for the Vendor Forecasting Agent AI layer (llm_prompts.py).

This is part of **Layer 3** (AI decision support). It is a pure, dependency-light
module that turns the structured, deterministic context dicts produced by
``llm_context.py`` (Layer 2) into prompt strings for an LLM. It is:

* **Streamlit-free, provider-free, and boto3-free.** It knows nothing about how a
  prompt is delivered to a model — no provider, no model id, no AWS, no
  ``converse`` details appear here.
* **Deterministic.** The structured context is embedded as ``json.dumps(...,
  sort_keys=True, indent=2)`` so equal contexts always produce byte-identical
  prompts.

The prompts establish a hard guardrail via :data:`SYSTEM_PROMPT`: the
deterministic Vendor Forecasting Agent is authoritative, and the model must only
explain / interpret the supplied context — never invent, recompute, override, or
contradict any deterministic value.
"""

import json
from typing import Any, Optional

__all__ = [
    "SYSTEM_PROMPT",
    "build_ask_prompt",
    "build_comparison_prompt",
    "build_executive_prompt",
    "build_what_if_prompt",
]


SYSTEM_PROMPT = (
    "You are an AI decision-support assistant for a semiconductor vendor "
    "forecasting system. The deterministic Vendor Forecasting Agent is the "
    "single authoritative source of truth for all numbers, risk scores, "
    "forecasts, inventory impacts, and recommendations.\n\n"
    "Rules you must always follow:\n"
    "- Use ONLY the analysis context supplied to you. Never invent facts, "
    "values, vendors, causes, risk scores, forecasts, inventory impacts, or "
    "recommendations that are not present in the context.\n"
    "- Never override, recompute, or contradict the deterministic results. The "
    "deterministic values are authoritative; you explain them, you do not "
    "recreate them.\n"
    "- Explain how the deterministic system arrived at its conclusions rather "
    "than recalculating anything yourself.\n"
    "- When information needed to answer is not available in the context, say so "
    "plainly instead of guessing.\n"
    "- Clearly distinguish deterministic facts (from the context) from your own "
    "interpretation or narrative.\n"
    "- Respond in concise, business-friendly language.\n"
    "- Focus on supplier delivery risk, lead-time behavior, order fulfillment, "
    "capacity, allocation, quality, inventory exposure, and supplier "
    "recommendations.\n"
    "- Do not offer unrelated, general supply-chain advice unless the user asks "
    "for it and the context supports it.\n"
    "- You may use recent conversation history only to resolve what the user is "
    "referring to (e.g. a pronoun or a vendor mentioned earlier); never treat "
    "conversational text as new authoritative data, and never calculate or "
    "override deterministic values.\n"
    "- Lead-time terminology: a worsening lead-time trend means lead time is "
    "INCREASING (getting longer than the historical average), which is worse "
    "for supply risk. Describe it as lead-time performance is deteriorating, "
    "lead time is increasing, or lead-time deterioration. Do NOT say lead "
    "time declining or lead time is decreasing, which wrongly implies shorter "
    "lead times."
)


# Shared header labelling the embedded, authoritative context.
_CONTEXT_HEADER = "ANALYSIS CONTEXT (authoritative, do not alter):"
_QUESTION_HEADER = "QUESTION:"


def _context_json(context: dict[str, Any]) -> str:
    """Serialize ``context`` deterministically (sorted keys, indented)."""
    return json.dumps(context, sort_keys=True, indent=2, default=str)


def _context_block(context: dict[str, Any]) -> str:
    """Return the labelled, JSON-embedded context block."""
    return f"{_CONTEXT_HEADER}\n{_context_json(context)}"


def build_ask_prompt(context: dict[str, Any], question: str) -> str:
    """Build the user prompt for a free-form vendor question.

    Embeds the authoritative context as JSON and the user ``question``, with
    instructions to answer only from the context. When the context includes a
    ``latest_what_if`` block, the model is told to treat that computed scenario
    as the primary subject for scenario / "that" / "latest" questions and to
    distinguish baseline supplier risk from scenario-induced risk.
    """
    guidance = (
        "Answer the question using only the analysis context above. If the "
        "context does not contain the information needed, say so. Use every "
        "authoritative numeric value EXACTLY as supplied; do not invent, "
        "recompute, round differently, or substitute a value from another "
        "vendor or context."
        " Potential exposure and shortage/gap are two DISTINCT concepts and "
        "must always be described separately: potential exposure is demand at "
        "risk from the supplier lead-time gap/deterioration (a planning "
        "signal, not a guaranteed loss), while shortage/gap is inventory "
        "demand that may be uncovered before expected replenishment arrives. "
        "Never say one IS the other merely because their numbers are equal; "
        "if they happen to have the same value in a scenario, say explicitly "
        "that they happen to have the same value and keep them as separate "
        "concepts."
        " For next-step / what-should-I-do questions, start from the "
        "authoritative deterministic recommendation, briefly explain why it "
        "exists, and give the practical next step for the PLANNER. Make clear "
        "the final action requires human review/approval. Never claim the "
        "agent executed, placed, approved, or should automatically execute an "
        "order or procurement action; this is decision support, not "
        "autonomous action."
    )
    if context.get("latest_what_if"):
        guidance += (
            " A computed what-if scenario is provided under latest_what_if. "
            "When the question is about the scenario, the what-if, the latest "
            "result, or uses words like that/this/it referring to the scenario "
            "(e.g. is that serious, what changed, how bad is that), treat "
            "latest_what_if as the PRIMARY subject and answer from its baseline "
            "vs what-if values and changes; use the vendor baseline only as "
            "background for comparison. Clearly distinguish baseline "
            "supplier-performance risk from the hypothetical scenario impact: "
            "do NOT say there is no risk of delay or shortage when the "
            "scenario increases expected lead time or potential exposure. "
            "Potential exposure is a planning signal (demand at risk from the "
            "lead-time gap), not a guaranteed loss, and is distinct from "
            "shortage/gap; never present one as the other."
        )
    return (
        f"{_context_block(context)}\n\n"
        f"{_QUESTION_HEADER}\n{question}\n\n"
        f"{guidance}"
    )


def build_comparison_prompt(
    context: dict[str, Any], question: Optional[str] = None
) -> str:
    """Build the user prompt for a vendor comparison.

    Embeds the comparison context. When ``question`` is supplied it is included;
    otherwise the model is asked for a general comparison grounded in the context.
    """
    parts = [_context_block(context)]
    if question:
        parts.append(f"{_QUESTION_HEADER}\n{question}")
        parts.append(
            "Compare the vendors above to answer the question using only the "
            "analysis context. Do not rank or re-score vendors yourself; rely on "
            "the deterministic risk scores and recommendations provided."
        )
    else:
        parts.append(
            "Compare the vendors above using only the analysis context, "
            "highlighting the differences in risk, lead time, fulfillment, and "
            "inventory exposure. Do not rank or re-score vendors yourself; rely "
            "on the deterministic risk scores and recommendations provided."
        )
    return "\n\n".join(parts)


def build_executive_prompt(context: dict[str, Any]) -> str:
    """Build the user prompt for an executive summary of the analysis."""
    return (
        f"{_context_block(context)}\n\n"
        "Write a concise, INTERPRETIVE executive summary of the vendor "
        "analysis above using only the analysis context. Do NOT simply "
        "restate the dashboard figures line by line; instead explain the "
        "business meaning: which vendor is the clearest supply-continuity "
        "concern and why, which is the strongest choice, and which is usable "
        "but warrants monitoring. You may reference the authoritative values "
        "(risk levels/scores, recommendations, exposure) exactly as supplied, "
        "but do not invent or recompute any numbers, and do not change any "
        "recommendation. Use each authoritative numeric value EXACTLY as "
        "provided in the context (do not recalculate, estimate, round "
        "differently, or substitute a value from another vendor). Distinguish "
        "the total potential exposure (summary.total_potential_exposure_units) "
        "from an individual vendors per-vendor exposure "
        "(vendors[].potential_exposure_units); never present one as the other. "
        "If a value is not present in the supplied context, do not invent it. "
        "Keep it to a short, business-friendly paragraph."
    )


def build_what_if_prompt(context: dict[str, Any]) -> str:
    """Build the user prompt explaining a what-if delay scenario.

    The what-if numbers were already computed deterministically (Layer 1); the
    model only explains them and must not recalculate anything.
    """
    return (
        f"{_context_block(context)}\n\n"
        "Explain the what-if delay scenario above in business terms using only "
        "the analysis context. Describe how the additional delay changes the "
        "expected lead time and inventory exposure, referencing the "
        "already-computed values. Do NOT recalculate any numbers yourself. "
        "Treat potential exposure and shortage/gap as two DISTINCT concepts: "
        "potential exposure is a planning signal (demand at risk from the "
        "lead-time gap), not a guaranteed loss; shortage/gap is inventory "
        "demand that may be uncovered before replenishment arrives. Never "
        "state that one IS the other; if the two values happen to be equal in "
        "this scenario, say they happen to have the same value and still keep "
        "them as separate concepts."
    )
