"""AI decision-support orchestration for the Vendor Forecasting Agent.

This is the top of **Layer 3** (AI decision support). It wires together the
deterministic results (Layer 1), the structured context builders (Layer 2,
``llm_context.py``), the prompt builders (``llm_prompts.py``) and a pluggable
:class:`~vendor_forecasting_agent.llm_provider.LLMProvider` to answer questions,
compare vendors, summarize for executives, and explain what-if scenarios.

Hard boundaries
---------------
* **Streamlit-free and boto3-free.** No UI, no top-level provider construction
  that requires AWS. The provider is passed in by the caller (Layer 4 chooses a
  :class:`NullLLMProvider` or a :class:`MistralBedrockProvider`).
* **The LLM only explains.** These operations never parse the model's text back
  into authoritative fields and never recompute or modify any deterministic
  value. The what-if numbers are computed by Layer 1 and passed in — this module
  does not recompute them.
* **Never raises for provider failures.** Any provider failure (an
  :class:`LLMProviderError`, any other exception, or a missing/``None`` provider)
  is turned into a graceful fallback :class:`LLMResponse` with ``success=False``.
  Raw AWS/credential details are never leaked into the response.
"""

from dataclasses import dataclass
from typing import Optional

from .llm_context import (
    build_comparison_context,
    build_executive_context,
    build_vendor_context,
    build_what_if_context,
)
from .llm_prompts import (
    SYSTEM_PROMPT,
    build_ask_prompt,
    build_comparison_prompt,
    build_executive_prompt,
    build_what_if_prompt,
)
from .llm_provider import (
    LLMConfig,
    LLMProvider,
    MistralAPIProvider,
    MistralBedrockProvider,
    NullLLMProvider,
    load_llm_config,
)
from .schema import DeterministicResults
from .upstream import UpstreamRequest
from .whatif import WhatIfResult
from .whatif_scenarios import ScenarioResult, build_scenario_context

__all__ = [
    "LLMResponse",
    "FALLBACK_TEXT",
    "ask_vendor_agent",
    "compare_vendors",
    "generate_executive_summary",
    "explain_what_if",
    "explain_scenario",
    "get_default_provider",
]


FALLBACK_TEXT = (
    "AI explanation temporarily unavailable. Showing deterministic analysis "
    "instead."
)


@dataclass(frozen=True)
class LLMResponse:
    """Immutable result of a Layer 3 AI decision-support operation.

    ``text`` is explanatory natural language only (never authoritative). On
    failure ``text`` is :data:`FALLBACK_TEXT`, ``success`` is ``False`` and
    ``error`` carries a short, credential-free description.
    """

    text: str
    provider: str
    success: bool
    error: Optional[str] = None


def _provider_name(provider: Optional[LLMProvider]) -> str:
    """Human-readable provider label for the response (class name, else 'none')."""
    if provider is None:
        return "none"
    return type(provider).__name__


def _run(
    provider: Optional[LLMProvider],
    user_prompt: str,
) -> LLMResponse:
    """Call ``provider.generate(user_prompt, SYSTEM_PROMPT)`` and wrap the result.

    Central failure boundary: a missing provider, empty text, or ANY exception
    becomes a graceful fallback :class:`LLMResponse` (never raises, never leaks
    raw AWS/credential details).
    """
    name = _provider_name(provider)

    if provider is None:
        return LLMResponse(
            text=FALLBACK_TEXT,
            provider=name,
            success=False,
            error="no provider configured",
        )

    try:
        text = provider.generate(user_prompt, SYSTEM_PROMPT)
    except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
        return LLMResponse(
            text=FALLBACK_TEXT,
            provider=name,
            success=False,
            error=str(exc),
        )

    if not text:
        return LLMResponse(
            text=FALLBACK_TEXT,
            provider=name,
            success=False,
            error="provider returned empty text",
        )

    return LLMResponse(text=text, provider=name, success=True)


def ask_vendor_agent(
    results: DeterministicResults,
    request: UpstreamRequest,
    question: str,
    provider: Optional[LLMProvider],
    *,
    vendor_id: Optional[str] = None,
    history: Optional[list[dict]] = None,
    what_if_context: Optional[dict] = None,
) -> LLMResponse:
    """Answer a free-form question about the analysis.

    Builds a combined context: the focused ``selected_vendor`` context (when a
    ``vendor_id`` is given and present) plus the full ``comparison`` context (so
    comparison-style questions still work). The LLM answers only from this
    context; it never recomputes or overrides any deterministic value.

    Two optional, keyword-only context enrichments are supported (both default to
    ``None``, so existing callers are completely unaffected):

    * ``history`` — a small, already-bounded/trimmed list of conversational turn
      dicts (``{"role": "user"/"assistant", "text": ...}``). It is embedded under
      ``conversation_history`` purely so the model can resolve what the user is
      referring to (e.g. a pronoun or a vendor named earlier). It is
      conversational context, NOT authoritative data — the model still explains
      only the supplied deterministic values and never recomputes anything.
    * ``what_if_context`` — the authoritative, already-computed dict from
      :func:`whatif_scenarios.build_scenario_context` for the latest what-if
      scenario. It is embedded under ``latest_what_if`` as deterministic facts;
      the model explains them and never recalculates them.
    """
    selected_vendor = None
    if vendor_id is not None:
        selected_vendor = build_vendor_context(results, request, vendor_id)

    comparison = build_comparison_context(results, request)
    combined: dict = {"selected_vendor": selected_vendor, "comparison": comparison}

    if history:
        combined["conversation_history"] = history
    if what_if_context:
        combined["latest_what_if"] = what_if_context

    prompt = build_ask_prompt(combined, question)
    return _run(provider, prompt)


def compare_vendors(
    results: DeterministicResults,
    request: UpstreamRequest,
    provider: Optional[LLMProvider],
    *,
    vendor_ids: Optional[list[str]] = None,
    question: Optional[str] = None,
) -> LLMResponse:
    """Compare vendors (optionally a subset), grounded in the comparison context.

    The deterministic risk scores / recommendations are authoritative; the LLM
    only narrates the comparison.
    """
    context = build_comparison_context(results, request, vendor_ids)
    prompt = build_comparison_prompt(context, question)
    return _run(provider, prompt)


def generate_executive_summary(
    results: DeterministicResults,
    request: UpstreamRequest,
    provider: Optional[LLMProvider],
) -> LLMResponse:
    """Produce an executive summary grounded in the executive context."""
    context = build_executive_context(results, request)
    prompt = build_executive_prompt(context)
    return _run(provider, prompt)


def explain_what_if(
    what_if: WhatIfResult,
    provider: Optional[LLMProvider],
    *,
    request: Optional[UpstreamRequest] = None,
) -> LLMResponse:
    """Explain an already-computed :class:`WhatIfResult` in business terms.

    The what-if numbers were computed deterministically by Layer 1 and are passed
    in; this function does NOT recompute them. It only builds a reflecting context
    and asks the LLM to explain it.
    """
    context = build_what_if_context(what_if, request)
    prompt = build_what_if_prompt(context)
    return _run(provider, prompt)


def explain_scenario(
    scenario_result: ScenarioResult,
    provider: Optional[LLMProvider],
    *,
    request: Optional[UpstreamRequest] = None,
) -> LLMResponse:
    """Explain an already-computed :class:`ScenarioResult` in business terms.

    Thin wrapper for the deterministic multi-scenario what-if
    (:mod:`vendor_forecasting_agent.whatif_scenarios`). The scenario numbers were
    computed deterministically and are passed in; this function does NOT recompute
    them. It builds a reflecting context via
    :func:`whatif_scenarios.build_scenario_context`, reuses the existing what-if
    prompt builder (which already instructs the model to explain, not recalculate)
    and routes through the SAME central failure boundary :func:`_run` (never
    raises, never leaks credentials, graceful fallback on failure / ``None``
    provider).
    """
    context = build_scenario_context(scenario_result, request)
    prompt = build_what_if_prompt(context)
    return _run(provider, prompt)


def get_default_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Return a usable default provider without ever raising or touching AWS.

    Selection order (all providers are constructed lazily -- no eager network,
    boto3, or mistralai client is created here, so this helper is safe to call in
    any environment):

    1. :class:`MistralAPIProvider` when ``MISTRAL_API_KEY`` is configured
       (a non-empty ``mistral_api_key``) -- the default production/demo path.
    2. :class:`MistralBedrockProvider` when a non-empty ``bedrock_model_id`` is
       configured -- preserved for backward compatibility.
    3. :class:`NullLLMProvider` otherwise (offline; deterministic analysis only).

    Any configuration error falls back to the offline :class:`NullLLMProvider`.
    """
    try:
        cfg = config if config is not None else load_llm_config()
    except Exception:  # noqa: BLE001 - config problems -> safe offline default
        return NullLLMProvider()

    if cfg.mistral_api_key:
        return MistralAPIProvider(config=cfg)
    if cfg.bedrock_model_id:
        return MistralBedrockProvider(config=cfg)
    return NullLLMProvider()
