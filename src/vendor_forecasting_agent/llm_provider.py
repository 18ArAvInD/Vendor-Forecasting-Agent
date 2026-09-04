"""Pluggable LLM provider layer for the Vendor Forecasting Agent (llm_provider.py).

This module is the ``LLM_Provider`` described in the design. It defines a small
:class:`LLMProvider` protocol so a concrete provider (offline pass-through or a
real Amazon Bedrock model) can be swapped in without touching the deterministic
Phase 1 pipeline (Req 15.3), and supplies the concrete implementations plus the
configuration plumbing the Phase 3 AI decision-support layer needs.

Pieces
------
* :class:`LLMProvider` — the pluggable protocol. Its ``generate`` method takes a
  ``user_prompt`` and an optional ``system_prompt`` so both the legacy
  single-argument callers (``explanation.py``) and the new Layer 3 callers
  (which supply a system prompt) work against the same interface.
* :class:`NullLLMProvider` — the offline, no-network **default** provider. A pure
  pass-through: it returns the ``user_prompt`` unchanged. ``explanation.py`` calls
  ``generate(text)`` and gets the deterministic template back verbatim, so the
  whole explanation path stays fully offline and deterministic.
* :class:`LLMConfig` — a small frozen dataclass holding the AWS region, Bedrock
  model id, temperature and max-token settings.
* :func:`load_llm_config` — reads those settings from an environment mapping
  (``os.environ`` by default). There is **no** hardcoded real model id: when
  ``BEDROCK_MODEL_ID`` is absent the model id is the empty string and the real
  provider refuses to run until it is configured.
* :class:`MistralBedrockProvider` — the real, network-capable provider that calls
  the Bedrock Runtime Converse API for the configured Mistral model. ``boto3`` is
  imported lazily inside the client factory (never at module import time), so
  importing this module never requires ``boto3`` and the whole test suite stays
  import-safe and offline.
* :class:`MistralAPIProvider` — the real, network-capable provider that calls the
  **official Mistral API** (SDK ``mistralai``) for the configured model
  (``MISTRAL_MODEL_ID``, default ``mistral-medium-latest``) using
  ``MISTRAL_API_KEY``. The SDK is imported lazily inside the client factory
  (never at module import time), so importing this module never requires
  ``mistralai`` and the suite stays import-safe and offline. The API key is never
  logged, printed, embedded in a prompt, or included in an error message.
* :class:`LLMProviderError` — a clean, application-level exception used to wrap
  any provider/config/invocation failure so callers never see raw ``boto3``/AWS
  or ``mistralai`` exceptions.
* :class:`BedrockLLMProvider` — the original deferred stub, kept for backward
  compatibility (raises ``NotImplementedError``).

Boundaries: no ``boto3`` import at module top level; providers only turn
already-computed results (embedded in the prompt) into words — they never
calculate or alter any risk/forecast/inventory/recommendation value.
"""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

__all__ = [
    "LLMProvider",
    "NullLLMProvider",
    "BedrockLLMProvider",
    "MistralBedrockProvider",
    "MistralAPIProvider",
    "LLMConfig",
    "LLMProviderError",
    "load_llm_config",
]


class LLMProviderError(Exception):
    """Application-level LLM provider error.

    Wraps any provider/config/invocation failure (missing configuration, missing
    ``boto3``, a raw AWS/``boto3`` error, or an unexpected response shape) so
    callers never have to import or handle ``boto3`` exception types. Messages
    are intentionally generic and never include credentials.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Pluggable provider interface (Req 15.3).

    A provider turns a prompt into natural-language text. The signature is
    backward compatible: legacy callers pass only ``user_prompt`` (e.g.
    ``explanation.py``'s ``generate(text)``), while the Layer 3 decision-support
    operations also pass a ``system_prompt``.
    """

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Return natural-language text for ``user_prompt`` (optionally guided by
        ``system_prompt``)."""
        ...


class NullLLMProvider:
    """Offline, no-network default provider — a pure pass-through of the
    ``user_prompt``.

    ``explanation.py`` constructs the complete, human-readable template
    explanation from the structured deterministic results and passes it in as the
    prompt. This provider simply returns that text unchanged: it adds no number,
    alters no number, and makes no network call. Using it keeps the entire
    explanation path fully offline and deterministic, which is why it is the
    default Phase 2 provider (``LLM_PROVIDER = null``) and the provider used in
    tests.

    The optional ``system_prompt`` is accepted (for interface compatibility with
    the Layer 3 callers) but ignored — the pass-through returns the user prompt so
    the deterministic template is echoed exactly.

    An optional configured-failure mode is supported to exercise the degraded
    path: when constructed with ``raise_error=True``, :meth:`generate` raises
    :class:`RuntimeError` instead of returning text. Defaults to ``False``.
    """

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Echo ``user_prompt`` unchanged (the ``system_prompt`` is ignored).

        Raises :class:`RuntimeError` when this provider was constructed with
        ``raise_error=True`` (to exercise the degraded path).
        """
        if self.raise_error:
            raise RuntimeError("NullLLMProvider configured to fail (raise_error=True)")
        return user_prompt


# ---- Configuration ----------------------------------------------------------

# Environment variable names read by :func:`load_llm_config`.
_ENV_AWS_REGION = "AWS_REGION"
_ENV_MODEL_ID = "BEDROCK_MODEL_ID"
_ENV_TEMPERATURE = "LLM_TEMPERATURE"
_ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
# Official Mistral API settings.
_ENV_MISTRAL_API_KEY = "MISTRAL_API_KEY"
_ENV_MISTRAL_MODEL_ID = "MISTRAL_MODEL_ID"

# Sensible defaults. NOTE: there is deliberately NO default Bedrock model id — a
# real model id must be supplied via the environment; absent it the id is "" and
# :class:`MistralBedrockProvider` refuses to run.
_DEFAULT_AWS_REGION = "us-east-1"
_DEFAULT_TEMPERATURE = 0.2
_DEFAULT_MAX_TOKENS = 512
# The official Mistral API has a safe, current default model id (unlike the
# Bedrock model id): the provider only runs when an API KEY is present, so a
# default model id here is harmless and matches the requested model.
_DEFAULT_MISTRAL_MODEL_ID = "mistral-medium-latest"


@dataclass(frozen=True)
class LLMConfig:
    """Immutable configuration for a Bedrock-backed provider.

    A stdlib frozen dataclass (matching the project's use of frozen models) so it
    is hashable/immutable without Pydantic-optional friction.
    """

    aws_region: str
    bedrock_model_id: str
    temperature: float
    max_tokens: int
    # Official Mistral API settings. Defaulted so existing constructors and tests
    # (which pass only the Bedrock fields) keep working unchanged. The API key is
    # held here only to select/enable the provider; it is never logged, printed,
    # or placed into any prompt or error message.
    mistral_api_key: str = ""
    mistral_model_id: str = "mistral-medium-latest"


def load_llm_config(env: Optional[Mapping[str, str]] = None) -> LLMConfig:
    """Build an :class:`LLMConfig` from an environment mapping.

    Reads ``AWS_REGION``, ``BEDROCK_MODEL_ID``, ``LLM_TEMPERATURE`` and
    ``LLM_MAX_TOKENS`` from ``env`` (default ``os.environ``). Defaults are applied
    only where safe:

    * ``aws_region`` -> ``"us-east-1"``
    * ``temperature`` -> ``0.2``
    * ``max_tokens`` -> ``512``
    * ``bedrock_model_id`` -> ``""`` (empty) when absent — NO real model id is
      hardcoded; the real provider raises :class:`LLMProviderError` at call time
      when the id is empty.

    It also reads the official Mistral API settings:

    * ``MISTRAL_API_KEY`` -> ``mistral_api_key`` (``""`` when absent; a non-empty
      key is what enables :class:`MistralAPIProvider`). Never logged.
    * ``MISTRAL_MODEL_ID`` -> ``mistral_model_id`` (defaults to
      ``"mistral-medium-latest"``).

    Raises :class:`LLMProviderError` when ``LLM_TEMPERATURE`` / ``LLM_MAX_TOKENS``
    are present but not valid numbers.
    """
    source: Mapping[str, str] = env if env is not None else os.environ

    aws_region = source.get(_ENV_AWS_REGION) or _DEFAULT_AWS_REGION
    bedrock_model_id = source.get(_ENV_MODEL_ID) or ""

    temperature_raw = source.get(_ENV_TEMPERATURE)
    if temperature_raw is None or temperature_raw == "":
        temperature = _DEFAULT_TEMPERATURE
    else:
        try:
            temperature = float(temperature_raw)
        except (TypeError, ValueError) as exc:
            raise LLMProviderError(
                f"invalid {_ENV_TEMPERATURE}: expected a number"
            ) from exc

    max_tokens_raw = source.get(_ENV_MAX_TOKENS)
    if max_tokens_raw is None or max_tokens_raw == "":
        max_tokens = _DEFAULT_MAX_TOKENS
    else:
        try:
            max_tokens = int(max_tokens_raw)
        except (TypeError, ValueError) as exc:
            raise LLMProviderError(
                f"invalid {_ENV_MAX_TOKENS}: expected an integer"
            ) from exc

    mistral_api_key = source.get(_ENV_MISTRAL_API_KEY) or ""
    mistral_model_id = source.get(_ENV_MISTRAL_MODEL_ID) or _DEFAULT_MISTRAL_MODEL_ID

    return LLMConfig(
        aws_region=aws_region,
        bedrock_model_id=bedrock_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        mistral_api_key=mistral_api_key,
        mistral_model_id=mistral_model_id,
    )


class MistralBedrockProvider:
    """Real Amazon Bedrock provider for a configured Mistral model.

    Uses the Bedrock Runtime **Converse** API (``client.converse``), the clean,
    model-agnostic conversational interface, so the request shape does not depend
    on any single model's native payload format.

    ``boto3`` is optional and imported lazily: the client is created on first
    :meth:`generate` (or an injected client is used), so importing this class
    never requires ``boto3`` and constructing it never touches AWS. Any failure
    (missing config, missing ``boto3``, a raw AWS error, or an unexpected response
    shape) is wrapped in :class:`LLMProviderError` — raw ``boto3``/AWS exceptions
    never propagate to callers.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Any = None,
    ) -> None:
        self.config = config if config is not None else load_llm_config()
        # Injected client (e.g. a test fake) short-circuits lazy creation; when
        # None the client is created lazily on first generate() so importing /
        # constructing this class never requires boto3 or AWS credentials.
        self._client = client

    def _get_client(self) -> Any:
        """Return the Bedrock Runtime client, creating it lazily if needed.

        ``boto3`` is imported *inside* this method so it is never a top-level
        import. Raises :class:`LLMProviderError` if ``boto3`` is not installed or
        the client cannot be created.
        """
        if self._client is not None:
            return self._client

        try:
            import boto3  # local import keeps boto3 optional
        except ImportError as exc:
            raise LLMProviderError(
                "boto3 is not installed; install the [llm] extra"
            ) from exc

        try:
            self._client = boto3.client(
                "bedrock-runtime", region_name=self.config.aws_region
            )
        except Exception as exc:  # noqa: BLE001 - wrap any client-creation failure
            raise LLMProviderError(
                f"failed to create Bedrock client: {exc}"
            ) from exc
        return self._client

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Invoke the configured Mistral model via the Converse API.

        Builds a Converse request (user message, optional system block, inference
        config from :class:`LLMConfig`), calls ``client.converse`` and extracts the
        assistant text defensively. Any failure is wrapped in
        :class:`LLMProviderError`.
        """
        if not self.config.bedrock_model_id:
            raise LLMProviderError("BEDROCK_MODEL_ID is not configured")

        client = self._get_client()

        kwargs: dict[str, Any] = {
            "modelId": self.config.bedrock_model_id,
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]}
            ],
            "inferenceConfig": {
                "temperature": self.config.temperature,
                "maxTokens": self.config.max_tokens,
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            resp = client.converse(**kwargs)
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak raw boto3/AWS errors
            raise LLMProviderError(f"Bedrock invocation failed: {exc}") from exc

        return self._extract_text(resp)

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """Safely pull the assistant text from a Converse response.

        Expected shape: ``resp["output"]["message"]["content"][0]["text"]``.
        Raises :class:`LLMProviderError` if the shape is unexpected.
        """
        try:
            content = resp["output"]["message"]["content"]
            text = content[0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("unexpected Bedrock response shape") from exc

        if not isinstance(text, str):
            raise LLMProviderError("unexpected Bedrock response shape")
        return text


class MistralAPIProvider:
    """Real provider that calls the **official Mistral API** directly.

    Uses the official Mistral Python SDK (``mistralai``) and the chat/completion
    mechanism. The SDK is imported *lazily* inside the client factory (never at
    module import time), so importing this module never requires ``mistralai``
    and the whole test suite stays import-safe and offline. An injected client
    (for tests) short-circuits SDK creation entirely.

    Configuration comes from :class:`LLMConfig` (populated from the environment by
    :func:`load_llm_config`):

    * ``mistral_api_key`` (``MISTRAL_API_KEY``) -- required; a missing/empty key
      raises :class:`LLMProviderError`. The key is never logged, printed, placed
      into any prompt, or included in any error message.
    * ``mistral_model_id`` (``MISTRAL_MODEL_ID``) -- defaults to
      ``"mistral-medium-latest"``.
    * ``temperature`` / ``max_tokens`` -- reused from the existing LLM config.

    The LLM remains an explanation/decision-support layer only: this provider
    turns the already-computed, prompt-embedded deterministic results into words
    and never calculates or alters any authoritative value. Any SDK/API failure
    (or unexpected response shape) is wrapped in :class:`LLMProviderError` so raw
    SDK exceptions never propagate.
    """

    def __init__(
        self,
        config=None,
        client=None,
    ):
        self.config = config if config is not None else load_llm_config()
        # Injected client (e.g. a test fake) short-circuits lazy creation; when
        # None the client is created lazily on first generate() so importing /
        # constructing this class never requires the mistralai SDK.
        self._client = client

    def _get_client(self):
        """Return the Mistral client, creating it lazily if needed.

        ``mistralai`` is imported *inside* this method so it is never a top-level
        import. Raises :class:`LLMProviderError` if the SDK is not installed, the
        API key is missing, or the client cannot be created. The API key is never
        included in any raised message.
        """
        if self._client is not None:
            return self._client

        if not self.config.mistral_api_key:
            raise LLMProviderError("MISTRAL_API_KEY is not configured")

        try:
            from mistralai import Mistral  # local import keeps the SDK optional
        except ImportError as exc:
            raise LLMProviderError(
                "mistralai is not installed; install the [mistral] extra"
            ) from exc

        try:
            self._client = Mistral(api_key=self.config.mistral_api_key)
        except Exception as exc:  # noqa: BLE001 - wrap any client-creation failure
            # Deliberately do NOT interpolate the exception: a client-init error
            # could echo the api_key. Keep the message generic.
            raise LLMProviderError("failed to create Mistral client") from exc
        return self._client

    def generate(self, user_prompt, system_prompt=None):
        """Invoke the configured Mistral model via the official chat API.

        Builds a chat/completion request (optional system message + user message,
        the configured model, temperature and max tokens), calls the SDK and
        extracts the assistant text defensively. Any failure is wrapped in
        :class:`LLMProviderError` (never leaking the API key or raw SDK error
        internals).
        """
        if not self.config.mistral_api_key:
            raise LLMProviderError("MISTRAL_API_KEY is not configured")

        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            resp = client.chat.complete(
                model=self.config.mistral_model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leak raw SDK/API errors
            raise LLMProviderError("Mistral API invocation failed") from exc

        return self._extract_text(resp)

    @staticmethod
    def _extract_text(resp):
        """Safely pull the assistant text from a Mistral chat response.

        Expected shape: ``resp.choices[0].message.content``. Raises
        :class:`LLMProviderError` if the shape is unexpected or the text is empty.
        """
        try:
            choices = resp.choices
            message = choices[0].message
            text = message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMProviderError("unexpected Mistral response shape") from exc

        if not isinstance(text, str) or not text:
            raise LLMProviderError("unexpected Mistral response shape")
        return text


class BedrockLLMProvider:
    """Amazon Bedrock implementation stub (deferred).

    Retained for backward compatibility. The real provider is
    :class:`MistralBedrockProvider`. This stub makes no network call and raises
    :class:`NotImplementedError`.
    """

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError("Bedrock provider is not implemented in this task")
