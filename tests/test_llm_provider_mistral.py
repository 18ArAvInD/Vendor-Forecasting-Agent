"""Tests for the official Mistral API provider in ``llm_provider.py``.

Standard pytest only (NO Hypothesis, NO real ``mistralai``/network). The provider
is exercised with an INJECTED fake Mistral client, so nothing here imports the
``mistralai`` SDK or contacts the Mistral API. The API key is a dummy string and
is never a real secret.
"""

import importlib
import sys

import pytest

from vendor_forecasting_agent.llm_provider import (
    LLMConfig,
    LLMProviderError,
    MistralAPIProvider,
    load_llm_config,
)

_DUMMY_KEY = "test-key-not-a-real-secret"


# ---- Fake Mistral SDK client ------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChat:
    def __init__(self, reply_text):
        self.reply_text = reply_text
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.reply_text)


class _FakeMistralClient:
    """Mimics ``mistralai.Mistral``: exposes a ``.chat.complete(**kwargs)``."""

    def __init__(self, reply_text="hello from mistral"):
        self.chat = _FakeChat(reply_text)


class _FailingChat:
    def complete(self, **kwargs):
        raise RuntimeError("simulated Mistral 429 rate limit at endpoint")


class _FailingMistralClient:
    def __init__(self):
        self.chat = _FailingChat()


def _config(api_key=_DUMMY_KEY, model_id="mistral-medium-latest"):
    return LLMConfig(
        aws_region="us-east-1",
        bedrock_model_id="",
        temperature=0.3,
        max_tokens=256,
        mistral_api_key=api_key,
        mistral_model_id=model_id,
    )


# ---- load_llm_config: Mistral settings --------------------------------------


def test_load_llm_config_reads_mistral_env():
    cfg = load_llm_config(
        {"MISTRAL_API_KEY": _DUMMY_KEY, "MISTRAL_MODEL_ID": "mistral-large-latest"}
    )
    assert cfg.mistral_api_key == _DUMMY_KEY
    assert cfg.mistral_model_id == "mistral-large-latest"


def test_load_llm_config_mistral_defaults_when_absent():
    cfg = load_llm_config({})
    assert cfg.mistral_api_key == ""  # no key -> provider disabled
    assert cfg.mistral_model_id == "mistral-medium-latest"  # requested default


# ---- Initialization ---------------------------------------------------------


def test_provider_construction_does_not_require_sdk():
    # Lazy client: constructing the provider must never import mistralai.
    provider = MistralAPIProvider(config=_config())
    assert provider is not None


def test_importing_llm_provider_does_not_import_mistralai():
    had = "mistralai" in sys.modules
    sys.modules.pop("vendor_forecasting_agent.llm_provider", None)
    importlib.import_module("vendor_forecasting_agent.llm_provider")
    if not had:
        assert "mistralai" not in sys.modules


# ---- Request construction + model configuration -----------------------------


def test_generate_builds_request_with_model_and_inference_settings():
    client = _FakeMistralClient("analysis text")
    provider = MistralAPIProvider(config=_config(model_id="mistral-medium-latest"), client=client)

    result = provider.generate("user question", "system guardrail")

    assert result == "analysis text"
    assert len(client.chat.calls) == 1
    call = client.chat.calls[0]
    assert call["model"] == "mistral-medium-latest"
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 256


def test_generate_includes_system_and_user_messages():
    client = _FakeMistralClient()
    provider = MistralAPIProvider(config=_config(), client=client)
    provider.generate("the user prompt", "the system prompt")
    messages = client.chat.calls[0]["messages"]
    assert messages == [
        {"role": "system", "content": "the system prompt"},
        {"role": "user", "content": "the user prompt"},
    ]


def test_generate_without_system_prompt_omits_system_message():
    client = _FakeMistralClient()
    provider = MistralAPIProvider(config=_config(), client=client)
    provider.generate("just a user prompt")
    messages = client.chat.calls[0]["messages"]
    assert messages == [{"role": "user", "content": "just a user prompt"}]


# ---- Successful response extraction -----------------------------------------


def test_generate_success_extracts_text():
    client = _FakeMistralClient("the extracted answer")
    provider = MistralAPIProvider(config=_config(), client=client)
    assert provider.generate("q", "s") == "the extracted answer"


def test_generate_unexpected_response_shape_raises_provider_error():
    class _BadClient:
        def __init__(self):
            self.chat = self

        def complete(self, **kwargs):
            return object()  # no .choices

    provider = MistralAPIProvider(config=_config(), client=_BadClient())
    with pytest.raises(LLMProviderError):
        provider.generate("q", "s")


def test_generate_empty_text_raises_provider_error():
    client = _FakeMistralClient("")  # empty content
    provider = MistralAPIProvider(config=_config(), client=client)
    with pytest.raises(LLMProviderError):
        provider.generate("q", "s")


# ---- API error handling -----------------------------------------------------


def test_generate_wraps_api_failure_in_provider_error():
    provider = MistralAPIProvider(config=_config(), client=_FailingMistralClient())
    with pytest.raises(LLMProviderError) as excinfo:
        provider.generate("q", "s")
    assert isinstance(excinfo.value, LLMProviderError)


# ---- Missing API key + key-safety -------------------------------------------


def test_generate_missing_api_key_raises_provider_error():
    # No injected client and no key -> must raise cleanly, never touch the SDK.
    provider = MistralAPIProvider(config=_config(api_key=""))
    with pytest.raises(LLMProviderError):
        provider.generate("q", "s")


def test_api_key_never_appears_in_error_messages():
    secret = "super-secret-key-value-1234567890"
    # Failing client so the invocation-failure path runs; assert the key is not
    # leaked into the wrapped error message or its chain.
    provider = MistralAPIProvider(
        config=_config(api_key=secret), client=_FailingMistralClient()
    )
    with pytest.raises(LLMProviderError) as excinfo:
        provider.generate("q", "s")
    err = excinfo.value
    assert secret not in str(err)
    assert secret not in repr(err)
    # Also check the missing-model / missing-key path message.
    provider2 = MistralAPIProvider(config=_config(api_key=""))
    with pytest.raises(LLMProviderError) as excinfo2:
        provider2.generate("q", "s")
    assert secret not in str(excinfo2.value)
