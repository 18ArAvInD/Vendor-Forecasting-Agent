"""Tests for the Layer 3 provider plumbing in ``llm_provider.py``.

Standard pytest only (NO Hypothesis, NO real boto3/AWS/network). The real
provider is exercised with an INJECTED fake Bedrock client, so nothing here ever
imports boto3 or contacts AWS.
"""

import importlib
import sys

import pytest

from vendor_forecasting_agent.llm_provider import (
    LLMConfig,
    LLMProviderError,
    MistralBedrockProvider,
    NullLLMProvider,
    load_llm_config,
)


# ---- Fake Bedrock client ----------------------------------------------------


class _FakeBedrockClient:
    """Records the converse() call and returns a well-formed Converse response."""

    def __init__(self, reply_text: str = "hello from bedrock") -> None:
        self.reply_text = reply_text
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": self.reply_text}],
                }
            }
        }


class _FailingBedrockClient:
    """A converse() that raises to simulate an AWS/API failure."""

    def converse(self, **kwargs):
        raise RuntimeError("simulated AWS ThrottlingException at endpoint")


def _config(model_id: str = "mistral.test-model") -> LLMConfig:
    return LLMConfig(
        aws_region="eu-west-1",
        bedrock_model_id=model_id,
        temperature=0.3,
        max_tokens=256,
    )


# ---- load_llm_config --------------------------------------------------------


def test_load_llm_config_reads_env_mapping():
    env = {
        "AWS_REGION": "ap-south-1",
        "BEDROCK_MODEL_ID": "mistral.large",
        "LLM_TEMPERATURE": "0.7",
        "LLM_MAX_TOKENS": "1024",
    }
    cfg = load_llm_config(env)
    assert cfg.aws_region == "ap-south-1"
    assert cfg.bedrock_model_id == "mistral.large"
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 1024


def test_load_llm_config_applies_defaults_when_absent():
    cfg = load_llm_config({})
    assert cfg.aws_region == "us-east-1"
    assert cfg.bedrock_model_id == ""  # NO hardcoded real model id
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 512


def test_load_llm_config_invalid_numbers_raise_provider_error():
    with pytest.raises(LLMProviderError):
        load_llm_config({"LLM_TEMPERATURE": "not-a-number"})
    with pytest.raises(LLMProviderError):
        load_llm_config({"LLM_MAX_TOKENS": "abc"})


# ---- MistralBedrockProvider.generate ---------------------------------------


def test_generate_uses_injected_client_and_converse_shape():
    client = _FakeBedrockClient("analysis text")
    provider = MistralBedrockProvider(config=_config("mistral.test-model"), client=client)

    result = provider.generate("user question", "system guardrail")

    assert result == "analysis text"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == "mistral.test-model"
    assert call["messages"] == [
        {"role": "user", "content": [{"text": "user question"}]}
    ]
    assert call["system"] == [{"text": "system guardrail"}]
    assert call["inferenceConfig"] == {"temperature": 0.3, "maxTokens": 256}


def test_generate_without_system_prompt_omits_system_block():
    client = _FakeBedrockClient()
    provider = MistralBedrockProvider(config=_config(), client=client)

    provider.generate("just a user prompt")

    assert "system" not in client.calls[0]


def test_generate_success_extracts_text():
    client = _FakeBedrockClient("the extracted answer")
    provider = MistralBedrockProvider(config=_config(), client=client)
    assert provider.generate("q", "s") == "the extracted answer"


def test_generate_wraps_api_failure_in_provider_error():
    provider = MistralBedrockProvider(config=_config(), client=_FailingBedrockClient())
    with pytest.raises(LLMProviderError) as excinfo:
        provider.generate("q", "s")
    # It must be our clean error type, not the raw RuntimeError.
    assert isinstance(excinfo.value, LLMProviderError)
    # Basic credential-safety check: no obvious secret markers in the message.
    message = str(excinfo.value).lower()
    assert "secret" not in message
    assert "aws_access_key" not in message


def test_generate_unexpected_response_shape_raises_provider_error():
    class _BadShapeClient:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": []}}}  # empty content

    provider = MistralBedrockProvider(config=_config(), client=_BadShapeClient())
    with pytest.raises(LLMProviderError):
        provider.generate("q", "s")


def test_generate_missing_model_id_raises_provider_error():
    provider = MistralBedrockProvider(config=_config(model_id=""), client=_FakeBedrockClient())
    with pytest.raises(LLMProviderError):
        provider.generate("q", "s")


# ---- NullLLMProvider backward-compat ----------------------------------------


def test_null_provider_two_args_returns_user_unchanged():
    provider = NullLLMProvider()
    assert provider.generate("user text", "system text") == "user text"


def test_null_provider_single_arg_backward_compat():
    provider = NullLLMProvider()
    assert provider.generate("template text") == "template text"


def test_null_provider_raise_error_mode():
    provider = NullLLMProvider(raise_error=True)
    with pytest.raises(RuntimeError):
        provider.generate("anything")


# ---- boto3 optionality / laziness ------------------------------------------


def test_importing_llm_provider_does_not_import_boto3():
    # Drop any cached copy, then re-import fresh and assert boto3 wasn't pulled in
    # by the import itself.
    had_boto3 = "boto3" in sys.modules
    sys.modules.pop("vendor_forecasting_agent.llm_provider", None)
    importlib.import_module("vendor_forecasting_agent.llm_provider")
    if not had_boto3:
        assert "boto3" not in sys.modules


def test_constructing_provider_without_boto3_does_not_raise():
    # Lazy client: construction must never require boto3 or AWS credentials.
    provider = MistralBedrockProvider(config=_config())
    assert provider is not None
