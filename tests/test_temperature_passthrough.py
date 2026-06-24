"""
Wire-payload regression tests for temperature passthrough.

vox is a pure API passthrough: it must NOT fabricate a temperature. When the
caller omits temperature, the parameter must be absent from the provider request
so the model applies its own server-side default. An explicitly-supplied
temperature is forwarded (clamped per the model's constraint). Reasoning models
that reject/ignore temperature must always omit it.

These tests assert on the actual outbound request kwargs.
"""

from unittest.mock import MagicMock, patch

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.deepseek import DeepSeekProvider
from providers.moonshot import MoonshotProvider
from providers.xai import XAIModelProvider


def _mock_anthropic_response():
    resp = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "ok"
    resp.content = [block]
    resp.usage = MagicMock()
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    resp.stop_reason = "end_turn"
    resp.model = "claude-3-opus-20240229"
    return resp


def _mock_chat_response(model="grok-4"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].finish_reason = "stop"
    resp.model = model
    resp.id = "test-id"
    resp.created = 1234567890
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


def _call_kwargs(mock_client):
    mock_client.chat.completions.create.assert_called_once()
    return mock_client.chat.completions.create.call_args[1]


# --------------------------------------------------------------------------- #
# Sampling-capable model (xAI Grok): omit when unset, forward when explicit.
# --------------------------------------------------------------------------- #


@patch("providers.openai_compatible.OpenAI")
def test_grok_omits_temperature_when_unset_but_keeps_max_tokens(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("grok-4")

    provider = XAIModelProvider("test-key")
    provider.generate_content(
        prompt="hi",
        model_name="grok-4",
        temperature=None,  # caller did not specify
        max_output_tokens=256,
    )

    kwargs = _call_kwargs(mock_client)
    # Temperature must be ABSENT so the API uses its own default...
    assert "temperature" not in kwargs
    # ...but other sampling params (max_tokens) must still be sent for a
    # sampling-capable model (the omit-temp path must not disable them).
    assert kwargs["max_tokens"] == 256


@patch("providers.openai_compatible.OpenAI")
def test_grok_forwards_explicit_temperature(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("grok-4")

    provider = XAIModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="grok-4", temperature=0.5)

    kwargs = _call_kwargs(mock_client)
    assert kwargs["temperature"] == 0.5


# --------------------------------------------------------------------------- #
# DeepSeek (always-on thinking): omit when unset; clamp+forward when explicit.
# --------------------------------------------------------------------------- #


@patch("providers.openai_compatible.OpenAI")
def test_deepseek_omits_temperature_when_unset(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("deepseek-v4-pro")

    provider = DeepSeekProvider("test-key")
    provider.generate_content(prompt="hi", model_name="deepseek", temperature=None)

    kwargs = _call_kwargs(mock_client)
    assert "temperature" not in kwargs


@patch("providers.openai_compatible.OpenAI")
def test_deepseek_forwards_explicit_temperature(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("deepseek-v4-pro")

    provider = DeepSeekProvider("test-key")
    provider.generate_content(prompt="hi", model_name="deepseek", temperature=0.5)

    kwargs = _call_kwargs(mock_client)
    assert kwargs["temperature"] == 0.5


# --------------------------------------------------------------------------- #
# Moonshot Kimi (thinking): temperature is NEVER sent, even when explicit.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("temp", [None, 0.5, 1.0])
@patch("providers.openai_compatible.OpenAI")
def test_kimi_never_sends_temperature(mock_openai_class, temp):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("kimi-k2.6")

    provider = MoonshotProvider("test-key")
    provider.generate_content(prompt="hi", model_name="kimi", temperature=temp)

    kwargs = _call_kwargs(mock_client)
    # Kimi K2 thinking: "temperature is not modifiable — do not pass it explicitly".
    assert "temperature" not in kwargs
    # Thinking is still forced on via extra_body.
    assert kwargs.get("extra_body", {}).get("thinking", {}).get("type") == "enabled"


# --------------------------------------------------------------------------- #
# Anthropic (native client): omit when unset; clamp+forward when explicit.
# Critical for Opus 4.7+/Fable 5 which 400 on any non-default temperature.
# --------------------------------------------------------------------------- #


@patch("providers.anthropic.Anthropic")
def test_anthropic_omits_temperature_when_unset(mock_anthropic_class):
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-3-opus", temperature=None)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert "temperature" not in kwargs


@patch("providers.anthropic.Anthropic")
def test_anthropic_forwards_explicit_temperature(mock_anthropic_class):
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-3-opus", temperature=0.5)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["temperature"] == 0.5
