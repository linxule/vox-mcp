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
    resp.model = "claude-sonnet-5"
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
# Anthropic (native client): omit when unset; adaptive-thinking models
# (Fable 5 / Opus 4.8 / Sonnet 5) 400 on ANY non-default temperature, so an
# explicit value must be dropped, and no thinking block may be fabricated for
# them (adaptive thinking is a server-side default). Haiku 4.5 keeps classic
# extended thinking: budget must fit under max_tokens, and temperature is
# dropped whenever a thinking block is sent. Claude 3 Opus (researcher access)
# still plainly samples: explicit temperature forwards.
#
# Each test patches the restriction service: the developer's own .env may set
# ANTHROPIC_ALLOWED_MODELS (runtime config), and catalog tests must not depend
# on it.
# --------------------------------------------------------------------------- #


def _permissive_restrictions():
    service = MagicMock()
    service.is_allowed.return_value = True
    return service


@patch("utils.model_restrictions.get_restriction_service")
@patch("providers.anthropic.Anthropic")
def test_anthropic_omits_temperature_when_unset(mock_anthropic_class, mock_restrictions):
    mock_restrictions.return_value = _permissive_restrictions()
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-sonnet-5", temperature=None)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert "temperature" not in kwargs
    # Adaptive thinking is server-side: no fabricated thinking block.
    assert "thinking" not in kwargs


@patch("utils.model_restrictions.get_restriction_service")
@patch("providers.anthropic.Anthropic")
def test_anthropic_drops_explicit_temperature_on_adaptive_models(mock_anthropic_class, mock_restrictions):
    mock_restrictions.return_value = _permissive_restrictions()
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-opus-4-8", temperature=0.5)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    # Opus 4.8 / Sonnet 5 / Fable 5 reject non-default temperature with a 400;
    # the explicit value must never reach the wire.
    assert "temperature" not in kwargs
    assert "thinking" not in kwargs


@patch("utils.model_restrictions.get_restriction_service")
@patch("providers.anthropic.Anthropic")
def test_anthropic_haiku_thinking_fits_max_tokens_and_drops_temperature(mock_anthropic_class, mock_restrictions):
    mock_restrictions.return_value = _permissive_restrictions()
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-haiku-4-5", temperature=0.3, thinking_mode="medium")

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    thinking = kwargs["thinking"]
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] >= 1024
    # Anthropic requires budget_tokens < max_tokens.
    assert thinking["budget_tokens"] < kwargs["max_tokens"]
    # ...and rejects non-default temperature alongside thinking.
    assert "temperature" not in kwargs


@patch("utils.model_restrictions.get_restriction_service")
@patch("providers.anthropic.Anthropic")
def test_anthropic_haiku_default_thinking_still_drops_temperature(mock_anthropic_class, mock_restrictions):
    """Haiku's thinking constraint resolves a default mode even when the caller
    omits thinking_mode, so thinking is always enabled for it in practice — and
    an explicit temperature must still be dropped on that path."""
    mock_restrictions.return_value = _permissive_restrictions()
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-haiku-4-5", temperature=0.5)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    # Even without an explicit thinking_mode, the constraint resolves its
    # default mode, thinking is enabled, and temperature is dropped.
    assert "thinking" in kwargs
    assert "temperature" not in kwargs


@patch("utils.model_restrictions.get_restriction_service")
@patch("providers.anthropic.Anthropic")
def test_anthropic_claude3_opus_forwards_explicit_temperature(mock_anthropic_class, mock_restrictions):
    """Claude 3 Opus (kept for researcher access) has no thinking and plain
    sampling: an explicit temperature is clamped and forwarded."""
    mock_restrictions.return_value = _permissive_restrictions()
    mock_client = MagicMock()
    mock_anthropic_class.return_value = mock_client
    mock_client.messages.create.return_value = _mock_anthropic_response()

    provider = AnthropicModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="claude-3-opus", temperature=0.5)

    mock_client.messages.create.assert_called_once()
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["temperature"] == 0.5
    assert "thinking" not in kwargs
