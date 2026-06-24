"""
Unit tests for the Gemini Interactions API adapter (stateless, store=False).

These validate vox's call construction and response parsing against a mocked
client, plus the graceful fallback to the (fully-supported) generateContent path
when the still-evolving Interactions surface fails. Live behavior must be
smoke-tested separately.
"""

import os
from unittest.mock import MagicMock, patch

from providers.gemini import GeminiModelProvider


def _fake_interaction(text="hello from interactions", in_tok=11, out_tok=7):
    interaction = MagicMock()
    interaction.output_text = text
    interaction.id = "interaction-123"
    status = MagicMock()
    status.name = "COMPLETED"
    interaction.status = status
    usage = MagicMock()
    usage.total_input_tokens = in_tok
    usage.total_output_tokens = out_tok
    interaction.usage = usage
    return interaction


def _fake_generate_content_response(text="fallback text", in_tok=5, out_tok=3):
    resp = MagicMock()
    resp.text = text
    cand = MagicMock()
    cand.finish_reason.name = "STOP"
    resp.candidates = [cand]
    resp.usage_metadata.prompt_token_count = in_tok
    resp.usage_metadata.candidates_token_count = out_tok
    return resp


def _provider_with_mock_client():
    provider = GeminiModelProvider("test-key")
    mock_client = MagicMock()
    provider._client = mock_client  # inject; bypass real genai.Client construction
    return provider, mock_client


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_interactions_path_omits_temperature_when_unset():
    provider, client = _provider_with_mock_client()
    client.interactions.create.return_value = _fake_interaction()

    resp = provider.generate_content(prompt="hi", model_name="gemini-2.5-flash", temperature=None)

    client.interactions.create.assert_called_once()
    kwargs = client.interactions.create.call_args[1]
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["input"] == "hi"
    assert kwargs["store"] is False
    # No fabricated temperature; the model uses its own default.
    assert "temperature" not in kwargs.get("generation_config", {})
    # generateContent path not used.
    client.models.generate_content.assert_not_called()
    # Parsed from output_text + usage.
    assert resp.content == "hello from interactions"
    assert resp.usage["input_tokens"] == 11
    assert resp.usage["output_tokens"] == 7
    assert resp.usage["total_tokens"] == 18
    assert resp.metadata["api"] == "interactions"
    assert resp.metadata["interaction_id"] == "interaction-123"


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_interactions_path_forwards_explicit_temperature():
    provider, client = _provider_with_mock_client()
    client.interactions.create.return_value = _fake_interaction()

    provider.generate_content(prompt="hi", model_name="gemini-2.5-flash", temperature=0.4)

    kwargs = client.interactions.create.call_args[1]
    assert kwargs["generation_config"]["temperature"] == 0.4


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_interactions_path_sets_thinking_level_for_gemini3():
    provider, client = _provider_with_mock_client()
    client.interactions.create.return_value = _fake_interaction()

    provider.generate_content(prompt="hi", model_name="gemini-3.1", temperature=None, thinking_mode="medium")

    kwargs = client.interactions.create.call_args[1]
    # Gemini 3 accepts low/medium/high on Interactions.
    assert kwargs["generation_config"]["thinking_level"] == "medium"


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_interactions_path_sets_thinking_level_for_gemini2x():
    """Regression (Kimi finding 3): Gemini 2.x thinking must map to thinking_level
    on Interactions (was silently dropped). 2.x accepts only low/high, so
    medium/high/max collapse to high and minimal/low to low."""
    provider, client = _provider_with_mock_client()
    client.interactions.create.return_value = _fake_interaction()

    provider.generate_content(prompt="hi", model_name="gemini-2.5-pro", temperature=None, thinking_mode="medium")

    kwargs = client.interactions.create.call_args[1]
    assert kwargs["generation_config"]["thinking_level"] == "high"


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_interactions_failure_falls_back_to_generate_content():
    provider, client = _provider_with_mock_client()
    # Non-retryable error so the retry loop fails fast (no sleeps).
    client.interactions.create.side_effect = TypeError("interactions surface unavailable")
    client.models.generate_content.return_value = _fake_generate_content_response()

    resp = provider.generate_content(prompt="hi", model_name="gemini-2.5-flash", temperature=None)

    client.interactions.create.assert_called_once()
    client.models.generate_content.assert_called_once()
    assert resp.content == "fallback text"
    # Interactions disabled for the rest of the session after one failure.
    assert provider._interactions_disabled is True


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "false"})
def test_disabled_flag_uses_generate_content_directly():
    provider, client = _provider_with_mock_client()
    client.models.generate_content.return_value = _fake_generate_content_response(text="direct gc")

    resp = provider.generate_content(prompt="hi", model_name="gemini-2.5-flash", temperature=None)

    client.interactions.create.assert_not_called()
    client.models.generate_content.assert_called_once()
    assert resp.content == "direct gc"


@patch.dict(os.environ, {"VOX_GEMINI_USE_INTERACTIONS": "true"})
def test_images_use_generate_content_not_interactions():
    """Image inputs are not wired on the interactions adapter; must use generateContent."""
    provider, client = _provider_with_mock_client()
    client.models.generate_content.return_value = _fake_generate_content_response(text="vision")

    # gemini-2.5-flash supports images per conf; provider falls through to models path.
    provider.generate_content(prompt="describe", model_name="gemini-2.5-flash", temperature=None, images=["/tmp/x.png"])

    client.interactions.create.assert_not_called()
    client.models.generate_content.assert_called_once()
