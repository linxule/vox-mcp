"""Current native model selection and request-shape regressions (offline)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from providers.deepseek import DeepSeekProvider
from providers.gemini import GeminiModelProvider
from tests.test_gemini_interactions import _fake_generate_content_response, _fake_interaction
from tests.test_temperature_passthrough import _mock_chat_response
from tools.models import ToolModelCategory


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
@pytest.mark.parametrize(
    "mode,effort", [("minimal", "low"), ("low", "low"), ("medium", "high"), ("high", "high"), ("max", "max")]
)
def test_deepseek_sends_native_model_and_configurable_effort(model, mode, effort):
    provider = DeepSeekProvider("test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _mock_chat_response(model)
    provider.generate_content(
        "hi", model, thinking_mode=mode, top_p=0.97, presence_penalty=0.5, max_output_tokens=393216
    )
    request = provider._client.chat.completions.create.call_args.kwargs
    assert request["model"] == model
    assert request["reasoning_effort"] == effort
    assert request["max_tokens"] == 393216
    assert request["extra_body"]["thinking"] == {"type": "enabled"}
    assert request["top_p"] == 0.97
    assert "presence_penalty" not in request


def test_deepseek_flash_alias_keeps_real_image_on_wire():
    provider = DeepSeekProvider("test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _mock_chat_response("deepseek-flash")
    provider.generate_content("describe", "deepseek", images=[str(Path(__file__).with_name("triangle.png"))])
    request = provider._client.chat.completions.create.call_args.kwargs
    assert request["model"] == "deepseek-flash"
    assert request["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert provider.get_capabilities("deepseek").max_image_size_mb == 32.0
    assert not provider.get_capabilities("v4-pro").supports_images


def test_deepseek_explicit_nonthinking_wins_over_default_effort():
    provider = DeepSeekProvider("test-key")
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = _mock_chat_response("deepseek-flash")
    caller_body = {"thinking": {"type": "disabled"}}
    provider.generate_content("hi", "deepseek", extra_body=caller_body)
    request = provider._client.chat.completions.create.call_args.kwargs
    # OpenAI SDK merges extra_body last when constructing the actual JSON body.
    wire = {**request, **request["extra_body"]}
    assert wire["reasoning_effort"] == "none"
    assert wire["thinking"] == {"type": "disabled"}
    assert caller_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("category", list(ToolModelCategory))
def test_deepseek_default_respects_available_models(category):
    provider = DeepSeekProvider("test-key")
    assert provider.get_preferred_model(category, ["deepseek-v4-pro", "deepseek-flash"]) == "deepseek-flash"
    assert provider.get_preferred_model(category, ["deepseek-v4-pro"]) == "deepseek-v4-pro"
    assert provider.get_preferred_model(category, []) is None


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-3.1-pro-preview"])
@pytest.mark.parametrize("mode,expected", [("minimal", "low"), ("medium", "medium"), ("max", "high")])
@pytest.mark.parametrize("path", ["interactions", "fallback", "images"])
def test_gemini_current_models_send_supported_thinking_levels(monkeypatch, model, mode, expected, path):
    monkeypatch.setenv("VOX_GEMINI_USE_INTERACTIONS", "true")
    provider = GeminiModelProvider("test-key")
    client = MagicMock()
    provider._client = client
    client.interactions.create.return_value = _fake_interaction()
    client.models.generate_content.return_value = _fake_generate_content_response()
    if path == "fallback":
        client.interactions.create.side_effect = TypeError("interactions unavailable")
    images = [str(Path(__file__).with_name("triangle.png"))] if path == "images" else None
    provider.generate_content("hi", model, thinking_mode=mode, images=images)
    if path == "interactions":
        request = client.interactions.create.call_args.kwargs
        assert request["generation_config"]["thinking_level"] == expected
        assert "temperature" not in request["generation_config"]
        client.models.generate_content.assert_not_called()
    else:
        request = client.models.generate_content.call_args.kwargs
        assert request["config"].thinking_config.thinking_level.value.lower() == expected
        assert request["config"].temperature is None
        if path == "images":
            assert len(request["contents"][0]["parts"]) == 2
    assert request["model"] == model


def test_gemini_catalog_and_category_defaults():
    provider = GeminiModelProvider("test-key")
    assert provider.get_capabilities("flash").model_name == "gemini-3.8-flash"
    for retired in ("gemini-2.0-flash", "gemini-2.0-flash-lite", "flash-2.0", "flashlite"):
        assert not provider.validate_model_name(retired)
    for legacy in ("gemini-2.5-flash", "gemini-2.5-pro"):
        assert "legacy" in provider.get_capabilities(legacy).description
    allowed = list(provider.get_all_model_capabilities())
    assert provider.get_preferred_model(ToolModelCategory.BALANCED, allowed) == "gemini-3.8-flash"
    assert provider.get_preferred_model(ToolModelCategory.FAST_RESPONSE, allowed) == "gemini-3.8-flash"
    assert provider.get_preferred_model(ToolModelCategory.EXTENDED_REASONING, allowed) == "gemini-3.1-pro-preview"
    assert provider.get_preferred_model(ToolModelCategory.BALANCED, ["gemini-2.5-flash"]) == "gemini-2.5-flash"


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro"])
@pytest.mark.parametrize("path", ["interactions", "generate_content", "fallback", "images"])
def test_gemini_omitted_thinking_uses_upstream_default(monkeypatch, model, path):
    monkeypatch.setenv("VOX_GEMINI_USE_INTERACTIONS", "false" if path == "generate_content" else "true")
    provider = GeminiModelProvider("test-key")
    client = MagicMock()
    provider._client = client
    client.interactions.create.return_value = _fake_interaction()
    client.models.generate_content.return_value = _fake_generate_content_response()
    if path == "fallback":
        client.interactions.create.side_effect = TypeError("interactions unavailable")
    images = [str(Path(__file__).with_name("triangle.png"))] if path == "images" else None
    provider.generate_content("hi", model, images=images)
    if path == "interactions":
        request = client.interactions.create.call_args.kwargs
        assert "thinking_level" not in request.get("generation_config", {})
        client.models.generate_content.assert_not_called()
    else:
        request = client.models.generate_content.call_args.kwargs
        assert request["config"].thinking_config is None
        assert "thinking_config" not in request["config"].model_dump(exclude_none=True)
        if path == "images":
            assert len(request["contents"][0]["parts"]) == 2
    assert request["model"] == model
