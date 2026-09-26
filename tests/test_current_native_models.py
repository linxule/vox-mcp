"""Current native model contracts: real routing plus mocked SDK wire payloads."""

import base64
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.moonshot import MoonshotProvider
from providers.xai import XAIModelProvider


def chat_response(model):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        model=model,
        id="test-response",
        created=123,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5"])
@pytest.mark.parametrize("mode,effort", [(None, None), ("minimal", "low"), ("medium", "medium"), ("max", "max")])
def test_adaptive_claude_uses_effort_and_ignores_empty_thinking(model, mode, effort):
    with patch("providers.anthropic.Anthropic") as sdk:
        client = sdk.return_value
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            model=model,
        )
        result = AnthropicModelProvider("test-key").generate_content(
            prompt="hello",
            model_name=model,
            thinking_mode=mode,
            temperature=0.4,
            max_output_tokens=700,
        )
    wire = client.messages.create.call_args.kwargs
    assert wire["model"] == model
    assert wire["max_tokens"] == 700
    assert "thinking" not in wire
    assert "temperature" not in wire
    if effort is None:
        assert "output_config" not in wire
    else:
        assert wire["output_config"] == {"effort": effort}
    assert result.content == "ok"


def test_versioned_claude_aliases_do_not_upgrade_silently():
    with patch("providers.anthropic.Anthropic"):
        provider = AnthropicModelProvider("test-key")
    assert provider.get_capabilities("fable").model_name == "claude-fable-5-1"
    assert provider.get_capabilities("fable5").model_name == "claude-fable-5"
    assert provider.get_capabilities("opus4.8").model_name == "claude-opus-4-8"
    assert provider.get_capabilities("opus5.5").model_name == "claude-opus-5-5"
    assert provider.get_capabilities("opus3").model_name == "claude-3-opus-20240229"


@pytest.mark.parametrize("mode,effort", [(None, None), ("minimal", "low"), ("medium", "high"), ("max", "max")])
def test_k3_wire_contract(mode, effort):
    with patch("providers.openai_compatible.OpenAI") as sdk:
        client = sdk.return_value
        client.chat.completions.create.return_value = chat_response("kimi-k3")
        provider = MoonshotProvider("test-key")
        provider.generate_content(
            prompt="hello",
            model_name="kimi",
            max_output_tokens=700,
            thinking_mode=mode,
            temperature=0.4,
            top_p=0.2,
            presence_penalty=1,
            frequency_penalty=1,
        )
    wire = client.chat.completions.create.call_args.kwargs
    assert wire["model"] == "kimi-k3"
    assert wire["max_completion_tokens"] == 700
    assert {"max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty", "extra_body"}.isdisjoint(
        wire
    )
    if effort is None:
        assert "reasoning_effort" not in wire
    else:
        assert wire["reasoning_effort"] == effort
    assert provider.get_capabilities("k2.6").model_name == "kimi-k2.6"
    assert provider.get_capabilities("kimi-k2").model_name == "kimi-k2.6"


def test_k26_keeps_its_native_contract():
    with patch("providers.openai_compatible.OpenAI") as sdk:
        client = sdk.return_value
        client.chat.completions.create.return_value = chat_response("kimi-k2.6")
        MoonshotProvider("test-key").generate_content(prompt="hello", model_name="k2.6", max_output_tokens=700)
    wire = client.chat.completions.create.call_args.kwargs
    assert wire["model"] == "kimi-k2.6"
    assert wire["max_tokens"] == 700
    assert wire["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "max_completion_tokens" not in wire


@pytest.mark.parametrize(
    "model,effort", [("grok", "xhigh"), ("grok46", "xhigh"), ("grok47", "xhigh"), ("grok45", "high")]
)
def test_grok_max_effort_is_model_specific(model, effort):
    with patch("providers.openai_compatible.OpenAI") as sdk:
        client = sdk.return_value
        client.chat.completions.create.return_value = chat_response("grok-4.7")
        XAIModelProvider("test-key").generate_content(
            prompt="hello",
            model_name=model,
            thinking_mode="max",
            max_output_tokens=700,
            frequency_penalty=1,
            presence_penalty=1,
            stop=["end"],
        )
    wire = client.chat.completions.create.call_args.kwargs
    assert wire["model"] == ({"grok45": "grok-4.5", "grok47": "grok-4.7"}.get(model, "grok-4.6"))
    assert wire["reasoning_effort"] == effort
    assert wire["max_tokens"] == 700
    assert {"frequency_penalty", "presence_penalty", "stop"}.isdisjoint(wire)


def test_current_provider_preferences_respect_allowlists_in_chat_category():
    from tools.models import ToolModelCategory

    with patch("providers.anthropic.Anthropic"):
        claude = AnthropicModelProvider("test-key")
    kimi = MoonshotProvider("test-key")
    grok = XAIModelProvider("test-key")
    category = ToolModelCategory.FAST_RESPONSE
    assert (
        claude.get_preferred_model(category, ["claude-haiku-4-5", "claude-fable-5-1", "claude-opus-5-5"])
        == "claude-opus-5-5"
    )
    assert claude.get_preferred_model(category, ["claude-fable-5-1"]) == "claude-fable-5-1"
    assert kimi.get_preferred_model(category, ["kimi-k2.6", "kimi-k3"]) == "kimi-k3"
    assert kimi.get_preferred_model(category, ["kimi-k2.6"]) == "kimi-k2.6"
    assert grok.get_preferred_model(category, ["grok-4.7", "grok-4.6"]) == "grok-4.6"
    assert grok.get_preferred_model(category, ["grok-4.7"]) == "grok-4.7"
    assert all(provider.get_preferred_model(category, []) is None for provider in [claude, kimi, grok])


@pytest.mark.parametrize("mode", [None, "medium"])
def test_haiku_thinking_is_enabled_only_when_requested(mode):
    with patch("providers.anthropic.Anthropic") as sdk:
        client = sdk.return_value
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            model="claude-haiku-4-5",
        )
        AnthropicModelProvider("test-key").generate_content(
            prompt="hello",
            model_name="claude-haiku-4-5",
            thinking_mode=mode,
            temperature=0.4,
            max_output_tokens=700,
        )
    wire = client.messages.create.call_args.kwargs
    if mode is None:
        assert "thinking" not in wire
        assert "output_config" not in wire
        assert wire["max_tokens"] == 700
        assert wire["temperature"] == 0.4
    else:
        assert wire["thinking"]["type"] == "enabled"
        assert 0 < wire["thinking"]["budget_tokens"] < wire["max_tokens"]
        assert "temperature" not in wire


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-3-opus"])
def test_anthropic_images_reach_native_wire_without_changing_prompt(tmp_path, model):
    # Actual tiny PNG fixtures exercise shared file/data-URL validation and encoding.
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    image_file = tmp_path / "image.png"
    image_file.write_bytes(base64.b64decode(png))
    data_url = "data:image/png;base64," + png
    prompt = "What changed?\nKeep this text exactly."
    with patch("providers.anthropic.Anthropic") as sdk:
        client = sdk.return_value
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            stop_reason="end_turn",
            model=model,
        )
        AnthropicModelProvider("test-key").generate_content(
            prompt=prompt,
            model_name=model,
            images=[str(image_file), data_url, str(image_file)],
        )
    wire = client.messages.create.call_args.kwargs
    assert "system" not in wire
    assert wire["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@pytest.mark.parametrize("invalid", ["missing", "encoded_size", "invalid_type"])
def test_anthropic_rejects_invalid_images_before_inference(tmp_path, invalid):
    image_file = tmp_path / "image.png"
    if invalid == "encoded_size":
        # 8MB decoded fits the helper's10MB cap, but exceeds10MB encoded.
        image_file.write_bytes(b"x" * (8 * 1024 * 1024))
    elif invalid == "invalid_type":
        image_file = tmp_path / "image.txt"
        image_file.write_text("not an image")
    with patch("providers.anthropic.Anthropic") as sdk:
        provider = AnthropicModelProvider("test-key")
        with pytest.raises(
            RuntimeError, match="Image file not found|Base64-encoded image too large|Unsupported image format"
        ):
            provider.generate_content(prompt="hello", model_name="claude-opus-5-5", images=[str(image_file)])
        sdk.return_value.messages.create.assert_not_called()


def test_claude_snapshot_id_and_alias_resolve_to_the_same_capability():
    with patch("providers.anthropic.Anthropic"):
        provider = AnthropicModelProvider("test-key")
    snapshot = "claude-3-opus-20240229"
    assert provider.validate_model_name(snapshot)
    assert provider.get_capabilities(snapshot) is provider.get_capabilities("opus3")
    assert provider._resolve_model_name(snapshot) == provider._resolve_model_name("claude-3-opus")
