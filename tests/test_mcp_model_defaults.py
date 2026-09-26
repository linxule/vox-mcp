"""Exercise the real MCP dispatch path through native providers and mocked SDKs."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.moonshot import MoonshotProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from providers.xai import XAIModelProvider
from server import handle_call_tool


@pytest.fixture
def native_wire(monkeypatch):
    def make_provider(model):
        client = Mock()
        if model.startswith("claude"):
            monkeypatch.setattr("providers.anthropic.Anthropic", lambda **kwargs: client)
            provider = AnthropicModelProvider("test-key")
            client.messages.create.return_value = SimpleNamespace(
                content=[SimpleNamespace(type="text", text="answer")],
                usage=SimpleNamespace(input_tokens=2, output_tokens=1),
                stop_reason="end_turn",
                model=model,
            )
            create = client.messages.create
        elif model.startswith("gpt"):
            provider = OpenAIModelProvider("test-key")
            provider._client = client
            client.responses.create.return_value = SimpleNamespace(
                output_text="answer",
                model=model,
                id="response",
                created_at=0,
                usage=SimpleNamespace(input_tokens=2, output_tokens=1, total_tokens=3),
            )
            create = client.responses.create
        else:
            if "/" in model:
                provider = OpenRouterProvider("test-key")
            else:
                provider = MoonshotProvider("test-key") if model.startswith("kimi") else XAIModelProvider("test-key")
            provider._client = client
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
                model=model,
                id="response",
                created=0,
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )
            create = client.chat.completions.create
        monkeypatch.setattr(
            ModelProviderRegistry,
            "get_provider_for_model",
            lambda name: provider if provider.validate_model_name(name) else None,
        )
        return create

    return make_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["kimi-k3", "grok-4.6", "claude-fable-5-1", "claude-opus-5-5", "gpt-6-astra"])
async def test_mcp_omitted_reasoning_preserves_native_default(native_wire, model):
    create = native_wire(model)
    result = await handle_call_tool("chat", {"prompt": "hello", "model": model})
    assert json.loads(result[0].text)["content"] == "answer"
    wire = create.call_args.kwargs
    assert wire["model"] == model
    assert {"temperature", "reasoning", "reasoning_effort", "output_config", "thinking"}.isdisjoint(wire)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,effort", [("kimi-k3", "max"), ("grok-4.6", "xhigh"), ("claude-fable-5-1", "max"), ("gpt-6-astra", "max")]
)
async def test_mcp_explicit_max_reaches_native_model_contract(native_wire, model, effort):
    create = native_wire(model)
    await handle_call_tool("chat", {"prompt": "hello", "model": model, "thinking_mode": "max"})
    wire = create.call_args.kwargs
    if model.startswith("claude"):
        assert wire["output_config"] == {"effort": effort}
    elif model.startswith("gpt"):
        assert wire["reasoning"] == {"effort": effort}
    else:
        assert wire["reasoning_effort"] == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking_mode", [None, "max"])
@pytest.mark.parametrize(
    "model,max_effort",
    [
        ("openai/gpt-6-astra", "max"),
        ("openai/gpt-6-sol", "max"),
        ("openai/gpt-6-luna", "max"),
        ("moonshotai/kimi-k3", "max"),
        ("google/gemini-3.8-flash", "high"),
        ("google/gemini-3.1-pro-preview", "high"),
        ("anthropic/claude-opus-5.5", "max"),
        ("anthropic/claude-fable-5.1", "max"),
        ("deepseek/deepseek-v4.1-flash", "max"),
        ("x-ai/grok-4.6", "xhigh"),
        ("x-ai/grok-4.7", "xhigh"),
    ],
)
async def test_openrouter_mcp_images_and_reasoning_reach_exact_model(native_wire, model, max_effort, thinking_mode):
    create = native_wire(model)
    image = Path(__file__).with_name("triangle.png")
    arguments = {"prompt": "describe the image", "model": model, "images": [str(image)]}
    if thinking_mode is not None:
        arguments["thinking_mode"] = thinking_mode
    result = await handle_call_tool("chat", arguments)
    assert json.loads(result[0].text)["content"] == "answer"
    wire = create.call_args.kwargs
    assert wire["model"] == model
    assert wire["messages"][0]["content"][0] == {"type": "text", "text": "describe the image"}
    images = [part for part in wire["messages"][0]["content"] if part["type"] == "image_url"]
    assert len(images) == 1
    data_url = images[0]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == image.read_bytes()
    if thinking_mode is None:
        assert "reasoning_effort" not in wire
    else:
        assert wire["reasoning_effort"] == max_effort


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-3-opus-20240229"])
async def test_anthropic_mcp_preserves_exact_image_bytes(native_wire, model):
    create = native_wire(model)
    image = Path(__file__).with_name("triangle.png")
    await handle_call_tool("chat", {"prompt": "describe the image", "model": model, "images": [str(image)]})
    wire = create.call_args.kwargs
    assert wire["model"] == model
    content = wire["messages"][0]["content"]
    assert {"type": "text", "text": "describe the image"} in content
    images = [part for part in content if part["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["type"] == "base64"
    assert images[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(images[0]["source"]["data"]) == image.read_bytes()
