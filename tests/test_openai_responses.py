"""Exercise real SDK serialization without sending provider requests."""

import json

import httpx
import pytest
from openai import OpenAI

import utils.model_restrictions as restrictions
from providers.openai import OpenAIModelProvider
from providers.registries.openai import OpenAIModelRegistry
from tools.models import ToolModelCategory


@pytest.fixture
def sdk_provider(monkeypatch):
    monkeypatch.setattr(restrictions, "_restriction_service", None)
    calls = []

    def respond(request):
        calls.append((str(request.url), json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": calls[-1][1]["model"],
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Unchanged answer.", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            },
        )

    provider = OpenAIModelProvider("test-key")
    provider._client = OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    yield provider, calls
    provider.close()


@pytest.mark.parametrize("tier", ["astra", "sol", "luna"])
def test_gpt6_uses_stateless_responses_without_fabricated_parameters(sdk_provider, tier):
    provider, calls = sdk_provider
    response = provider.generate_content("Exact prompt.", f"gpt-6-{tier}", max_output_tokens=345, temperature=0.3)
    url, body = calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert body == {
        "model": f"gpt-6-{tier}",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Exact prompt."}]}],
        "store": False,
        "max_output_tokens": 345,
    }
    assert response.content == "Unchanged answer."
    assert response.usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


@pytest.mark.parametrize("tier", ["astra", "sol", "luna"])
@pytest.mark.parametrize(
    "mode,effort", [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "max")]
)
def test_gpt6_explicit_thinking_mapping_reaches_wire(sdk_provider, tier, mode, effort):
    provider, calls = sdk_provider
    provider.generate_content("Prompt", f"gpt-6-{tier}", thinking_mode=mode, top_p=0.2)
    assert calls[0][1]["reasoning"] == {"effort": effort}
    assert "top_p" not in calls[0][1]


def test_responses_converts_images_without_nesting_chat_blocks(sdk_provider, monkeypatch):
    provider, calls = sdk_provider
    image_url = "data:image/png;base64,dGVzdA=="
    monkeypatch.setattr(provider, "_process_image", lambda _: {"type": "image_url", "image_url": {"url": image_url}})
    provider.generate_content("Look at this.", "gpt-6-astra", images=["image.png"])
    assert calls[0][1]["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at this."},
                {"type": "input_image", "image_url": image_url, "detail": "auto"},
            ],
        }
    ]


def test_responses_preserves_explicit_roles_and_omits_output_limit(sdk_provider):
    provider, calls = sdk_provider
    provider._generate_with_responses_endpoint(
        "gpt-6-astra",
        [
            {"role": "system", "content": "Explicit caller instruction."},
            {"role": "assistant", "content": "Earlier answer."},
            {"role": "user", "content": "Next prompt."},
        ],
    )
    body = calls[0][1]
    assert [message["role"] for message in body["input"]] == ["system", "assistant", "user"]
    assert body["input"][1]["content"] == [{"type": "output_text", "text": "Earlier answer."}]
    assert "max_output_tokens" not in body
    assert "reasoning" not in body


def test_gpt6_catalog_and_default_preserve_explicit_legacy_models(sdk_provider):
    provider, _ = sdk_provider
    models = provider.list_models()
    for category in ToolModelCategory:
        assert provider.get_preferred_model(category, models) == "gpt-6-astra"
    for model in ("gpt-5.1", "gpt-5.1-codex", "gpt-5", "o3-pro"):
        assert provider._resolve_model_name(model) == model
    for model in ("gpt-6-astra", "gpt-6-sol", "gpt-6-luna"):
        capabilities = provider.get_capabilities(model)
        assert capabilities.context_window == 1_050_000
        assert capabilities.max_output_tokens == 128_000
        assert capabilities.supports_images


def test_effort_map_is_loaded_per_model_and_unknown_modes_rejected():
    registry = OpenAIModelRegistry()
    raw = {
        "model_name": "mapping-test",
        "thinking_constraint": "effort_level",
        "effort_map": {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "xhigh"},
        "default_thinking_mode": "high",
    }
    capabilities = registry._convert_entry(raw)
    assert capabilities.get_effective_thinking_params("max") == {"effort": "xhigh"}
    assert capabilities.thinking_constraint.get_default_mode() == "high"
    raw["effort_map"] = {"invalid": "high"}
    with pytest.raises(ValueError, match="every thinking mode"):
        registry._convert_entry(raw)
