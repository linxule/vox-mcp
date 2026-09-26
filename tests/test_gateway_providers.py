"""Offline wire-contract tests for explicit gateway routes."""

import json
import os

import httpx
import pytest

from providers.gateway import CloudflareGatewayProvider, VercelGatewayProvider
from providers.shared import ProviderType


@pytest.fixture(autouse=True)
def gateway_environment(monkeypatch):
    """Neither local .env files nor real account settings affect these tests."""
    settings = {}

    def lookup(name, default=None):
        return settings.get(name, default)

    monkeypatch.setattr("providers.gateway.get_env", lookup)
    monkeypatch.setattr("providers.openai_compatible.get_env", lookup)
    monkeypatch.setattr("utils.model_restrictions.get_env", lookup)
    monkeypatch.setattr("utils.model_restrictions._restriction_service", None)
    return settings


def record_requests(provider):
    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gateway-test",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "  untouched\n"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    provider._test_transport = httpx.MockTransport(respond)
    return requests


@pytest.mark.parametrize(
    "provider_type,route,upstream,url,headers",
    [
        (
            CloudflareGatewayProvider,
            "cloudflare/openai/gpt-4.1",
            "openai/gpt-4.1",
            "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1/chat/completions",
            {"cf-aig-gateway-id": "named-gateway"},
        ),
        (
            VercelGatewayProvider,
            "vercel/anthropic/claude-sonnet-4",
            "anthropic/claude-sonnet-4",
            "https://ai-gateway.vercel.sh/v1/chat/completions",
            {},
        ),
    ],
)
def test_gateway_wire_request_preserves_prompt_defaults_and_route(
    gateway_environment, provider_type, route, upstream, url, headers
):
    gateway_environment.update(CLOUDFLARE_ACCOUNT_ID="test-account", CLOUDFLARE_GATEWAY_ID="named-gateway")
    provider = provider_type(api_key="test-key")
    requests = record_requests(provider)
    response = provider.generate_content("  original prompt\n", route)

    assert len(requests) == 1
    assert str(requests[0].url) == url
    assert requests[0].headers["authorization"] == "Bearer test-key"
    for name, value in headers.items():
        assert requests[0].headers[name] == value
    assert json.loads(requests[0].content) == {
        "model": upstream,
        "messages": [{"role": "user", "content": "  original prompt\n"}],
        "stream": False,
    }
    assert response.content == "  untouched\n"
    assert response.model_name == route
    assert response.metadata["model"] == upstream
    assert provider.get_capabilities(route).model_name == route


def test_cloudflare_workers_model_retains_at_cf_and_has_gateway_header():
    provider = CloudflareGatewayProvider("test-key", account_id="account")
    requests = record_requests(provider)
    provider.generate_content("Hello", "cloudflare/@cf/meta/llama-example", temperature=0.6, max_output_tokens=100)
    assert requests[0].headers["cf-aig-gateway-id"] == "default"
    body = json.loads(requests[0].content)
    assert body["model"] == "@cf/meta/llama-example"
    assert body["temperature"] == 0.6
    assert body["max_tokens"] == 100


def test_cloudflare_headers_are_isolated_between_instances():
    first = CloudflareGatewayProvider("test-key", account_id="account", gateway_id="first")
    second = CloudflareGatewayProvider("test-key", account_id="account", gateway_id="second")
    assert first.DEFAULT_HEADERS == {"cf-aig-gateway-id": "first"}
    assert second.DEFAULT_HEADERS == {"cf-aig-gateway-id": "second"}


def test_gateway_client_ignores_proxies_without_mutating_process_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:1")
    provider = VercelGatewayProvider("test-key")
    requests = record_requests(provider)
    provider.generate_content("Hello", "vercel/creator/model")
    assert len(requests) == 1
    assert os.environ["HTTPS_PROXY"] == "http://localhost:1"
    assert provider.client._client._trust_env is False


@pytest.mark.parametrize("account_id", [None, "", "account/other", "account?query", "account\n"])
def test_invalid_cloudflare_account_fails_early(account_id):
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        CloudflareGatewayProvider("test-key", account_id=account_id)


@pytest.mark.parametrize("gateway_id", ["name\nheader", "name/path", "a" * 65])
def test_invalid_cloudflare_gateway_fails_early(gateway_id):
    with pytest.raises(ValueError, match="CLOUDFLARE_GATEWAY_ID"):
        CloudflareGatewayProvider("test-key", account_id="account", gateway_id=gateway_id)


@pytest.mark.parametrize("key", ["", "   "])
def test_gateway_requires_nonempty_credentials(key):
    with pytest.raises(ValueError, match="requires an API key"):
        VercelGatewayProvider(key)


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4.1",
        "cloudflare/openai/gpt-4.1",
        "vercel/model",
        "vercel//model",
        "vercel/openai/",
        "vercel/../model",
        "vercel/openai/model?query",
        "vercel/openai/model name",
        "vercel/cloudflare/openai/model",
        "vercel/@cf/meta/model",
    ],
)
def test_vercel_does_not_claim_unprefixed_or_invalid_routes(model):
    assert not VercelGatewayProvider("test-key").validate_model_name(model)


def test_cloudflare_does_not_accept_dynamic_routes_on_rest_api():
    provider = CloudflareGatewayProvider("test-key", account_id="account")
    assert not provider.validate_model_name("cloudflare/dynamic/my-route")


def test_configured_catalog_does_not_claim_verified_capabilities(gateway_environment):
    gateway_environment["VERCEL_MODELS"] = "openai/model, vercel/anthropic/model,openai/model"
    provider = VercelGatewayProvider("test-key")
    assert provider.list_models() == ["vercel/anthropic/model", "vercel/openai/model"]
    assert provider.validate_model_name("vercel/future/new-model")
    assert provider.get_provider_type() == ProviderType.VERCEL
    for capabilities in provider.get_all_model_capabilities().values():
        assert capabilities._is_generic
        assert "unverified" in capabilities.description
        assert "budgeting estimates" in capabilities.description
        assert not capabilities.supports_images
        assert not capabilities.supports_extended_thinking
        assert not capabilities.use_openai_response_api


def test_empty_catalog_does_not_invent_available_models():
    provider = VercelGatewayProvider("test-key")
    assert provider.list_models() == []
    assert provider.validate_model_name("vercel/creator/new-model")


@pytest.mark.parametrize("allowed", ["openai/allowed", "vercel/openai/allowed"])
def test_gateway_allowlist_accepts_upstream_or_routed_names(gateway_environment, allowed):
    gateway_environment.update(VERCEL_ALLOWED_MODELS=allowed, VERCEL_MODELS="openai/allowed,openai/blocked")
    provider = VercelGatewayProvider("test-key")
    assert provider.validate_model_name("vercel/openai/allowed")
    assert not provider.validate_model_name("vercel/openai/blocked")
    assert provider.list_models() == ["vercel/openai/allowed"]
    assert len(provider.list_models(respect_restrictions=False)) == 2
    with pytest.raises(ValueError):
        provider.generate_content("blocked", "vercel/openai/blocked")


@pytest.mark.parametrize("extra", [{"images": ["image.png"]}, {"thinking_mode": "high"}])
def test_unverified_modalities_and_thinking_fail_before_network(extra):
    provider = VercelGatewayProvider("test-key")
    requests = record_requests(provider)
    with pytest.raises(ValueError, match="unverified"):
        provider.generate_content("Hello", "vercel/creator/model", **extra)
    assert requests == []
