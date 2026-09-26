"""Offline startup, discovery, and routing tests for independent providers."""

import json
from unittest.mock import Mock

import httpx
import pytest

import config
import utils.env as env_config
import utils.model_restrictions as restrictions
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from tools.listmodels import ListModelsTool

pytestmark = pytest.mark.no_mock_provider


@pytest.fixture(autouse=True)
def isolated_providers(monkeypatch):
    """Keep real keys, process-wide caches, and provider network calls out of tests."""
    for variable in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MOONSHOT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "CUSTOM_API_URL",
        "CUSTOM_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_GATEWAY_ID",
        "CLOUDFLARE_MODELS",
        "VERCEL_AI_GATEWAY_API_KEY",
        "AI_GATEWAY_API_KEY",
        "VERCEL_MODELS",
        *restrictions.ModelRestrictionService.ENV_VARS.values(),
    ):
        monkeypatch.delenv(variable, raising=False)
    env_config.reload_env({"VOX_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setattr(restrictions, "_restriction_service", None)
    registry = ModelProviderRegistry()
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_initialized_providers", {})
    monkeypatch.setattr(config, "IS_AUTO_MODE", True)

    def no_network(*args, **kwargs):
        raise AssertionError("Provider discovery must not make a network request")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)

    import server

    monkeypatch.setattr(server.atexit, "register", lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    ("provider_type", "environment", "model_name"),
    [
        (ProviderType.ANTHROPIC, {"ANTHROPIC_API_KEY": "test-key"}, "claude-opus-4-8"),
        (ProviderType.MOONSHOT, {"MOONSHOT_API_KEY": "test-key"}, "kimi-k2.6"),
        (ProviderType.DEEPSEEK, {"DEEPSEEK_API_KEY": "test-key"}, "deepseek-v4-pro"),
        (
            ProviderType.CLOUDFLARE,
            {
                "CLOUDFLARE_API_TOKEN": "test-key",
                "CLOUDFLARE_ACCOUNT_ID": "test-account",
                "CLOUDFLARE_MODELS": "openai/test-model",
            },
            "cloudflare/openai/test-model",
        ),
        (
            ProviderType.VERCEL,
            {"VERCEL_AI_GATEWAY_API_KEY": "test-key", "VERCEL_MODELS": "openai/test-model"},
            "vercel/openai/test-model",
        ),
        (
            ProviderType.VERCEL,
            {"AI_GATEWAY_API_KEY": "test-key", "VERCEL_MODELS": "openai/test-model"},
            "vercel/openai/test-model",
        ),
    ],
)
def test_provider_can_start_and_route_without_another_provider(monkeypatch, provider_type, environment, model_name):
    from server import configure_providers

    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    configure_providers()

    assert ModelProviderRegistry.get_available_providers() == [provider_type]
    provider = ModelProviderRegistry.get_provider_for_model(model_name)
    assert provider is not None
    assert provider.get_provider_type() == provider_type
    assert model_name in ModelProviderRegistry.get_available_models()


@pytest.mark.parametrize("prefix", ["cloudflare", "vercel"])
@pytest.mark.parametrize("configured", [False, True])
def test_explicit_gateway_never_falls_through_to_openrouter(monkeypatch, prefix, configured):
    from providers.gateway import CloudflareGatewayProvider, VercelGatewayProvider

    provider_type = ProviderType(prefix)
    if configured:
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
        monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-key")
        monkeypatch.setenv(f"{prefix.upper()}_ALLOWED_MODELS", "openai/allowed")
        provider_class = CloudflareGatewayProvider if prefix == "cloudflare" else VercelGatewayProvider
        ModelProviderRegistry.register_provider(provider_type, provider_class)

    catch_all = Mock()
    catch_all.validate_model_name.return_value = True
    registry = ModelProviderRegistry()
    registry._providers[ProviderType.OPENROUTER] = Mock()
    registry._initialized_providers[ProviderType.OPENROUTER] = catch_all

    assert ModelProviderRegistry.get_provider_for_model(f"{prefix}/openai/denied") is None
    assert ModelProviderRegistry.get_provider_for_model(f"{prefix}/missing-author") is None
    catch_all.validate_model_name.assert_not_called()
    if configured:
        routed = ModelProviderRegistry.get_provider_for_model(f"{prefix}/openai/allowed")
        assert routed.get_provider_type() == provider_type
        catch_all.validate_model_name.assert_not_called()


@pytest.mark.parametrize("prefix", ["cloudflare", "vercel"])
@pytest.mark.parametrize("qualified_allowlist", [False, True])
@pytest.mark.asyncio
async def test_gateway_catalog_and_allowlist_agree_with_listing(monkeypatch, prefix, qualified_allowlist):
    from server import configure_providers

    if prefix == "cloudflare":
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
    else:
        monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-key")
    allowed = f"{prefix}/openai/allowed"
    denied = f"{prefix}/openai/denied"
    monkeypatch.setenv(f"{prefix.upper()}_MODELS", f"openai/allowed,{denied}")
    monkeypatch.setenv(f"{prefix.upper()}_ALLOWED_MODELS", allowed if qualified_allowlist else "openai/allowed")
    configure_providers()

    assert ModelProviderRegistry.get_available_models() == {allowed: ProviderType(prefix)}
    assert set(ModelProviderRegistry.get_available_models(respect_restrictions=False)) == {allowed, denied}
    assert ModelProviderRegistry.get_provider_for_model(denied) is None
    assert ModelProviderRegistry.get_provider_for_model(allowed) is not None

    result = json.loads((await ListModelsTool().execute({}))[0].text)
    assert result["status"] == "success"
    assert result["metadata"]["configured_providers"] == 1
    assert f"`{allowed}`" in result["content"]
    assert denied not in result["content"]
    assert "budgeting" in result["content"]


@pytest.mark.parametrize("variable", ["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"])
def test_incomplete_cloudflare_configuration_is_not_registered(monkeypatch, variable):
    from server import configure_providers

    monkeypatch.setenv(variable, "test-value")
    with pytest.raises(ValueError, match="At least one API configuration"):
        configure_providers()
    assert ModelProviderRegistry.get_available_providers() == []


def test_vercel_specific_key_takes_precedence_over_generic_key(monkeypatch):
    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "specific-test-key")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "generic-test-key")
    assert ModelProviderRegistry._get_api_key_for_provider(ProviderType.VERCEL) == "specific-test-key"


@pytest.mark.parametrize("prefix", ["cloudflare", "vercel"])
@pytest.mark.asyncio
async def test_gateway_only_auto_mode_starts_without_a_catalog(monkeypatch, prefix):
    from server import configure_providers
    from tools.chat import ChatTool

    if prefix == "cloudflare":
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
    else:
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    configure_providers()

    assert ModelProviderRegistry.get_available_models() == {}
    assert ModelProviderRegistry.get_provider_for_model(f"{prefix}/openai/explicit-model") is not None
    assert prefix + "/" in ChatTool()._format_available_models_list()
    result = json.loads((await ListModelsTool().execute({}))[0].text)
    assert result["metadata"]["configured_providers"] == 1
    assert f"{prefix.upper()}_MODELS" in result["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["vercel/vendor/model:version", "cloudflare/vendor/model:version"])
async def test_gateway_version_suffix_survives_mcp_and_continuation(monkeypatch, route):
    from providers.shared import ModelResponse
    from server import configure_providers, handle_call_tool
    from utils.conversation_memory import get_thread

    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    configure_providers()
    provider = ModelProviderRegistry.get_provider_for_model(route)
    received = []

    def generate(**kwargs):
        received.append(kwargs["model_name"])
        return ModelResponse(content="answer", model_name=route, provider=provider.get_provider_type())

    monkeypatch.setattr(provider, "generate_content", generate)
    first = await handle_call_tool("chat", {"prompt": "first", "model": route})
    first_payload = json.loads(first[0].text)
    thread_id = first_payload["continuation_offer"]["continuation_id"]
    await handle_call_tool("chat", {"prompt": "second", "continuation_id": thread_id})
    assert received == [route, route]
    assert get_thread(thread_id).turns[-1].model_name == route
