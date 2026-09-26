"""Explicit, text-only routes through Cloudflare and Vercel AI Gateway.

Gateway catalogs change independently of Vox. Configured model IDs are exposed
without inventing model-specific capabilities or making discovery requests at
startup. Token limits below are local budgeting estimates, not provider claims.
"""

import re

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType


class _GatewayProvider(OpenAICompatibleProvider):
    """Share explicit route validation and conservative capability metadata."""

    ROUTE_PREFIX = ""
    MODELS_ENV = ""
    PROVIDER_TYPE: ProviderType

    def __init__(self, api_key: str, base_url: str, **kwargs):
        if not api_key or not api_key.strip():
            raise ValueError(f"{self.FRIENDLY_NAME} requires an API key")
        super().__init__(api_key, base_url=base_url, **kwargs)
        self._models: dict[str, ModelCapabilities] = {}
        for raw in (get_env(self.MODELS_ENV, "") or "").split(","):
            name = raw.strip()
            if not name:
                continue
            route = name if name.startswith(self.ROUTE_PREFIX) else self.ROUTE_PREFIX + name
            upstream = self._resolve_model_name(route)
            self._models[route] = self._generic_capabilities(upstream)

    def get_provider_type(self) -> ProviderType:
        return self.PROVIDER_TYPE

    def _resolve_model_name(self, model_name: str) -> str:
        if not model_name.startswith(self.ROUTE_PREFIX):
            raise ValueError(f"{self.FRIENDLY_NAME} model names must start with '{self.ROUTE_PREFIX}'")
        upstream = model_name[len(self.ROUTE_PREFIX) :]
        parts = upstream.split("/")
        if (
            len(parts) < 2
            or any(not part or part in {".", ".."} for part in parts)
            or any(character.isspace() or character in "?#\\" for character in upstream)
            or upstream.startswith(("cloudflare/", "vercel/", "dynamic/"))
            or (upstream.startswith("@cf/") and (self.ROUTE_PREFIX != "cloudflare/" or len(parts) < 3))
        ):
            raise ValueError(f"Invalid {self.FRIENDLY_NAME} model ID: use '{self.ROUTE_PREFIX}author/model'")
        return upstream

    def _generic_capabilities(self, upstream: str) -> ModelCapabilities:
        route = self.ROUTE_PREFIX + upstream
        capabilities = ModelCapabilities(
            provider=self.PROVIDER_TYPE,
            model_name=route,
            friendly_name=f"{self.FRIENDLY_NAME} ({upstream})",
            intelligence_score=1,
            description=(
                "Gateway passthrough; model capabilities unverified. "
                "32,768 context / 4,096 output tokens are conservative local budgeting estimates. "
                "Text only; provider defaults apply unless a request parameter is supplied."
            ),
            context_window=32_768,
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=False,
            supports_streaming=False,
            supports_function_calling=False,
            supports_images=False,
        )
        capabilities._is_generic = True
        return capabilities

    def _lookup_capabilities(self, canonical_name: str, requested_name: str | None = None) -> ModelCapabilities:
        return self._models.get(self.ROUTE_PREFIX + canonical_name) or self._generic_capabilities(canonical_name)

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        return dict(self._models)

    def list_models(
        self,
        *,
        respect_restrictions: bool = True,
        include_aliases: bool = True,
        lowercase: bool = False,
        unique: bool = False,
    ) -> list[str]:
        models = {
            name: capabilities
            for name, capabilities in self._models.items()
            if not respect_restrictions or self.validate_model_name(name)
        }
        return ModelCapabilities.collect_model_names(
            models, include_aliases=include_aliases, lowercase=lowercase, unique=unique
        )

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        # Never silently discard an image or fabricate a reasoning mapping for an
        # unverified gateway model. Normal tool calls already gate these flags.
        if images:
            raise ValueError("Gateway models currently support text only; image capabilities are unverified")
        if kwargs.get("thinking_mode") is not None:
            raise ValueError("Gateway thinking_mode mapping is unverified; use the provider's default")
        response = super().generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        # Conversation continuation must keep the gateway selection, even though
        # the OpenAI-compatible request and response use the upstream model ID.
        response.model_name = self.ROUTE_PREFIX + self._resolve_model_name(model_name)
        return response


class CloudflareGatewayProvider(_GatewayProvider):
    """Cloudflare's current account REST API (not the deprecated /compat API)."""

    FRIENDLY_NAME = "Cloudflare AI Gateway"
    ROUTE_PREFIX = "cloudflare/"
    MODELS_ENV = "CLOUDFLARE_MODELS"
    PROVIDER_TYPE = ProviderType.CLOUDFLARE

    def __init__(self, api_key: str, account_id: str | None = None, gateway_id: str | None = None, **kwargs):
        account_id = account_id or get_env("CLOUDFLARE_ACCOUNT_ID")
        if not account_id or not re.fullmatch(r"[A-Za-z0-9_-]+", account_id):
            raise ValueError("CLOUDFLARE_ACCOUNT_ID must be a nonempty account ID")
        gateway_id = gateway_id or get_env("CLOUDFLARE_GATEWAY_ID") or "default"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", gateway_id):
            raise ValueError("CLOUDFLARE_GATEWAY_ID must contain 1-64 letters, digits, underscores, or hyphens")
        # Headers belong to this instance; accounts and gateway choices must not
        # leak between providers created in the same process.
        self.DEFAULT_HEADERS = {"cf-aig-gateway-id": gateway_id}
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        super().__init__(api_key, base_url=base_url, **kwargs)


class VercelGatewayProvider(_GatewayProvider):
    """Vercel's OpenAI-compatible gateway using an AI Gateway API key."""

    FRIENDLY_NAME = "Vercel AI Gateway"
    ROUTE_PREFIX = "vercel/"
    MODELS_ENV = "VERCEL_MODELS"
    PROVIDER_TYPE = ProviderType.VERCEL

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, base_url="https://ai-gateway.vercel.sh/v1", **kwargs)
