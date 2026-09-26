"""DeepSeek AI provider implementation for DeepSeek models."""

import logging

from .openai_compatible import OpenAICompatibleProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType
from .shared.temperature import RangeTemperatureConstraint
from .shared.thinking import EffortLevelThinkingConstraint

logger = logging.getLogger(__name__)


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek AI provider for chat and reasoning models.

    DeepSeek AI provides OpenAI-compatible APIs for their models.
    The current default is V4.1 Flash, which exposes thinking mode via an
    extra_body toggle on a single endpoint.
    """

    FRIENDLY_NAME = "DeepSeek"

    # Native API IDs and limits: https://api-docs.deepseek.com/api/create-chat-completion/
    _EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "max": "max"}

    MODEL_CAPABILITIES = {
        "deepseek-flash": ModelCapabilities(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-flash",
            friendly_name="DeepSeek V4.1 Flash",
            context_window=1_000_000,
            max_output_tokens=393_216,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            supports_json_mode=True,
            supports_function_calling=True,
            supports_extended_thinking=True,
            thinking_constraint=EffortLevelThinkingConstraint(effort_map=_EFFORT_MAP),
            supports_images=True,
            max_image_size_mb=32.0,
            unsupported_params=["presence_penalty", "frequency_penalty"],
            aliases=["deepseek", "deepseek-v4.1", "v4.1-flash"],
            description="DeepSeek V4.1 Flash - Configurable reasoning and vision (1M context, 384Ki output)",
        ),
        "deepseek-v4-pro": ModelCapabilities(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-v4-pro",
            friendly_name="DeepSeek V4 Pro",
            context_window=1_000_000,
            max_output_tokens=393_216,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            supports_json_mode=True,
            supports_function_calling=True,
            supports_extended_thinking=True,
            thinking_constraint=EffortLevelThinkingConstraint(effort_map=_EFFORT_MAP),
            unsupported_params=["presence_penalty", "frequency_penalty"],
            aliases=["deepseek-v4", "v4", "v4-pro"],
            description="DeepSeek V4 Pro - Configurable reasoning (1M context, 384Ki output, text-only)",
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize DeepSeek AI provider.

        Args:
            api_key: DeepSeek AI API key from https://platform.deepseek.com
            **kwargs: Additional configuration passed to OpenAI-compatible provider
        """
        # DeepSeek AI API endpoint
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        super().__init__(api_key, **kwargs)

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get model capabilities for the specified model."""
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.MODEL_CAPABILITIES:
            raise ValueError(f"Unsupported DeepSeek model: {model_name}")

        # Apply model restrictions if configured
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.DEEPSEEK, resolved_name, model_name):
            raise ValueError(f"DeepSeek model '{model_name}' is not allowed by current restrictions.")

        return self.MODEL_CAPABILITIES[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.DEEPSEEK

    def validate_model_name(self, model_name: str) -> bool:
        """Check if the model name is supported by this provider."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in self.MODEL_CAPABILITIES

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using DeepSeek AI models.

        Args:
            prompt: User prompt
            model_name: DeepSeek model name or alias
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0), or ``None`` to omit it
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Resolve model aliases before API call
        resolved_model_name = self._resolve_model_name(model_name)

        # Validate the model
        if not self.validate_model_name(model_name):
            raise ValueError(f"Model '{model_name}' is not supported by DeepSeek AI provider")

        # Get model capabilities to adjust parameters
        capabilities = self.get_capabilities(model_name)

        # Clamp only an explicitly-supplied temperature. When the caller omits it
        # (None) we pass None through so the parent omits it on the wire and the
        # API applies its own default, rather than us inventing one. Both models take
        # temperature in [0.0, 2.0]; the constraint on the model, not this comment,
        # is the authority for the range.
        if temperature is not None and capabilities.temperature_constraint:
            temperature = capabilities.temperature_constraint.get_corrected_value(temperature)

        # DeepSeek exposes thinking mode via extra_body on a single
        # endpoint; enable it by default. Merge into any
        # caller-supplied extra_body to avoid silently dropping the toggle.
        if capabilities.supports_extended_thinking:
            extra_body = kwargs.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body = dict(extra_body)
                kwargs["extra_body"] = extra_body
                extra_body.setdefault("thinking", {"type": "enabled"})
                if extra_body.get("thinking") == {"type": "disabled"}:
                    # The SDK merges extra_body after named parameters. Prevent
                    # the default reasoning effort from re-enabling thinking.
                    extra_body.setdefault("reasoning_effort", "none")

        # Call parent implementation with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def get_preferred_model(self, category, allowed_models: list[str]) -> str | None:
        """Prefer current Flash; preserve explicit Pro choices and restrictions."""
        for model in ("deepseek-flash", "deepseek-v4-pro"):
            if model in allowed_models:
                return model
        return None

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        Both supported DeepSeek models offer configurable reasoning effort.
        """
        resolved_name = self._resolve_model_name(model_name)
        if resolved_name in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[resolved_name].supports_extended_thinking
        return False
