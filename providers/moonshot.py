"""Moonshot AI provider implementation for Kimi models."""

import logging

from .openai_compatible import OpenAICompatibleProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType
from .shared.temperature import RangeTemperatureConstraint
from .shared.thinking import AlwaysOnThinkingConstraint

logger = logging.getLogger(__name__)


class MoonshotProvider(OpenAICompatibleProvider):
    """Moonshot AI provider for Kimi K2 models.

    Moonshot AI provides OpenAI-compatible APIs for their Kimi models,
    which are optimized for agentic capabilities, tool calling, and research tasks.
    """

    FRIENDLY_NAME = "Moonshot AI"

    # Define Kimi models with their capabilities
    MODEL_CAPABILITIES = {
        "kimi-k2-thinking-turbo": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2-thinking-turbo",
            friendly_name="Kimi K2 Thinking Turbo",
            context_window=262_144,
            max_output_tokens=65_536,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 1.0),
            supports_json_mode=True,
            supports_function_calling=True,
            supports_extended_thinking=True,
            thinking_constraint=AlwaysOnThinkingConstraint(),
            aliases=["kimi", "kimi-k2"],
            intelligence_score=20,
            description="Kimi K2 Thinking Turbo (262K context) - Text-only model with always-on thinking",
        ),
        "kimi-k2.5": ModelCapabilities(
            provider=ProviderType.MOONSHOT,
            model_name="kimi-k2.5",
            friendly_name="Kimi K2.5",
            context_window=262_144,
            max_output_tokens=65_536,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 1.0),
            supports_json_mode=True,
            supports_function_calling=True,
            supports_extended_thinking=True,
            thinking_constraint=AlwaysOnThinkingConstraint(),
            supports_images=True,
            max_image_size_mb=20.0,
            aliases=["k2.5", "kimi-k25"],
            intelligence_score=20,
            description="Kimi K2.5 (262K context) - Multimodal model with vision and always-on thinking",
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Moonshot AI provider.

        Args:
            api_key: Moonshot AI API key from https://platform.moonshot.cn
            **kwargs: Additional configuration passed to OpenAI-compatible provider
        """
        # Moonshot AI API endpoint
        # Use .cn endpoint (where account/API key is registered)
        kwargs.setdefault("base_url", "https://api.moonshot.cn/v1")
        super().__init__(api_key, **kwargs)

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get model capabilities for the specified model."""
        resolved_name = self._resolve_model_name(model_name)

        if resolved_name not in self.MODEL_CAPABILITIES:
            raise ValueError(f"Unsupported Moonshot model: {model_name}")

        # Apply model restrictions if configured
        from utils.model_restrictions import get_restriction_service

        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.MOONSHOT, resolved_name, model_name):
            raise ValueError(f"Moonshot model '{model_name}' is not allowed by current restrictions.")

        return self.MODEL_CAPABILITIES[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.MOONSHOT

    def validate_model_name(self, model_name: str) -> bool:
        """Check if the model name is supported by this provider."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in self.MODEL_CAPABILITIES

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.6,
        max_output_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using Moonshot AI models.

        Args:
            prompt: User prompt
            model_name: Moonshot model name or alias
            system_prompt: Optional system prompt
            temperature: Temperature for generation (0.0-1.0)
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Resolve model aliases before API call
        resolved_model_name = self._resolve_model_name(model_name)

        # Validate the model
        if not self.validate_model_name(model_name):
            raise ValueError(f"Model '{model_name}' is not supported by Moonshot AI provider")

        # Get model capabilities to adjust parameters
        capabilities = self.get_capabilities(model_name)

        # Adjust temperature according to model constraints
        if hasattr(capabilities, "temperature_constraint") and capabilities.temperature_constraint:
            temperature = capabilities.temperature_constraint.get_corrected_value(temperature)

        # Moonshot API requires explicit extra_body to control thinking mode
        # Thinking is on by default for K2.5 and K2 Thinking models
        if capabilities.supports_extended_thinking:
            kwargs.setdefault("extra_body", {"thinking": {"type": "enabled"}})

        # Call parent implementation with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        Returns True for models with extended thinking capabilities (e.g., kimi-k2-thinking-turbo).
        """
        try:
            resolved_name = self._resolve_model_name(model_name)
            if resolved_name in self.MODEL_CAPABILITIES:
                capabilities = self.MODEL_CAPABILITIES[resolved_name]
                return getattr(capabilities, "supports_extended_thinking", False)
            return False
        except Exception:
            return False
