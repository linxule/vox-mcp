"""DeepSeek AI provider implementation for DeepSeek models."""

import logging
from typing import Optional

from .openai_compatible import OpenAICompatibleProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType
from .shared.temperature import RangeTemperatureConstraint
from .shared.thinking import AlwaysOnThinkingConstraint

logger = logging.getLogger(__name__)


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek AI provider for chat and reasoning models.

    DeepSeek AI provides OpenAI-compatible APIs for their models,
    including both standard chat models and advanced reasoning models (R1).
    """

    FRIENDLY_NAME = "DeepSeek"

    # Define DeepSeek models with their capabilities
    MODEL_CAPABILITIES = {
        "deepseek-reasoner": ModelCapabilities(
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-reasoner",
            friendly_name="DeepSeek Reasoner (R1)",
            context_window=128_000,
            max_output_tokens=64_000,
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            supports_json_mode=True,
            supports_function_calling=True,
            supports_extended_thinking=True,
            thinking_constraint=AlwaysOnThinkingConstraint(),
            aliases=["deepseek", "deepseek-r1", "ds-reasoner", "r1"],
            description="DeepSeek Reasoner (R1) - Advanced reasoning model with chain-of-thought thinking (128K context, text-only)",
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
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using DeepSeek AI models.

        Args:
            prompt: User prompt
            model_name: DeepSeek model name or alias
            system_prompt: Optional system prompt
            temperature: Temperature for generation (0.0-2.0)
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

        # Adjust temperature according to model constraints
        if hasattr(capabilities, "temperature_constraint") and capabilities.temperature_constraint:
            temperature = capabilities.temperature_constraint.get_corrected_value(temperature)

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

        DeepSeek Reasoner (R1) model supports thinking capabilities.
        """
        resolved_name = self._resolve_model_name(model_name)
        if resolved_name in self.MODEL_CAPABILITIES:
            return self.MODEL_CAPABILITIES[resolved_name].supports_extended_thinking
        return False
