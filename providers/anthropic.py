"""Anthropic Claude model provider implementation."""

import logging
import os
import time
from typing import List, Optional

import anthropic
from anthropic import Anthropic

from .base import ModelProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType
from .shared.temperature import RangeTemperatureConstraint

logger = logging.getLogger(__name__)


class AnthropicModelProvider(ModelProvider):
    """Anthropic Claude model provider implementation."""

    # Model configurations using ModelCapabilities objects
    SUPPORTED_MODELS = MODEL_CAPABILITIES = {
        "claude-3-opus": ModelCapabilities(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            friendly_name="Claude 3 Opus",
            context_window=200_000,
            max_output_tokens=4_096,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=32.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.8),
            aliases=["opus3", "claude3-opus", "claude-3-opus"],
            description="Claude 3 Opus (200K context) - Original frontier model with vision (researcher access required, retired Jan 2026)",
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        if not api_key:
            raise ValueError("Anthropic API key is required")

        try:
            self.client = Anthropic(api_key=api_key)
            self.api_key = api_key
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific model."""
        resolved_name = self._resolve_model_name(model_name)
        if resolved_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown Anthropic model: {model_name}")

        # Check restrictions
        from utils.model_restrictions import get_restriction_service
        restriction_service = get_restriction_service()
        if not restriction_service.is_allowed(ProviderType.ANTHROPIC, resolved_name, model_name):
            raise ValueError(f"Anthropic model '{model_name}' is not allowed by current restrictions.")

        return self.SUPPORTED_MODELS[resolved_name]

    def get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.ANTHROPIC

    def validate_model_name(self, model_name: str) -> bool:
        """Check if the model name is supported by this provider."""
        resolved_name = self._resolve_model_name(model_name)
        return resolved_name in self.SUPPORTED_MODELS

    def _resolve_model_name(self, model_name: str) -> Optional[str]:
        """Resolve aliases to actual model names."""
        # Direct match
        if model_name in self.SUPPORTED_MODELS:
            return model_name

        # Check aliases
        for model_key, capabilities in self.SUPPORTED_MODELS.items():
            if model_name.lower() in [alias.lower() for alias in capabilities.aliases]:
                return model_key

        return None

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_output_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate content using Anthropic API.

        Args:
            prompt: User prompt to send to the model
            model_name: Model name or alias
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (thinking_mode, images, etc.)

        Returns:
            ModelResponse with generated content
        """
        try:
            # Get model capabilities
            capabilities = self.get_capabilities(model_name)
            actual_model_name = capabilities.model_name

            # Convert prompt to Anthropic messages format
            messages = [{"role": "user", "content": prompt}]

            # Prepare request parameters
            request_params = {
                "model": actual_model_name,
                "messages": messages,
                "max_tokens": min(max_output_tokens or 4000, capabilities.max_output_tokens),
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            # Add temperature if supported
            if capabilities.supports_temperature:
                # Apply temperature constraints
                if hasattr(capabilities, 'temperature_constraint') and capabilities.temperature_constraint:
                    temperature = capabilities.temperature_constraint.get_corrected_value(temperature)
                request_params["temperature"] = temperature

            # Add extended thinking if supported, using ThinkingConstraint
            thinking_mode = kwargs.get("thinking_mode")
            thinking_params = capabilities.get_effective_thinking_params(thinking_mode)
            if thinking_params is not None and "thinking_budget" in thinking_params:
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_params["thinking_budget"],
                }
            elif capabilities.supports_extended_thinking:
                # Fallback for models without a constraint configured
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": capabilities.max_thinking_tokens or 16_000,
                }

            # Make API call
            response = self.client.messages.create(**request_params)

            # Extract response content (including thinking blocks)
            content = ""
            thinking_content = ""

            if response.content:
                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == 'thinking':
                            thinking_content += f"[THINKING]\n{block.thinking}\n\n"
                        elif block.type == 'text':
                            content += block.text
                    elif hasattr(block, 'text'):
                        content += block.text

            # Include thinking content in final response if present
            final_content = f"{thinking_content}{content}" if thinking_content else content

            # Build usage dict
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

            return ModelResponse(
                content=final_content,
                usage=usage,
                model_name=actual_model_name,
                friendly_name=capabilities.friendly_name,
                provider=self.get_provider_type(),
                metadata={
                    "stop_reason": response.stop_reason,
                    "model": response.model,
                },
            )

        except Exception as e:
            logger.error(f"Anthropic API error for model {model_name}: {e}")
            raise RuntimeError(f"Anthropic API error for model {model_name}: {e}")

    def _convert_messages(self, messages: List[dict]) -> List[dict]:
        """Convert messages to Anthropic format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert OpenAI format to Anthropic format
            if role == "system":
                # System messages are handled separately in Anthropic
                continue
            elif role == "assistant":
                role = "assistant"
            else:  # user, function, etc.
                role = "user"
            
            converted.append({
                "role": role,
                "content": content
            })
        
        return converted

    def get_supported_models(self) -> List[str]:
        """Get list of supported model names including aliases."""
        models = list(self.SUPPORTED_MODELS.keys())
        for capabilities in self.SUPPORTED_MODELS.values():
            models.extend(capabilities.aliases)
        return models 
