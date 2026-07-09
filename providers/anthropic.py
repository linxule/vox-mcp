"""Anthropic Claude model provider implementation."""

import logging

from anthropic import Anthropic

from .base import ModelProvider
from .shared import ModelCapabilities, ModelResponse, ProviderType
from .shared.temperature import RangeTemperatureConstraint
from .shared.thinking import TokenBudgetThinkingConstraint

logger = logging.getLogger(__name__)


class AnthropicModelProvider(ModelProvider):
    """Anthropic Claude model provider implementation."""

    # Model configurations using ModelCapabilities objects.
    #
    # Catalog refreshed 2026-07-09 against the Claude API models overview
    # (https://platform.claude.com/docs/en/about-claude/models/overview):
    #   - Fable 5 / Opus 4.8 / Sonnet 5 use ADAPTIVE thinking (server-side,
    #     effort defaults to high) and 400 on any non-default temperature /
    #     top_p / top_k, so supports_temperature=False keeps the wire clean.
    #     Their dateless IDs are pinned snapshots, not evergreen pointers.
    #   - Haiku 4.5 keeps classic extended thinking (budget_tokens) and does
    #     accept temperature — but not alongside thinking; generate_content
    #     drops temperature whenever a thinking block is sent.
    #   - max_image_size_mb is the per-image limit on the direct Claude API
    #     (10 MB base64); the old 32.0 was the whole-request cap.
    #   - Bare "opus"/"sonnet"/"haiku" aliases are deliberately NOT claimed
    #     here: the OpenRouter registry already binds them, and colliding
    #     would silently re-route users who configure both providers.
    SUPPORTED_MODELS = MODEL_CAPABILITIES = {
        "claude-fable-5": ModelCapabilities(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-fable-5",
            friendly_name="Claude Fable 5",
            intelligence_score=20,
            context_window=1_000_000,
            max_output_tokens=128_000,
            supports_extended_thinking=False,  # adaptive thinking, always on server-side
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=False,  # rejects non-default temperature/top_p/top_k
            allow_code_generation=True,
            aliases=["fable", "fable5", "fable-5"],
            description=(
                "Claude Fable 5 (1M context) - Anthropic's most capable widely released model; "
                "adaptive thinking always on. Availability is policy-sensitive "
                "(limited/premium terms since July 2026)"
            ),
        ),
        "claude-opus-4-8": ModelCapabilities(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-opus-4-8",
            friendly_name="Claude Opus 4.8",
            intelligence_score=19,
            context_window=1_000_000,
            max_output_tokens=128_000,
            supports_extended_thinking=False,  # adaptive thinking; effort defaults to high
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=False,  # rejects non-default temperature/top_p/top_k
            allow_code_generation=True,
            aliases=["opus4.8", "opus-4.8", "claude-opus-4.8"],
            description="Claude Opus 4.8 (1M context) - Complex agentic coding and enterprise work; adaptive thinking",
        ),
        "claude-sonnet-5": ModelCapabilities(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-sonnet-5",
            friendly_name="Claude Sonnet 5",
            intelligence_score=17,
            context_window=1_000_000,
            max_output_tokens=128_000,
            supports_extended_thinking=False,  # adaptive thinking; effort defaults to high
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=False,  # rejects non-default temperature/top_p/top_k
            aliases=["sonnet5", "sonnet-5", "claude-sonnet5"],
            description="Claude Sonnet 5 (1M context) - Best combination of speed and intelligence; adaptive thinking",
        ),
        "claude-haiku-4-5": ModelCapabilities(
            provider=ProviderType.ANTHROPIC,
            # Dateless alias; resolves to the pinned claude-haiku-4-5-20251001 snapshot.
            model_name="claude-haiku-4-5",
            friendly_name="Claude Haiku 4.5",
            intelligence_score=13,
            context_window=200_000,
            max_output_tokens=64_000,
            supports_extended_thinking=True,
            max_thinking_tokens=48_000,
            thinking_constraint=TokenBudgetThinkingConstraint(max_tokens=48_000),
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            supports_images=True,
            max_image_size_mb=10.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.8),
            aliases=["haiku4.5", "haiku-4.5", "claude-haiku-4.5"],
            description=(
                "Claude Haiku 4.5 (200K context) - Fastest model with near-frontier intelligence; "
                "extended thinking via thinking_mode (temperature is dropped when thinking is enabled)"
            ),
        ),
        # Retired for general access (Jan 2026) but kept in the catalog:
        # researcher-access keys can still call the pinned snapshot.
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
            max_image_size_mb=10.0,
            supports_temperature=True,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.8),
            aliases=["opus3", "claude3-opus", "claude-3-opus"],
            description="Claude 3 Opus (200K context) - Original frontier model with vision (researcher access required, retired for general access Jan 2026)",
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

    def _resolve_model_name(self, model_name: str) -> str | None:
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
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using Anthropic API.

        Args:
            prompt: User prompt to send to the model
            model_name: Model name or alias
            system_prompt: Optional system prompt
            temperature: Sampling temperature, or ``None`` to omit it
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

            # Add temperature only when the caller explicitly supplied one and the
            # model supports it. Omitting it lets the API use its own default —
            # required for Opus 4.7+/Fable 5, which reject any non-default
            # temperature/top_p/top_k with a 400.
            if capabilities.supports_temperature and temperature is not None:
                # Apply temperature constraints to the explicit value
                if hasattr(capabilities, "temperature_constraint") and capabilities.temperature_constraint:
                    temperature = capabilities.temperature_constraint.get_corrected_value(temperature)
                request_params["temperature"] = temperature

            # Add extended thinking if supported, using ThinkingConstraint.
            # Adaptive-thinking models (Fable 5, Opus 4.8, Sonnet 5) carry no
            # constraint and supports_extended_thinking=False, so nothing is
            # sent for them — adaptive thinking is a server-side default and
            # fabricating parameters would violate the passthrough principle.
            thinking_mode = kwargs.get("thinking_mode")
            thinking_params = capabilities.get_effective_thinking_params(thinking_mode)
            budget = None
            if thinking_params is not None and "thinking_budget" in thinking_params:
                budget = thinking_params["thinking_budget"]
            elif capabilities.supports_extended_thinking:
                # Fallback for models without a constraint configured
                budget = capabilities.max_thinking_tokens or 16_000

            if budget is not None:
                # Anthropic requires budget_tokens < max_tokens: grow max_tokens
                # to fit the budget (it is a cap, not a spend target), bounded by
                # the model's own output limit. If the budget still cannot fit,
                # skip thinking rather than send a request the API will 400.
                if request_params["max_tokens"] <= budget:
                    request_params["max_tokens"] = min(capabilities.max_output_tokens, budget + 8_192)
                if request_params["max_tokens"] > budget:
                    request_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                    # Anthropic rejects non-default temperature alongside an
                    # enabled thinking block; thinking wins, temperature is
                    # dropped (the model then applies its server default).
                    if request_params.pop("temperature", None) is not None:
                        logger.debug(
                            "Dropping explicit temperature for %s: not permitted with extended thinking",
                            actual_model_name,
                        )
                else:
                    logger.debug(
                        "Skipping extended thinking for %s: budget %d cannot fit under max_tokens %d",
                        actual_model_name,
                        budget,
                        request_params["max_tokens"],
                    )

            # Make API call
            response = self.client.messages.create(**request_params)

            # Extract response content (including thinking blocks)
            content = ""
            thinking_content = ""

            if response.content:
                for block in response.content:
                    if hasattr(block, "type"):
                        if block.type == "thinking":
                            thinking_content += f"[THINKING]\n{block.thinking}\n\n"
                        elif block.type == "text":
                            content += block.text
                    elif hasattr(block, "text"):
                        content += block.text

            # Include thinking content in final response if present
            final_content = f"{thinking_content}{content}" if thinking_content else content

            # Build usage dict
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
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

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
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

            converted.append({"role": role, "content": content})

        return converted

    def get_supported_models(self) -> list[str]:
        """Get list of supported model names including aliases."""
        models = list(self.SUPPORTED_MODELS.keys())
        for capabilities in self.SUPPORTED_MODELS.values():
            models.extend(capabilities.aliases)
        return models
