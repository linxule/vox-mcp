"""Gemini model provider implementation."""

import base64
import logging
import threading
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from google import genai
from google.genai import types

from utils.env import get_env
from utils.image_utils import validate_image

from .base import ModelProvider
from .registries.gemini import GeminiModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class GeminiModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Gemini integration built on the official Google SDK.

    The provider advertises detailed thinking-mode budgets, handles optional
    custom endpoints, and performs image pre-processing before forwarding a
    request to the Gemini APIs.
    """

    REGISTRY_CLASS = GeminiModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Note: Thinking mode budgets are now managed via ThinkingConstraint
    # objects attached to each model's ModelCapabilities. See
    # providers/shared/thinking.py for the TokenBudgetThinkingConstraint.

    def __init__(self, api_key: str, **kwargs):
        """Initialize Gemini provider with API key and optional base URL."""
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._client = None
        # Guards lazy client creation: the provider instance is cached and its
        # generate_content runs in worker threads (asyncio.to_thread), so without
        # this two threads could race and build duplicate clients/connection pools.
        self._client_lock = threading.Lock()
        # Set once if an Interactions API call fails, to skip it (and avoid
        # repeated failed attempts) for the rest of this process; see
        # generate_content's fallback to generateContent.
        self._interactions_disabled = False
        self._token_counters = {}  # Cache for token counting
        self._base_url = kwargs.get("base_url", None)  # Optional custom endpoint
        self._timeout_override = self._resolve_http_timeout()
        self._invalidate_capability_cache()

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Lazy, thread-safe initialization of the Gemini client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def _build_client(self):
        """Construct the genai.Client (called under _client_lock)."""
        http_options_kwargs: dict[str, object] = {}
        if self._base_url:
            http_options_kwargs["base_url"] = self._base_url
        if self._timeout_override is not None:
            http_options_kwargs["timeout"] = self._timeout_override

        if http_options_kwargs:
            http_options = types.HttpOptions(**http_options_kwargs)
            logger.debug(
                "Initializing Gemini client with options: base_url=%s timeout=%s",
                http_options_kwargs.get("base_url"),
                http_options_kwargs.get("timeout"),
            )
            return genai.Client(api_key=self.api_key, http_options=http_options)
        return genai.Client(api_key=self.api_key)

    def _resolve_http_timeout(self) -> float | None:
        """Compute timeout override from shared custom timeout environment variables."""

        timeouts: list[float] = []
        for env_var in [
            "CUSTOM_CONNECT_TIMEOUT",
            "CUSTOM_READ_TIMEOUT",
            "CUSTOM_WRITE_TIMEOUT",
            "CUSTOM_POOL_TIMEOUT",
        ]:
            raw_value = get_env(env_var)
            if raw_value:
                try:
                    timeouts.append(float(raw_value))
                except (TypeError, ValueError):
                    logger.warning("Invalid %s value '%s'; ignoring.", env_var, raw_value)

        if timeouts:
            # Use the largest timeout to best approximate long-running requests
            resolved = max(timeouts)
            logger.debug("Using custom Gemini HTTP timeout: %ss", resolved)
            return resolved

        return None

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_mode: str = "medium",
        images: list[str] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate content using Gemini model.

        Args:
            prompt: The main user prompt/query to send to the model
            model_name: Canonical model name or its alias (e.g., "gemini-2.5-pro", "flash", "gemini-3.1")
            system_prompt: Optional system instructions to prepend to the prompt for context/behavior
            temperature: Controls randomness in generation (0.0=deterministic, 1.0=creative), default 0.3
            max_output_tokens: Optional maximum number of tokens to generate in the response
            thinking_mode: Thinking budget level for models that support it ("minimal", "low", "medium", "high", "max"), default "medium"
            images: Optional list of image paths or data URLs to include with the prompt (for vision models)
            **kwargs: Additional keyword arguments (reserved for future use)

        Returns:
            ModelResponse: Contains the generated content, token usage stats, model metadata, and safety information
        """
        # Fetch capabilities; validate only an explicitly-supplied temperature.
        capabilities = self.get_capabilities(model_name)
        if temperature is not None:
            self.validate_parameters(model_name, temperature)

        resolved_model_name = self._resolve_model_name(model_name)

        # Resolve the temperature once. None => omit it so Gemini applies its own
        # default. Google strongly recommends keeping Gemini 3 at the default
        # (1.0) and warns that lowering it can cause looping/degraded reasoning,
        # so we never fabricate a value — an explicit caller value is clamped per
        # the model's constraint.
        effective_temperature = capabilities.get_effective_temperature(temperature)

        # Prefer the Interactions API (Google's GA, recommended surface) when
        # enabled, used statelessly (store=False) — vox keeps owning conversation
        # memory. The surface is still evolving, so on any failure we fall back to
        # the fully-supported generateContent path below and skip Interactions for
        # the rest of this process to avoid repeated failed attempts. Image inputs
        # always use generateContent (multimodal input is not wired on the
        # Interactions adapter yet).
        if self._use_interactions_api() and not self._interactions_disabled and not images:
            try:
                return self._generate_via_interactions(
                    prompt=prompt,
                    resolved_model_name=resolved_model_name,
                    system_prompt=system_prompt,
                    effective_temperature=effective_temperature,
                    max_output_tokens=max_output_tokens,
                    thinking_mode=thinking_mode,
                    capabilities=capabilities,
                )
            except Exception as exc:
                self._interactions_disabled = True
                logger.warning(
                    "Gemini Interactions API failed for %s (%s); falling back to "
                    "generateContent for the rest of this session. Set "
                    "VOX_GEMINI_USE_INTERACTIONS=false to skip Interactions entirely.",
                    resolved_model_name,
                    exc,
                )

        # Prepare content parts (text and potentially images)
        parts = []

        # Add system and user prompts as text
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        parts.append({"text": full_prompt})

        # Add images if provided and model supports vision
        if images and capabilities.supports_images:
            for image_path in images:
                try:
                    image_part = self._process_image(image_path)
                    if image_part:
                        parts.append(image_part)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    # Continue with other images and text
                    continue
        elif images and not capabilities.supports_images:
            logger.warning(f"Model {resolved_model_name} does not support images, ignoring {len(images)} image(s)")

        # Create contents structure
        contents = [{"parts": parts}]

        # Prepare generation config. effective_temperature is resolved above the
        # Interactions dispatch (None => omit so Gemini uses its own default).
        generation_config = types.GenerateContentConfig(candidate_count=1)
        if effective_temperature is not None:
            generation_config.temperature = effective_temperature

        # Add max output tokens if specified
        if max_output_tokens:
            generation_config.max_output_tokens = max_output_tokens

        # Add thinking configuration based on model generation
        effective_thinking_mode = thinking_mode
        if resolved_model_name.startswith("gemini-3"):
            # Gemini 3+ uses thinking_level (string) instead of thinking_budget (int)
            # Map thinking_mode to thinking_level (no "max" in ThinkingLevel, map to "high")
            level_map = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "max": "high"}
            level = level_map.get(thinking_mode, "high")
            effective_thinking_mode = level
            generation_config.thinking_config = types.ThinkingConfig(thinking_level=level)
        else:
            # Gemini 2.x uses token-based thinking budget
            thinking_params = capabilities.get_effective_thinking_params(effective_thinking_mode)
            if thinking_params and "thinking_budget" in thinking_params:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_params["thinking_budget"]
                )

        # Retry logic with progressive delays
        max_retries = 4  # Total of 4 attempts
        retry_delays = [1, 3, 5, 8]  # Progressive delays: 1s, 3s, 5s, 8s
        attempt_counter = {"value": 0}

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self.client.models.generate_content(
                model=resolved_model_name,
                contents=contents,
                config=generation_config,
            )

            usage = self._extract_usage(response)

            finish_reason_str = "UNKNOWN"
            is_blocked_by_safety = False
            safety_feedback_details = None

            if response.candidates:
                candidate = response.candidates[0]

                try:
                    finish_reason_enum = candidate.finish_reason
                    if finish_reason_enum:
                        try:
                            finish_reason_str = finish_reason_enum.name
                        except AttributeError:
                            finish_reason_str = str(finish_reason_enum)
                    else:
                        finish_reason_str = "STOP"
                except AttributeError:
                    finish_reason_str = "STOP"

                if not response.text:
                    try:
                        safety_ratings = candidate.safety_ratings
                        if safety_ratings:
                            for rating in safety_ratings:
                                try:
                                    if rating.blocked:
                                        is_blocked_by_safety = True
                                        category_name = "UNKNOWN"
                                        probability_name = "UNKNOWN"

                                        try:
                                            category_name = rating.category.name
                                        except (AttributeError, TypeError):
                                            pass

                                        try:
                                            probability_name = rating.probability.name
                                        except (AttributeError, TypeError):
                                            pass

                                        safety_feedback_details = (
                                            f"Category: {category_name}, Probability: {probability_name}"
                                        )
                                        break
                                except (AttributeError, TypeError):
                                    continue
                    except (AttributeError, TypeError):
                        pass

            elif response.candidates is not None and len(response.candidates) == 0:
                is_blocked_by_safety = True
                finish_reason_str = "SAFETY"
                safety_feedback_details = "Prompt blocked, reason unavailable"

                try:
                    prompt_feedback = response.prompt_feedback
                    if prompt_feedback and prompt_feedback.block_reason:
                        try:
                            block_reason_name = prompt_feedback.block_reason.name
                        except AttributeError:
                            block_reason_name = str(prompt_feedback.block_reason)
                        safety_feedback_details = f"Prompt blocked, reason: {block_reason_name}"
                except (AttributeError, TypeError):
                    pass

            return ModelResponse(
                content=response.text,
                usage=usage,
                model_name=resolved_model_name,
                friendly_name="Gemini",
                provider=ProviderType.GOOGLE,
                metadata={
                    "thinking_mode": effective_thinking_mode if capabilities.supports_extended_thinking else None,
                    "finish_reason": finish_reason_str,
                    "is_blocked_by_safety": is_blocked_by_safety,
                    "safety_feedback": safety_feedback_details,
                },
            )

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix=f"Gemini API ({resolved_model_name})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            error_msg = (
                f"Gemini API error for model {resolved_model_name} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            )
            raise RuntimeError(error_msg) from exc

    # ------------------------------------------------------------------
    # Interactions API (stateless) path
    # ------------------------------------------------------------------

    @staticmethod
    def _use_interactions_api() -> bool:
        """Whether to route Gemini through the Interactions API (default on).

        Set ``VOX_GEMINI_USE_INTERACTIONS=false`` to use the classic
        generateContent endpoint directly.
        """
        return (get_env("VOX_GEMINI_USE_INTERACTIONS", "true") or "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _resolve_thinking_level(resolved_model_name: str, thinking_mode: str | None, capabilities) -> str | None:
        """Map a thinking_mode to an Interactions ``thinking_level`` string, or None.

        The Interactions API expresses thinking via an enum ``thinking_level`` for
        ALL thinking-capable models (there is no token ``thinking_budget`` on this
        surface), so this must run for Gemini 2.x as well as Gemini 3 — otherwise
        2.x thinking is silently dropped. Allowed levels differ by tier (verified
        against the live API): Gemini 2.x accepts only low/high; Gemini 3 accepts
        low/medium/high (``minimal`` is Flash-only, so it is mapped to ``low`` to
        stay valid across pro+flash). ``max`` collapses to ``high``. Returns None
        for models without extended thinking (omit the field).
        """
        if not getattr(capabilities, "supports_extended_thinking", False):
            return None
        mode = (thinking_mode or "medium").lower()
        if resolved_model_name.startswith("gemini-3"):
            return {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "high"}.get(mode, "high")
        # Gemini 2.x: only low/high are accepted on the Interactions API.
        return "low" if mode in ("minimal", "low") else "high"

    def _generate_via_interactions(
        self,
        *,
        prompt: str,
        resolved_model_name: str,
        system_prompt: str | None,
        effective_temperature: float | None,
        max_output_tokens: int | None,
        thinking_mode: str | None,
        capabilities,
    ) -> ModelResponse:
        """Generate via the stateless (store=False) Interactions API.

        vox keeps owning conversation memory, so we pass the full prompt as the
        interaction ``input`` and never use ``previous_interaction_id``.
        """
        generation_config: dict[str, object] = {}
        if effective_temperature is not None:
            generation_config["temperature"] = effective_temperature
        if max_output_tokens:
            generation_config["max_output_tokens"] = max_output_tokens
        thinking_level = self._resolve_thinking_level(resolved_model_name, thinking_mode, capabilities)
        if thinking_level is not None:
            generation_config["thinking_level"] = thinking_level

        create_kwargs: dict[str, object] = {
            "model": resolved_model_name,
            "input": prompt,
            "store": False,
        }
        if system_prompt:
            create_kwargs["system_instruction"] = system_prompt
        if generation_config:
            create_kwargs["generation_config"] = generation_config

        # Single attempt: the generateContent fallback in generate_content is the
        # safety net (and has its own retry loop), so we don't retry here — a
        # failure should fall back fast rather than sleep through retries.
        interaction = self.client.interactions.create(**create_kwargs)
        text = getattr(interaction, "output_text", None) or ""
        usage = self._extract_interaction_usage(interaction)
        status = getattr(interaction, "status", None)
        try:
            finish_reason_str = (
                status.name if (status is not None and hasattr(status, "name")) else (str(status) if status else "STOP")
            )
        except Exception:
            finish_reason_str = "STOP"
        return ModelResponse(
            content=text,
            usage=usage,
            model_name=resolved_model_name,
            friendly_name="Gemini",
            provider=ProviderType.GOOGLE,
            metadata={
                "thinking_mode": thinking_mode if capabilities.supports_extended_thinking else None,
                "finish_reason": finish_reason_str,
                "interaction_id": getattr(interaction, "id", None),
                "api": "interactions",
            },
        )

    @staticmethod
    def _extract_interaction_usage(interaction) -> dict[str, int]:
        """Extract token usage from an Interaction (total_input/output_tokens)."""
        usage: dict[str, int] = {}
        meta = getattr(interaction, "usage", None)
        if not meta:
            return usage
        input_tokens = getattr(meta, "total_input_tokens", None)
        output_tokens = getattr(meta, "total_output_tokens", None)
        if input_tokens is not None:
            usage["input_tokens"] = input_tokens
        if output_tokens is not None:
            usage["output_tokens"] = output_tokens
        if input_tokens is not None and output_tokens is not None:
            usage["total_tokens"] = input_tokens + output_tokens
        return usage

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.GOOGLE

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        usage = {}

        # Try to extract usage metadata from response
        # Note: The actual structure depends on the SDK version and response format
        try:
            metadata = response.usage_metadata
            if metadata:
                # Extract token counts with explicit None checks
                input_tokens = None
                output_tokens = None

                try:
                    value = metadata.prompt_token_count
                    if value is not None:
                        input_tokens = value
                        usage["input_tokens"] = value
                except (AttributeError, TypeError):
                    pass

                try:
                    value = metadata.candidates_token_count
                    if value is not None:
                        output_tokens = value
                        usage["output_tokens"] = value
                except (AttributeError, TypeError):
                    pass

                # Calculate total only if both values are available and valid
                if input_tokens is not None and output_tokens is not None:
                    usage["total_tokens"] = input_tokens + output_tokens
        except (AttributeError, TypeError):
            # response doesn't have usage_metadata
            pass

        return usage

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes.

        Uses Gemini API error structure instead of text pattern matching for reliability.

        Args:
            error: Exception from Gemini API call

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Check for 429 errors first - these need special handling
        if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            # For Gemini, check for specific non-retryable error indicators
            # These typically indicate permanent failures or quota/size limits
            non_retryable_indicators = [
                "quota exceeded",
                "resource exhausted",
                "context length",
                "token limit",
                "request too large",
                "invalid request",
                "quota_exceeded",
                "resource_exhausted",
            ]

            # Also check if this is a structured error from Gemini SDK
            try:
                # Try to access error details if available
                error_details = None
                try:
                    error_details = error.details
                except AttributeError:
                    try:
                        error_details = error.reason
                    except AttributeError:
                        pass

                if error_details:
                    error_details_str = str(error_details).lower()
                    # Check for non-retryable error codes/reasons
                    if any(indicator in error_details_str for indicator in non_retryable_indicators):
                        logger.debug(f"Non-retryable Gemini error: {error_details}")
                        return False
            except Exception:
                pass

            # Check main error string for non-retryable patterns
            if any(indicator in error_str for indicator in non_retryable_indicators):
                logger.debug(f"Non-retryable Gemini error based on message: {error_str[:200]}...")
                return False

            # If it's a 429/quota error but doesn't match non-retryable patterns, it might be retryable rate limiting
            logger.debug(f"Retryable Gemini rate limiting error: {error_str[:100]}...")
            return True

        # For non-429 errors, check if they're retryable
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "retry",
            "internal error",
            "408",  # Request timeout
            "500",  # Internal server error
            "502",  # Bad gateway
            "503",  # Service unavailable
            "504",  # Gateway timeout
            "ssl",  # SSL errors
            "handshake",  # Handshake failures
        ]

        return any(indicator in error_str for indicator in retryable_indicators)

    def _process_image(self, image_path: str) -> dict | None:
        """Process an image for Gemini API."""
        try:
            # Use base class validation
            image_bytes, mime_type = validate_image(image_path)

            # For data URLs, extract the base64 data directly
            if image_path.startswith("data:"):
                # Extract base64 data from data URL
                _, data = image_path.split(",", 1)
                return {"inline_data": {"mime_type": mime_type, "data": data}}
            else:
                # For file paths, encode the bytes
                image_data = base64.b64encode(image_bytes).decode()
                return {"inline_data": {"mime_type": mime_type, "data": image_data}}

        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get Gemini's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        capability_map = self.get_all_model_capabilities()

        # Helper to find best model from candidates
        def find_best(candidates: list[str]) -> str | None:
            """Return best model from candidates (sorted for consistency)."""
            return sorted(candidates, reverse=True)[0] if candidates else None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # For extended reasoning, prefer models with thinking support
            # First try Pro models that support thinking
            pro_thinking = [
                m
                for m in allowed_models
                if "pro" in m and m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if pro_thinking:
                return find_best(pro_thinking)

            # Then any model that supports thinking
            any_thinking = [
                m for m in allowed_models if m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if any_thinking:
                return find_best(any_thinking)

            # Finally, just prefer Pro models even without thinking
            pro_models = [m for m in allowed_models if "pro" in m]
            if pro_models:
                return find_best(pro_models)

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Flash models for speed
            flash_models = [m for m in allowed_models if "flash" in m]
            if flash_models:
                return find_best(flash_models)

        # Default for BALANCED or as fallback
        # Prefer Flash for balanced use, then Pro, then anything
        flash_models = [m for m in allowed_models if "flash" in m]
        if flash_models:
            return find_best(flash_models)

        pro_models = [m for m in allowed_models if "pro" in m]
        if pro_models:
            return find_best(pro_models)

        # Ultimate fallback to best available model
        return find_best(allowed_models)


# Load registry data at import time for registry consumers
GeminiModelProvider._ensure_registry()
