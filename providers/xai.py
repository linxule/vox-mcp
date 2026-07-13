"""X.AI (GROK) model provider implementation."""

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.xai import XAIModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class XAIModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for X.AI's GROK models exposed over an OpenAI-style API.

    Publishes capability metadata for the officially supported deployments and
    maps tool-category preferences to the appropriate GROK model.
    """

    FRIENDLY_NAME = "X.AI"

    REGISTRY_CLASS = XAIModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    def __init__(self, api_key: str, **kwargs):
        """Initialize X.AI provider with API key."""
        # Set X.AI base URL
        kwargs.setdefault("base_url", "https://api.x.ai/v1")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.XAI

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get XAI's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Grok 4.5 has configurable reasoning; 4.3 is the 1M-context fallback
            preferences = ["grok-4.5", "grok-4.3"]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # No non-reasoning Grok remains in the catalogue: the only one was
            # grok-4.20-0309-non-reasoning, dropped with the rest of the 4.20 line.
            # xAI calls 4.5 "the most intelligent and fastest model we've built",
            # so it carries this category too rather than leaving it unserved.
            preferences = ["grok-4.5"]

        else:  # BALANCED or default
            preferences = ["grok-4.5", "grok-4.3"]

        for model in preferences:
            if model in allowed_models:
                return model

        # Fall back to any available model
        return allowed_models[0]


# Load registry data at import time
XAIModelProvider._ensure_registry()
