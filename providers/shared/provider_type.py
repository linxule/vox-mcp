"""Enumeration describing which backend owns a given model."""

from enum import Enum

__all__ = ["ProviderType"]


class ProviderType(Enum):
    """Canonical identifiers for every supported provider backend."""

    GOOGLE = "google"
    OPENAI = "openai"
    XAI = "xai"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"
    ANTHROPIC = "anthropic"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
