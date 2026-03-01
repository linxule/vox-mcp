"""
Base models for Vox MCP tools.

This module contains shared Pydantic models used across all tools,
extracted to avoid circular imports and promote code reuse.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Shared field descriptions to avoid duplication
COMMON_FIELD_DESCRIPTIONS = {
    "model": "Model name to use. Use `listmodels` for available options. The server validates model availability and returns errors for unknown models.",
    "temperature": "0 = deterministic · 1 = creative.",
    "thinking_mode": "Reasoning depth: minimal, low, medium, high, or max.",
    "continuation_id": (
        "Unique thread continuation ID for multi-turn conversations. Works across different tools. "
        "Reuse the last continuation_id you were given to preserve full conversation context, "
        "files, and history across turns. Threads are held in memory and expire after inactivity."
    ),
    "images": "Optional absolute image paths or base64 blobs for visual context.",
    "absolute_file_paths": "Full paths to relevant code",
}


class ToolRequest(BaseModel):
    """
    Base request model for all Vox MCP tools.

    This model defines common fields that all tools accept, including
    model selection, temperature control, and conversation threading.
    Tool-specific request models should inherit from this class.
    """

    # Model configuration
    model: str | None = Field(None, description=COMMON_FIELD_DESCRIPTIONS["model"])
    temperature: float | None = Field(None, ge=0.0, le=1.0, description=COMMON_FIELD_DESCRIPTIONS["temperature"])
    thinking_mode: Literal["minimal", "low", "medium", "high", "max"] | None = Field(
        None, description=COMMON_FIELD_DESCRIPTIONS["thinking_mode"]
    )

    # Conversation support
    continuation_id: str | None = Field(None, description=COMMON_FIELD_DESCRIPTIONS["continuation_id"])

    # Visual context
    images: list[str] | None = Field(None, description=COMMON_FIELD_DESCRIPTIONS["images"])


# Tool-specific field descriptions are declared in each tool file
# This keeps concerns separated and makes each tool self-contained
