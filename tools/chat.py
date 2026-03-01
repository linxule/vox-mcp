"""
Chat tool - Multi-model AI gateway with conversation memory

This tool routes prompts to external AI models via configured providers
(Gemini, OpenAI, Anthropic, DeepSeek, Moonshot, xAI, OpenRouter, custom endpoints).
It supports file context embedding, images, and multi-turn conversation threads.
"""

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_BALANCED
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS, ToolRequest

from .simple.base import SimpleTool

CHAT_FIELD_DESCRIPTIONS = {
    "prompt": (
        "Your question or task for the external model. "
        "Prefer passing code and large content via absolute_file_paths rather than inlining it here."
    ),
    "absolute_file_paths": (
        "Full, absolute file paths to relevant code in order to share with the external model. "
        "Accepts both files and directories (directories are expanded recursively). "
        "Content is read and embedded into the prompt context."
    ),
    "images": "Image paths (absolute) or base64 strings for optional visual context.",
}


class ChatRequest(ToolRequest):
    """Request model for Chat tool"""

    prompt: str = Field(..., description=CHAT_FIELD_DESCRIPTIONS["prompt"])
    absolute_file_paths: list[str] | None = Field(
        default_factory=list,
        description=CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
    )
    images: list[str] | None = Field(default_factory=list, description=CHAT_FIELD_DESCRIPTIONS["images"])


class ChatTool(SimpleTool):
    """
    Multi-model AI gateway with conversation memory.

    Routes prompts to external AI models with optional file context, image support,
    and multi-turn conversation threads via continuation_id.
    """

    def get_name(self) -> str:
        return "chat"

    def get_description(self) -> str:
        return (
            "Multi-model AI gateway. Routes prompts to external AI models "
            "(Gemini, OpenAI, Anthropic, DeepSeek, Moonshot, xAI, OpenRouter, custom endpoints) "
            "with conversation memory. Supports file context embedding, images, and multi-turn threads via continuation_id."
        )

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_model_category(self) -> "ToolModelCategory":
        """Chat prioritizes fast responses and cost efficiency"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.FAST_RESPONSE

    def get_request_model(self):
        """Return the Chat-specific request model"""
        return ChatRequest

    # === Schema Generation Utilities ===

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema matching the original Chat tool expectations."""

        required_fields = ["prompt"]
        if self.is_effective_auto_mode():
            required_fields.append("model")

        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
                },
                "absolute_file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["images"],
                },
                "model": self.get_model_field_schema(),
                "temperature": {
                    "type": "number",
                    "description": COMMON_FIELD_DESCRIPTIONS["temperature"],
                    "minimum": 0,
                    "maximum": 1,
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": COMMON_FIELD_DESCRIPTIONS["thinking_mode"],
                },
                "continuation_id": {
                    "type": "string",
                    "description": COMMON_FIELD_DESCRIPTIONS["continuation_id"],
                },
            },
            "required": required_fields,
            "additionalProperties": False,
        }

        return schema

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Tool-specific field definitions used by SimpleTool scaffolding."""

        return {
            "prompt": {
                "type": "string",
                "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
            },
            "absolute_file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["images"],
            },
        }

    def get_required_fields(self) -> list[str]:
        """Required fields for ChatSimple tool"""
        return ["prompt"]

    # === Hook Method Implementations ===

    async def prepare_prompt(self, request: ChatRequest) -> str:
        """Prepare the prompt with optional file context embedding."""
        # Use SimpleTool's Chat-style prompt preparation
        return self.prepare_chat_style_prompt(request)

    def _validate_file_paths(self, request) -> str | None:
        """Expand ~ in file paths before validation."""

        files = self.get_request_files(request)
        if files:
            expanded_files: list[str] = []
            for file_path in files:
                expanded = os.path.expanduser(file_path)
                if not os.path.isabs(expanded):
                    return (
                        "Error: All file paths must be full absolute paths to real files or folders. "
                        f"Received: {file_path}"
                    )
                expanded_files.append(expanded)
            self.set_request_files(request, expanded_files)

        return super()._validate_file_paths(request)
