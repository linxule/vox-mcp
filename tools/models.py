"""Data models for tool responses and model-selection categories."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolModelCategory(Enum):
    """Categories for tool model selection based on requirements."""

    EXTENDED_REASONING = "extended_reasoning"  # Requires deep thinking capabilities
    FAST_RESPONSE = "fast_response"  # Speed and cost efficiency preferred
    BALANCED = "balanced"  # Balance of capability and performance


class ContinuationOffer(BaseModel):
    """Offer for CLI agent to continue conversation when model did not ask a follow-up."""

    continuation_id: str = Field(
        ..., description="Thread continuation ID for multi-turn conversations across different tools"
    )
    note: str = Field(..., description="Message explaining continuation opportunity to CLI agent")
    remaining_turns: int = Field(..., description="Number of conversation turns remaining")


class ToolOutput(BaseModel):
    """Standardized output format for all tools."""

    status: Literal[
        "success",
        "error",
        "files_required_to_continue",
        "resend_prompt",
        "code_too_large",
        "continuation_available",
    ] = "success"
    content: str | None = Field(None, description="The main content/response from the tool")
    content_type: Literal["text", "markdown", "json"] = "text"
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    continuation_offer: ContinuationOffer | None = Field(
        None, description="Optional offer for agent to continue conversation"
    )
