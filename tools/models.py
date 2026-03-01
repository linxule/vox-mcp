"""Data models for tool responses and model-selection categories."""

from enum import Enum
from typing import Any, Literal, Optional

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
    content: Optional[str] = Field(None, description="The main content/response from the tool")
    content_type: Literal["text", "markdown", "json"] = "text"
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    continuation_offer: Optional[ContinuationOffer] = Field(
        None, description="Optional offer for agent to continue conversation"
    )


class FilesNeededRequest(BaseModel):
    """Request for missing files or code needed to continue."""

    status: Literal["files_required_to_continue"] = "files_required_to_continue"
    mandatory_instructions: str = Field(..., description="Critical instructions for agent regarding required context")
    files_needed: Optional[list[str]] = Field(
        default_factory=list, description="Specific files that are needed for analysis"
    )
    suggested_next_action: Optional[dict[str, Any]] = Field(
        None,
        description="Suggested tool call with parameters after getting clarification",
    )


class CodeTooLargeRequest(BaseModel):
    """Request to reduce file selection due to size constraints."""

    status: Literal["code_too_large"] = "code_too_large"
    content: str = Field(..., description="Message explaining the size constraint")
    content_type: Literal["text"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResendPromptRequest(BaseModel):
    """Request to resend prompt via file due to size limits."""

    status: Literal["resend_prompt"] = "resend_prompt"
    content: str = Field(..., description="Instructions for handling large prompt")
    content_type: Literal["text"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


# Registry mapping status strings to their corresponding Pydantic models
SPECIAL_STATUS_MODELS = {
    "files_required_to_continue": FilesNeededRequest,
    "resend_prompt": ResendPromptRequest,
    "code_too_large": CodeTooLargeRequest,
}
