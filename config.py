"""
Configuration and constants for Vox MCP Server

This module centralizes all configuration settings for the Vox MCP Server.
It defines model configurations, token limits, temperature defaults, and other
constants used throughout the application.

Configuration values can be overridden by environment variables where appropriate.
"""

import importlib.metadata
from pathlib import Path

from utils.env import get_env

# Version and metadata
# Version is derived from pyproject.toml via importlib.metadata (single source of truth)
try:
    __version__ = importlib.metadata.version("vox-mcp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Model configuration
# DEFAULT_MODEL: The default model used for all AI operations
# Can be overridden by setting DEFAULT_MODEL environment variable
# Special value "auto" means Claude should pick the best model for each task
DEFAULT_MODEL = get_env("DEFAULT_MODEL", "auto") or "auto"

# Auto mode detection - when DEFAULT_MODEL is "auto", Claude picks the model
IS_AUTO_MODE = DEFAULT_MODEL.lower() == "auto"

# Each provider (gemini.py, openai.py, xai.py, openrouter.py, custom.py)
# defines its own MODEL_CAPABILITIES
# with detailed descriptions. Tools use ModelProviderRegistry.get_available_model_names()
# to get models only from enabled providers (those with valid API keys).
#
# This architecture ensures:
# - No namespace collisions (models only appear when their provider is enabled)
# - API key-based filtering (prevents wrong models from being shown to Claude)
# - Proper provider routing (models route to the correct API endpoint)
# - Clean separation of concerns (providers own their model definitions)


# Temperature defaults
# NOTE: Gemini 3.0 Pro notes suggest temperature should be set at 1.0
# in most cases. Lowering it can affect the models 'reasoning' abilities.
# Newer models / inference stacks are able to handle their randomness better.

# Temperature controls the randomness/creativity of model responses
# Lower values (0.0-0.3) produce more deterministic, focused responses
# Higher values (0.7-1.0) produce more creative, varied responses

# TEMPERATURE_BALANCED: Middle ground for general conversations
# Provides a good balance between consistency and helpful variety
TEMPERATURE_BALANCED = 1.0  # For general chat

# MCP Protocol Transport Limits
#
# This limit applies only to the Claude CLI ↔ MCP Server transport boundary.
# It does NOT limit internal MCP Server operations like system prompts, file embeddings,
# conversation history, or content sent to external models (Gemini/OpenAI/OpenRouter).
#
# MCP Protocol Architecture:
# Claude CLI ←→ MCP Server ←→ External Model (Gemini/OpenAI/etc.)
#     ↑                              ↑
#     │                              │
# MCP transport                Internal processing
# (token limit from MAX_MCP_OUTPUT_TOKENS)    (No MCP limit - can be 1M+ tokens)
#
# MCP_PROMPT_SIZE_LIMIT: Maximum character size for USER INPUT crossing MCP transport
# The MCP protocol has a combined request+response limit controlled by MAX_MCP_OUTPUT_TOKENS.
# To ensure adequate space for MCP Server → Claude CLI responses, we limit user input
# to roughly 60% of the total token budget converted to characters. Larger user prompts
# must be sent as prompt.txt files to bypass MCP's transport constraints.
#
# Token to character conversion ratio: ~4 characters per token (average for code/text)
# Default allocation: 60% of tokens for input, 40% for response
#
# What IS limited by this constant:
# - request.prompt field content (user input from Claude CLI)
# - prompt.txt file content (alternative user input method)
# - Any other direct user input fields
#
# What is NOT limited by this constant:
# - System prompts added internally by tools
# - File content embedded by tools
# - Conversation history loaded from storage
# - Web search instructions or other internal additions
# - Complete prompts sent to external models (managed by model-specific token limits)
#
# This ensures MCP transport stays within protocol limits while allowing internal
# processing to use full model context windows (200K-1M+ tokens).


def _calculate_mcp_prompt_limit() -> int:
    """
    Calculate MCP prompt size limit based on MAX_MCP_OUTPUT_TOKENS environment variable.

    Returns:
        Maximum character count for user input prompts
    """
    # Check for Claude's MAX_MCP_OUTPUT_TOKENS environment variable
    max_tokens_str = get_env("MAX_MCP_OUTPUT_TOKENS")

    if max_tokens_str:
        try:
            max_tokens = int(max_tokens_str)
            # Allocate 60% of tokens for input, convert to characters (~4 chars per token)
            input_token_budget = int(max_tokens * 0.6)
            character_limit = input_token_budget * 4
            return character_limit
        except (ValueError, TypeError):
            # Fall back to default if MAX_MCP_OUTPUT_TOKENS is not a valid integer
            pass

    # Default fallback: 60,000 characters (equivalent to ~15k tokens input of 25k total)
    return 60_000


MCP_PROMPT_SIZE_LIMIT = _calculate_mcp_prompt_limit()

# Thread persistence
# Shadow persistence layer that writes every thread to disk as append-only JSONL.
# Threads are still served from memory for performance; disk is for durability and cold reload.
VOX_THREADS_DIR = Path(get_env("VOX_THREADS_DIR", "~/.vox/threads/") or "~/.vox/threads/").expanduser().resolve()

# Threading configuration
# In-memory conversation threading with shadow persistence to disk (JSONL)
# Memory serves as primary store; disk provides durability and cold-reload
