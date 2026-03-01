"""
Vox MCP Server - Main server implementation

This module implements the core MCP (Model Context Protocol) server that provides
AI-powered tools for chat, model listing, and thread export.

The server follows the MCP specification to expose callable tools for MCP clients
(like Claude). Each tool provides specialized functionality for interactive
engineering support and model/provider visibility.

Key Components:
- MCP Server: Handles protocol communication and tool discovery
- Tool Registry: Maps tool names to their implementations
- Request Handler: Processes incoming tool calls and returns formatted responses
- Configuration: Manages API keys and model settings

The server runs on stdio (standard input/output) and communicates using JSON-RPC messages
as defined by the MCP protocol.
"""

import asyncio
import atexit
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server  # noqa: E402
from mcp.server.models import InitializationOptions  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import (  # noqa: E402
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptsCapability,
    ServerCapabilities,
    TextContent,
    Tool,
    ToolAnnotations,
    ToolsCapability,
)

from config import (  # noqa: E402
    DEFAULT_MODEL,
    __version__,
)
from tools import (  # noqa: E402
    ChatTool,
    ListModelsTool,
)
from tools.dump_threads import DumpThreadsTool  # noqa: E402
from tools.models import ToolOutput  # noqa: E402
from tools.shared.exceptions import ToolExecutionError  # noqa: E402
from utils.env import env_override_enabled, get_env  # noqa: E402

# Configure logging for server operations
# Can be controlled via LOG_LEVEL environment variable (DEBUG, INFO, WARNING, ERROR)
log_level = (get_env("LOG_LEVEL", "DEBUG") or "DEBUG").upper()

# Create timezone-aware formatter


class LocalTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        """Override to use local timezone instead of UTC"""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            s = f"{t},{record.msecs:03.0f}"
        return s


# Configure both console and file logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Clear any existing handlers first
root_logger = logging.getLogger()
root_logger.handlers.clear()

# Create and configure stderr handler explicitly
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(getattr(logging, log_level, logging.INFO))
stderr_handler.setFormatter(LocalTimeFormatter(log_format))
root_logger.addHandler(stderr_handler)

# Note: MCP stdio_server interferes with stderr during tool execution
# All logs are properly written to logs/mcp_server.log for monitoring

# Set root logger level
root_logger.setLevel(getattr(logging, log_level, logging.INFO))

# Add rotating file handler for local log monitoring

try:
    # Create logs directory in project root
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Main server log with size-based rotation (20MB max per file)
    # This ensures logs don't grow indefinitely and are properly managed
    file_handler = RotatingFileHandler(
        log_dir / "mcp_server.log",
        maxBytes=20 * 1024 * 1024,  # 20MB max file size
        backupCount=5,  # Keep 5 rotated files (120MB total with active file)
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(LocalTimeFormatter(log_format))
    logging.getLogger().addHandler(file_handler)

    # Create a special logger for MCP activity tracking with size-based rotation
    mcp_logger = logging.getLogger("mcp_activity")
    mcp_file_handler = RotatingFileHandler(
        log_dir / "mcp_activity.log",
        maxBytes=10 * 1024 * 1024,  # 10MB max file size
        backupCount=2,  # Keep 2 rotated files (30MB total with active file)
        encoding="utf-8",
    )
    mcp_file_handler.setLevel(logging.INFO)
    mcp_file_handler.setFormatter(LocalTimeFormatter("%(asctime)s - %(message)s"))
    mcp_logger.addHandler(mcp_file_handler)
    mcp_logger.setLevel(logging.INFO)
    # Ensure MCP activity also goes to stderr
    mcp_logger.propagate = True

    # Log setup info directly to root logger since logger isn't defined yet
    logging.info(f"Logging to: {log_dir / 'mcp_server.log'}")
    logging.info(f"Process PID: {os.getpid()}")

except Exception as e:
    print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)

logger = logging.getLogger(__name__)

# Log VOX_FORCE_ENV_OVERRIDE configuration for transparency
if env_override_enabled():
    logger.info("VOX_FORCE_ENV_OVERRIDE enabled - .env file values will override system environment variables")
    logger.debug("Environment override prevents conflicts between different AI tools passing cached API keys")
else:
    logger.debug("VOX_FORCE_ENV_OVERRIDE disabled - system environment variables take precedence")


# Create the MCP server instance with a unique name identifier
# This name is used by MCP clients to identify and connect to this specific server
server: Server = Server("vox-server")


# Constants for tool filtering
ESSENTIAL_TOOLS = {"listmodels"}


def parse_disabled_tools_env() -> set[str]:
    """
    Parse the DISABLED_TOOLS environment variable into a set of tool names.

    Returns:
        Set of lowercase tool names to disable, empty set if none specified
    """
    disabled_tools_env = (get_env("DISABLED_TOOLS", "") or "").strip()
    if not disabled_tools_env:
        return set()
    return {t.strip().lower() for t in disabled_tools_env.split(",") if t.strip()}


def validate_disabled_tools(disabled_tools: set[str], all_tools: dict[str, Any]) -> None:
    """
    Validate the disabled tools list and log appropriate warnings.

    Args:
        disabled_tools: Set of tool names requested to be disabled
        all_tools: Dictionary of all available tool instances
    """
    essential_disabled = disabled_tools & ESSENTIAL_TOOLS
    if essential_disabled:
        logger.warning(f"Cannot disable essential tools: {sorted(essential_disabled)}")
    unknown_tools = disabled_tools - set(all_tools.keys())
    if unknown_tools:
        logger.warning(f"Unknown tools in DISABLED_TOOLS: {sorted(unknown_tools)}")


def apply_tool_filter(all_tools: dict[str, Any], disabled_tools: set[str]) -> dict[str, Any]:
    """
    Apply the disabled tools filter to create the final tools dictionary.

    Args:
        all_tools: Dictionary of all available tool instances
        disabled_tools: Set of tool names to disable

    Returns:
        Dictionary containing only enabled tools
    """
    enabled_tools = {}
    for tool_name, tool_instance in all_tools.items():
        if tool_name in ESSENTIAL_TOOLS or tool_name not in disabled_tools:
            enabled_tools[tool_name] = tool_instance
        else:
            logger.debug(f"Tool '{tool_name}' disabled via DISABLED_TOOLS")
    return enabled_tools


def log_tool_configuration(disabled_tools: set[str], enabled_tools: dict[str, Any]) -> None:
    """
    Log the final tool configuration for visibility.

    Args:
        disabled_tools: Set of tool names that were requested to be disabled
        enabled_tools: Dictionary of tools that remain enabled
    """
    if not disabled_tools:
        logger.info("All tools enabled (DISABLED_TOOLS not set)")
        return
    actual_disabled = disabled_tools - ESSENTIAL_TOOLS
    if actual_disabled:
        logger.debug(f"Disabled tools: {sorted(actual_disabled)}")
        logger.info(f"Active tools: {sorted(enabled_tools.keys())}")


def filter_disabled_tools(all_tools: dict[str, Any]) -> dict[str, Any]:
    """
    Filter tools based on DISABLED_TOOLS environment variable.

    Args:
        all_tools: Dictionary of all available tool instances

    Returns:
        dict: Filtered dictionary containing only enabled tools
    """
    disabled_tools = parse_disabled_tools_env()
    if not disabled_tools:
        log_tool_configuration(disabled_tools, all_tools)
        return all_tools
    validate_disabled_tools(disabled_tools, all_tools)
    enabled_tools = apply_tool_filter(all_tools, disabled_tools)
    log_tool_configuration(disabled_tools, enabled_tools)
    return enabled_tools


# Initialize the tool registry with all available AI-powered tools
# Each tool provides specialized functionality for different development tasks
# Tools are instantiated once and reused across requests (stateless design)
TOOLS = {
    "chat": ChatTool(),  # Multi-model AI gateway with conversation memory
    "listmodels": ListModelsTool(),  # List all available AI models by provider
    "dump_threads": DumpThreadsTool(),  # Export active conversation threads as JSON
}
TOOLS = filter_disabled_tools(TOOLS)

# Rich prompt templates for all tools
PROMPT_TEMPLATES = {
    "chat": {
        "name": "chat",
        "description": "Open the chat tool",
        "template": "chat",
    },
    "listmodels": {
        "name": "listmodels",
        "description": "List available AI models",
        "template": "listmodels",
    },
}


def configure_providers():
    """
    Configure and validate AI providers based on available API keys.

    This function checks for API keys and registers the appropriate providers.
    At least one valid API key (Gemini or OpenAI) is required.

    Raises:
        ValueError: If no valid API keys are found or conflicting configurations detected
    """
    # Log environment variable status for debugging
    logger.debug("Checking environment variables for API keys...")
    api_keys_to_check = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "CUSTOM_API_URL"]
    for key in api_keys_to_check:
        value = get_env(key)
        logger.debug(f"  {key}: {'[PRESENT]' if value else '[MISSING]'}")
    from providers import ModelProviderRegistry
    from providers.anthropic import AnthropicModelProvider
    from providers.custom import CustomProvider
    from providers.deepseek import DeepSeekProvider
    from providers.gemini import GeminiModelProvider
    from providers.moonshot import MoonshotProvider
    from providers.openai import OpenAIModelProvider
    from providers.openrouter import OpenRouterProvider
    from providers.shared import ProviderType
    from providers.xai import XAIModelProvider
    from utils.model_restrictions import get_restriction_service

    valid_providers = []
    has_native_apis = False
    has_openrouter = False
    has_custom = False

    # Check for Gemini API key
    gemini_key = get_env("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        valid_providers.append("Gemini")
        has_native_apis = True
        logger.info("Gemini API key found - Gemini models available")

    # Check for OpenAI API key
    openai_key = get_env("OPENAI_API_KEY")
    logger.debug(f"OpenAI key check: key={'[PRESENT]' if openai_key else '[MISSING]'}")
    if openai_key and openai_key != "your_openai_api_key_here":
        valid_providers.append("OpenAI")
        has_native_apis = True
        logger.info("OpenAI API key found")
    else:
        if not openai_key:
            logger.debug("OpenAI API key not found in environment")
        else:
            logger.debug("OpenAI API key is placeholder value")

    # Check for X.AI API key
    xai_key = get_env("XAI_API_KEY")
    if xai_key and xai_key != "your_xai_api_key_here":
        valid_providers.append("X.AI (GROK)")
        has_native_apis = True
        logger.info("X.AI API key found - GROK models available")

    # Check for OpenRouter API key
    openrouter_key = get_env("OPENROUTER_API_KEY")
    logger.debug(f"OpenRouter key check: key={'[PRESENT]' if openrouter_key else '[MISSING]'}")
    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        valid_providers.append("OpenRouter")
        has_openrouter = True
        logger.info("OpenRouter API key found - Multiple models available via OpenRouter")
    else:
        if not openrouter_key:
            logger.debug("OpenRouter API key not found in environment")
        else:
            logger.debug("OpenRouter API key is placeholder value")

    # Check for custom API endpoint (Ollama, vLLM, etc.)
    custom_url = get_env("CUSTOM_API_URL")
    if custom_url:
        # Always read CUSTOM_API_KEY even if empty
        # - Some providers (vLLM, LM Studio, enterprise APIs) require authentication
        # - Others (Ollama) work without authentication (empty key)
        # - This variable is required by the provider factory function
        custom_key = get_env("CUSTOM_API_KEY", "") or ""  # Default to empty (Ollama doesn't need auth)
        custom_model = get_env("CUSTOM_MODEL_NAME", "llama3.2") or "llama3.2"
        valid_providers.append(f"Custom API ({custom_url})")
        has_custom = True
        logger.info(f"Custom API endpoint found: {custom_url} with model {custom_model}")
        if custom_key:
            logger.debug("Custom API key provided for authentication")
        else:
            logger.debug("No custom API key provided (using unauthenticated access)")

    # Register providers in priority order:
    # 1. Native APIs first (most direct and efficient)
    registered_providers = []

    if has_native_apis:
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
            registered_providers.append(ProviderType.GOOGLE.value)
            logger.debug(f"Registered provider: {ProviderType.GOOGLE.value}")
        if openai_key and openai_key != "your_openai_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
            registered_providers.append(ProviderType.OPENAI.value)
            logger.debug(f"Registered provider: {ProviderType.OPENAI.value}")
        if xai_key and xai_key != "your_xai_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)
            registered_providers.append(ProviderType.XAI.value)
            logger.debug(f"Registered provider: {ProviderType.XAI.value}")

        # Custom providers (Anthropic, Moonshot, DeepSeek)
        anthropic_key = get_env("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.ANTHROPIC, AnthropicModelProvider)
            registered_providers.append(ProviderType.ANTHROPIC.value)
            logger.debug(f"Registered provider: {ProviderType.ANTHROPIC.value}")

        moonshot_key = get_env("MOONSHOT_API_KEY")
        if moonshot_key and moonshot_key != "your_moonshot_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.MOONSHOT, MoonshotProvider)
            registered_providers.append(ProviderType.MOONSHOT.value)
            logger.debug(f"Registered provider: {ProviderType.MOONSHOT.value}")

        deepseek_key = get_env("DEEPSEEK_API_KEY")
        if deepseek_key and deepseek_key != "your_deepseek_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.DEEPSEEK, DeepSeekProvider)
            registered_providers.append(ProviderType.DEEPSEEK.value)
            logger.debug(f"Registered provider: {ProviderType.DEEPSEEK.value}")

    # 2. Custom provider second (for local/private models)
    if has_custom:
        # Factory function that creates CustomProvider with proper parameters
        def custom_provider_factory(api_key=None):
            # api_key is CUSTOM_API_KEY (can be empty for Ollama), base_url from CUSTOM_API_URL
            base_url = get_env("CUSTOM_API_URL", "") or ""
            return CustomProvider(api_key=api_key or "", base_url=base_url)  # Use provided API key or empty string

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)
        registered_providers.append(ProviderType.CUSTOM.value)
        logger.debug(f"Registered provider: {ProviderType.CUSTOM.value}")

    # 3. OpenRouter last (catch-all for everything else)
    if has_openrouter:
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        registered_providers.append(ProviderType.OPENROUTER.value)
        logger.debug(f"Registered provider: {ProviderType.OPENROUTER.value}")

    # Log all registered providers
    if registered_providers:
        logger.info(f"Registered providers: {', '.join(registered_providers)}")

    # Require at least one valid provider
    if not valid_providers:
        raise ValueError(
            "At least one API configuration is required. Please set either:\n"
            "- GEMINI_API_KEY for Gemini models\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- XAI_API_KEY for X.AI GROK models\n"
            "- OPENROUTER_API_KEY for OpenRouter (multiple models)\n"
            "- CUSTOM_API_URL for local models (Ollama, vLLM, etc.)"
        )

    logger.info(f"Available providers: {', '.join(valid_providers)}")

    # Log provider priority
    priority_info = []
    if has_native_apis:
        priority_info.append("Native APIs (Gemini, OpenAI)")
    if has_custom:
        priority_info.append("Custom endpoints")
    if has_openrouter:
        priority_info.append("OpenRouter (catch-all)")

    if len(priority_info) > 1:
        logger.info(f"Provider priority: {' → '.join(priority_info)}")

    # Register cleanup function for providers
    def cleanup_providers():
        """Clean up all registered providers on shutdown."""
        try:
            registry = ModelProviderRegistry()
            if hasattr(registry, "_initialized_providers"):
                # Iterate over provider instances (values), not (type, instance) tuples
                for provider in list(registry._initialized_providers.values()):
                    try:
                        if provider and hasattr(provider, "close"):
                            provider.close()
                    except Exception:
                        # Logger might be closed during shutdown
                        pass
        except Exception:
            # Silently ignore any errors during cleanup
            pass

    atexit.register(cleanup_providers)

    # Check and log model restrictions
    restriction_service = get_restriction_service()
    restrictions = restriction_service.get_restriction_summary()

    if restrictions:
        logger.info("Model restrictions configured:")
        for provider_name, allowed_models in restrictions.items():
            if isinstance(allowed_models, list):
                logger.info(f"  {provider_name}: {', '.join(allowed_models)}")
            else:
                logger.info(f"  {provider_name}: {allowed_models}")

        # Validate restrictions against known models
        provider_instances = {}
        provider_types_to_validate = [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.XAI]
        for provider_type in provider_types_to_validate:
            provider = ModelProviderRegistry.get_provider(provider_type)
            if provider:
                provider_instances[provider_type] = provider

        if provider_instances:
            restriction_service.validate_against_known_models(provider_instances)
    else:
        logger.info("No model restrictions configured - all models allowed")

    # Check if auto mode has any models available after restrictions
    from config import IS_AUTO_MODE

    if IS_AUTO_MODE:
        available_models = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        if not available_models:
            logger.error(
                "Auto mode is enabled but no models are available after applying restrictions. "
                "Please check your OPENAI_ALLOWED_MODELS and GOOGLE_ALLOWED_MODELS settings."
            )
            raise ValueError(
                "No models available for auto mode due to restrictions. "
                "Please adjust your allowed model settings or disable auto mode."
            )


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    List all available tools with their descriptions and input schemas.

    This handler is called by MCP clients during initialization to discover
    what tools are available. Each tool provides:
    - name: Unique identifier for the tool
    - description: Detailed explanation of what the tool does
    - inputSchema: JSON Schema defining the expected parameters

    Returns:
        List of Tool objects representing all available tools
    """
    logger.debug("MCP client requested tool list")

    # Try to log client info if available (this happens early in the handshake)
    try:
        from utils.client_info import format_client_info, get_client_info_from_context

        client_info = get_client_info_from_context(server)
        if client_info:
            formatted = format_client_info(client_info)
            logger.info(f"MCP Client Connected: {formatted}")

            # Log to activity file as well
            try:
                mcp_activity_logger = logging.getLogger("mcp_activity")
                friendly_name = client_info.get("friendly_name", "CLI Agent")
                raw_name = client_info.get("name", "Unknown")
                version = client_info.get("version", "Unknown")
                mcp_activity_logger.info(f"MCP_CLIENT_INFO: {friendly_name} (raw={raw_name} v{version})")
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Could not log client info during list_tools: {e}")
    tools = []

    # Add all registered AI-powered tools from the TOOLS registry
    for tool in TOOLS.values():
        # Get optional annotations from the tool
        annotations = tool.get_annotations()
        tool_annotations = ToolAnnotations(**annotations) if annotations else None

        tools.append(
            Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.get_input_schema(),
                annotations=tool_annotations,
            )
        )

    # Log cache efficiency info
    openrouter_key_for_cache = get_env("OPENROUTER_API_KEY")
    if openrouter_key_for_cache and openrouter_key_for_cache != "your_openrouter_api_key_here":
        logger.debug("OpenRouter registry cache used efficiently across all tool schemas")

    logger.debug(f"Returning {len(tools)} tools to MCP client")
    return tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle incoming tool execution requests from MCP clients.

    This is the main request dispatcher that routes tool calls to their appropriate handlers.
    It supports both AI-powered tools (from TOOLS registry) and utility tools (implemented as
    static functions).

    CONVERSATION LIFECYCLE MANAGEMENT:
    This function serves as the central orchestrator for multi-turn AI-to-AI conversations:

    1. THREAD RESUMPTION: When continuation_id is present, it reconstructs complete conversation
       context from in-memory storage including conversation history and file references

    2. CROSS-TOOL CONTINUATION: Enables seamless handoffs between different tools while
       preserving full conversation context and file references

    3. CONTEXT INJECTION: Reconstructed conversation history is embedded into tool prompts
       using the dual prioritization strategy:
       - Files: Newest-first prioritization (recent file versions take precedence)
       - Turns: Newest-first collection for token efficiency, chronological presentation for LLM

    4. FOLLOW-UP GENERATION: After tool execution, generates continuation offers for ongoing
       AI-to-AI collaboration with natural language instructions

    STATELESS TO STATEFUL BRIDGE:
    The MCP protocol is inherently stateless, but this function bridges the gap by:
    - Loading persistent conversation state from in-memory storage
    - Reconstructing full multi-turn context for tool execution
    - Enabling tools to access previous exchanges and file references
    - Supporting conversation chains across different tool types

    Args:
        name: The name of the tool to execute (e.g., "chat", "listmodels", "dump_threads")
        arguments: Dictionary of arguments to pass to the tool, potentially including:
                  - continuation_id: UUID for conversation thread resumption
                  - files: File paths for analysis (subject to deduplication)
                  - prompt: User request or follow-up question
                  - model: Specific AI model to use (optional)

    Returns:
        List of TextContent objects containing:
        - Tool's primary response with analysis/results
        - Continuation offers for follow-up conversations (when applicable)
        - Structured JSON responses with status and content

    Raises:
        ValueError: If continuation_id is invalid or conversation thread not found
        Exception: For tool-specific errors or execution failures

    Example Conversation Flow:
        1. The CLI calls chat with files → creates new thread
        2. Thread ID returned in continuation offer
        3. The CLI continues with dump_threads + continuation_id → full context preserved
        4. Multiple tools can collaborate using the same thread ID
    """
    logger.info(f"MCP tool call: {name}")
    logger.debug(f"MCP tool arguments: {list(arguments.keys())}")

    # Log to activity file for monitoring
    try:
        mcp_activity_logger = logging.getLogger("mcp_activity")
        mcp_activity_logger.info(f"TOOL_CALL: {name} with {len(arguments)} arguments")
    except Exception:
        pass

    # Handle thread context reconstruction if continuation_id is present
    if "continuation_id" in arguments and arguments["continuation_id"]:
        continuation_id = arguments["continuation_id"]
        logger.debug(f"Resuming conversation thread: {continuation_id}")
        logger.debug(
            f"[CONVERSATION_DEBUG] Tool '{name}' resuming thread {continuation_id} with {len(arguments)} arguments"
        )
        logger.debug(f"[CONVERSATION_DEBUG] Original arguments keys: {list(arguments.keys())}")

        # Log to activity file for monitoring
        try:
            mcp_activity_logger = logging.getLogger("mcp_activity")
            mcp_activity_logger.info(f"CONVERSATION_RESUME: {name} resuming thread {continuation_id}")
        except Exception:
            pass

        arguments = await reconstruct_thread_context(arguments)
        logger.debug(f"[CONVERSATION_DEBUG] After thread reconstruction, arguments keys: {list(arguments.keys())}")
        if "_remaining_tokens" in arguments:
            logger.debug(f"[CONVERSATION_DEBUG] Remaining token budget: {arguments['_remaining_tokens']:,}")

    # Route to AI-powered tools that require Gemini API calls
    if name in TOOLS:
        logger.info(f"Executing tool '{name}' with {len(arguments)} parameter(s)")
        tool = TOOLS[name]

        # EARLY MODEL RESOLUTION AT MCP BOUNDARY
        # Resolve model before passing to tool - this ensures consistent model handling
        from providers.registry import ModelProviderRegistry
        from utils.file_utils import check_total_file_size
        from utils.model_context import ModelContext

        # Get model from arguments or use default
        model_name = arguments.get("model") or DEFAULT_MODEL
        logger.debug(f"Initial model for {name}: {model_name}")

        # Parse model:option format if present
        model_name, model_option = parse_model_option(model_name)
        if model_option:
            logger.info(f"Parsed model format - model: '{model_name}', option: '{model_option}'")
        else:
            logger.info(f"Parsed model format - model: '{model_name}'")

        # Skip model resolution for tools that don't require models.
        if not tool.requires_model():
            logger.debug(f"Tool {name} doesn't require model resolution - skipping model validation")
            # Execute tool directly without model context
            return await tool.execute(arguments)

        # Handle auto mode at MCP boundary - resolve to specific model
        if model_name.lower() == "auto":
            # Get tool category to determine appropriate model
            tool_category = tool.get_model_category()
            resolved_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)
            logger.info(f"Auto mode resolved to {resolved_model} for {name} (category: {tool_category.value})")
            model_name = resolved_model
            # Update arguments with resolved model
            arguments["model"] = model_name

        # Validate model availability at MCP boundary
        provider = ModelProviderRegistry.get_provider_for_model(model_name)
        if not provider:
            # Get list of available models for error message
            available_models = list(ModelProviderRegistry.get_available_models(respect_restrictions=True).keys())
            tool_category = tool.get_model_category()
            suggested_model = ModelProviderRegistry.get_preferred_fallback_model(tool_category)

            error_message = (
                f"Model '{model_name}' is not available with current API keys. "
                f"Available models: {', '.join(available_models)}. "
                f"Suggested model for {name}: '{suggested_model}' "
                f"(category: {tool_category.value})"
            )
            error_output = ToolOutput(
                status="error",
                content=error_message,
                content_type="text",
                metadata={"tool_name": name, "requested_model": model_name},
            )
            raise ToolExecutionError(error_output.model_dump_json())

        # Create model context with resolved model and option
        model_context = ModelContext(model_name, model_option)
        arguments["_model_context"] = model_context
        arguments["_resolved_model_name"] = model_name
        logger.debug(
            f"Model context created for {model_name} with {model_context.capabilities.context_window} token capacity"
        )
        if model_option:
            logger.debug(f"Model option stored in context: '{model_option}'")

        # EARLY FILE SIZE VALIDATION AT MCP BOUNDARY
        # Check file sizes before tool execution using resolved model
        argument_files = arguments.get("absolute_file_paths")
        if argument_files:
            logger.debug(f"Checking file sizes for {len(argument_files)} files with model {model_name}")
            file_size_check = check_total_file_size(argument_files, model_name)
            if file_size_check:
                logger.warning(f"File size check failed for {name} with model {model_name}")
                raise ToolExecutionError(ToolOutput(**file_size_check).model_dump_json())

        # Execute tool with pre-resolved model context
        result = await tool.execute(arguments)
        logger.info(f"Tool '{name}' execution completed")

        # Log completion to activity file
        try:
            mcp_activity_logger = logging.getLogger("mcp_activity")
            mcp_activity_logger.info(f"TOOL_COMPLETED: {name}")
        except Exception:
            pass
        return result

    # Handle unknown tool requests as errors
    else:
        raise ToolExecutionError(f"Unknown tool: {name}")


def parse_model_option(model_string: str) -> tuple[str, Optional[str]]:
    """
    Parse model:option format into model name and option.

    Handles different formats:
    - OpenRouter models: preserve :free, :beta, :preview suffixes as part of model name
    - Ollama/Custom models: split on : to extract tags like :latest

    Args:
        model_string: String that may contain "model:option" format

    Returns:
        tuple: (model_name, option) where option may be None
    """
    if ":" in model_string and not model_string.startswith("http"):  # Avoid parsing URLs
        # Check if this looks like an OpenRouter model (contains /)
        if "/" in model_string and model_string.count(":") == 1:
            # Could be openai/gpt-4:something - check what comes after colon
            parts = model_string.split(":", 1)
            suffix = parts[1].strip().lower()

            # Known OpenRouter suffixes to preserve
            if suffix in ["free", "beta", "preview"]:
                return model_string.strip(), None

        # For other patterns (for example, local model tags), split normally
        parts = model_string.split(":", 1)
        model_name = parts[0].strip()
        model_option = parts[1].strip() if len(parts) > 1 else None
        return model_name, model_option
    return model_string.strip(), None


async def reconstruct_thread_context(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Reconstruct conversation context for stateless-to-stateful thread continuation.

    This is a critical function that transforms the inherently stateless MCP protocol into
    stateful multi-turn conversations. It loads persistent conversation state from in-memory
    storage and rebuilds complete conversation context using the sophisticated dual prioritization
    strategy implemented in the conversation memory system.

    CONTEXT RECONSTRUCTION PROCESS:

    1. THREAD RETRIEVAL: Loads complete ThreadContext from storage using continuation_id
       - Includes all conversation turns with tool attribution
       - Preserves file references and cross-tool context
       - Handles conversation chains across multiple linked threads

    2. CONVERSATION HISTORY BUILDING: Uses build_conversation_history() to create
       comprehensive context with intelligent prioritization:

       FILE PRIORITIZATION (Newest-First Throughout):
       - When same file appears in multiple turns, newest reference wins
       - File embedding prioritizes recent versions, excludes older duplicates
       - Token budget management ensures most relevant files are preserved

       CONVERSATION TURN PRIORITIZATION (Dual Strategy):
       - Collection Phase: Processes turns newest-to-oldest for token efficiency
       - Presentation Phase: Presents turns chronologically for LLM understanding
       - Ensures recent context is preserved when token budget is constrained

    3. CONTEXT INJECTION: Embeds reconstructed history into tool request arguments
       - Conversation history becomes part of the tool's prompt context
       - Files referenced in previous turns are accessible to current tool
       - Cross-tool knowledge transfer is seamless and comprehensive

    4. TOKEN BUDGET MANAGEMENT: Applies model-specific token allocation
       - Balances conversation history vs. file content vs. response space
       - Gracefully handles token limits with intelligent exclusion strategies
       - Preserves most contextually relevant information within constraints

    CROSS-TOOL CONTINUATION SUPPORT:
    This function enables seamless handoffs between different tools while
    preserving conversation context and file references.

    ERROR HANDLING & RECOVERY:
    - Thread expiration: Provides clear instructions for conversation restart
    - Storage unavailability: Graceful degradation with error messaging
    - Invalid continuation_id: Security validation and user-friendly errors

    Args:
        arguments: Original request arguments dictionary containing:
                  - continuation_id (required): UUID of conversation thread to resume
                  - Other tool-specific arguments that will be preserved

    Returns:
        dict[str, Any]: Enhanced arguments dictionary with conversation context:
        - Original arguments preserved
        - Conversation history embedded in appropriate format for tool consumption
        - File context from previous turns made accessible
        - Cross-tool knowledge transfer enabled

    Raises:
        ValueError: When continuation_id is invalid, thread not found, or expired
                   Includes user-friendly recovery instructions

    Performance Characteristics:
        - O(1) thread lookup in memory
        - O(n) conversation history reconstruction where n = number of turns
        - Intelligent token budgeting prevents context window overflow
        - Optimized file deduplication minimizes redundant content

    Example Usage Flow:
        1. CLI: "Continue with this thread" + continuation_id
        2. reconstruct_thread_context() loads the previous conversation
        3. The next tool receives full context including previous file analysis
        4. Natural cross-tool collaboration continues without context loss
    """
    from utils.conversation_memory import add_turn, build_conversation_history, get_thread

    continuation_id = arguments["continuation_id"]

    # Get thread context from storage
    logger.debug(f"[CONVERSATION_DEBUG] Looking up thread {continuation_id} in storage")
    context = get_thread(continuation_id)
    if not context:
        logger.warning(f"Thread not found: {continuation_id}")
        logger.debug(f"[CONVERSATION_DEBUG] Thread {continuation_id} not found in storage or expired")

        # Log to activity file for monitoring
        try:
            mcp_activity_logger = logging.getLogger("mcp_activity")
            mcp_activity_logger.info(f"CONVERSATION_ERROR: Thread {continuation_id} not found or expired")
        except Exception:
            pass

        # Return error asking CLI to restart conversation with full context
        raise ValueError(
            f"Conversation thread '{continuation_id}' was not found or has expired. "
            f"This may happen if the conversation was created more than 24 hours ago or if the "
            f"server was restarted. "
            f"Please restart the conversation by providing your full question/prompt without the "
            f"continuation_id parameter. "
            f"This will create a new conversation thread that can continue with follow-up exchanges."
        )

    # Add user's new input to the conversation
    user_prompt = arguments.get("prompt", "")
    if user_prompt:
        # Capture files referenced in this turn
        user_files = arguments.get("absolute_file_paths") or []
        logger.debug(f"[CONVERSATION_DEBUG] Adding user turn to thread {continuation_id}")
        from utils.token_utils import estimate_tokens

        user_prompt_tokens = estimate_tokens(user_prompt)
        logger.debug(
            f"[CONVERSATION_DEBUG] User prompt length: {len(user_prompt)} chars (~{user_prompt_tokens:,} tokens)"
        )
        logger.debug(f"[CONVERSATION_DEBUG] User files: {user_files}")
        success = add_turn(continuation_id, "user", user_prompt, files=user_files)
        if not success:
            logger.warning(f"Failed to add user turn to thread {continuation_id}")
            logger.debug("[CONVERSATION_DEBUG] Failed to add user turn - thread may be at turn limit or expired")
        else:
            logger.debug(f"[CONVERSATION_DEBUG] Successfully added user turn to thread {continuation_id}")

    # Create model context early to use for history building
    from utils.model_context import ModelContext

    tool = TOOLS.get(context.tool_name)
    requires_model = tool.requires_model() if tool else True

    # Check if we should use the model from the previous conversation turn
    model_from_args = arguments.get("model")
    if requires_model and not model_from_args and context.turns:
        # Find the last assistant turn to get the model used
        for turn in reversed(context.turns):
            if turn.role == "assistant" and turn.model_name:
                arguments["model"] = turn.model_name
                logger.debug(f"[CONVERSATION_DEBUG] Using model from previous turn: {turn.model_name}")
                break

    # Resolve an effective model for context reconstruction when DEFAULT_MODEL=auto
    model_context = arguments.get("_model_context")

    if requires_model:
        if model_context is None:
            try:
                model_context = ModelContext.from_arguments(arguments)
                arguments.setdefault("_resolved_model_name", model_context.model_name)
            except ValueError as exc:
                from providers.registry import ModelProviderRegistry

                fallback_model = None
                if tool is not None:
                    try:
                        fallback_model = ModelProviderRegistry.get_preferred_fallback_model(tool.get_model_category())
                    except Exception as fallback_exc:  # pragma: no cover - defensive log
                        logger.debug(
                            f"[CONVERSATION_DEBUG] Unable to resolve fallback model for {context.tool_name}: {fallback_exc}"
                        )

                if fallback_model is None:
                    available_models = ModelProviderRegistry.get_available_model_names()
                    if available_models:
                        fallback_model = available_models[0]

                if fallback_model is None:
                    raise

                logger.debug(
                    f"[CONVERSATION_DEBUG] Falling back to model '{fallback_model}' for context reconstruction after error: {exc}"
                )
                model_context = ModelContext(fallback_model)
                arguments["_model_context"] = model_context
                arguments["_resolved_model_name"] = fallback_model

        from providers.registry import ModelProviderRegistry

        provider = ModelProviderRegistry.get_provider_for_model(model_context.model_name)
        if provider is None:
            fallback_model = None
            if tool is not None:
                try:
                    fallback_model = ModelProviderRegistry.get_preferred_fallback_model(tool.get_model_category())
                except Exception as fallback_exc:  # pragma: no cover - defensive log
                    logger.debug(
                        f"[CONVERSATION_DEBUG] Unable to resolve fallback model for {context.tool_name}: {fallback_exc}"
                    )

            if fallback_model is None:
                available_models = ModelProviderRegistry.get_available_model_names()
                if available_models:
                    fallback_model = available_models[0]

            if fallback_model is None:
                raise ValueError(
                    f"Conversation continuation failed: model '{model_context.model_name}' is not available with current API keys."
                )

            logger.debug(
                f"[CONVERSATION_DEBUG] Model '{model_context.model_name}' unavailable; swapping to '{fallback_model}' for context reconstruction"
            )
            model_context = ModelContext(fallback_model)
            arguments["_model_context"] = model_context
            arguments["_resolved_model_name"] = fallback_model
    else:
        if model_context is None:
            from providers.registry import ModelProviderRegistry

            fallback_model = None
            if tool is not None:
                try:
                    fallback_model = ModelProviderRegistry.get_preferred_fallback_model(tool.get_model_category())
                except Exception as fallback_exc:  # pragma: no cover - defensive log
                    logger.debug(
                        f"[CONVERSATION_DEBUG] Unable to resolve fallback model for {context.tool_name}: {fallback_exc}"
                    )

            if fallback_model is None:
                available_models = ModelProviderRegistry.get_available_model_names()
                if available_models:
                    fallback_model = available_models[0]

            if fallback_model is None:
                raise ValueError(
                    "Conversation continuation failed: no available models detected for context reconstruction."
                )

            logger.debug(
                f"[CONVERSATION_DEBUG] Using fallback model '{fallback_model}' for context reconstruction of tool without model requirement"
            )
            model_context = ModelContext(fallback_model)
            arguments["_model_context"] = model_context
            arguments["_resolved_model_name"] = fallback_model

    # Build conversation history with model-specific limits
    logger.debug(f"[CONVERSATION_DEBUG] Building conversation history for thread {continuation_id}")
    logger.debug(f"[CONVERSATION_DEBUG] Thread has {len(context.turns)} turns, tool: {context.tool_name}")
    logger.debug(f"[CONVERSATION_DEBUG] Using model: {model_context.model_name}")
    conversation_history, conversation_tokens = build_conversation_history(context, model_context)
    logger.debug(f"[CONVERSATION_DEBUG] Conversation history built: {conversation_tokens:,} tokens")
    logger.debug(
        f"[CONVERSATION_DEBUG] Conversation history length: {len(conversation_history)} chars (~{conversation_tokens:,} tokens)"
    )

    # All tools now use standardized 'prompt' field
    original_prompt = arguments.get("prompt", "")
    logger.debug("[CONVERSATION_DEBUG] Extracting user input from 'prompt' field")
    original_prompt_tokens = estimate_tokens(original_prompt) if original_prompt else 0
    logger.debug(
        f"[CONVERSATION_DEBUG] User input length: {len(original_prompt)} chars (~{original_prompt_tokens:,} tokens)"
    )

    # Merge original context with the new prompt
    if conversation_history:
        enhanced_prompt = f"{conversation_history}\n\n=== NEW USER INPUT ===\n{original_prompt}"
    else:
        enhanced_prompt = original_prompt

    # Update arguments with enhanced context and remaining token budget
    enhanced_arguments = arguments.copy()

    # Store the enhanced prompt in the prompt field
    enhanced_arguments["prompt"] = enhanced_prompt
    # Store the original user prompt separately for size validation
    enhanced_arguments["_original_user_prompt"] = original_prompt
    logger.debug("[CONVERSATION_DEBUG] Storing enhanced prompt in 'prompt' field")
    logger.debug("[CONVERSATION_DEBUG] Storing original user prompt in '_original_user_prompt' field")

    # Calculate remaining token budget based on current model
    # (model_context was already created above for history building)
    token_allocation = model_context.calculate_token_allocation()

    # Calculate remaining tokens for files/new content
    # History has already consumed some of the content budget
    remaining_tokens = token_allocation.content_tokens - conversation_tokens
    enhanced_arguments["_remaining_tokens"] = max(0, remaining_tokens)  # Ensure non-negative
    enhanced_arguments["_model_context"] = model_context  # Pass context for use in tools

    logger.debug("[CONVERSATION_DEBUG] Token budget calculation:")
    logger.debug(f"[CONVERSATION_DEBUG]   Model: {model_context.model_name}")
    logger.debug(f"[CONVERSATION_DEBUG]   Total capacity: {token_allocation.total_tokens:,}")
    logger.debug(f"[CONVERSATION_DEBUG]   Content allocation: {token_allocation.content_tokens:,}")
    logger.debug(f"[CONVERSATION_DEBUG]   Conversation tokens: {conversation_tokens:,}")
    logger.debug(f"[CONVERSATION_DEBUG]   Remaining tokens: {remaining_tokens:,}")

    # Merge original context parameters (files, etc.) with new request
    if context.initial_context:
        logger.debug(f"[CONVERSATION_DEBUG] Merging initial context with {len(context.initial_context)} parameters")
        for key, value in context.initial_context.items():
            if key not in enhanced_arguments and key not in ["temperature", "thinking_mode", "model"]:
                enhanced_arguments[key] = value
                logger.debug(f"[CONVERSATION_DEBUG] Merged initial context param: {key}")

    logger.info(f"Reconstructed context for thread {continuation_id} (turn {len(context.turns)})")
    logger.debug(f"[CONVERSATION_DEBUG] Final enhanced arguments keys: {list(enhanced_arguments.keys())}")

    if "absolute_file_paths" in enhanced_arguments:
        logger.debug(
            f"[CONVERSATION_DEBUG] Final files in enhanced arguments: {enhanced_arguments['absolute_file_paths']}"
        )

    # Log to activity file for monitoring
    try:
        mcp_activity_logger = logging.getLogger("mcp_activity")
        mcp_activity_logger.info(
            f"CONVERSATION_CONTINUATION: Thread {continuation_id} turn {len(context.turns)} - "
            f"{len(context.turns)} previous turns loaded"
        )
    except Exception:
        pass

    return enhanced_arguments


@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """
    List all available prompts for CLI Code shortcuts.

    This handler returns prompts that enable shortcuts like /vox:chat.
    We automatically generate prompts from all tools (1:1 mapping) plus add
    a few marketing aliases with richer templates for commonly used tools.

    Returns:
        List of Prompt objects representing all available prompts
    """
    logger.debug("MCP client requested prompt list")
    prompts = []

    # Add a prompt for each tool with rich templates
    for tool_name, tool in TOOLS.items():
        if tool_name in PROMPT_TEMPLATES:
            # Use the rich template
            template_info = PROMPT_TEMPLATES[tool_name]
            prompts.append(
                Prompt(
                    name=template_info["name"],
                    description=template_info["description"],
                    arguments=[],  # MVP: no structured args
                )
            )
        else:
            # Fallback for any tools without templates (shouldn't happen)
            prompts.append(
                Prompt(
                    name=tool_name,
                    description=f"Use {tool.name} tool",
                    arguments=[],
                )
            )

    # Add special "continue" prompt only if chat is available
    if "chat" in TOOLS:
        prompts.append(
            Prompt(
                name="continue",
                description="Continue the previous conversation using the chat tool",
                arguments=[],
            )
        )

    logger.debug(f"Returning {len(prompts)} prompts to MCP client")
    return prompts


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict[str, Any] = None) -> GetPromptResult:
    """
    Get prompt details and generate the actual prompt text.

    This handler is called when a user invokes a prompt (e.g., /vox:chat).
    It generates the appropriate text that CLI will then use to call the
    underlying tool.

    Args:
        name: The name of the prompt to execute
        arguments: Optional arguments for the prompt (e.g., model, thinking_mode)

    Returns:
        GetPromptResult with the prompt details and generated message

    Raises:
        ValueError: If the prompt name is unknown
    """
    logger.debug(f"MCP client requested prompt: {name} with args: {arguments}")

    # Handle special "continue" case
    if name.lower() == "continue":
        # This is "/vox:continue" - use chat tool as default for continuation
        tool_name = "chat"
        template_info = {
            "name": "continue",
            "description": "Continue the previous conversation",
            "template": "Continue the conversation",
        }
        logger.debug("Using /vox:continue - defaulting to chat tool")
    else:
        # Find the corresponding tool by checking prompt names
        tool_name = None
        template_info = None

        # Check if it's a known prompt name
        for t_name, t_info in PROMPT_TEMPLATES.items():
            if t_info["name"] == name:
                tool_name = t_name
                template_info = t_info
                break

        # If not found, check if it's a direct tool name
        if not tool_name and name in TOOLS:
            tool_name = name
            template_info = {
                "name": name,
                "description": f"Use {name} tool",
                "template": f"Use {name}",
            }

        if not tool_name:
            logger.error(f"Unknown prompt requested: {name}")
            raise ValueError(f"Unknown prompt: {name}")

    # Get the template
    template = template_info.get("template", f"Use {tool_name}")

    # Safe template expansion with defaults
    final_model = arguments.get("model", "auto") if arguments else "auto"

    prompt_args = {
        "model": final_model,
        "thinking_mode": arguments.get("thinking_mode", "medium") if arguments else "medium",
    }

    logger.debug(f"Using model '{final_model}' for prompt '{name}'")

    # Safely format the template
    try:
        prompt_text = template.format(**prompt_args)
    except KeyError as e:
        logger.warning(f"Missing template argument {e} for prompt {name}, using raw template")
        prompt_text = template  # Fallback to raw template

    # Generate tool call text
    tool_instruction = prompt_text

    return GetPromptResult(
        prompt=Prompt(
            name=name,
            description=template_info["description"],
            arguments=[],
        ),
        messages=[
            PromptMessage(
                role="user",
                content={"type": "text", "text": tool_instruction},
            )
        ],
    )


async def main():
    """
    Main entry point for the MCP server.

    Configures AI providers and starts the server using stdio transport.
    The server will continue running until the client disconnects or an error occurs.
    """
    # Validate and configure providers based on available API keys
    configure_providers()

    # Log startup message
    logger.info("Vox MCP Server starting up...")
    logger.info(f"Log level: {log_level}")

    # Note: MCP client info will be logged during the protocol handshake
    # (when handle_list_tools is called)

    # Log current model mode
    from config import IS_AUTO_MODE

    if IS_AUTO_MODE:
        logger.info("Model mode: AUTO (CLI will select the best model for each task)")
    else:
        logger.info(f"Model mode: Fixed model '{DEFAULT_MODEL}'")

    logger.info(f"Available tools: {list(TOOLS.keys())}")
    logger.info("Server ready - waiting for tool requests...")

    handshake_instructions = ""

    # Run the server using stdio transport (standard input/output)
    # This allows the server to be launched by MCP clients as a subprocess
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="vox",
                server_version=__version__,
                instructions=handshake_instructions,
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(),  # Advertise tool support capability
                    prompts=PromptsCapability(),  # Advertise prompt support capability
                ),
            ),
        )


def run():
    """Console script entry point for vox-mcp."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle graceful shutdown
        pass


if __name__ == "__main__":
    run()
