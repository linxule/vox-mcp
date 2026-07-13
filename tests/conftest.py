"""
Pytest configuration for Vox MCP Server tests
"""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

# Ensure the parent directory is in the Python path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils.env as env_config  # noqa: E402

# Ensure tests operate with runtime environment rather than .env overrides during imports
env_config.reload_env({"VOX_FORCE_ENV_OVERRIDE": "false"})

# Set default model to a specific value for tests to avoid auto mode
# This prevents all tests from failing due to missing model parameter
os.environ["DEFAULT_MODEL"] = "gemini-2.5-flash"

# Force reload of config module to pick up the env var
import config  # noqa: E402

importlib.reload(config)

# Filesystem isolation is NOT automatic just because tests use tmp_path fixtures —
# any code path that resolves VOX_THREADS_DIR reaches the user's real ~/.vox/threads
# unless something redirects it. See the isolate_vox_threads_dir autouse fixture below.

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Register providers for all tests
from providers.gemini import GeminiModelProvider  # noqa: E402
from providers.openai import OpenAIModelProvider  # noqa: E402
from providers.registry import ModelProviderRegistry  # noqa: E402
from providers.shared import ProviderType  # noqa: E402
from providers.xai import XAIModelProvider  # noqa: E402

# Register providers at test startup
ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)


@pytest.fixture
def project_path(tmp_path):
    """
    Provides a temporary directory for tests.
    This ensures all file operations during tests are isolated.
    """
    # Create a subdirectory for this specific test
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)

    return test_dir


@pytest.fixture(autouse=True)
def isolate_vox_threads_dir(tmp_path, monkeypatch):
    """
    Keep the test suite out of the user's real thread store (~/.vox/threads).

    Without this, `add_turn()` reaches through conversation_memory into the
    persistence layer, which resolves VOX_THREADS_DIR and appends JSONL to the
    user's actual conversation history. That is bad on its own, but the second
    effect is worse: it makes a test pass for the wrong reason.

    test_add_turn_success asserted `get` was called once. On a clean machine the
    thread's JSONL does not exist, so the persistence layer takes its
    retroactive-header path and calls storage.get() a SECOND time — the assertion
    fails. But that same run writes the file, and every run afterwards finds it,
    short-circuits the retroactive path, and goes green. So the test failed once
    on a fresh checkout, healed itself into a false pass on every dev machine,
    and only ever told the truth on a fresh CI runner.

    config.VOX_THREADS_DIR is the single correct seam: both consumers
    (utils.thread_persistence and utils.markdown_export) import it from `config`
    at call time, so patching the module attribute covers both. Patching
    thread_persistence._get_threads_dir alone would leave markdown exports still
    writing to the real home directory.
    """
    threads_dir = tmp_path / "vox-threads"
    monkeypatch.setattr(config, "VOX_THREADS_DIR", threads_dir)
    return threads_dir


def _set_dummy_keys_if_missing():
    """Set dummy API keys only when they are completely absent."""
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        if not os.environ.get(var):
            os.environ[var] = "dummy-key-for-tests"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "no_mock_provider: disable automatic provider mocking")
    # Assume we need dummy keys until we learn otherwise
    config._needs_dummy_keys = True


def pytest_collection_modifyitems(session, config, items):
    """Hook that runs after test collection to check for no_mock_provider markers."""
    # Always set dummy keys if real keys are missing
    # This ensures tests work in CI even with no_mock_provider marker
    _set_dummy_keys_if_missing()


@pytest.fixture(autouse=True)
def mock_provider_availability(request, monkeypatch):
    """
    Automatically mock provider availability for all tests to prevent
    effective auto mode from being triggered when DEFAULT_MODEL is unavailable.

    This fixture ensures that when tests run with dummy API keys,
    the tools don't require model selection unless explicitly testing auto mode.
    """
    # Skip this fixture for tests that need real providers
    if hasattr(request, "node"):
        marker = request.node.get_closest_marker("no_mock_provider")
        if marker:
            return

    # Ensure providers are registered (in case other tests cleared the registry)
    from providers.shared import ProviderType

    registry = ModelProviderRegistry()

    if ProviderType.GOOGLE not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    if ProviderType.OPENAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    if ProviderType.XAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)

    # Mock is_effective_auto_mode for all BaseTool instances to return False
    # unless we're specifically testing auto mode behavior
    from tools.shared.base_tool import BaseTool

    def mock_is_effective_auto_mode(self):
        # If this is an auto mode test file or specific auto mode test, use the real logic
        test_file = request.node.fspath.basename if hasattr(request, "node") and hasattr(request.node, "fspath") else ""
        test_name = request.node.name if hasattr(request, "node") else ""

        # Allow auto mode for tests in auto mode files or with auto in the name
        if "auto_mode" in test_file.lower() or "auto" in test_name.lower():
            # Call original method logic
            from config import DEFAULT_MODEL

            if DEFAULT_MODEL.lower() == "auto":
                return True
            provider = ModelProviderRegistry.get_provider_for_model(DEFAULT_MODEL)
            return provider is None
        # For all other tests, return False to disable auto mode
        return False

    monkeypatch.setattr(BaseTool, "is_effective_auto_mode", mock_is_effective_auto_mode)


@pytest.fixture(autouse=True)
def clear_model_restriction_env(monkeypatch):
    """Ensure per-test isolation from user-defined model restriction env vars."""

    restriction_vars = [
        "OPENAI_ALLOWED_MODELS",
        "GOOGLE_ALLOWED_MODELS",
        "XAI_ALLOWED_MODELS",
        "OPENROUTER_ALLOWED_MODELS",
    ]

    for var in restriction_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def disable_force_env_override(monkeypatch):
    """Default tests to runtime environment visibility unless they explicitly opt in."""

    monkeypatch.setenv("VOX_FORCE_ENV_OVERRIDE", "false")
    env_config.reload_env({"VOX_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setenv("DEFAULT_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("MAX_CONVERSATION_TURNS", "50")

    import importlib
    import sys

    import config
    import utils.conversation_memory as conversation_memory

    importlib.reload(config)
    importlib.reload(conversation_memory)

    test_conversation_module = sys.modules.get("tests.test_conversation_memory")
    if test_conversation_module is not None:
        test_conversation_module.MAX_CONVERSATION_TURNS = conversation_memory.MAX_CONVERSATION_TURNS

    try:
        yield
    finally:
        env_config.reload_env()
