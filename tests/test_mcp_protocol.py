"""Exercise the MCP surface through the SDK's own in-process client.

Covers both connection eras the SDK 2 server speaks: ``legacy`` (the
``initialize`` handshake stdio clients use today) and ``auto`` (2026-07-28).
No provider is called.
"""

import json
from types import SimpleNamespace

import pytest
from mcp.client import Client

from config import __version__
from providers.registry import ModelProviderRegistry
from server import TOOLS
from server import server as mcp_server

MODES = ["legacy", "auto"]


def _install_dummy_provider(monkeypatch):
    """Ensure preflight model checks succeed without real provider configuration."""

    class DummyProvider:
        def get_provider_type(self):
            return SimpleNamespace(value="dummy")

        def get_capabilities(self, model_name):
            return SimpleNamespace(
                supports_extended_thinking=False,
                allow_code_generation=False,
                supports_images=False,
                context_window=1_000_000,
                max_image_size_mb=10,
            )

    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_provider_for_model",
        classmethod(lambda cls, model_name: DummyProvider()),
    )
    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_available_models",
        classmethod(lambda cls, respect_restrictions=False: {"gemini-2.5-flash": None}),
    )


@pytest.mark.parametrize("mode", MODES)
async def test_list_tools_returns_registered_tools(mode):
    async with Client(mcp_server, mode=mode) as client:
        result = await client.list_tools()

    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == set(TOOLS)
    for tool in tools.values():
        assert tool.input_schema["type"] == "object"
        assert tool.description
    assert tools["listmodels"].annotations is not None
    assert tools["listmodels"].annotations.read_only_hint is True


async def test_legacy_handshake_reports_vox_server_info():
    async with Client(mcp_server, mode="legacy") as client:
        assert client.server_info is not None
        assert client.server_info.name == "vox"
        assert client.server_info.version == __version__


@pytest.mark.parametrize("mode", MODES)
async def test_listmodels_call_returns_text_result(mode):
    async with Client(mcp_server, mode=mode) as client:
        result = await client.call_tool("listmodels", {})

    assert result.is_error is False
    assert result.content
    assert result.content[0].type == "text"
    assert result.content[0].text


@pytest.mark.parametrize("mode", MODES)
async def test_schema_violation_is_an_error_result(mode):
    async with Client(mcp_server, mode=mode) as client:
        result = await client.call_tool("chat", {})

    assert result.is_error is True
    assert result.content[0].text.startswith("Input validation error:")
    assert "prompt" in result.content[0].text


@pytest.mark.parametrize("mode", MODES)
async def test_unknown_tool_is_an_error_result(mode):
    async with Client(mcp_server, mode=mode) as client:
        result = await client.call_tool("no_such_tool", {})

    assert result.is_error is True
    assert "Unknown tool: no_such_tool" in result.content[0].text


@pytest.mark.parametrize("mode", MODES)
async def test_tool_execution_error_is_an_error_result(monkeypatch, mode):
    """ToolExecutionError reaches the client as isError=True content, not a JSON-RPC error."""
    _install_dummy_provider(monkeypatch)

    arguments = {
        "prompt": "Trigger file path validation failure",
        "absolute_file_paths": ["relative/path"],  # Not absolute -> ToolExecutionError from ChatTool
        "model": "gemini-2.5-flash",
    }
    async with Client(mcp_server, mode=mode) as client:
        result = await client.call_tool("chat", arguments)

    assert result.is_error is True
    data = json.loads(result.content[0].text)
    assert data["status"] == "error"
    assert "absolute" in data["content"].lower()
