"""
Regression test for the concurrent-call connection-drop bug.

Root cause: the MCP low-level server dispatches each tool call concurrently
(anyio task group / ``tg.start_soon``). vox previously called the synchronous,
blocking ``provider.generate_content()`` directly on the event loop, which froze
the single loop — including the stdio read/write streams — for the whole
duration of a provider HTTP call. Under concurrent calls this starved the
transport and the client dropped the connection.

Fix: ``tools/simple/base.py`` offloads ``provider.generate_content`` via
``asyncio.to_thread``, keeping the loop responsive.

This test fails (heartbeat frozen, elapsed ~2x) if the offload is removed.
"""

import asyncio
import time
from unittest.mock import Mock

import pytest

from providers.shared import ModelCapabilities, ProviderType, RangeTemperatureConstraint
from tools.chat import ChatTool
from utils.model_context import ModelContext

BLOCK_SECONDS = 0.5


def _make_blocking_provider():
    """A mock provider whose generate_content blocks (simulates a slow HTTP call)."""
    provider = Mock()
    caps = ModelCapabilities(
        provider=ProviderType.GOOGLE,
        model_name="gemini-2.5-flash",
        friendly_name="Gemini",
        context_window=1_048_576,
        max_output_tokens=8192,
        supports_extended_thinking=False,
        temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 0.7),
    )
    provider.get_capabilities.return_value = caps
    provider.get_provider_type.return_value = ProviderType.GOOGLE
    provider.validate_model_name.return_value = True

    response = Mock()
    response.content = "ok"
    response.usage = {"input_tokens": 1, "output_tokens": 1}
    response.model_name = "gemini-2.5-flash"
    response.friendly_name = "Gemini"
    response.provider = ProviderType.GOOGLE
    response.metadata = {"finish_reason": "STOP"}

    def blocking_generate(**kwargs):
        time.sleep(BLOCK_SECONDS)
        return response

    provider.generate_content.side_effect = blocking_generate
    return provider, caps


def _model_context(provider, caps):
    ctx = ModelContext("gemini-2.5-flash")
    ctx._provider = provider
    ctx._capabilities = caps
    return ctx


@pytest.mark.asyncio
async def test_blocking_provider_call_does_not_freeze_event_loop():
    provider, caps = _make_blocking_provider()

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)  # establish a baseline
    ticks_before = ticks

    async def run_one():
        tool = ChatTool()
        return await tool.execute(
            {
                "prompt": "hello",
                "model": "gemini-2.5-flash",
                "_model_context": _model_context(provider, caps),
            }
        )

    start = time.monotonic()
    results = await asyncio.gather(run_one(), run_one())
    elapsed = time.monotonic() - start

    stop.set()
    await hb

    # Both concurrent calls completed successfully.
    assert len(results) == 2
    for r in results:
        assert r and r[0].text

    # The event loop stayed responsive *while the blocking provider calls ran*:
    # the heartbeat kept ticking (~50 ticks expected for a 0.5s window at 10ms).
    # If the loop had been frozen by an inline blocking call this would be ~0.
    ticks_during = ticks - ticks_before
    assert ticks_during >= 10, f"event loop appears frozen (only {ticks_during} heartbeats during blocking calls)"

    # Two 0.5s blocking calls run concurrently in worker threads finish in ~0.5s,
    # not ~1.0s (which inline serial execution on the loop would produce).
    assert elapsed < BLOCK_SECONDS * 1.8, f"calls did not run concurrently (elapsed {elapsed:.2f}s)"
