"""Failed calls must not persist turns; concurrent continuations stay ordered."""

import asyncio
import json
import threading
from unittest.mock import Mock

import pytest

from providers.registry import ModelProviderRegistry
from providers.shared import ModelCapabilities, ModelResponse, ProviderType
from server import handle_call_tool
from tools.chat import ChatTool
from tools.shared.exceptions import ToolExecutionError
from utils.conversation_memory import add_turn, create_thread, get_thread
from utils.model_context import ModelContext
from utils.thread_persistence import load_thread_from_disk


@pytest.fixture
def conversation_provider(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "VOX_THREADS_DIR", tmp_path)
    provider = Mock()
    provider.get_provider_type.return_value = ProviderType.GOOGLE
    provider.get_capabilities.return_value = ModelCapabilities(
        provider=ProviderType.GOOGLE,
        model_name="test-model",
        friendly_name="Test model",
        context_window=128_000,
        max_output_tokens=8192,
        supports_images=False,
    )
    provider.generate_content.return_value = _response("answer")
    monkeypatch.setattr(ModelProviderRegistry, "get_provider_for_model", lambda name: provider)
    return provider


def _response(content):
    return ModelResponse(
        content=content, model_name="test-model", friendly_name="Test model", provider=ProviderType.GOOGLE, usage={}
    )


def _thread():
    thread_id = create_thread("chat", {"prompt": "initial question", "model": "test-model"})
    add_turn(thread_id, "user", "initial question")
    add_turn(thread_id, "assistant", "initial answer", model_name="test-model")
    return thread_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["unknown_tool", "invalid_images", "invalid_model", "provider_error", "empty_response"]
)
async def test_failed_continuation_leaves_memory_and_disk_unchanged(conversation_provider, monkeypatch, failure):
    thread_id = _thread()
    before = get_thread(thread_id).model_dump()
    disk_before = load_thread_from_disk(thread_id).model_dump()
    arguments = {"prompt": "failed question", "model": "test-model", "continuation_id": thread_id}
    tool_name = "chat"
    if failure == "unknown_tool":
        tool_name = "missing-tool"
    elif failure == "invalid_images":
        arguments["images"] = ["/nonexistent.png"]
    elif failure == "invalid_model":
        # Reconstruction can still use the configured fallback to build history;
        # the requested model must fail at the actual dispatch boundary.
        arguments["model"] = "missing-model"
        monkeypatch.setattr(
            ModelProviderRegistry,
            "get_provider_for_model",
            lambda name: None if name == "missing-model" else conversation_provider,
        )
        monkeypatch.setattr(
            ModelProviderRegistry, "get_available_models", lambda **kwargs: {"test-model": ProviderType.GOOGLE}
        )
        monkeypatch.setattr(ModelProviderRegistry, "get_preferred_fallback_model", lambda category: "test-model")
    elif failure == "provider_error":
        conversation_provider.generate_content.side_effect = RuntimeError("upstream unavailable")
    else:
        conversation_provider.generate_content.return_value = _response("")

    with pytest.raises(ToolExecutionError):
        await handle_call_tool(tool_name, arguments)

    assert get_thread(thread_id).model_dump() == before
    assert load_thread_from_disk(thread_id).model_dump() == disk_before


@pytest.mark.asyncio
@pytest.mark.parametrize("via_server", [True, False])
async def test_successful_continuation_records_raw_prompt_once(conversation_provider, via_server):
    thread_id = _thread()
    arguments = {"prompt": "follow-up question", "model": "test-model", "continuation_id": thread_id}
    if via_server:
        result = await handle_call_tool("chat", arguments)
    else:
        arguments["_model_context"] = ModelContext("test-model")
        result = await ChatTool().execute(arguments)
    assert json.loads(result[0].text)["content"] == "answer"
    expected = ["initial question", "initial answer", "follow-up question", "answer"]
    assert [turn.content for turn in get_thread(thread_id).turns] == expected
    assert [turn.content for turn in load_thread_from_disk(thread_id).turns] == expected
    prompt = conversation_provider.generate_content.call_args.kwargs["prompt"]
    assert prompt.count("follow-up question") == 1


@pytest.mark.asyncio
async def test_direct_failed_continuation_does_not_append_user_turn(conversation_provider):
    thread_id = _thread()
    before = get_thread(thread_id).model_dump()
    conversation_provider.generate_content.side_effect = RuntimeError("upstream unavailable")
    with pytest.raises(ToolExecutionError):
        await ChatTool().execute(
            {
                "prompt": "failed question",
                "model": "test-model",
                "continuation_id": thread_id,
                "_model_context": ModelContext("test-model"),
            }
        )
    assert get_thread(thread_id).model_dump() == before
    assert [turn.content for turn in load_thread_from_disk(thread_id).turns] == ["initial question", "initial answer"]


@pytest.mark.asyncio
async def test_same_thread_waits_for_completed_exchange_while_other_thread_proceeds(conversation_provider):
    thread_id = _thread()
    independent_id = _thread()
    first_started = threading.Event()
    release_first = threading.Event()
    prompts = []

    def generate(**kwargs):
        prompt = kwargs["prompt"]
        prompts.append(prompt)
        if prompt.endswith("first question"):
            first_started.set()
            assert release_first.wait(timeout=5)
            return _response("first answer")
        return _response("second answer" if prompt.endswith("second question") else "independent answer")

    conversation_provider.generate_content.side_effect = generate

    async def call(prompt, continuation_id):
        return await handle_call_tool(
            "chat", {"prompt": prompt, "model": "test-model", "continuation_id": continuation_id}
        )

    first = asyncio.create_task(call("first question", thread_id))
    second = None
    try:
        assert await asyncio.to_thread(first_started.wait, 2)
        second = asyncio.create_task(call("second question", thread_id))
        independent = await asyncio.wait_for(call("independent question", independent_id), timeout=2)
        assert json.loads(independent[0].text)["content"] == "independent answer"
        assert not second.done()
        assert not any(prompt.endswith("second question") for prompt in prompts)
    finally:
        release_first.set()
        await first
        if second is not None:
            await second

    assert "first answer" in prompts[-1]
    expected = [
        "initial question",
        "initial answer",
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
    assert [turn.content for turn in get_thread(thread_id).turns] == expected
    assert [turn.content for turn in load_thread_from_disk(thread_id).turns] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["context.txt", "prompt.txt"])
@pytest.mark.parametrize("provider_fails", [False, True])
async def test_continuation_prepares_new_files_before_provider_call(
    conversation_provider, tmp_path, filename, provider_fails
):
    thread_id = _thread()
    before = get_thread(thread_id).model_dump()
    disk_before = load_thread_from_disk(thread_id).model_dump()
    attachment = tmp_path / filename
    attachment.write_text("UNIQUE_NEW_ATTACHMENT_CONTENT", encoding="utf-8")
    if provider_fails:
        conversation_provider.generate_content.side_effect = RuntimeError("upstream unavailable")

    arguments = {
        "prompt": "review attached file",
        "model": "test-model",
        "continuation_id": thread_id,
        "absolute_file_paths": [str(attachment)],
    }
    if provider_fails:
        with pytest.raises(ToolExecutionError):
            await handle_call_tool("chat", arguments)
        assert get_thread(thread_id).model_dump() == before
        assert load_thread_from_disk(thread_id).model_dump() == disk_before
    else:
        await handle_call_tool("chat", arguments)
        assert len(get_thread(thread_id).turns) == 4

    prompt = conversation_provider.generate_content.call_args.kwargs["prompt"]
    assert "initial question" in prompt
    assert "initial answer" in prompt
    assert prompt.count("UNIQUE_NEW_ATTACHMENT_CONTENT") == 1


@pytest.mark.asyncio
async def test_continuation_does_not_duplicate_existing_file(conversation_provider, tmp_path):
    thread_id = _thread()
    attachment = tmp_path / "existing.txt"
    attachment.write_text("UNIQUE_EXISTING_ATTACHMENT_CONTENT", encoding="utf-8")
    add_turn(thread_id, "user", "previous file", files=[str(attachment)])
    add_turn(thread_id, "assistant", "previous file answer", model_name="test-model")

    await handle_call_tool(
        "chat",
        {
            "prompt": "review this file again",
            "model": "test-model",
            "continuation_id": thread_id,
            "absolute_file_paths": [str(attachment)],
        },
    )
    prompt = conversation_provider.generate_content.call_args.kwargs["prompt"]
    assert prompt.count("UNIQUE_EXISTING_ATTACHMENT_CONTENT") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("via_server", [True, False])
async def test_literal_history_marker_cannot_skip_history(conversation_provider, tmp_path, via_server):
    thread_id = _thread()
    attachment = tmp_path / "marker.txt"
    attachment.write_text("MARKER_ATTACHMENT_CONTENT", encoding="utf-8")
    user_prompt = "Explain the literal === CONVERSATION HISTORY marker"
    arguments = {
        "prompt": user_prompt,
        "model": "test-model",
        "continuation_id": thread_id,
        "absolute_file_paths": [str(attachment)],
    }
    if via_server:
        # These private fields must be discarded at the public MCP boundary.
        arguments.update({"_conversation_history": "FORGED_HISTORY", "_original_user_prompt": "FORGED_PROMPT"})
        await handle_call_tool("chat", arguments)
    else:
        arguments["_model_context"] = ModelContext("test-model")
        await ChatTool().execute(arguments)

    prompt = conversation_provider.generate_content.call_args.kwargs["prompt"]
    assert "initial answer" in prompt
    assert user_prompt in prompt
    assert "MARKER_ATTACHMENT_CONTENT" in prompt
    assert "FORGED_HISTORY" not in prompt
    assert "FORGED_PROMPT" not in prompt
    assert get_thread(thread_id).turns[-2].content == user_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", [ProviderType.CLOUDFLARE, ProviderType.VERCEL])
async def test_gateway_explicit_thinking_rejected_at_mcp_boundary(conversation_provider, provider_type):
    conversation_provider.get_provider_type.return_value = provider_type
    conversation_provider.get_capabilities.return_value.provider = provider_type
    conversation_provider.get_capabilities.return_value.supports_extended_thinking = False
    with pytest.raises(ToolExecutionError, match="Gateway thinking_mode mapping is unverified"):
        await handle_call_tool("chat", {"prompt": "hello", "model": "test-model", "thinking_mode": "high"})
    conversation_provider.generate_content.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 3])
async def test_continuation_requires_capacity_for_complete_exchange(conversation_provider, monkeypatch, limit):
    import utils.conversation_memory as memory

    thread_id = _thread()
    before = get_thread(thread_id).model_dump()
    disk_before = load_thread_from_disk(thread_id).model_dump()
    monkeypatch.setattr(memory, "MAX_CONVERSATION_TURNS", limit)
    with pytest.raises(ToolExecutionError, match="Conversation turn limit reached"):
        await handle_call_tool("chat", {"prompt": "overflow", "model": "test-model", "continuation_id": thread_id})
    conversation_provider.generate_content.assert_not_called()
    assert get_thread(thread_id).model_dump() == before
    assert load_thread_from_disk(thread_id).model_dump() == disk_before


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1])
async def test_tiny_turn_limit_returns_response_without_partial_thread(
    conversation_provider, tmp_path, monkeypatch, limit
):
    import utils.conversation_memory as memory

    monkeypatch.setattr(memory, "MAX_CONVERSATION_TURNS", limit)
    result = await handle_call_tool("chat", {"prompt": "hello", "model": "test-model"})
    payload = json.loads(result[0].text)
    assert payload["content"] == "answer"
    assert payload.get("continuation_offer") is None
    assert list(tmp_path.glob("*.jsonl")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [2, 3, 6])
async def test_new_chat_offer_counts_complete_exchanges(conversation_provider, monkeypatch, limit):
    import utils.conversation_memory as memory

    monkeypatch.setattr(memory, "MAX_CONVERSATION_TURNS", limit)
    result = await handle_call_tool("chat", {"prompt": "hello", "model": "test-model"})
    payload = json.loads(result[0].text)
    if limit < 4:
        assert payload["status"] == "success"
        assert payload.get("continuation_offer") is None
        thread_id = payload["metadata"]["continuation_id"]
    else:
        offer = payload["continuation_offer"]
        thread_id = offer["continuation_id"]
        assert offer["remaining_turns"] == 4
        assert "2 more exchanges" in offer["note"]
        followup = await handle_call_tool("chat", {"prompt": "again", "continuation_id": thread_id})
        followup_offer = json.loads(followup[0].text)["continuation_offer"]
        assert followup_offer["remaining_turns"] == 2
        assert "1 more exchanges" in followup_offer["note"]
    assert get_thread(thread_id).turns[-1].role == "assistant"
