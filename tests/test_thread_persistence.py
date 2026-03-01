"""Tests for thread persistence (JSONL) and markdown export."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.conversation_memory import ConversationTurn, ThreadContext
from utils.markdown_export import (
    _aggregate_metadata,
    _format_duration,
    _yaml_escape,
    _yaml_list,
    export_thread_to_file,
    render_thread_markdown,
)
from utils.thread_persistence import (
    _get_filename,
    append_turn,
    list_thread_files,
    load_all_threads_from_disk,
    load_thread_from_disk,
    save_thread_header,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_threads_dir(tmp_path):
    """Patch VOX_THREADS_DIR to a temp directory for isolation."""
    with patch("utils.thread_persistence._get_threads_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def sample_context():
    """A minimal ThreadContext for testing."""
    return ThreadContext(
        thread_id="cf23ea29-3166-42d2-b0fb-7de5a08de1f9",
        created_at="2026-02-12T14:30:22+00:00",
        last_updated_at="2026-02-12T14:42:56+00:00",
        tool_name="chat",
        turns=[],
        initial_context={"prompt": "hello"},
        client_name="claude-code",
        client_version="2.1.39",
    )


@pytest.fixture
def sample_turn_user():
    return ConversationTurn(
        role="user",
        content="Hello, how are you?",
        timestamp="2026-02-12T14:30:25+00:00",
        files=["server.py"],
        tool_name="chat",
    )


@pytest.fixture
def sample_turn_assistant():
    return ConversationTurn(
        role="assistant",
        content="I'm doing well! I can see your server code.",
        timestamp="2026-02-12T14:30:30+00:00",
        model_name="gemini-2.5-pro",
        model_provider="google",
        tool_name="chat",
        model_metadata={"usage": {"input": 142, "output": 87}},
    )


@pytest.fixture
def context_with_turns(sample_context, sample_turn_user, sample_turn_assistant):
    """Context pre-populated with turns."""
    ctx = sample_context.model_copy()
    ctx.turns = [sample_turn_user, sample_turn_assistant]
    return ctx


# ---------------------------------------------------------------------------
# thread_persistence tests
# ---------------------------------------------------------------------------


class TestGetFilename:
    def test_full_uuid_in_filename(self):
        name = _get_filename("cf23ea29-3166-42d2-b0fb-7de5a08de1f9", "2026-02-12T14:30:22+00:00")
        assert name == "20260212-143022-cf23ea29-3166-42d2-b0fb-7de5a08de1f9.jsonl"

    def test_fallback_on_bad_timestamp(self):
        tid = "abcdef12-0000-0000-0000-000000000000"
        name = _get_filename(tid, "not-a-date")
        assert name.endswith(f"-{tid}.jsonl")


class TestSaveThreadHeader:
    def test_creates_file_with_header(self, tmp_threads_dir, sample_context):
        save_thread_header(sample_context)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            header = json.loads(f.readline())

        assert header["type"] == "header"
        assert header["thread_id"] == sample_context.thread_id
        assert header["client_name"] == "claude-code"
        assert header["client_version"] == "2.1.39"
        assert header["tool_name"] == "chat"
        assert header["parent_thread_id"] is None

    def test_header_preserves_initial_context(self, tmp_threads_dir, sample_context):
        save_thread_header(sample_context)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        with open(files[0]) as f:
            header = json.loads(f.readline())

        assert header["initial_context"] == {"prompt": "hello"}

    def test_header_preserves_parent_thread_id(self, tmp_threads_dir):
        ctx = ThreadContext(
            thread_id="aaaaaaaa-0000-0000-0000-000000000000",
            parent_thread_id="bbbbbbbb-1111-1111-1111-111111111111",
            created_at="2026-02-12T10:00:00+00:00",
            last_updated_at="2026-02-12T10:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
        )
        save_thread_header(ctx)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        with open(files[0]) as f:
            header = json.loads(f.readline())

        assert header["parent_thread_id"] == "bbbbbbbb-1111-1111-1111-111111111111"


class TestAppendTurn:
    def test_appends_user_turn(self, tmp_threads_dir, sample_context, sample_turn_user):
        save_thread_header(sample_context)
        append_turn(sample_context.thread_id, sample_context.created_at, sample_turn_user)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 turn

        turn_data = json.loads(lines[1])
        assert turn_data["type"] == "turn"
        assert turn_data["role"] == "user"
        assert turn_data["content"] == "Hello, how are you?"
        assert turn_data["files"] == ["server.py"]

    def test_appends_assistant_turn_with_metadata(self, tmp_threads_dir, sample_context, sample_turn_assistant):
        save_thread_header(sample_context)
        append_turn(sample_context.thread_id, sample_context.created_at, sample_turn_assistant)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        turn_data = json.loads(lines[1])

        assert turn_data["model_name"] == "gemini-2.5-pro"
        assert turn_data["model_provider"] == "google"
        assert turn_data["model_metadata"]["usage"]["input"] == 142

    def test_creates_retroactive_header_when_no_file(self, tmp_threads_dir, sample_turn_user):
        """When JSONL file is missing, append_turn creates a retroactive header."""
        tid = "retroactive-0000-0000-0000-000000000000"
        created = "2026-02-12T10:00:00+00:00"
        append_turn(tid, created, sample_turn_user)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2  # retroactive header + turn
        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert header["thread_id"] == tid

    def test_omits_none_optional_fields(self, tmp_threads_dir, sample_context):
        save_thread_header(sample_context)
        bare_turn = ConversationTurn(
            role="user",
            content="just text",
            timestamp="2026-02-12T15:00:00+00:00",
        )
        append_turn(sample_context.thread_id, sample_context.created_at, bare_turn)

        files = list(tmp_threads_dir.glob("*.jsonl"))
        lines = files[0].read_text().strip().split("\n")
        turn_data = json.loads(lines[1])

        assert "files" not in turn_data
        assert "images" not in turn_data
        assert "model_name" not in turn_data


class TestLoadThreadFromDisk:
    def test_cold_reload(self, tmp_threads_dir, sample_context, sample_turn_user, sample_turn_assistant):
        save_thread_header(sample_context)
        append_turn(sample_context.thread_id, sample_context.created_at, sample_turn_user)
        append_turn(sample_context.thread_id, sample_context.created_at, sample_turn_assistant)

        loaded = load_thread_from_disk(sample_context.thread_id)
        assert loaded is not None
        assert loaded.thread_id == sample_context.thread_id
        assert loaded.client_name == "claude-code"
        assert len(loaded.turns) == 2
        assert loaded.turns[0].role == "user"
        assert loaded.turns[1].model_name == "gemini-2.5-pro"

    def test_cold_reload_preserves_initial_context(self, tmp_threads_dir, sample_context):
        save_thread_header(sample_context)

        loaded = load_thread_from_disk(sample_context.thread_id)
        assert loaded is not None
        assert loaded.initial_context == {"prompt": "hello"}

    def test_returns_none_for_unknown_id(self, tmp_threads_dir):
        assert load_thread_from_disk("00000000-0000-0000-0000-000000000000") is None

    def test_returns_none_for_empty_file(self, tmp_threads_dir):
        # Create an empty file that matches the pattern
        tid = "deadbeef-0000-0000-0000-000000000000"
        (tmp_threads_dir / f"20260212-000000-{tid}.jsonl").write_text("")
        assert load_thread_from_disk(tid) is None

    def test_validates_thread_id_on_reload(self, tmp_threads_dir):
        """Cold-reload rejects files where header thread_id doesn't match requested ID."""
        wrong_tid = "aaaaaaaa-0000-0000-0000-000000000000"
        right_tid = "bbbbbbbb-0000-0000-0000-000000000000"
        # Manually create a file that looks like it matches wrong_tid but has right_tid inside
        # This simulates the 8-char prefix fallback finding a wrong match
        header = {"type": "header", "thread_id": right_tid, "created_at": "2026-01-01T00:00:00+00:00"}
        # File named after wrong_tid (simulating old 8-char prefix match)
        filepath = tmp_threads_dir / f"20260101-000000-{wrong_tid[:8]}.jsonl"
        filepath.write_text(json.dumps(header) + "\n")

        # Load with wrong_tid — should fail because header has right_tid
        assert load_thread_from_disk(wrong_tid) is None


class TestLoadAllThreadsFromDisk:
    def test_loads_multiple_threads(self, tmp_threads_dir, sample_context, sample_turn_user):
        # Create two threads
        save_thread_header(sample_context)
        append_turn(sample_context.thread_id, sample_context.created_at, sample_turn_user)

        ctx2 = ThreadContext(
            thread_id="aaaaaaaa-0000-0000-0000-000000000000",
            created_at="2026-02-12T15:00:00+00:00",
            last_updated_at="2026-02-12T15:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
            client_name="gemini-cli",
        )
        save_thread_header(ctx2)

        all_threads = load_all_threads_from_disk()
        assert len(all_threads) == 2
        ids = {t.thread_id for t in all_threads}
        assert sample_context.thread_id in ids
        assert ctx2.thread_id in ids


class TestListThreadFiles:
    def test_lists_newest_first(self, tmp_threads_dir):
        (tmp_threads_dir / "20260101-000000-aaaa.jsonl").write_text("{}")
        (tmp_threads_dir / "20260201-000000-bbbb.jsonl").write_text("{}")
        (tmp_threads_dir / "20260212-000000-cccc.jsonl").write_text("{}")

        files = list_thread_files()
        names = [f.name for f in files]
        assert names[0].startswith("20260212")
        assert names[-1].startswith("20260101")


# ---------------------------------------------------------------------------
# markdown_export tests
# ---------------------------------------------------------------------------


class TestYamlEscape:
    def test_plain_string_unchanged(self):
        assert _yaml_escape("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_colon_gets_quoted(self):
        assert _yaml_escape("key: value") == '"key: value"'

    def test_brackets_get_quoted(self):
        assert _yaml_escape("[test]") == '"[test]"'

    def test_quotes_get_escaped(self):
        assert _yaml_escape('say "hello"') == '"say \\"hello\\""'

    def test_comma_gets_quoted(self):
        assert _yaml_escape("a, b") == '"a, b"'


class TestYamlList:
    def test_empty(self):
        assert _yaml_list([]) == "[]"

    def test_simple_items(self):
        assert _yaml_list(["a", "b"]) == "[a, b]"

    def test_items_with_special_chars(self):
        result = _yaml_list(["normal", "has: colon", "has, comma"])
        assert result == '[normal, "has: colon", "has, comma"]'


class TestFormatDuration:
    def test_minutes_and_seconds(self):
        assert _format_duration("2026-02-12T14:30:00+00:00", "2026-02-12T14:42:34+00:00") == "12m 34s"

    def test_hours(self):
        assert _format_duration("2026-02-12T10:00:00+00:00", "2026-02-12T12:30:00+00:00") == "2h 30m"

    def test_zero(self):
        assert _format_duration("2026-02-12T10:00:00+00:00", "2026-02-12T10:00:00+00:00") == "0s"

    def test_bad_input(self):
        assert _format_duration("bad", "bad") == "unknown"


class TestAggregateMetadata:
    def test_collects_models_and_tokens(self, context_with_turns):
        meta = _aggregate_metadata(context_with_turns)
        assert meta["models_used"] == ["gemini-2.5-pro"]
        assert meta["providers_used"] == ["google"]
        assert meta["files_referenced"] == ["server.py"]
        assert meta["tokens_in"] == 142
        assert meta["tokens_out"] == 87

    def test_empty_turns(self, sample_context):
        meta = _aggregate_metadata(sample_context)
        assert meta["models_used"] == []
        assert meta["tokens_in"] == 0


class TestRenderThreadMarkdown:
    def test_has_yaml_frontmatter(self, context_with_turns):
        md = render_thread_markdown(context_with_turns)
        assert md.startswith("---\n")
        assert "\ntype: vox-thread\n" in md
        assert "\nproject: vox\n" in md
        assert "\nhas_memo: false\n" in md

    def test_frontmatter_fields(self, context_with_turns):
        md = render_thread_markdown(context_with_turns)
        assert "thread_id: cf23ea29" in md
        assert "client: claude-code v2.1.39" in md
        assert "turns: 2" in md
        assert "models_used: [gemini-2.5-pro]" in md

    def test_turn_rendering(self, context_with_turns):
        md = render_thread_markdown(context_with_turns)
        assert "### Turn 1 — User" in md
        assert "### Turn 2 — gemini-2.5-pro (google)" in md
        assert "> Hello, how are you?" in md
        assert "I'm doing well!" in md

    def test_files_in_turns(self, context_with_turns):
        md = render_thread_markdown(context_with_turns)
        assert "**Files:** `server.py`" in md

    def test_summary_table(self, context_with_turns):
        md = render_thread_markdown(context_with_turns)
        assert "| **Client** | claude-code v2.1.39 |" in md
        assert "| **Turns** | 2 |" in md

    def test_special_chars_in_files_escaped_in_frontmatter(self):
        """File paths with special YAML chars are properly escaped in frontmatter."""
        ctx = ThreadContext(
            thread_id="aaaaaaaa-0000-0000-0000-000000000000",
            created_at="2026-02-12T10:00:00+00:00",
            last_updated_at="2026-02-12T10:00:00+00:00",
            tool_name="chat",
            turns=[
                ConversationTurn(
                    role="user",
                    content="test",
                    timestamp="2026-02-12T10:00:00+00:00",
                    files=["path/to: file.py", "normal.py"],
                ),
            ],
            initial_context={},
        )
        md = render_thread_markdown(ctx)
        # The colon-containing path should be quoted in frontmatter
        assert '"path/to: file.py"' in md


class TestExportThreadToFile:
    def test_writes_file(self, context_with_turns, tmp_path):
        path = export_thread_to_file(context_with_turns, output_dir=tmp_path)
        assert os.path.exists(path)
        assert path.endswith(".md")
        content = Path(path).read_text()
        assert content.startswith("---\n")
        assert "vox-thread" in content
