"""
Thread Persistence — append-only JSONL storage for conversation threads.

Each thread gets one JSONL file: YYYYMMDD-HHMMSS-<full-uuid>.jsonl
Line 1 is a header with thread metadata; subsequent lines are individual turns
appended as they arrive. This shadow layer runs alongside in-memory storage
for durability, archival, and cold-reload after memory expiry.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.conversation_memory import ConversationTurn, ThreadContext

logger = logging.getLogger(__name__)


def _get_threads_dir() -> Path:
    """Get the configured threads directory, creating it if needed."""
    from config import VOX_THREADS_DIR

    VOX_THREADS_DIR.mkdir(parents=True, exist_ok=True)
    return VOX_THREADS_DIR


def _get_filename(thread_id: str, created_at: str) -> str:
    """Build filename from thread metadata: YYYYMMDD-HHMMSS-<full-uuid>.jsonl"""
    try:
        dt = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        dt = datetime.now(timezone.utc)
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{thread_id}.jsonl"


def _parse_jsonl_file(filepath: Path) -> Optional[tuple[dict, list[ConversationTurn], Optional[str]]]:
    """Parse a JSONL thread file into (header, turns, last_timestamp).

    Returns None if no valid header is found. Malformed lines are silently skipped.
    """
    header = None
    turns: list[ConversationTurn] = []
    last_timestamp: Optional[str] = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") == "header":
                header = record
            elif record.get("type") == "turn":
                turns.append(
                    ConversationTurn(
                        role=record["role"],
                        content=record["content"],
                        timestamp=record.get("timestamp", ""),
                        files=record.get("files"),
                        images=record.get("images"),
                        tool_name=record.get("tool_name"),
                        model_provider=record.get("model_provider"),
                        model_name=record.get("model_name"),
                        model_metadata=record.get("model_metadata"),
                    )
                )
                last_timestamp = record.get("timestamp", last_timestamp)

    if not header:
        return None
    return header, turns, last_timestamp


def _header_to_context(header: dict, turns: list[ConversationTurn], last_timestamp: Optional[str]) -> ThreadContext:
    """Build a ThreadContext from parsed JSONL header and turns."""
    return ThreadContext(
        thread_id=header["thread_id"],
        parent_thread_id=header.get("parent_thread_id"),
        created_at=header["created_at"],
        last_updated_at=last_timestamp or header["created_at"],
        tool_name=header.get("tool_name", "chat"),
        turns=turns,
        initial_context=header.get("initial_context", {}),
        client_name=header.get("client_name"),
        client_version=header.get("client_version"),
    )


def save_thread_header(context: ThreadContext) -> None:
    """Write the JSONL header line when a thread is created."""
    try:
        threads_dir = _get_threads_dir()
        filename = _get_filename(context.thread_id, context.created_at)
        filepath = threads_dir / filename

        header = {
            "type": "header",
            "thread_id": context.thread_id,
            "created_at": context.created_at,
            "tool_name": context.tool_name,
            "client_name": context.client_name,
            "client_version": context.client_version,
            "parent_thread_id": context.parent_thread_id,
            "initial_context": context.initial_context,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")

        logger.debug(f"[PERSIST] Wrote header for thread {context.thread_id[:8]} -> {filename}")
    except Exception as e:
        logger.warning(f"[PERSIST] Failed to write thread header: {e}")


def append_turn(thread_id: str, created_at: str, turn: ConversationTurn) -> None:
    """Append a single turn line to the thread's JSONL file."""
    try:
        threads_dir = _get_threads_dir()
        filename = _get_filename(thread_id, created_at)
        filepath = threads_dir / filename

        if not filepath.exists():
            # Retroactively create header so turns are not silently lost.
            # Try to enrich with full metadata from memory if available.
            logger.info(f"[PERSIST] JSONL file not found for {thread_id[:8]}, creating retroactively")
            header: dict = {"type": "header", "thread_id": thread_id, "created_at": created_at}
            try:
                from utils.conversation_memory import get_storage

                storage = get_storage()
                data = storage.get(f"thread:{thread_id}")
                if data:
                    ctx_dict = json.loads(data)
                    for key in ("tool_name", "client_name", "client_version", "parent_thread_id", "initial_context"):
                        if key in ctx_dict and ctx_dict[key] is not None:
                            header[key] = ctx_dict[key]
            except Exception:
                pass
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(header, ensure_ascii=False) + "\n")

        turn_data = {
            "type": "turn",
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp,
        }

        # Optional fields — only include when present
        if turn.files:
            turn_data["files"] = turn.files
        if turn.images:
            turn_data["images"] = turn.images
        if turn.tool_name:
            turn_data["tool_name"] = turn.tool_name
        if turn.model_provider:
            turn_data["model_provider"] = turn.model_provider
        if turn.model_name:
            turn_data["model_name"] = turn.model_name
        if turn.model_metadata:
            turn_data["model_metadata"] = turn.model_metadata

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")

        logger.debug(f"[PERSIST] Appended {turn.role} turn to {filename}")
    except Exception as e:
        logger.warning(f"[PERSIST] Failed to append turn: {e}")


def load_thread_from_disk(thread_id: str) -> Optional[ThreadContext]:
    """Cold-reload a thread from its JSONL file by full UUID match."""
    try:
        threads_dir = _get_threads_dir()

        # Match by full UUID in filename
        matches = sorted(threads_dir.glob(f"*-{thread_id}.jsonl"), reverse=True)
        if not matches:
            # Fallback: try 8-char prefix for files created before full-UUID migration
            short_id = thread_id[:8]
            matches = sorted(threads_dir.glob(f"*-{short_id}.jsonl"), reverse=True)
            if not matches:
                logger.debug(f"[PERSIST] No JSONL file found for thread {thread_id[:8]}")
                return None

        filepath = matches[0]
        logger.debug(f"[PERSIST] Cold-loading thread from {filepath.name}")

        parsed = _parse_jsonl_file(filepath)
        if not parsed:
            logger.warning(f"[PERSIST] No header found in {filepath.name}")
            return None

        header, turns, last_timestamp = parsed

        # Validate thread_id matches to prevent returning wrong thread
        if header.get("thread_id") != thread_id:
            logger.warning(f"[PERSIST] Thread ID mismatch in {filepath.name}: expected {thread_id[:8]}, got {header.get('thread_id', 'none')[:8]}")
            return None

        context = _header_to_context(header, turns, last_timestamp)
        logger.info(f"[PERSIST] Cold-loaded thread {thread_id[:8]} with {len(turns)} turns from disk")
        return context

    except Exception as e:
        logger.warning(f"[PERSIST] Failed to load thread from disk: {e}")
        return None


def list_thread_files() -> list[Path]:
    """List all JSONL thread files, newest first."""
    try:
        threads_dir = _get_threads_dir()
        files = sorted(threads_dir.glob("*.jsonl"), reverse=True)
        return files
    except Exception as e:
        logger.warning(f"[PERSIST] Failed to list thread files: {e}")
        return []


def load_all_threads_from_disk() -> list[ThreadContext]:
    """Load all threads from disk. Used by dump_threads to include expired threads."""
    contexts = []
    for filepath in list_thread_files():
        try:
            parsed = _parse_jsonl_file(filepath)
            if not parsed:
                continue
            header, turns, last_timestamp = parsed
            contexts.append(_header_to_context(header, turns, last_timestamp))
        except Exception as e:
            logger.debug(f"[PERSIST] Skipping {filepath.name}: {e}")
            continue

    return contexts
