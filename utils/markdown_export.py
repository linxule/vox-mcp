"""
Markdown Export — render conversation threads as clean markdown with YAML frontmatter.

Produces memex-compatible files with frontmatter fields that work with
Obsidian Dataview queries. Each exported thread gets its own .md file.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.conversation_memory import ThreadContext

logger = logging.getLogger(__name__)


def _format_duration(start_iso: str, end_iso: str) -> str:
    """Human-readable duration between two ISO timestamps."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        delta = end - start
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "0s"
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        return " ".join(parts)
    except (ValueError, TypeError):
        return "unknown"


def _aggregate_metadata(context: ThreadContext) -> dict[str, Any]:
    """Extract aggregate stats from a thread's turns."""
    models_used: list[str] = []
    providers_used: list[str] = []
    files_referenced: list[str] = []
    tokens_in = 0
    tokens_out = 0
    seen_models: set[str] = set()
    seen_providers: set[str] = set()
    seen_files: set[str] = set()

    for turn in context.turns:
        if turn.model_name and turn.model_name not in seen_models:
            seen_models.add(turn.model_name)
            models_used.append(turn.model_name)
        if turn.model_provider and turn.model_provider not in seen_providers:
            seen_providers.add(turn.model_provider)
            providers_used.append(turn.model_provider)
        if turn.files:
            for f in turn.files:
                if f not in seen_files:
                    seen_files.add(f)
                    files_referenced.append(f)
        if turn.model_metadata and isinstance(turn.model_metadata, dict):
            usage = turn.model_metadata.get("usage", {})
            if isinstance(usage, dict):
                tokens_in += usage.get("input", 0) or 0
                tokens_out += usage.get("output", 0) or 0

    # Duration
    last_ts = context.turns[-1].timestamp if context.turns else context.created_at
    duration_seconds = 0
    try:
        start = datetime.fromisoformat(context.created_at)
        end = datetime.fromisoformat(last_ts)
        duration_seconds = max(0, int((end - start).total_seconds()))
    except (ValueError, TypeError):
        pass

    return {
        "models_used": models_used,
        "providers_used": providers_used,
        "files_referenced": files_referenced,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "duration_seconds": duration_seconds,
    }


def _yaml_escape(value: str) -> str:
    """Escape a string for safe YAML inline inclusion."""
    if any(c in value for c in ':[]{},\n"\'#&*!|>\\'):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return value


def _yaml_list(items: list[str]) -> str:
    """Format a list for YAML frontmatter inline notation with proper escaping."""
    if not items:
        return "[]"
    return "[" + ", ".join(_yaml_escape(item) for item in items) + "]"


def render_thread_markdown(context: ThreadContext) -> str:
    """Render a thread as markdown with YAML frontmatter for memex compatibility."""
    meta = _aggregate_metadata(context)
    short_id = context.thread_id[:8]

    # Date for frontmatter (YYYY-MM-DD)
    try:
        date_str = datetime.fromisoformat(context.created_at).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Client string
    client_str = context.client_name or "unknown"
    if context.client_version:
        client_str += f" v{context.client_version}"

    last_ts = context.turns[-1].timestamp if context.turns else context.created_at
    duration_str = _format_duration(context.created_at, last_ts)

    # Build YAML frontmatter
    lines = [
        "---",
        "type: vox-thread",
        f"thread_id: {context.thread_id}",
        f"date: {date_str}",
        f"created_at: {context.created_at}",
        f"client: {_yaml_escape(client_str)}",
        f"turns: {len(context.turns)}",
        f"models_used: {_yaml_list(meta['models_used'])}",
        f"providers_used: {_yaml_list(meta['providers_used'])}",
        f"files_referenced: {_yaml_list(meta['files_referenced'])}",
        f"tokens_in: {meta['tokens_in']}",
        f"tokens_out: {meta['tokens_out']}",
        f"duration_seconds: {meta['duration_seconds']}",
        "project: vox",
        "topics: []",
        "has_memo: false",
        "---",
        "",
        f"# Thread {short_id} — {context.tool_name}",
        "",
    ]

    # Summary table
    models_display = ", ".join(
        f"{m} ({p})" if p else m
        for m, p in _zip_models_providers(meta["models_used"], meta["providers_used"])
    )
    files_display = ", ".join(meta["files_referenced"]) if meta["files_referenced"] else "none"
    token_display = f"{meta['tokens_in']:,} in / {meta['tokens_out']:,} out" if (meta["tokens_in"] or meta["tokens_out"]) else "not tracked"

    lines.extend([
        "| Field | Value |",
        "|-------|-------|",
        f"| **Client** | {client_str} |",
        f"| **Created** | {context.created_at} |",
        f"| **Duration** | {duration_str} |",
        f"| **Turns** | {len(context.turns)} |",
        f"| **Models** | {models_display or 'none'} |",
        f"| **Tokens** | {token_display} |",
        f"| **Files** | {files_display} |",
        "",
    ])

    # Render turns
    for i, turn in enumerate(context.turns, 1):
        if turn.role == "user":
            role_label = "User"
        elif turn.model_name:
            provider_suffix = f" ({turn.model_provider})" if turn.model_provider else ""
            role_label = f"{turn.model_name}{provider_suffix}"
        else:
            role_label = "Assistant"

        lines.append("---")
        lines.append("")
        lines.append(f"### Turn {i} — {role_label}")
        lines.append("")

        if turn.files:
            files_str = ", ".join(f"`{f}`" for f in turn.files)
            lines.append(f"**Files:** {files_str}")
            lines.append("")

        # Quote user messages, render assistant messages as-is
        if turn.role == "user":
            for content_line in turn.content.split("\n"):
                lines.append(f"> {content_line}")
        else:
            lines.append(turn.content)

        lines.append("")

    return "\n".join(lines)


def _zip_models_providers(models: list[str], providers: list[str]) -> list[tuple[str, Optional[str]]]:
    """Zip models with their providers, padding with None if lists differ in length."""
    result = []
    for i, model in enumerate(models):
        provider = providers[i] if i < len(providers) else None
        result.append((model, provider))
    return result


def export_thread_to_file(context: ThreadContext, output_dir: Optional[Path] = None) -> str:
    """Write a thread as a markdown file and return the file path."""
    if output_dir is None:
        from config import VOX_THREADS_DIR
        output_dir = VOX_THREADS_DIR / "exports"

    output_dir.mkdir(parents=True, exist_ok=True)

    short_id = context.thread_id[:8]
    try:
        dt = datetime.fromisoformat(context.created_at)
    except (ValueError, TypeError):
        dt = datetime.now(timezone.utc)
    stamp = dt.strftime("%Y%m%d-%H%M%S")
    filename = f"{stamp}-{short_id}.md"
    filepath = output_dir / filename

    content = render_thread_markdown(context)
    filepath.write_text(content, encoding="utf-8")

    logger.info(f"[EXPORT] Wrote markdown export to {filepath}")
    return str(filepath)
