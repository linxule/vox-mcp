"""
Dump Threads Tool - Export conversation threads as JSON or Markdown

Utility tool for extracting conversation threads. Supports filtering by
thread ID, JSON export (existing behavior), and clean markdown export
with YAML frontmatter for memex ingestion. No AI model required.
"""

import json
import logging
from typing import Any, Optional

from mcp.types import TextContent

from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


class DumpThreadsTool(BaseTool):
    """Export conversation threads from memory (or cold-reload from disk)."""

    def get_name(self) -> str:
        return "dump_threads"

    def get_description(self) -> str:
        return (
            "Export conversation threads as JSON or Markdown. "
            "Threads persist to disk and can be cold-reloaded after memory expiry. "
            "Use thread_ids to filter specific threads, format to choose output."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thread_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific thread UUIDs. Omit for all active threads.",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "default": "markdown",
                    "description": "Output format: 'markdown' (clean export with YAML frontmatter, written to disk) or 'json' (raw thread data, inline).",
                },
            },
            "required": [],
            "additionalProperties": False,
        }

    def get_annotations(self) -> Optional[dict[str, Any]]:
        return {"readOnlyHint": True}

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return ToolRequest

    def requires_model(self) -> bool:
        return False

    async def prepare_prompt(self, request: ToolRequest) -> str:
        return ""

    def format_response(self, response: str, request: ToolRequest, model_info: dict = None) -> str:
        return response

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Dump conversation threads with optional filtering and format selection."""
        from utils.conversation_memory import ThreadContext, get_thread
        from utils.storage_backend import get_storage_backend

        thread_ids = arguments.get("thread_ids")
        output_format = arguments.get("format", "markdown")

        storage = get_storage_backend()

        if thread_ids:
            # Filtered: resolve specific threads (memory first, then disk cold-reload)
            contexts: list[ThreadContext] = []
            missing: list[str] = []
            for tid in thread_ids:
                ctx = get_thread(tid)
                if ctx:
                    contexts.append(ctx)
                else:
                    missing.append(tid)

            if output_format == "markdown":
                return self._export_markdown(contexts, missing)
            else:
                return self._export_json_filtered(contexts, missing)
        else:
            # All threads: memory + disk-only (expired from memory)
            raw = storage.dump_all()

            # Collect memory-resident threads
            contexts: list[ThreadContext] = []
            seen_ids: set[str] = set()
            for key, value_json in raw.items():
                try:
                    ctx = ThreadContext.model_validate_json(value_json)
                    contexts.append(ctx)
                    seen_ids.add(ctx.thread_id)
                except Exception:
                    pass

            # Merge disk-only threads not in memory
            try:
                from utils.thread_persistence import load_all_threads_from_disk

                for disk_ctx in load_all_threads_from_disk():
                    if disk_ctx.thread_id not in seen_ids:
                        contexts.append(disk_ctx)
                        seen_ids.add(disk_ctx.thread_id)
            except Exception as e:
                logger.debug(f"[PERSIST] Failed to load disk threads: {e}")

            if output_format == "markdown":
                return self._export_markdown(contexts, [])
            else:
                return self._export_json_all_merged(contexts, raw)

    def _export_json_all_merged(self, contexts: list, raw: dict[str, str]) -> list[TextContent]:
        """Dump all threads (memory + disk) as JSON."""
        threads = {}
        for ctx in contexts:
            threads[f"thread:{ctx.thread_id}"] = json.loads(ctx.model_dump_json())

        # Count how many are disk-only
        memory_count = len(raw)
        disk_only_count = len(contexts) - memory_count

        tool_output = ToolOutput(
            status="success",
            content=json.dumps(threads, indent=2, ensure_ascii=False),
            content_type="json",
            metadata={
                "tool_name": self.name,
                "thread_count": len(threads),
                "memory_threads": memory_count,
                "disk_only_threads": max(0, disk_only_count),
            },
        )
        return [TextContent(type="text", text=tool_output.model_dump_json())]

    def _export_json_filtered(self, contexts: list, missing: list[str]) -> list[TextContent]:
        """Dump specific threads as JSON."""
        threads = {}
        for ctx in contexts:
            threads[f"thread:{ctx.thread_id}"] = json.loads(ctx.model_dump_json())

        metadata: dict[str, Any] = {
            "tool_name": self.name,
            "thread_count": len(threads),
        }
        if missing:
            metadata["missing_thread_ids"] = missing

        tool_output = ToolOutput(
            status="success",
            content=json.dumps(threads, indent=2, ensure_ascii=False),
            content_type="json",
            metadata=metadata,
        )
        return [TextContent(type="text", text=tool_output.model_dump_json())]

    def _export_markdown(self, contexts: list, missing: list[str]) -> list[TextContent]:
        """Export threads as markdown files with YAML frontmatter."""
        from utils.markdown_export import export_thread_to_file

        exported_paths: list[str] = []
        errors: list[str] = []

        for ctx in contexts:
            try:
                path = export_thread_to_file(ctx)
                exported_paths.append(path)
            except Exception as e:
                errors.append(f"{ctx.thread_id[:8]}: {e}")

        metadata: dict[str, Any] = {
            "tool_name": self.name,
            "exported_count": len(exported_paths),
            "exported_paths": exported_paths,
        }
        if missing:
            metadata["missing_thread_ids"] = missing
        if errors:
            metadata["errors"] = errors

        summary_lines = [f"Exported {len(exported_paths)} thread(s) to markdown:"]
        for p in exported_paths:
            summary_lines.append(f"  {p}")
        if missing:
            summary_lines.append(f"\nNot found: {', '.join(tid[:8] for tid in missing)}")

        tool_output = ToolOutput(
            status="success",
            content="\n".join(summary_lines),
            content_type="text",
            metadata=metadata,
        )
        return [TextContent(type="text", text=tool_output.model_dump_json())]

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.FAST_RESPONSE
