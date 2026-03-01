# Vox MCP Development Guide

## Design Principles

- **Pure API passthrough** — prompts go to providers unmodified, responses come back unmodified
- **No system prompt injection** — `system_prompt` is always `None` at the provider boundary
- **No response modification** — no formatting, no coaching, no behavioral directives
- **Conversation memory is the sole value-add** — in-memory threads via `continuation_id`

## Quick Commands

```bash
uv sync                          # Install dependencies
uv run python server.py          # Run server (stdio MCP)
uv run python -c "import server" # Smoke test
uv run pytest                    # Run tests
```

## Tool Surface

| Tool | File | Purpose |
|------|------|---------|
| `chat` | `tools/chat.py` | Multi-model AI gateway with file/image context |
| `listmodels` | `tools/listmodels.py` | Show available models and capabilities |
| `dump_threads` | `tools/dump_threads.py` | Export threads as JSON or Markdown |

## File Structure

- `server.py` — MCP server, tool registration, request handling
- `config.py` — shared constants (temperatures, limits, env defaults)
- `providers/` — model provider implementations (gemini, openai, anthropic, deepseek, moonshot, xai, openrouter, custom)
- `providers/registry.py` — provider resolution and model routing
- `providers/shared/temperature.py` — TemperatureConstraint ABC (Fixed, Range, Discrete)
- `providers/shared/thinking.py` — ThinkingConstraint ABC (TokenBudget, EffortLevel, AlwaysOn)
- `providers/shared/model_capabilities.py` — ModelCapabilities dataclass with constraint fields
- `conf/` — provider model capability JSON files
- `tools/chat.py` — chat tool and ChatRequest model
- `tools/shared/base_tool.py` — BaseTool with model resolution, token budgeting, file handling
- `tools/shared/base_models.py` — ToolRequest base model, shared field descriptions
- `tools/simple/base.py` — SimpleTool (request -> model -> response pattern)
- `utils/conversation_memory.py` — thread creation, turn management, file tracking
- `utils/storage_backend.py` — in-memory storage with TTL
- `utils/thread_persistence.py` — append-only JSONL shadow persistence, cold-reload from disk
- `utils/markdown_export.py` — markdown rendering with YAML frontmatter
- `utils/model_context.py` — token budget calculation per model
- `tests/` — unit tests

## Key Environment Variables

- `GEMINI_API_KEY`, `OPENAI_API_KEY`, etc. — provider API keys (at least one required)
- `DEFAULT_MODEL` — `auto` (default) or a specific model name
- `VOX_FORCE_ENV_OVERRIDE` — when `true`, `.env` values override system env vars
- `CONVERSATION_TIMEOUT_HOURS` — thread TTL (default: 24)
- `MAX_CONVERSATION_TURNS` — thread length limit (default: 100)
- `VOX_THREADS_DIR` — durable thread storage path (default: `~/.vox/threads/`)

## Constraint Architecture

Temperature and thinking parameters use parallel constraint abstractions on `ModelCapabilities`.

**Temperature** (`providers/shared/temperature.py`):
- `FixedTemperatureConstraint` — models locked to a single value
- `RangeTemperatureConstraint` — continuous min/max clamping
- `DiscreteTemperatureConstraint` — specific allowed values

**Thinking** (`providers/shared/thinking.py`):
- `TokenBudgetThinkingConstraint` — maps `thinking_mode` to token counts (Gemini, Anthropic)
- `EffortLevelThinkingConstraint` — maps `thinking_mode` to effort strings (OpenAI)
- `AlwaysOnThinkingConstraint` — thinking is inherent to the model (DeepSeek, Moonshot)

## Provider-Specific Notes

### Moonshot (Kimi K2.5, K2 Thinking)
- Thinking mode requires `extra_body={'thinking': {'type': 'enabled'}}`
- Temperature must be 1.0 for thinking mode
- API endpoint: `api.moonshot.cn/v1`
