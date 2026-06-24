# Vox MCP Development Guide

## Design Principles

- **Pure API passthrough** — prompts go to providers unmodified, responses come back unmodified
- **No system prompt injection** — `system_prompt` is always `None` at the provider boundary
- **No response modification** — no formatting, no coaching, no behavioral directives
- **No fabricated sampling params** — `temperature` is omitted unless the caller sets one, so the model uses its own server-side default
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
- `VOX_GEMINI_USE_INTERACTIONS` — when `true` (default), Gemini uses the stateless Interactions API; `false` forces `generateContent`
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

Temperature is a **pure passthrough**: vox never fabricates a default. When the caller omits
`temperature` it is omitted from the provider request (the model uses its own default); an
explicit value is validated/clamped per the model's constraint. In `openai_compatible.py`,
"model supports sampling params" (gates `max_tokens`/`top_p` for o3/o4) is decoupled from
"send temperature" (only when explicitly resolved). Guard: `tests/test_temperature_passthrough.py`.

## Concurrency

The MCP low-level server dispatches each tool call concurrently (anyio `tg.start_soon`). Provider
SDK calls are synchronous/blocking, so `tools/simple/base.py` runs `provider.generate_content` via
`asyncio.to_thread` — otherwise a multi-second call freezes the single event loop (incl. the stdio
streams) and concurrent calls stall until the client drops the connection. Lazy client init is
guarded by a `threading.Lock` (double-checked) since cached providers now run in worker threads.
Guard: `tests/test_concurrency_event_loop.py`.

## Provider-Specific Notes

### Gemini (Interactions API)
- Default path is the stateless **Interactions API** (`client.interactions.create`, `store=False`) —
  Google's GA surface. vox keeps owning conversation memory (full prompt is the interaction `input`).
- Falls back to `generateContent` (legacy but fully supported) on any failure; image inputs always
  use `generateContent`. Toggle with `VOX_GEMINI_USE_INTERACTIONS` (default on).
- Gemini 3 uses an enum `thinking_level` (valid on google-genai ≥2.x). Interactions accepts only
  `low`/`high` for Gemini 2.x and `low`/`medium`/`high` for Gemini 3 (`minimal` is Flash-only).

### Moonshot (Kimi K2.6)
- Thinking mode requires `extra_body={'thinking': {'type': 'enabled'}}`
- Temperature is not sent — Kimi K2 thinking ignores it ("not modifiable — do not pass explicitly"); `moonshot.py` always omits it
- API endpoint: `api.moonshot.cn/v1`
- `kimi-k2-thinking-turbo` was removed in v0.3.0 (deprecated upstream)

### DeepSeek (V4 Pro)
- Thinking mode requires `extra_body={'thinking': {'type': 'enabled'}}` (single endpoint with toggle, defaults on)
- Temperature is ignored when thinking is enabled
- API endpoint: `api.deepseek.com`
