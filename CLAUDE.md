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
explicit value is validated/clamped per the model's constraint.
Guard: `tests/test_temperature_passthrough.py`.

### Request-parameter support is DECLARED, never inferred

`supports_temperature` used to do two unrelated jobs — it governed temperature *and* was read
as a proxy for "this model accepts `max_tokens` and the sampling penalties". Those are
different questions. A model can reject temperature and happily accept a token cap; a model
can accept temperature and reject penalties. The coupling meant o3's `max_tokens` was dropped
because of a flag about temperature.

Each model now declares its own exclusions in **`unsupported_params`**, and
`openai_compatible.py` filters `SAMPLING_PASSTHROUGH_PARAMS` against that list. Nothing is
inferred from anything else.

- **`None` means NOT DECLARED. `[]` means "declared: accepts everything else."** They behave
  identically on the wire (`or []` at the single read site) and differently to the gate. The
  default is `None` precisely so an omission cannot masquerade as a considered answer.
- A model with `supports_temperature=False` **must** declare `unsupported_params`.
  Enforced by `tests/test_config_integrity.py`. Guard: `tests/test_sampling_params.py`.

## Invariants

Three rules that are load-bearing, each one paid for by a bug that shipped.

### 1. The test suite must never touch the real `~/.vox`

The suite was writing into the user's **actual conversation store** — 151 test artifacts
accumulated alongside 976 real threads. Worse, one test was **self-concealing**: it failed
once on a clean machine, wrote the artifact that satisfied its own lookup, and was green
forever after. Only a fresh CI runner ever told the truth.

`tests/conftest.py` has an autouse fixture that repoints `config.VOX_THREADS_DIR` at a
`tmp_path`. **Patch `config.VOX_THREADS_DIR`, not a helper** — both `utils/thread_persistence.py`
and `utils/markdown_export.py` do a late `from config import VOX_THREADS_DIR` *inside* the
function, so patching either module's helper leaves the other one still writing to the home
directory. `tests/test_home_isolation.py` is the guard, and it includes a
`test_the_guard_is_not_vacuous`.

### 2. A model in the catalog is a promise vox can call it

`grok-build-0.1` sat in the xAI chat catalog while xAI documents it **only** on the Code API
(`/v1/responses`). vox advertised a model it could not call: the request passed validation and
would have died at the wire.

An ID that cannot be served is **worse than an absent one** — it fails late, at the provider,
opaquely, instead of early, at validation, legibly. Before adding a model, confirm the vendor
documents it on the endpoint the provider actually calls. A vendor's *model index* is not that
evidence; the endpoint's own page is.

Related: **never alias a retired ID onto a live model.** Silently redirecting `grok-4` →
`grok-4.5` is a lie about which model answered. A wrong ID that errors loudly teaches the
caller something; a silent substitution hands them plausible output from a model they did not
ask for and nobody ever finds out.

### 3. Models are declared THREE ways — a gate keyed on one of them is blind

| Mechanism | Providers | Class `MODEL_CAPABILITIES` |
|---|---|---|
| `conf/*_models.json` via `RegistryBackedProviderMixin` | gemini, openai, xai | populated |
| hardcoded Python dict in the provider module | anthropic, deepseek, moonshot | populated |
| instance-level registry, built in `__init__` | openrouter, custom | **empty** |

The first version of the integrity gate globbed `conf/*_models.json` — so it covered mechanisms
1 and 3 and was structurally blind to 2, while reporting green. DeepSeek and Moonshot are
`OpenAICompatibleProvider` subclasses whose wire payload *is* governed by `unsupported_params`,
and they got zero coverage. A naive rewrite keyed on the class dict would have flipped the blind
spot onto mechanism 3 instead.

**The discovery mechanism was itself the enumeration**, wearing a costume that looked like
discovery. `tests/test_config_integrity.py` now iterates providers and keys coverage on the
**`ProviderType` enum** — the authority for what exists — so a provider that yields no
capabilities *fails* rather than being skipped. If you add a fourth declaration mechanism, that
test is what tells you.

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
