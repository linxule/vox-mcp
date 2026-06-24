# Changelog

## 0.4.0 — Temperature passthrough, Gemini Interactions API, concurrency fix

### Behavior changes

- **Temperature is now a pure passthrough.** Vox no longer fabricates a default temperature (previously `1.0`) — when the caller omits `temperature` it is omitted from the provider request so the model applies its own server-side default. This aligns with 2026 provider guidance (Gemini 3 degrades below 1.0; OpenAI reasoning and Anthropic Opus 4.7+ reject non-default values; Kimi/DeepSeek reasoners ignore it). An explicit temperature is still validated and clamped per the model's constraint. **Kimi K2 thinking now omits temperature entirely** (Moonshot guidance: "not modifiable — do not pass it explicitly") rather than sending `1.0`, correcting the 0.3.0 note. The MCP temperature schema ceiling is raised `0–1` → `0–2` to match provider capability (per-model clamping still applies).
- **Gemini uses the stateless Interactions API by default.** Gemini requests now go through Google's GA Interactions API (`store=false`); vox keeps owning conversation memory. On any failure it falls back to the (fully supported) `generateContent` endpoint and skips Interactions for the rest of the session. Set `VOX_GEMINI_USE_INTERACTIONS=false` to force `generateContent`. Gemini 3 `thinking_level` (minimal/low/medium/high) is now supported; image inputs use `generateContent`.

### Fixes

- **Concurrent tool calls no longer drop the MCP connection.** The blocking provider call is offloaded via `asyncio.to_thread`, keeping the event loop (and stdio streams) responsive under the MCP SDK's concurrent request dispatch. Lazy provider-client initialization is now thread-safe (double-checked locking).

### Dependencies & CI

- Bumped `google-genai` 1.x → 2.x (required for the Interactions API and the corrected Gemini 3 `thinking_level` enum), plus `mcp`, `openai`, `anthropic`, and `pydantic` to current.
- Added a verify CI workflow (ruff lint + format check + pytest on Python 3.10 & 3.13) and grouped Dependabot.

## 0.3.0 — Drop deprecated kimi-k2-thinking-turbo

### Breaking

- **Removed `kimi-k2-thinking-turbo`** (deprecated upstream by Moonshot). The aliases `kimi` and `kimi-k2` now resolve to `kimi-k2.6`, so callers using `model: kimi` continue to work but now hit the multimodal K2.6 endpoint (vision-capable, `temperature=1.0` enforced via `FixedTemperatureConstraint`). Callers that depended on the wider temperature range (`0.0–1.0`) on the old turbo model will have temperature clamped to `1.0`.

## 0.2.0 — Model swap: Kimi K2.6 + DeepSeek V4 Pro

### Breaking

- **Removed `kimi-k2.5`** (and its aliases `k2.5`, `kimi-k25`). Replaced by `kimi-k2.6` (256K context, 64K output, multimodal with vision, always-on thinking). Aliases: `k2.6`, `kimi-k26`. Thinking mode requires `temperature=1.0` (now enforced via `FixedTemperatureConstraint`).
- **Removed `deepseek-reasoner`** (and its aliases `deepseek-r1`, `ds-reasoner`, `r1`). Replaced by `deepseek-v4-pro` (1M context, 384K output, always-on thinking). Aliases: `deepseek`, `deepseek-v4`, `v4`, `v4-pro`. Provider now injects `extra_body={"thinking": {"type": "enabled"}}` since V4 Pro is a single endpoint with a thinking toggle, unlike R1.

### Notes

- OpenRouter routing is unaffected; R1 references in `_TEMP_UNSUPPORTED_PATTERNS` retained for OpenRouter's `deepseek/deepseek-r1-*` models.

## 0.1.0 — Initial public release

First public release of Vox MCP, extracted from a private development repository.

### Features

- **8 providers**: Google Gemini, OpenAI, Anthropic, xAI (Grok), DeepSeek, Moonshot (Kimi), OpenRouter, and custom OpenAI-compatible endpoints (Ollama, vLLM, LM Studio)
- **3 tools**: `chat` (multi-model gateway with file/image context), `listmodels` (model discovery), `dump_threads` (thread export as JSON or Markdown)
- **Conversation memory**: in-memory threads with `continuation_id` for multi-turn exchanges across any provider
- **Thread persistence**: append-only JSONL shadow persistence with cold-reload from disk
- **Markdown export**: thread export with YAML frontmatter
- **Thinking mode support**: per-model constraint architecture (token budgets, effort levels, always-on reasoning)
- **Temperature constraints**: fixed, range, and discrete temperature handling per model
- **Model restrictions**: per-provider allowlists via environment variables
- **Auto mode**: agent-driven model selection when `DEFAULT_MODEL=auto`

### Architecture

- Pure API passthrough — no system prompt injection, no response modification
- Provider-independent model capability system via JSON configs in `conf/`
- Each provider registers independently based on its own API key presence
