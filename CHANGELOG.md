# Changelog

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
