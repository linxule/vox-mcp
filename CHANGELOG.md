# Changelog

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
