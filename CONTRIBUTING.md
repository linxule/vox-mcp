# Contributing to Vox MCP

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/linxule/vox-mcp.git
cd vox-mcp
uv sync
cp .env.example .env
# Add at least one API key to .env
```

Verify everything works:

```bash
uv run python -c "import server"   # smoke test
uv run pytest                       # run tests (321 tests)
```

## Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

- Line length: 120
- Target: Python 3.10+
- Rules: E, W, F, I, B, C4, UP

Run before submitting:

```bash
uv run ruff check .
uv run ruff format .
```

## Project structure

- `server.py` — MCP server entry point and tool registration
- `providers/` — one module per provider (gemini, openai, anthropic, etc.)
- `tools/` — MCP tool implementations (chat, listmodels, dump_threads)
- `utils/` — conversation memory, thread persistence, file handling
- `conf/` — model capability JSON configs per provider
- `tests/` — unit tests

## Adding a new provider

1. Create `providers/your_provider.py` implementing the provider interface
2. Add model capabilities to `conf/your_provider_models.json`
3. Register in `server.py` with the appropriate API key check
4. Add the env variable to `.env.example`
5. Add tests

## Submitting changes

1. Fork the repo and create a branch
2. Make your changes
3. Run `uv run pytest` — all tests must pass
4. Run `uv run ruff check . && uv run ruff format .`
5. Open a pull request with a clear description of what and why

## Reporting bugs

Open an issue with:
- What you expected vs what happened
- Your Python version and OS
- Which MCP client you're using
- Relevant log output from `logs/mcp_server.log`

## Design principles

Before contributing, understand the core philosophy:

- **Pure API passthrough** — prompts go to providers unmodified
- **No system prompt injection** — ever
- **No response modification** — no formatting, coaching, or directives
- **Conversation memory is the sole value-add**

If a proposed change would violate these principles, it won't be merged.
