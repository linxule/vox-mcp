# Vox MCP

Multi-model AI gateway for [MCP](https://modelcontextprotocol.io) clients. Routes prompts to external AI providers with conversation memory — no system prompt injection, no response modification, pure API passthrough.

## What it does

Vox gives any MCP client (Claude Code, Claude Desktop, Cursor, etc.) access to external AI models through a single `chat` tool. You send a prompt, optionally attach files or images, and get back the model's response. Conversation threads are maintained in memory via `continuation_id` for multi-turn exchanges.

**3 tools:**

| Tool | Description |
|------|-------------|
| `chat` | Send prompts to any configured AI model with optional file/image context |
| `listmodels` | Show available models, aliases, and capabilities |
| `dump_threads` | Export conversation threads as JSON or Markdown |

**8 providers:**

| Provider | Env Variable | Example Models |
|----------|-------------|----------------|
| Google Gemini | `GEMINI_API_KEY` | gemini-2.5-pro |
| OpenAI | `OPENAI_API_KEY` | gpt-5.1, gpt-5, o3, o4-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-4-opus, claude-4-sonnet |
| xAI | `XAI_API_KEY` | grok-3, grok-3-fast |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | kimi-k2-turbo, kimi-k2.5 |
| OpenRouter | `OPENROUTER_API_KEY` | Any OpenRouter model |
| Custom | `CUSTOM_API_URL` | Ollama, vLLM, LM Studio, etc. |

## Quick start

```bash
git clone https://github.com/linxule/vox-mcp.git
cd vox-mcp
cp .env.example .env
# Edit .env — add at least one API key
uv sync
uv run python server.py
```

## MCP client configuration

Vox runs as a stdio MCP server. Each client needs to know how to launch it.

Replace `/path/to/vox-mcp` with the absolute path to your cloned repo.

### Claude Code (CLI)

```bash
claude mcp add vox-mcp \
  -e GEMINI_API_KEY=your-key-here \
  -- uv run --directory /path/to/vox-mcp python server.py
```

Or add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "vox-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vox-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Claude Desktop

Add to `claude_desktop_config.json`:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "vox-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vox-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "vox-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vox-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "vox-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vox-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Any MCP client

The canonical stdio configuration:

```json
{
  "mcpServers": {
    "vox-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/vox-mcp", "python", "server.py"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

**Tips:**
- Paths must be absolute
- You only need one API key to start — add more providers later via `.env`
- The `.env` file in the vox-mcp directory is loaded automatically, so API keys can go there instead of in the client config
- Use `VOX_FORCE_ENV_OVERRIDE=true` in `.env` if client-passed env vars conflict with your `.env` values

## Configuration

Copy `.env.example` to `.env` and configure:

- **API keys** — at least one provider key is required
- **`DEFAULT_MODEL`** — `auto` (default, agent picks) or a specific model name
- **Model restrictions** — `GOOGLE_ALLOWED_MODELS`, `OPENAI_ALLOWED_MODELS`, etc.
- **`CONVERSATION_TIMEOUT_HOURS`** — thread TTL (default: 24h)
- **`MAX_CONVERSATION_TURNS`** — thread length limit (default: 100)

See `.env.example` for the full reference.

## Development

```bash
uv sync
uv run python -c "import server"   # smoke test
uv run pytest                       # run tests
```

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Derived from [pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server) by Beehive Innovations.
