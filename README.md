# Vox MCP

Multi-model AI gateway for [MCP](https://modelcontextprotocol.io) clients.

## Why

MCP clients like Claude Code, Claude Desktop, and Cursor are locked to their host model. Vox gives them access to every other model — Gemini, GPT, Grok, DeepSeek, Kimi, or your local Ollama — through a single `chat` tool.

The design is deliberately minimal: prompts go to providers unmodified, responses come back unmodified. No system prompt injection. No response formatting. No behavioral directives. The only value Vox adds is routing and conversation memory — everything else is pure passthrough.

## What it does

Send a prompt, optionally attach files or images, pick a model (or let the agent pick), and get back the model's raw response. Conversation threads persist in memory via `continuation_id` for multi-turn exchanges across any provider — start a thread with Gemini, continue it with GPT. Threads are shadow-persisted to disk as JSONL for durability and can be exported as Markdown.

**3 tools:**

| Tool | Description |
|------|-------------|
| `chat` | Send prompts to any configured AI model with optional file/image context |
| `listmodels` | Show available models, aliases, and capabilities |
| `dump_threads` | Export conversation threads as JSON or Markdown |

**10 providers:**

| Provider | Env Variable | Example Models |
|----------|-------------|----------------|
| Google Gemini | `GEMINI_API_KEY` | gemini-3.8-flash, gemini-3.1-pro-preview |
| OpenAI | `OPENAI_API_KEY` | gpt-6-astra (default), gpt-6-sol, gpt-6-luna |
| Anthropic | `ANTHROPIC_API_KEY` | claude-opus-5-5 (default), claude-fable-5-1, claude-sonnet-5, claude-haiku-4-5 |
| xAI | `XAI_API_KEY` | grok-4.6 (default), grok-4.7 |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-flash (V4.1 Flash), deepseek-v4-pro |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | kimi-k3, kimi-k2.6 |
| OpenRouter | `OPENROUTER_API_KEY` | Any OpenRouter model |
| Cloudflare AI Gateway | `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | `cloudflare/openai/gpt-5.5` |
| Vercel AI Gateway | `VERCEL_AI_GATEWAY_API_KEY` or `AI_GATEWAY_API_KEY` | `vercel/anthropic/claude-sonnet-4.6` |
| Custom | `CUSTOM_API_URL` | Ollama, vLLM, LM Studio, etc. |

See [model selections and verification sources](https://github.com/linxule/vox-mcp/blob/main/MODEL_SELECTION.md) for provider preferences, exact API IDs, and reasoning behavior.

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

### Set your default model

You or your agent can save a default without editing MCP client configuration:

```bash
uvx vox-mcp config set-default deepseek-flash
uvx vox-mcp config show
uvx vox-mcp config reset-default
```

For a source checkout, use `uv run vox-mcp config ...`. Preferences are stored in
`~/.vox/config.json`; set `VOX_CONFIG_PATH` to use another file. This file contains
model preferences, not API keys. Commands make no model requests and do not need
provider credentials. Use `listmodels` in your connected MCP client to choose an
available ID or alias. Explicit gateway routes such as
`vercel/google/gemini-3.1-pro-preview` can also be saved. IDs are preserved exactly;
saving a default does not verify your account's access to that model.

Restart or reconnect Vox after changing settings. `config show` prints the saved
default, effective default, and its source for the command's launch environment.
The running MCP client's environment may differ; `listmodels` reports the default
that its server actually loaded.

Selection order is: explicit `chat.model`, the previous model for a continued
thread, `DEFAULT_MODEL` from the server environment, the saved default, then `auto`.
A saved default therefore changes new conversations; it does not switch existing
threads. To override saved preferences for one MCP client, add
`"DEFAULT_MODEL": "your-model-id"` to the Vox server's `env` object. Existing source
checkout `.env` settings also retain their environment precedence. The automatically
loaded `.env` belongs to the Vox installation, not the current directory; use the
settings command or MCP client environment for `uvx` installations.

With `auto`, the agent is asked to select a model for each call. If a caller passes
`model: "auto"`, Vox uses its built-in provider priority and preferences; this is
not a live comparison of all providers. Explicit model selections always remain
available. A useful request to your agent is: “Use Vox listmodels to find my
preferred model, save it as my default, and tell me how to reconnect Vox.”

## Cloudflare and Vercel AI Gateway

Use an explicit gateway prefix in `chat.model`. Vox removes only that first prefix
before sending the request and keeps it in conversation memory. A missing gateway
configuration or disallowed model fails without falling through to OpenRouter or a
native provider. Bare model names keep their existing routing behavior.

### Cloudflare

```dotenv
CLOUDFLARE_API_TOKEN=your-cloudflare-token
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_GATEWAY_ID=default
CLOUDFLARE_MODELS=openai/gpt-5.5
```

```json
{"prompt": "Explain quorum consensus.", "model": "cloudflare/openai/gpt-5.5"}
```

Vox uses the [Cloudflare account REST API](https://developers.cloudflare.com/ai-gateway/usage/rest-api/)
at `https://api.cloudflare.com/client/v4/accounts/<account>/ai/v1`, authenticates
with a bearer token, and sets `cf-aig-gateway-id` (`default` unless configured).
The token needs **Workers AI Read** permission; an AI Gateway-only token is not
sufficient. Third-party models use Cloudflare Unified Billing. Workers AI model
IDs retain their `@cf/` prefix, for example `cloudflare/@cf/moonshotai/kimi-k2.6`.
Legacy `/compat`, provider-key forwarding, and `dynamic/` routes are not supported
by this adapter.

### Vercel

```dotenv
VERCEL_AI_GATEWAY_API_KEY=your-vercel-gateway-key
VERCEL_MODELS=anthropic/claude-sonnet-4.6
```

```json
{"prompt": "Explain quorum consensus.", "model": "vercel/anthropic/claude-sonnet-4.6"}
```

Vox uses [Vercel's OpenAI-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/python)
at `https://ai-gateway.vercel.sh/v1`. `AI_GATEWAY_API_KEY` is also accepted;
`VERCEL_AI_GATEWAY_API_KEY` takes precedence when both are set.

### Catalogs and limits

`CLOUDFLARE_MODELS` and `VERCEL_MODELS` are optional comma-separated upstream IDs
for `listmodels` and agent discovery. They do not restrict access. Explicit gateway
model IDs work without a catalog, including with the default `DEFAULT_MODEL=auto`;
the caller must supply the gateway model. Use `CLOUDFLARE_ALLOWED_MODELS` or
`VERCEL_ALLOWED_MODELS` to restrict access. Both upstream IDs and fully prefixed
Vox routes are accepted in catalogs and allowlists. Use the exact model ID
published by the gateway; native-provider and gateway IDs can differ.

Gateway requests currently support **text only**. Vox does not infer vision or
thinking controls from a model name; explicit gateway `thinking_mode` requests are
rejected before inference. Its 32,768-token context and 4,096-token output
budgets are conservative local estimates, not advertised upstream limits; these
numbers are not sent as generation parameters. Omitted temperature and reasoning
settings use upstream defaults. No catalog or model availability request is made
at startup. The adapters are covered by mocked HTTP tests; live inference requires
a configured account and has not been exercised as part of the release checks.

## Development

Dependencies are maintained in `pyproject.toml` and `uv.lock`; Dependabot updates
the lock through its `uv` integration while respecting the supported version ranges.
Changing those ranges requires a deliberate compatibility review. CI checks the lockfile, runs the offline test
suite on Python 3.10 and 3.13, and audits locked packages with `pip-audit`.
The supported SDK lines are MCP 1.x, OpenAI 2.x, and Anthropic 0.x.
MCP SDK 2 requires a separate server API migration; provider major upgrades
are kept separate from dependency maintenance.

```bash
uv sync
uv run python -c "import server"   # smoke test
uv run pytest                       # run tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style, project structure, and how to add providers.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Derived from [pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server) by Beehive Innovations.

<!-- mcp-name: io.github.linxule/vox-mcp -->
