# Model selection and verification

Verified against public provider documentation on 2026-09-26. These are curated
provider preferences, not a live ranking or a promise of access on every account.
No paid inference was used for this release; endpoint requests are covered by
SDK/HTTP mocks and the published package is checked for MCP startup and discovery.

## Built-in preferences

| Provider | Preferred model | Other current choices |
| --- | --- | --- |
| OpenAI | `gpt-6-astra` for every category | `gpt-6-sol`, `gpt-6-luna` |
| Moonshot | `kimi-k3` | Explicit `kimi-k2.6` retained |
| xAI | `grok-4.6` | `grok-4.7` remains explicitly selectable |
| Anthropic | `claude-opus-5-5`, then `claude-fable-5-1` | Sonnet 5, Haiku 4.5, pinned Claude 3 Opus |
| DeepSeek | `deepseek-flash` (V4.1 Flash) | Explicit `deepseek-v4-pro` retained |
| Gemini | `gemini-3.8-flash` for fast/balanced; `gemini-3.1-pro-preview` for extended reasoning | Gemini 2.5 entries marked legacy |

`chat` uses the fast category for its automatic fallback. An explicit model,
continuation model, or configured default takes precedence. With multiple providers,
Vox keeps its existing provider priority (Google, OpenAI, xAI, Anthropic, Moonshot,
DeepSeek, custom, Cloudflare, Vercel, OpenRouter); these preferences do not reorder
providers. See [default configuration](README.md#set-your-default-model).

Generic aliases such as `deepseek`, `kimi`, `grok`, `flash`, `fable`, and OpenRouter
`opus` advance to these selections. Exact historical IDs and versioned aliases do
not silently switch models. Claude 3 Opus's pinned snapshot and aliases remain
unchanged for accounts with access. Gemini 2.0 Flash/Lite were removed following
their June 1 shutdown; Gemini 2.5 requires prior access and is not the default.

## Provider contracts

- [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model)
  and [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra),
  [Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
  [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna): native Responses,
  text/image input, 1,050,000 context and 128,000 output. Vox leaves reasoning effort
  unset unless requested; universal `minimal` maps to `low`, and `max` to `max`.
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)
  and [thinking](https://api-docs.deepseek.com/guides/thinking_mode/): the native ID
  is `deepseek-flash`, with vision and a 393,216 output ceiling. Effort accepts
  low/high/max; Vox maps minimal/low to low and medium/high to high.
- [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash/),
  [3.1 Pro Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview),
  [Interactions](https://ai.google.dev/gemini-api/docs/interactions-overview), and
  [deprecations](https://ai.google.dev/gemini-api/docs/deprecations): both current
  models support Interactions. Images/fallback use generateContent. Minimal maps
  to low; max maps to high on either path.
- [Kimi K3 quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart):
  native Chat Completions uses `max_completion_tokens`, fixed sampling values
  omitted from requests, and low/high/max effort with upstream default max.
- [Grok 4.6](https://docs.x.ai/developers/models/grok-4.6),
  [API examples](https://x.ai/api), and
  [reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning):
  Chat Completions, 500,000 context, images, no published text output ceiling.
  Both 4.6 and 4.7 accept xhigh; Vox maps max to xhigh. Default high is left upstream.
- [Claude lineup](https://platform.claude.com/docs/en/models/overview),
  [Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/overview), and
  [Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide):
  Messages API, adaptive thinking and effort, 1M context / 128K output.
  Fable defaults to high effort; Opus to medium. Vox omits effort unless requested.
- [OpenRouter's public catalog](https://openrouter.ai/api/v1/models) independently
  supplies gateway IDs, limits, and effort support. For example native
  `deepseek-flash` is `deepseek/deepseek-v4.1-flash` there, and native
  `claude-opus-5-5` is `anthropic/claude-opus-5.5`. Gateway output limits differ
  from native limits; their metadata is kept separate. Mistral Large 2512 remains
  current. Cloudflare/Vercel routes continue to use explicitly configured IDs.

OpenRouter vision entries use a local combined image budget of 10 MiB where no
limit was configured. This is a Vox upload budget, not a provider guarantee;
upstream per-image and request limits can be stricter. Image blocks follow the
[OpenRouter image input contract](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding).
