# Security Policy

## Reporting a vulnerability

If you find a security vulnerability, please report it privately:

- **Email:** Open a [GitHub Security Advisory](https://github.com/linxule/vox-mcp/security/advisories/new) (preferred)
- Do **not** open a public issue for security vulnerabilities

I'll acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## API key handling

Vox MCP handles API keys for multiple providers. The security model:

- Keys are read from environment variables or `.env` files
- Keys are **never** logged, stored in memory beyond process lifetime, or sent to any service other than the intended provider
- `.env` is in `.gitignore` — never commit your keys
- When running via MCP clients, keys passed in the client config are available only to the server process

## What to watch for

- If you notice API keys appearing in logs, that's a bug — please report it
- If a provider receives requests it shouldn't (e.g., your OpenAI key being sent to a custom endpoint), that's a critical bug
- Test cassettes and fixtures should never contain real API keys or org identifiers

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
