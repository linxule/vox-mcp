"""Vox executable: stdio MCP by default, local model preferences on request."""

from __future__ import annotations

import argparse
import json
import sys

from model_settings import SettingsError, read_settings, resolve_default_model, settings_path, write_default_model


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        # Config commands must work without importing providers or starting MCP.
        from server import run as run_server

        run_server()
        return 0

    parser = argparse.ArgumentParser(prog="vox-mcp")
    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser("config", help="Manage non-secret model preferences")
    actions = config_parser.add_subparsers(dest="action", required=True)
    actions.add_parser("show", help="Show saved and effective defaults as JSON")
    set_parser = actions.add_parser("set-default", help="Save an exact model ID, alias, or auto")
    set_parser.add_argument("model")
    actions.add_parser("reset-default", help="Remove the saved default")
    parsed = parser.parse_args(arguments)

    # Match server environment/.env precedence without loading the server.
    from utils.env import get_env

    try:
        if parsed.action == "set-default":
            write_default_model(parsed.model)
        elif parsed.action == "reset-default":
            write_default_model(None)

        environment_default = get_env("DEFAULT_MODEL")
        saved = read_settings().get("default_model")
        effective, source = resolve_default_model(environment_default)
        print(
            json.dumps(
                {
                    "settings_path": str(settings_path()),
                    "saved_default_model": saved,
                    "effective_default_model": effective,
                    "source": source,
                    "environment_overrides_saved": bool(environment_default and saved),
                    "note": "Restart or reconnect Vox to apply changes. DEFAULT_MODEL in the MCP client's env overrides saved settings; this command shows its own launch environment.",
                },
                indent=2,
            )
        )
        return 0
    except SettingsError as exc:
        print(f"vox-mcp: {exc}", file=sys.stderr)
        return 2


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
