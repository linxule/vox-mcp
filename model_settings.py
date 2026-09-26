"""Persistent, non-secret model preferences. No providers or server imports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG_PATH = "~/.vox/config.json"


class SettingsError(ValueError):
    """A settings file cannot be safely read or updated."""


def settings_path() -> Path:
    """Resolve at call time so callers and tests can isolate their preferences."""
    return Path(os.environ.get("VOX_CONFIG_PATH") or _DEFAULT_CONFIG_PATH).expanduser().resolve()


def validate_default_model(value: Any) -> str:
    """Check shape only; exact provider IDs need no credentials or network lookup."""
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise SettingsError("default_model must be a nonempty model ID or alias without whitespace, or 'auto'.")
    return value


def read_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SettingsError(f"Cannot read Vox settings at {path}. Fix or move this file and retry.") from exc
    if not isinstance(data, dict):
        raise SettingsError(f"Vox settings at {path} must be a JSON object. Fix or move this file and retry.")
    if "default_model" in data:
        try:
            validate_default_model(data["default_model"])
        except SettingsError as exc:
            raise SettingsError(f"Invalid default_model in {path}: {exc}") from exc
    return data


def resolve_default_model(environment_default: str | None) -> tuple[str, str]:
    """Environment wins; unread saved settings cannot block an explicit override."""
    if environment_default:
        return environment_default, "environment (DEFAULT_MODEL)"
    saved = read_settings().get("default_model")
    if saved:
        return saved, "settings file"
    return "auto", "built-in"


def write_default_model(model: str | None) -> Path:
    """Atomically set/reset only the default; preserve unrelated settings."""
    if model is not None:
        validate_default_model(model)
    path = settings_path()
    data = read_settings(path)
    if model is None:
        data.pop("default_model", None)
        if not path.exists():
            return path
    else:
        data["default_model"] = model
    temporary_path: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        raise SettingsError(f"Cannot write Vox settings at {path}. Check directory and file permissions.") from exc
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
    return path
