"""Persistent defaults work offline and never touch the real user settings."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cli
import model_settings


def test_set_show_reset_and_environment_shadowing(isolate_vox_settings, monkeypatch, capsys):
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    exact = "vercel/google/gemini-3.1-pro-preview"
    assert cli.main(["config", "set-default", exact]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["effective_default_model"] == exact
    assert output["source"] == "settings file"
    assert json.loads(isolate_vox_settings.read_text()) == {"default_model": exact}
    monkeypatch.setenv("DEFAULT_MODEL", "opus")
    assert cli.main(["config", "show"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["effective_default_model"] == "opus"
    assert output["saved_default_model"] == exact
    assert output["environment_overrides_saved"]
    assert "reconnect" in output["note"]
    assert cli.main(["config", "reset-default"]) == 0
    assert json.loads(capsys.readouterr().out)["saved_default_model"] is None
    monkeypatch.delenv("DEFAULT_MODEL")
    assert model_settings.resolve_default_model(None) == ("auto", "built-in")


@pytest.mark.parametrize("contents", ["{broken", "[]", '{"default_model": 4}', '{"default_model": "bad model"}'])
def test_malformed_settings_fail_without_overwriting(contents, isolate_vox_settings, capsys):
    isolate_vox_settings.parent.mkdir()
    isolate_vox_settings.write_text(contents)
    assert cli.main(["config", "set-default", "new-model"]) == 2
    assert isolate_vox_settings.read_text() == contents
    assert str(isolate_vox_settings) in capsys.readouterr().err
    # An explicitly configured server can still start despite an unused broken file.
    assert model_settings.resolve_default_model("explicit") == ("explicit", "environment (DEFAULT_MODEL)")


def test_atomic_write_preserves_existing_file_on_failure(isolate_vox_settings, monkeypatch):
    isolate_vox_settings.parent.mkdir()
    original = '{"default_model": "old", "other": {"preserve": true}}'
    isolate_vox_settings.write_text(original)

    def fail_replace(*_args):
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(model_settings.os, "replace", fail_replace)
        with pytest.raises(model_settings.SettingsError, match="Cannot write"):
            model_settings.write_default_model("new")
    assert isolate_vox_settings.read_text() == original
    assert list(isolate_vox_settings.parent.iterdir()) == [isolate_vox_settings]
    model_settings.write_default_model("new")
    assert json.loads(isolate_vox_settings.read_text())["other"] == {"preserve": True}


def test_settings_isolation_survives_environment_clear(monkeypatch, isolate_vox_settings):
    monkeypatch.delenv("VOX_CONFIG_PATH")
    assert model_settings.settings_path() == isolate_vox_settings
    assert Path.home() / ".vox" not in isolate_vox_settings.parents


def test_config_commands_run_in_fresh_process_without_provider_imports(tmp_path):
    # Copy only the CLI's files; server/providers/SDKs are deliberately unavailable.
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / "package"
    package.mkdir()
    for name in ("cli.py", "model_settings.py"):
        shutil.copyfile(root / name, package / name)
    utils = package / "utils"
    utils.mkdir()
    (utils / "__init__.py").write_text("")
    shutil.copyfile(root / "utils" / "env.py", utils / "env.py")
    environment = {"HOME": str(tmp_path / "home"), "PATH": os.environ.get("PATH", "")}
    result = subprocess.run(
        [sys.executable, "-S", str(package / "cli.py"), "config", "set-default", "Exact/Route-ID"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["effective_default_model"] == "Exact/Route-ID"
    assert output["settings_path"] == str(tmp_path / "home" / ".vox" / "config.json")
    result = subprocess.run(
        [sys.executable, "-S", str(package / "cli.py"), "config", "show"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["saved_default_model"] == "Exact/Route-ID"


@pytest.mark.asyncio
async def test_listmodels_reports_loaded_default_and_source(monkeypatch):
    import config
    from tools.listmodels import ListModelsTool

    monkeypatch.setattr(config, "DEFAULT_MODEL", "explicit-route")
    monkeypatch.setattr(config, "DEFAULT_MODEL_SOURCE", "settings file")
    result = await ListModelsTool().execute({})
    output = json.loads(result[0].text)
    assert output["metadata"]["default_model"] == "explicit-route"
    assert output["metadata"]["default_model_source"] == "settings file"
    assert "`explicit-route` (settings file)" in output["content"]


@pytest.mark.no_mock_provider
def test_saved_default_loads_into_config_and_schema(monkeypatch):
    import config
    import utils.model_restrictions as restrictions
    from providers.deepseek import DeepSeekProvider
    from providers.registry import ModelProviderRegistry
    from providers.shared import ProviderType
    from tools.chat import ChatTool

    monkeypatch.setattr(ModelProviderRegistry, "_instance", None)
    monkeypatch.setattr(restrictions, "_restriction_service", None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key-for-tests")
    monkeypatch.delenv("DEEPSEEK_ALLOWED_MODELS", raising=False)
    ModelProviderRegistry.register_provider(ProviderType.DEEPSEEK, DeepSeekProvider)
    assert isinstance(ModelProviderRegistry.get_provider_for_model("deepseek-flash"), DeepSeekProvider)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    model_settings.write_default_model("deepseek-flash")
    try:
        importlib.reload(config)
        assert config.DEFAULT_MODEL == "deepseek-flash"
        assert config.DEFAULT_MODEL_SOURCE == "settings file"
        schema = ChatTool().get_input_schema()
        assert "model" not in schema["required"]
        assert "deepseek-flash" in schema["properties"]["model"]["description"]
        monkeypatch.setenv("DEFAULT_MODEL", "auto")
        importlib.reload(config)
        assert config.DEFAULT_MODEL == "auto"
        assert config.DEFAULT_MODEL_SOURCE == "environment (DEFAULT_MODEL)"
        assert "model" in ChatTool().get_input_schema()["required"]
    finally:
        monkeypatch.setenv("DEFAULT_MODEL", "gemini-2.5-flash")
        importlib.reload(config)
