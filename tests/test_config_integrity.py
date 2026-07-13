"""Offline integrity gate for the model configs.

These checks are DERIVED from conf/*.json rather than restating it, so they cannot
rot the way a hardcoded assertion does. They deliberately do NOT ask "is this model
still served upstream?" -- that is a question only the network can answer, it is not
the PR author's fault when the answer changes, and it therefore belongs in a separate
liveness job rather than in the suite that gates every commit.

What they DO catch is the class of defect that is always our own fault:

  * an alias pointing at a model the file no longer declares (a dangling alias)
  * the same alias claimed by two different models (an ambiguous resolve, where
    which model you get depends on dict ordering)
  * an alias that collides with some OTHER model's real ID -- the nastiest shape,
    because the caller asks for model A, silently receives model B, and gets
    plausible output from the wrong model with no error to notice. A wrong ID
    fails loudly and the caller learns; a silent substitution never teaches anyone
    anything.

Every one of these is decidable offline, from the files themselves.
"""

import json
from pathlib import Path

import pytest

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

# Discovered, not enumerated. A hard-coded list is a silent skip waiting to happen:
# add conf/newprovider_models.json and an enumerated list would simply not look at it,
# while still reporting green.
CONFIGS = sorted(CONF_DIR.glob("*_models.json"))


def _models(path: Path) -> list[dict]:
    return json.loads(path.read_text()).get("models", [])


def test_configs_were_actually_discovered():
    """Coverage before verdict: a run that checked nothing must not report green."""
    assert CONFIGS, f"no *_models.json found under {CONF_DIR} - this suite verified nothing"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_alias_collides_with_another_models_id(path: Path):
    """An alias must never name a DIFFERENT model's real ID.

    This is the silent-substitution trap. grok-code-fast-1 (a real, distinct xAI
    model) was briefly aliased onto grok-build-0.1: anyone asking for the former
    would have quietly received the latter.
    """
    models = _models(path)
    declared_ids = {m["model_name"] for m in models}

    offenders = []
    for m in models:
        for alias in m.get("aliases", []):
            if alias in declared_ids and alias != m["model_name"]:
                offenders.append(f"{path.name}: alias {alias!r} on {m['model_name']!r} is another model's ID")

    assert not offenders, "aliases silently shadowing other models:\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_duplicate_aliases_within_a_config(path: Path):
    """Two models claiming one alias means the winner depends on iteration order."""
    seen: dict[str, str] = {}
    clashes = []
    for m in _models(path):
        for alias in m.get("aliases", []):
            key = alias.lower()
            if key in seen:
                clashes.append(f"{path.name}: {alias!r} claimed by both {seen[key]!r} and {m['model_name']!r}")
            seen[key] = m["model_name"]

    assert not clashes, "ambiguous aliases:\n" + "\n".join(clashes)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_no_model_declares_a_zero_context_window(path: Path):
    """A missing context_window silently becomes 0 and breaks token budgeting."""
    bad = [m["model_name"] for m in _models(path) if not m.get("context_window")]
    assert not bad, f"{path.name}: models with no context_window: {bad}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_a_reasoning_model_declares_which_params_it_rejects(path: Path):
    """
    A model that rejects temperature almost certainly rejects other params too —
    o-series models take max_completion_tokens rather than max_tokens, and reject
    the sampling penalties. That used to be INFERRED from supports_temperature,
    which is why nobody had to write it down.

    Now that the inference is gone, an entry that says `supports_temperature: false`
    and stops there would quietly start receiving max_tokens and penalties. The
    omission is invisible at runtime — the request just fails at the provider, or
    worse, silently ignores the cap. So the declaration is required, not optional:
    if a model really does accept everything else, it says so with an explicit
    empty list.
    """
    undeclared = [
        m["model_name"]
        for m in _models(path)
        if m.get("supports_temperature") is False and "unsupported_params" not in m
    ]
    assert not undeclared, (
        f"{path.name}: these models reject temperature but do not declare "
        f"`unsupported_params`, so max_tokens and the penalties would now be sent "
        f"to them: {undeclared}\n"
        "Declare the list explicitly (use [] to mean 'accepts everything else')."
    )
