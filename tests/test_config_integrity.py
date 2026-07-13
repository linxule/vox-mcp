"""Offline integrity gate for model declarations.

These checks are DERIVED from what the providers actually serve rather than
restating it, so they cannot rot the way a hardcoded assertion does. They
deliberately do NOT ask "is this model still served upstream?" -- that is a
question only the network can answer, it is not the PR author's fault when the
answer changes, and it therefore belongs in a separate liveness job rather than
in the suite that gates every commit.

What they DO catch is the class of defect that is always our own fault:

  * an alias pointing at a model the provider no longer declares (a dangling alias)
  * the same alias claimed by two different models (an ambiguous resolve, where
    which model you get depends on dict ordering)
  * an alias that collides with some OTHER model's real ID -- the nastiest shape,
    because the caller asks for model A, silently receives model B, and gets
    plausible output from the wrong model with no error to notice. A wrong ID
    fails loudly and the caller learns; a silent substitution never teaches anyone
    anything.
  * a model that rejects temperature without declaring what else it rejects

Every one of these is decidable offline, from the code itself.


WHY THIS FILE ITERATES PROVIDERS AND NOT `conf/*.json`
------------------------------------------------------

The first version of this gate globbed `conf/*_models.json`, and its docstring
congratulated itself for discovering configs rather than enumerating them:

    "Discovered, not enumerated. A hard-coded list is a silent skip waiting to
     happen: add conf/newprovider_models.json and an enumerated list would
     simply not look at it, while still reporting green."

It was wrong, and wrong in exactly the way it was warning about. vox has THREE
ways a provider declares its models, not one:

  1. `conf/*_models.json`, loaded via RegistryBackedProviderMixin into the class
     dict                                            -- gemini, openai, xai
  2. a hardcoded Python dict in the provider module  -- anthropic, deepseek, moonshot
  3. an instance-level registry, leaving the class dict EMPTY
                                                     -- openrouter, custom

Globbing `conf/` saw (1) and (3) and was structurally blind to (2) -- so DeepSeek
and Moonshot, both OpenAICompatibleProvider subclasses whose wire payload IS
governed by `unsupported_params`, got zero coverage while the gate reported green.
A naive rewrite keyed on the class dict would have flipped the blind spot onto (3).

The lesson is not "glob harder". It is that **the discovery mechanism was itself
the enumeration**, wearing a costume that looked like discovery. Coverage must be
keyed on the authority for what EXISTS -- the ProviderType enum -- and not on
whichever files happen to be lying around. `test_every_provider_is_covered` is that
key, and it is the reason a fourth declaration mechanism cannot slip in unwatched:
a provider that yields no capabilities fails the suite instead of being skipped.
"""

import importlib
import inspect
import pkgutil

import pytest

import providers
from providers.base import ModelProvider
from providers.openai_compatible import OpenAICompatibleProvider
from providers.registries.base import CustomModelRegistryBase
from providers.registry_provider_mixin import RegistryBackedProviderMixin
from providers.shared import ModelCapabilities, ProviderType

# The two abstract layers every concrete provider inherits from. Named rather than
# detected because `inspect.isabstract` returns False for them (they have no
# unimplemented abstractmethods), so they would otherwise be mistaken for providers.
ABSTRACT_BASES = {"ModelProvider", "OpenAICompatibleProvider"}


def _discover_provider_classes() -> list[type[ModelProvider]]:
    """Every concrete ModelProvider defined under `providers/`."""
    found: list[type[ModelProvider]] = []
    for module_info in pkgutil.iter_modules(providers.__path__):
        module = importlib.import_module(f"providers.{module_info.name}")
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, ModelProvider)
                and obj.__module__ == module.__name__  # defined here, not imported
                and obj.__name__ not in ABSTRACT_BASES
                and not inspect.isabstract(obj)
            ):
                found.append(obj)
    return sorted(found, key=lambda c: c.__name__)


def _declared_registry_class(provider: type) -> type | None:
    """The registry an instance-registry provider builds in __init__ (mechanism 3)."""
    module = importlib.import_module(provider.__module__)
    for obj in vars(module).values():
        if inspect.isclass(obj) and issubclass(obj, CustomModelRegistryBase) and obj is not CustomModelRegistryBase:
            return obj
    return None


def _effective_capabilities(provider: type) -> dict[str, ModelCapabilities]:
    """What this provider actually serves -- across all three declaration mechanisms.

    Returns {} when a provider declares models some fourth way we do not know about.
    That empty result is deliberately NOT swallowed: `test_every_provider_is_covered`
    turns it into a failure, so a new mechanism announces itself instead of quietly
    shrinking the gate's reach.
    """
    if issubclass(provider, RegistryBackedProviderMixin):
        provider._ensure_registry()  # mechanism 1: populate the class dict from conf/

    capabilities = dict(getattr(provider, "MODEL_CAPABILITIES", {}) or {})
    if capabilities:  # mechanisms 1 and 2
        return capabilities

    registry_class = _declared_registry_class(provider)  # mechanism 3
    if registry_class is None:
        return {}
    registry = registry_class()
    return {name: registry.get_capabilities(name) for name in registry.list_models()}


PROVIDERS = _discover_provider_classes()
CAPABILITIES = {p: _effective_capabilities(p) for p in PROVIDERS}

# Parametrize on (provider, capabilities) so a failure names the provider, not an index.
CASES = [pytest.param(p, CAPABILITIES[p], id=p.__name__) for p in PROVIDERS]


# --------------------------------------------------------------------------- #
# Coverage before verdict. A gate that is green because it is not looking is
# worse than no gate, so the suite must fail rather than skip.
# --------------------------------------------------------------------------- #


def test_every_provider_is_covered():
    """Every ProviderType must map to exactly one discovered provider that serves models.

    The enum is the authority for what EXISTS. Keying coverage on it -- rather than
    on a file glob -- is what makes a new provider (or a fourth way of declaring
    models) impossible to add silently: it shows up here as a hole, immediately.
    """
    assert PROVIDERS, "no provider classes discovered - this suite verified NOTHING"

    covered: dict[ProviderType, str] = {}
    empty: list[str] = []
    for provider, capabilities in CAPABILITIES.items():
        if not capabilities:
            empty.append(provider.__name__)
            continue
        for provider_type in {c.provider for c in capabilities.values()}:
            covered.setdefault(provider_type, provider.__name__)

    assert not empty, (
        f"these providers yielded no capabilities, so nothing below inspected them: {empty}\n"
        "Either they declare models a way _effective_capabilities() does not know about "
        "(add the mechanism), or they genuinely serve nothing (delete them). A provider "
        "the gate cannot see is a provider the gate does not guard."
    )

    missing = set(ProviderType) - set(covered)
    assert not missing, (
        f"ProviderType members with no provider serving them: {sorted(m.value for m in missing)}\n"
        "A new backend was added to the enum but this gate cannot reach its models."
    )


# --------------------------------------------------------------------------- #
# The integrity checks themselves.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("provider", "capabilities"), CASES)
def test_no_alias_collides_with_another_models_id(provider, capabilities):
    """An alias must never name a DIFFERENT model's real ID.

    This is the silent-substitution trap. grok-code-fast-1 (a real, distinct xAI
    model) was briefly aliased onto grok-build-0.1: anyone asking for the former
    would have quietly received the latter.
    """
    declared_ids = set(capabilities)

    offenders = [
        f"{provider.__name__}: alias {alias!r} on {name!r} is another model's ID"
        for name, capability in capabilities.items()
        for alias in (capability.aliases or [])
        if alias in declared_ids and alias != name
    ]
    assert not offenders, "aliases silently shadowing other models:\n" + "\n".join(offenders)


@pytest.mark.parametrize(("provider", "capabilities"), CASES)
def test_no_duplicate_aliases_within_a_provider(provider, capabilities):
    """Two models claiming one alias means the winner depends on iteration order."""
    seen: dict[str, str] = {}
    clashes = []
    for name, capability in capabilities.items():
        for alias in capability.aliases or []:
            key = alias.lower()
            if key in seen:
                clashes.append(f"{provider.__name__}: {alias!r} claimed by both {seen[key]!r} and {name!r}")
            seen[key] = name

    assert not clashes, "ambiguous aliases:\n" + "\n".join(clashes)


@pytest.mark.parametrize(("provider", "capabilities"), CASES)
def test_no_model_declares_a_zero_context_window(provider, capabilities):
    """A missing context_window silently becomes 0 and breaks token budgeting."""
    bad = [name for name, c in capabilities.items() if not c.context_window]
    assert not bad, f"{provider.__name__}: models with no context_window: {bad}"


@pytest.mark.parametrize(("provider", "capabilities"), CASES)
def test_a_model_that_rejects_temperature_declares_what_else_it_rejects(provider, capabilities):
    """
    A model that rejects temperature almost certainly rejects other params too --
    o-series models take max_completion_tokens rather than max_tokens, and reject
    the sampling penalties. That used to be INFERRED from supports_temperature,
    which is why nobody had to write it down.

    Now that the inference is gone, an entry that says `supports_temperature: false`
    and stops there would quietly start receiving max_tokens and the penalties. The
    omission is invisible at runtime -- the request just fails at the provider, or
    worse, silently ignores the cap. So the declaration is required, not optional:
    a model that really does accept everything else says so with an explicit [].

    SCOPE, and why it is computed rather than listed. `unsupported_params` is read
    by OpenAICompatibleProvider when it builds the wire payload. A provider with its
    own client (Anthropic, Gemini) never consults it, so demanding the declaration
    there would be cargo cult -- it would pin a field nothing reads.

    That scope is derived with `issubclass`, NOT written as an exemption list, and
    the difference matters: an exemption inside a gate is a hole in the gate. If
    Anthropic is ever ported onto the OpenAI-compatible base, this check starts
    covering it on that same commit, with nobody remembering to update a list.

    `is None`, NOT `not ...`. The rule is authorship discipline -- "you must have
    thought about this and written it down" -- so it has to distinguish an omission
    from a considered empty answer. A model may legitimately reject temperature and
    accept every other param; its honest declaration is `[]`, and flagging that would
    be a false positive. `unsupported_params` therefore defaults to None (undeclared)
    rather than [], which is the whole reason this check can live on the capabilities
    object at all instead of only on the JSON that happens to have a key to be absent.
    """
    if not issubclass(provider, OpenAICompatibleProvider):
        pytest.skip(f"{provider.__name__} has its own client; unsupported_params is never read for it")

    undeclared = [
        name for name, c in capabilities.items() if c.supports_temperature is False and c.unsupported_params is None
    ]
    assert not undeclared, (
        f"{provider.__name__}: these models reject temperature but declare no "
        f"`unsupported_params`, so max_tokens and the penalties would now be sent "
        f"to them: {undeclared}\n"
        "Declare the list explicitly (use [] to mean 'accepts everything else')."
    )
