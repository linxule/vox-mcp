"""
Wire-payload tests for request-parameter support — and specifically for the fact
that it is DECLARED, never inferred.

`supports_temperature` used to do two unrelated jobs. It governed temperature (its
actual meaning), and it was also read as a proxy for "this model accepts max_tokens
and the sampling penalties":

    model_accepts_max_tokens = bool(capabilities.supports_temperature)
    if not capabilities.supports_temperature:
        unsupported.update({"top_p", "frequency_penalty", "presence_penalty", "stop", "stream"})

Those are different questions. A model can reject temperature and happily accept a
token cap; a model can accept temperature and reject penalties. Coupling them meant
o3's max_tokens was dropped because of a flag about temperature — invisible today
only because no tool passes max_output_tokens yet, and primed to fire silently the
moment one does.

Support is now declared per model in conf/*_models.json via `unsupported_params`.
These tests pin that, in both directions, so the coupling cannot creep back.
"""

from unittest.mock import MagicMock, patch

from providers.openai import OpenAIModelProvider
from providers.shared import ModelCapabilities, ProviderType
from providers.shared.temperature import RangeTemperatureConstraint
from providers.xai import XAIModelProvider


def _mock_chat_response(model="grok-4.5"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].finish_reason = "stop"
    resp.model = model
    resp.id = "test-id"
    resp.created = 1234567890
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


def _call_kwargs(mock_client):
    mock_client.chat.completions.create.assert_called_once()
    return mock_client.chat.completions.create.call_args[1]


def _caps(name, **over):
    """A synthetic model, so the two flags can be varied independently."""
    base = {
        "model_name": name,
        "friendly_name": name,
        "provider": ProviderType.XAI,
        "context_window": 100_000,
        "max_output_tokens": 8_192,
        "temperature_constraint": RangeTemperatureConstraint(0.0, 2.0, 1.0),
    }
    base.update(over)
    return ModelCapabilities(**base)


# --------------------------------------------------------------------------- #
# The decoupling, in both directions. These are the regression guards.
# --------------------------------------------------------------------------- #


@patch("providers.openai_compatible.OpenAI")
def test_no_temperature_support_does_not_imply_no_max_tokens(mock_openai_class):
    """
    THE BUG. A model that rejects temperature but says nothing about max_tokens
    must still receive max_tokens.

    Under the old code this assertion failed: max_tokens was gated on
    supports_temperature, so declaring "no temperature" silently confiscated the
    caller's token cap.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response()

    provider = XAIModelProvider("test-key")
    caps = _caps("reasoner-x", supports_temperature=False, unsupported_params=[])
    with (
        patch.object(provider, "get_capabilities", return_value=caps),
        patch.object(provider, "validate_model_name", return_value=True),
        patch.object(provider, "_resolve_model_name", return_value="reasoner-x"),
    ):
        provider.generate_content(prompt="hi", model_name="reasoner-x", temperature=0.5, max_output_tokens=256)

    kwargs = _call_kwargs(mock_client)
    assert "temperature" not in kwargs, "a model that rejects temperature must not receive it"
    assert kwargs["max_tokens"] == 256, "max_tokens must NOT be gated on temperature support"


@patch("providers.openai_compatible.OpenAI")
def test_temperature_support_does_not_imply_every_penalty_is_accepted(mock_openai_class):
    """
    The mirror image. A model that accepts temperature but declares a penalty
    unsupported must have that penalty stripped — the old code only ever excluded
    penalties for models that also rejected temperature, so a temperature-capable
    model could not opt out of one.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response()

    provider = XAIModelProvider("test-key")
    caps = _caps(
        "picky-x",
        supports_temperature=True,
        unsupported_params=["frequency_penalty", "presence_penalty"],
    )
    with (
        patch.object(provider, "get_capabilities", return_value=caps),
        patch.object(provider, "validate_model_name", return_value=True),
        patch.object(provider, "_resolve_model_name", return_value="picky-x"),
    ):
        provider.generate_content(
            prompt="hi",
            model_name="picky-x",
            temperature=0.5,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            top_p=0.9,
        )

    kwargs = _call_kwargs(mock_client)
    assert kwargs["temperature"] == 0.5, "temperature support is independent of penalty support"
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs
    assert kwargs["top_p"] == 0.9, "an undeclared param must still pass through"


# --------------------------------------------------------------------------- #
# The real configured models still behave exactly as before.
# --------------------------------------------------------------------------- #


@patch("providers.openai_compatible.OpenAI")
def test_o3_still_drops_max_tokens_because_it_now_says_so(mock_openai_class):
    """
    o3 rejects max_tokens on chat/completions (it takes max_completion_tokens).
    That was true before this change and is true after — the difference is that
    o3's entry now DECLARES it instead of it being inferred from a temperature flag.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("o3")

    provider = OpenAIModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="o3", temperature=0.5, max_output_tokens=256)

    kwargs = _call_kwargs(mock_client)
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


@patch("providers.openai_compatible.OpenAI")
def test_grok_keeps_max_tokens(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("grok-4.5")

    provider = XAIModelProvider("test-key")
    provider.generate_content(prompt="hi", model_name="grok-4.5", temperature=0.5, max_output_tokens=256)

    kwargs = _call_kwargs(mock_client)
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.5


@patch("providers.openai_compatible.OpenAI")
def test_grok_reasoning_model_strips_exactly_the_three_params_xai_rejects(mock_openai_class):
    """
    Sourced from docs.x.ai's chat request schema: `frequency_penalty`,
    `presence_penalty` and `stop` each carry "Not supported by reasoning models."
    `temperature` (0-2) and `top_p` carry no such note, and `max_tokens` is marked
    DEPRECATED (in favour of max_completion_tokens) but not rejected — xAI marks
    rejections explicitly and did not mark that one.

    So the boundary is precise, and this pins it: three params out, the rest through.
    Grok had NO exclusions declared before, which meant the penalties would have been
    forwarded to a model that rejects them the moment any caller supplied one.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("grok-4.5")

    provider = XAIModelProvider("test-key")
    provider.generate_content(
        prompt="hi",
        model_name="grok-4.5",
        temperature=0.5,
        max_output_tokens=256,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop=["\n"],
        top_p=0.9,
    )

    kwargs = _call_kwargs(mock_client)
    # Rejected by xAI on reasoning models — must never reach the wire.
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs
    assert "stop" not in kwargs
    # Accepted — must still be forwarded.
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert kwargs["max_tokens"] == 256


@patch("providers.openai_compatible.OpenAI")
def test_grok_non_reasoning_model_accepts_the_penalties(mock_openai_class):
    """
    The restriction is on REASONING models specifically. grok-4.20-0309-non-reasoning
    is tagged "Reasoning: No" on its docs.x.ai page, so it takes the full sampling
    set — a blanket per-provider exclusion would have been wrong.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("grok-4.20-0309-non-reasoning")

    provider = XAIModelProvider("test-key")
    provider.generate_content(
        prompt="hi",
        model_name="grok-4.20-0309-non-reasoning",
        temperature=0.5,
        frequency_penalty=0.3,
        presence_penalty=0.3,
    )

    kwargs = _call_kwargs(mock_client)
    assert kwargs["frequency_penalty"] == 0.3
    assert kwargs["presence_penalty"] == 0.3


# --------------------------------------------------------------------------- #
# The dead branch that would have raised if anything reached it.
# --------------------------------------------------------------------------- #


@patch("providers.openai_compatible.OpenAI")
def test_absent_capabilities_with_a_temperature_does_not_raise(mock_openai_class):
    """
    When get_capabilities() raises, generate_content falls back to capabilities=None.
    The old fallback branch then called `capabilities.get_effective_temperature(...)`
    on that very None — an AttributeError waiting for the first caller to reach it.
    It was unreachable only by luck (validate_model_name happened to reject the same
    models first), which is a coincidence, not a guard.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_chat_response("mystery")

    provider = XAIModelProvider("test-key")
    with (
        patch.object(provider, "get_capabilities", side_effect=RuntimeError("no metadata")),
        patch.object(provider, "validate_model_name", return_value=True),
        patch.object(provider, "_resolve_model_name", return_value="mystery"),
    ):
        provider.generate_content(prompt="hi", model_name="mystery", temperature=0.5)

    kwargs = _call_kwargs(mock_client)
    # No metadata to clamp against: pass the caller's value through untouched.
    assert kwargs["temperature"] == 0.5
