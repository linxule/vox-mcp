"""Tests for X.AI provider implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from providers.shared import ProviderType
from providers.xai import XAIModelProvider

# Model IDs xAI retired. None of these are served any more; a regression that
# reintroduces one would make the whole provider non-functional, so they are
# pinned here as *negatives*.
RETIRED_MODELS = ["grok-4", "grok-3", "grok-3-fast", "grok4", "grok3", "grokfast", "grok3fast"]

# Aliases xAI's own model table attests, which vox deliberately does not carry:
# cross-model aliases that would silently substitute a different model, and
# *-latest pointers whose upstream target moves. See the test below.
UNSOURCEABLE_ALIASES = [
    "grok-code-fast-1",
    "grok-code-fast",
    "grok-latest",
    "grok-4.5-latest",
    "grok-build-latest",
]


class TestXAIProvider:
    """Test X.AI provider functionality."""

    def setup_method(self):
        """Set up clean state before each test."""
        # Clear restriction service cache before each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        """Clean up after each test to avoid singleton issues."""
        # Clear restriction service cache after each test
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        # And drop any provider INSTANCE the registry cached while XAI_API_KEY was
        # patched into the environment. @patch.dict restores the env var, but the
        # registry has already memoised a live provider built from it — so without
        # this, X.AI stays "configured" for the rest of the session and later tests
        # (tests/test_listmodels.py) see a provider that should not exist.
        #
        # conftest's autouse fixture re-registers the provider CLASSES for the next
        # test, so resetting here is safe; what it cannot do is evict a cached
        # instance, which is exactly what leaks.
        #
        # This only ever bit us in a subset run: pytest collects test_listmodels
        # before test_xai_provider alphabetically, so the full suite passed green
        # while the pollution sat there. An order-dependent green is not a green.
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reset_for_testing()

    @patch.dict(os.environ, {"XAI_API_KEY": "test-key"})
    def test_initialization(self):
        """Test provider initialization."""
        provider = XAIModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.XAI
        assert provider.base_url == "https://api.x.ai/v1"

    def test_initialization_with_custom_url(self):
        """Test provider initialization with custom base URL."""
        provider = XAIModelProvider("test-key", base_url="https://custom.x.ai/v1")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.x.ai/v1"

    def test_model_validation(self):
        """Test model name validation."""
        provider = XAIModelProvider("test-key")

        # Test valid models
        assert provider.validate_model_name("grok-4.5") is True
        assert provider.validate_model_name("grok") is True
        assert provider.validate_model_name("grok45") is True
        assert provider.validate_model_name("grok-4.3") is True
        assert provider.validate_model_name("grok-4.20-0309-reasoning") is True
        assert provider.validate_model_name("grok-4.20") is True
        assert provider.validate_model_name("grok-build-0.1") is True

        # Test invalid model
        assert provider.validate_model_name("invalid-model") is False
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    @pytest.mark.parametrize("retired", RETIRED_MODELS)
    def test_retired_models_are_rejected(self, retired):
        """Retired xAI IDs must not validate.

        grok-3 / grok-4 / grok-3-fast were removed from the xAI API. Accepting
        them would let the server send a model name the API will reject.
        """
        provider = XAIModelProvider("test-key")
        assert provider.validate_model_name(retired) is False

    @pytest.mark.parametrize("name", UNSOURCEABLE_ALIASES)
    def test_unsourceable_aliases_are_not_carried(self, name):
        """Two alias families xAI attests but vox deliberately does not carry.

        Cross-model aliases (grok-code-fast-1 -> grok-build-0.1) name a real,
        historically distinct model. xAI resolves them onto Grok Build today,
        but encoding that here asserts an equivalence vox does not control: if
        xAI un-aliases them, vox would go on silently substituting. A wrong ID
        that errors loudly teaches the caller; a silent substitution returns
        plausible output from the wrong model and nobody finds out.

        Moving pointers (*-latest) mean something different upstream over time
        (grok-latest resolves to grok-4.3, not the flagship). Baking one into a
        static registry is the same rot-by-standing-still this catalog fixes.
        """
        provider = XAIModelProvider("test-key")
        assert provider.validate_model_name(name) is False

    def test_resolve_model_name(self):
        """Test model name resolution."""
        provider = XAIModelProvider("test-key")

        # Shorthand resolution — "grok" points at the current flagship
        assert provider._resolve_model_name("grok") == "grok-4.5"
        assert provider._resolve_model_name("grok45") == "grok-4.5"
        assert provider._resolve_model_name("grok4.5") == "grok-4.5"
        assert provider._resolve_model_name("grok4.3") == "grok-4.3"
        assert provider._resolve_model_name("grok-4.20") == "grok-4.20-0309-reasoning"
        assert provider._resolve_model_name("grok-build") == "grok-build-0.1"

        # Test full name passthrough
        assert provider._resolve_model_name("grok-4.5") == "grok-4.5"
        assert provider._resolve_model_name("grok-4.3") == "grok-4.3"
        assert provider._resolve_model_name("grok-4.20-0309-non-reasoning") == "grok-4.20-0309-non-reasoning"

    def test_get_capabilities_grok_4_5(self):
        """Test getting model capabilities for Grok 4.5 (flagship)."""
        provider = XAIModelProvider("test-key")

        capabilities = provider.get_capabilities("grok-4.5")
        assert capabilities.model_name == "grok-4.5"
        assert capabilities.friendly_name == "X.AI (Grok 4.5)"
        assert capabilities.context_window == 500_000
        assert capabilities.provider == ProviderType.XAI
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_system_prompts is True
        assert capabilities.supports_streaming is True
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_json_mode is True
        assert capabilities.supports_images is True
        assert capabilities.allow_code_generation is True

        # Test temperature range
        assert capabilities.temperature_constraint.min_temp == 0.0
        assert capabilities.temperature_constraint.max_temp == 2.0
        assert capabilities.temperature_constraint.default_temp == 0.3

    def test_grok_4_5_uses_effort_level_reasoning(self):
        """Grok 4.5 accepts reasoning_effort (low/medium/high) per xAI's reasoning guide."""
        provider = XAIModelProvider("test-key")

        capabilities = provider.get_capabilities("grok-4.5")
        assert capabilities.get_effective_thinking_params("high") == {"effort": "high"}
        assert capabilities.get_effective_thinking_params("low") == {"effort": "low"}

    def test_get_capabilities_grok_4_3(self):
        """Test getting model capabilities for Grok 4.3."""
        provider = XAIModelProvider("test-key")

        capabilities = provider.get_capabilities("grok-4.3")
        assert capabilities.model_name == "grok-4.3"
        assert capabilities.friendly_name == "X.AI (Grok 4.3)"
        assert capabilities.context_window == 1_000_000
        assert capabilities.provider == ProviderType.XAI
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_function_calling is True

    def test_get_capabilities_grok_4_20_variants(self):
        """The 4.20 family splits reasoning from non-reasoning."""
        provider = XAIModelProvider("test-key")

        reasoning = provider.get_capabilities("grok-4.20-0309-reasoning")
        assert reasoning.context_window == 1_000_000
        assert reasoning.supports_extended_thinking is True

        non_reasoning = provider.get_capabilities("grok-4.20-0309-non-reasoning")
        assert non_reasoning.context_window == 1_000_000
        assert non_reasoning.supports_extended_thinking is False

        multi_agent = provider.get_capabilities("grok-4.20-multi-agent-0309")
        assert multi_agent.context_window == 1_000_000
        assert multi_agent.supports_extended_thinking is True

    def test_get_capabilities_grok_build(self):
        """Grok Build 0.1 is the code-focused model."""
        provider = XAIModelProvider("test-key")

        capabilities = provider.get_capabilities("grok-build-0.1")
        assert capabilities.model_name == "grok-build-0.1"
        assert capabilities.context_window == 256_000
        assert capabilities.allow_code_generation is True

    def test_get_capabilities_with_shorthand(self):
        """Test getting model capabilities with shorthand."""
        provider = XAIModelProvider("test-key")

        capabilities = provider.get_capabilities("grok")
        assert capabilities.model_name == "grok-4.5"  # Should resolve to full name
        assert capabilities.context_window == 500_000

    def test_unsupported_model_capabilities(self):
        """Test error handling for unsupported models."""
        provider = XAIModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider xai"):
            provider.get_capabilities("invalid-model")

    @pytest.mark.parametrize("retired", ["grok-4", "grok-3", "grok-3-fast"])
    def test_retired_model_capabilities_raise(self, retired):
        """Asking for a retired model must fail loudly, not silently resolve."""
        provider = XAIModelProvider("test-key")

        with pytest.raises(ValueError, match="Unsupported model"):
            provider.get_capabilities(retired)

    def test_extended_thinking_flags(self):
        """X.AI capabilities should expose extended thinking support correctly."""
        provider = XAIModelProvider("test-key")

        thinking_aliases = ["grok-4.5", "grok", "grok-4.3", "grok-4.20"]
        for alias in thinking_aliases:
            assert provider.get_capabilities(alias).supports_extended_thinking is True

        non_thinking = ["grok-4.20-0309-non-reasoning", "grok-4.20-non-reasoning"]
        for alias in non_thinking:
            assert provider.get_capabilities(alias).supports_extended_thinking is False

    def test_provider_type(self):
        """Test provider type identification."""
        provider = XAIModelProvider("test-key")
        assert provider.get_provider_type() == ProviderType.XAI

    def test_preferred_models_are_live(self):
        """Every category preference must name a model the registry actually serves."""
        from tools.models import ToolModelCategory

        provider = XAIModelProvider("test-key")
        allowed = list(provider.MODEL_CAPABILITIES.keys())

        for category in ToolModelCategory:
            preferred = provider.get_preferred_model(category, allowed)
            assert preferred in provider.MODEL_CAPABILITIES, f"{category} prefers unknown model {preferred!r}"

    @patch.dict(os.environ, {"XAI_ALLOWED_MODELS": "grok-4.3"})
    def test_model_restrictions(self):
        """Test model restrictions functionality."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = XAIModelProvider("test-key")

        # grok-4.3 should be allowed
        assert provider.validate_model_name("grok-4.3") is True
        assert provider.validate_model_name("grok43") is True  # Shorthand for grok-4.3

        # grok should be blocked (resolves to grok-4.5 which is not allowed)
        assert provider.validate_model_name("grok") is False

        # grok-4.5 should be blocked by restrictions
        assert provider.validate_model_name("grok-4.5") is False

    @patch.dict(os.environ, {"XAI_ALLOWED_MODELS": "grok,grok-4.3"})
    def test_multiple_model_restrictions(self):
        """Test multiple models in restrictions."""
        # Clear cached restriction service
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = XAIModelProvider("test-key")

        # Shorthand "grok" should be allowed (resolves to grok-4.5)
        assert provider.validate_model_name("grok") is True

        # Full name "grok-4.5" should NOT be allowed (only shorthand "grok" is in restriction list)
        assert provider.validate_model_name("grok-4.5") is False

        # "grok-4.3" should be allowed (explicitly listed)
        assert provider.validate_model_name("grok-4.3") is True

        # Shorthand "grok43" should be allowed (resolves to grok-4.3)
        assert provider.validate_model_name("grok43") is True

    @patch.dict(os.environ, {"XAI_ALLOWED_MODELS": "grok,grok-4.3,grok-4.5"})
    def test_both_shorthand_and_full_name_allowed(self):
        """Test that both shorthand and full name can be allowed."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = XAIModelProvider("test-key")

        # Both shorthand and full name should be allowed
        assert provider.validate_model_name("grok") is True  # Resolves to grok-4.5
        assert provider.validate_model_name("grok-4.3") is True
        assert provider.validate_model_name("grok-4.5") is True

        # Other models should not be allowed
        assert provider.validate_model_name("grok-build-0.1") is False

    @patch.dict(os.environ, {"XAI_ALLOWED_MODELS": ""})
    def test_empty_restrictions_allows_all(self):
        """Test that empty restrictions allow all models."""
        # Clear cached restriction service
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

        provider = XAIModelProvider("test-key")

        assert provider.validate_model_name("grok-4.5") is True
        assert provider.validate_model_name("grok-4.3") is True
        assert provider.validate_model_name("grok-4.20-0309-reasoning") is True
        assert provider.validate_model_name("grok") is True
        assert provider.validate_model_name("grok-build-0.1") is True

    def test_friendly_name(self):
        """Test friendly name constant."""
        provider = XAIModelProvider("test-key")
        assert provider.FRIENDLY_NAME == "X.AI"

        capabilities = provider.get_capabilities("grok-4.3")
        assert capabilities.friendly_name == "X.AI (Grok 4.3)"

    def test_supported_models_structure(self):
        """Test that MODEL_CAPABILITIES has the correct structure."""
        provider = XAIModelProvider("test-key")

        # Check that all expected base models are present
        assert "grok-4.5" in provider.MODEL_CAPABILITIES
        assert "grok-4.3" in provider.MODEL_CAPABILITIES
        assert "grok-4.20-0309-reasoning" in provider.MODEL_CAPABILITIES
        assert "grok-4.20-0309-non-reasoning" in provider.MODEL_CAPABILITIES
        assert "grok-4.20-multi-agent-0309" in provider.MODEL_CAPABILITIES
        assert "grok-build-0.1" in provider.MODEL_CAPABILITIES

        # No retired model may reappear as a base model
        for retired in ["grok-4", "grok-3", "grok-3-fast"]:
            assert retired not in provider.MODEL_CAPABILITIES

        # Check model configs have required fields
        from providers.shared import ModelCapabilities

        flagship = provider.MODEL_CAPABILITIES["grok-4.5"]
        assert isinstance(flagship, ModelCapabilities)
        assert flagship.context_window == 500_000
        assert flagship.supports_extended_thinking is True

        # "grok" is the ergonomic shorthand and must point at the flagship
        assert "grok" in flagship.aliases

        # No alias anywhere may name a retired model
        all_aliases = {alias for caps in provider.MODEL_CAPABILITIES.values() for alias in caps.aliases}
        assert all_aliases.isdisjoint({"grok-4", "grok4", "grok-3", "grok3", "grok-3-fast", "grokfast", "grok3fast"})

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_resolves_alias_before_api_call(self, mock_openai_class):
        """Test that generate_content resolves aliases before making API calls.

        This is the CRITICAL test that ensures aliases like 'grok' get resolved
        to 'grok-4.5' before being sent to X.AI API.
        """
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "grok-4.5"  # API returns the resolved model name
        mock_response.id = "test-id"
        mock_response.created = 1234567890
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        provider = XAIModelProvider("test-key")

        # Call generate_content with alias 'grok'
        result = provider.generate_content(
            prompt="Test prompt",
            model_name="grok",
            temperature=0.7,  # This should be resolved to "grok-4.5"
        )

        # Verify the API was called with the RESOLVED model name
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        # CRITICAL ASSERTION: The API should receive "grok-4.5", not "grok"
        assert call_kwargs["model"] == "grok-4.5", f"Expected 'grok-4.5' but API received '{call_kwargs['model']}'"

        # Verify other parameters
        assert call_kwargs["temperature"] == 0.7
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

        # Verify response
        assert result.content == "Test response"
        assert result.model_name == "grok-4.5"  # Should be the resolved name

    @patch("providers.openai_compatible.OpenAI")
    def test_generate_content_other_aliases(self, mock_openai_class):
        """Test other alias resolutions in generate_content."""
        from unittest.mock import MagicMock

        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = XAIModelProvider("test-key")

        # Test grok45 -> grok-4.5
        mock_response.model = "grok-4.5"
        provider.generate_content(prompt="Test", model_name="grok45", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-4.5"

        # Test grok-4.5 -> grok-4.5
        provider.generate_content(prompt="Test", model_name="grok-4.5", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-4.5"

        # Test grok4.3 -> grok-4.3
        mock_response.model = "grok-4.3"
        provider.generate_content(prompt="Test", model_name="grok4.3", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-4.3"

        # Test grok-4.20 -> grok-4.20-0309-reasoning
        mock_response.model = "grok-4.20-0309-reasoning"
        provider.generate_content(prompt="Test", model_name="grok-4.20", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-4.20-0309-reasoning"

        # Test grok-build -> grok-build-0.1
        mock_response.model = "grok-build-0.1"
        provider.generate_content(prompt="Test", model_name="grok-build", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "grok-build-0.1"
