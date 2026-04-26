"""Unit tests for the ThinkingConstraint abstraction.

Tests the ABC, all three concrete implementations, the factory method,
normalization, and integration with ModelCapabilities.get_effective_thinking_params().
"""

import pytest

from providers.shared.model_capabilities import ModelCapabilities
from providers.shared.provider_type import ProviderType
from providers.shared.thinking import (
    DEFAULT_EFFORT_MAP,
    DEFAULT_TOKEN_BUDGETS,
    VALID_THINKING_MODES,
    AlwaysOnThinkingConstraint,
    EffortLevelThinkingConstraint,
    ThinkingConstraint,
    TokenBudgetThinkingConstraint,
)

# ---------------------------------------------------------------------------
# ThinkingConstraint.normalize_mode
# ---------------------------------------------------------------------------


class TestNormalizeMode:
    """Tests for the static normalize_mode method."""

    @pytest.mark.parametrize("mode", sorted(VALID_THINKING_MODES))
    def test_valid_modes_pass_through(self, mode):
        assert ThinkingConstraint.normalize_mode(mode) == mode

    def test_strips_whitespace(self):
        assert ThinkingConstraint.normalize_mode("  high  ") == "high"

    def test_case_insensitive(self):
        assert ThinkingConstraint.normalize_mode("HIGH") == "high"
        assert ThinkingConstraint.normalize_mode("Medium") == "medium"

    def test_unknown_mode_defaults_to_medium(self):
        assert ThinkingConstraint.normalize_mode("turbo") == "medium"
        assert ThinkingConstraint.normalize_mode("") == "medium"
        assert ThinkingConstraint.normalize_mode("999") == "medium"


# ---------------------------------------------------------------------------
# ThinkingConstraint.create  (factory)
# ---------------------------------------------------------------------------


class TestFactory:
    """Tests for the static create factory method."""

    def test_creates_token_budget(self):
        c = ThinkingConstraint.create("token_budget", max_thinking_tokens=32768)
        assert isinstance(c, TokenBudgetThinkingConstraint)
        assert c.max_tokens == 32768

    def test_creates_effort_level(self):
        c = ThinkingConstraint.create("effort_level")
        assert isinstance(c, EffortLevelThinkingConstraint)

    def test_creates_always_on(self):
        c = ThinkingConstraint.create("always_on")
        assert isinstance(c, AlwaysOnThinkingConstraint)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown thinking constraint type"):
            ThinkingConstraint.create("quantum")

    def test_token_budget_zero_tokens_gets_min_floor(self):
        c = ThinkingConstraint.create("token_budget", max_thinking_tokens=0)
        assert isinstance(c, TokenBudgetThinkingConstraint)
        # min_tokens floor (1024) prevents degenerate zero budgets
        assert c.resolve("max") == {"thinking_budget": 1024}


# ---------------------------------------------------------------------------
# TokenBudgetThinkingConstraint
# ---------------------------------------------------------------------------


class TestTokenBudgetThinkingConstraint:
    """Tests for the token-budget implementation (Gemini, Anthropic)."""

    def setup_method(self):
        self.constraint = TokenBudgetThinkingConstraint(max_tokens=32768)

    def test_resolve_medium(self):
        result = self.constraint.resolve("medium")
        expected = int(32768 * DEFAULT_TOKEN_BUDGETS["medium"])
        assert result == {"thinking_budget": expected}

    def test_resolve_all_valid_modes(self):
        for mode in VALID_THINKING_MODES:
            result = self.constraint.resolve(mode)
            raw_budget = int(32768 * DEFAULT_TOKEN_BUDGETS[mode])
            expected_budget = max(1024, raw_budget)  # min_tokens floor
            assert result == {"thinking_budget": expected_budget}, f"Failed for mode={mode}"

    def test_resolve_max_returns_full_budget(self):
        result = self.constraint.resolve("max")
        assert result == {"thinking_budget": 32768}

    def test_resolve_minimal_clamps_to_min_tokens(self):
        """Minimal mode on 32768 tokens = 163, but floor is 1024."""
        result = self.constraint.resolve("minimal")
        assert result == {"thinking_budget": 1024}

    def test_invalid_mode_falls_back_to_medium(self):
        result = self.constraint.resolve("turbo")
        medium_result = self.constraint.resolve("medium")
        assert result == medium_result

    def test_default_mode(self):
        assert self.constraint.get_default_mode() == "medium"

    def test_custom_default_mode(self):
        c = TokenBudgetThinkingConstraint(max_tokens=10000, default_mode="high")
        assert c.get_default_mode() == "high"

    def test_custom_budgets(self):
        custom = {"minimal": 0.0, "low": 0.1, "medium": 0.5, "high": 0.9, "max": 1.0}
        c = TokenBudgetThinkingConstraint(max_tokens=10000, budgets=custom)
        assert c.resolve("medium") == {"thinking_budget": 5000}
        assert c.resolve("max") == {"thinking_budget": 10000}
        # 0% of 10000 = 0, but min_tokens floor applies
        assert c.resolve("minimal") == {"thinking_budget": 1024}

    def test_description_contains_max_tokens(self):
        desc = self.constraint.get_description()
        assert "32768" in desc

    def test_gemini_budget_backward_compat(self):
        """Ensure the default budgets match the old Gemini THINKING_BUDGETS exactly."""
        # These values were previously hardcoded in gemini.py
        expected = {
            "minimal": 0.005,
            "low": 0.08,
            "medium": 0.33,
            "high": 0.67,
            "max": 1.0,
        }
        assert DEFAULT_TOKEN_BUDGETS == expected

    def test_resolve_returns_int(self):
        """Budget should always be an integer (no fractional tokens)."""
        c = TokenBudgetThinkingConstraint(max_tokens=50000)
        result = c.resolve("medium")
        assert isinstance(result["thinking_budget"], int)

    def test_min_tokens_floor_prevents_degenerate_values(self):
        """Small max_tokens still produces at least min_tokens."""
        c = TokenBudgetThinkingConstraint(max_tokens=100)
        # 100 * 0.33 = 33, but floor is 1024
        assert c.resolve("medium") == {"thinking_budget": 1024}

    def test_custom_min_tokens(self):
        """Custom min_tokens floor is respected."""
        c = TokenBudgetThinkingConstraint(max_tokens=32768, min_tokens=2048)
        # minimal: 32768 * 0.005 = 163, but floor is 2048
        assert c.resolve("minimal") == {"thinking_budget": 2048}
        # medium: 32768 * 0.33 = 10813, above floor
        assert c.resolve("medium") == {"thinking_budget": 10813}

    def test_min_tokens_zero_disables_floor(self):
        """Setting min_tokens=0 allows zero budgets (opt-out)."""
        c = TokenBudgetThinkingConstraint(max_tokens=100, min_tokens=0)
        result = c.resolve("minimal")
        assert result == {"thinking_budget": 0}


# ---------------------------------------------------------------------------
# EffortLevelThinkingConstraint
# ---------------------------------------------------------------------------


class TestEffortLevelThinkingConstraint:
    """Tests for the effort-level implementation (OpenAI)."""

    def setup_method(self):
        self.constraint = EffortLevelThinkingConstraint()

    def test_resolve_all_valid_modes(self):
        expected = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        }
        for mode, effort in expected.items():
            assert self.constraint.resolve(mode) == {"effort": effort}, f"Failed for mode={mode}"

    def test_invalid_mode_falls_back_to_medium(self):
        result = self.constraint.resolve("turbo")
        assert result == {"effort": "medium"}

    def test_default_mode(self):
        assert self.constraint.get_default_mode() == "medium"

    def test_custom_default_mode(self):
        c = EffortLevelThinkingConstraint(default_mode="high")
        assert c.get_default_mode() == "high"

    def test_custom_effort_map(self):
        custom = {"minimal": "low", "low": "low", "medium": "low", "high": "medium", "max": "high"}
        c = EffortLevelThinkingConstraint(effort_map=custom)
        assert c.resolve("medium") == {"effort": "low"}
        assert c.resolve("high") == {"effort": "medium"}

    def test_description_lists_effort_levels(self):
        desc = self.constraint.get_description()
        for level in ("low", "medium", "high"):
            assert level in desc

    def test_default_effort_map_values(self):
        """Ensure the default map covers all 5 thinking modes."""
        assert set(DEFAULT_EFFORT_MAP.keys()) == VALID_THINKING_MODES
        assert set(DEFAULT_EFFORT_MAP.values()) <= {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# AlwaysOnThinkingConstraint
# ---------------------------------------------------------------------------


class TestAlwaysOnThinkingConstraint:
    """Tests for the always-on implementation (DeepSeek, Moonshot)."""

    def setup_method(self):
        self.constraint = AlwaysOnThinkingConstraint()

    @pytest.mark.parametrize("mode", sorted(VALID_THINKING_MODES))
    def test_resolve_returns_empty_dict(self, mode):
        assert self.constraint.resolve(mode) == {}

    def test_resolve_invalid_mode_still_empty(self):
        assert self.constraint.resolve("turbo") == {}

    def test_default_mode(self):
        assert self.constraint.get_default_mode() == "medium"

    def test_description(self):
        desc = self.constraint.get_description()
        assert "always" in desc.lower()


# ---------------------------------------------------------------------------
# ModelCapabilities.get_effective_thinking_params integration
# ---------------------------------------------------------------------------


class TestModelCapabilitiesThinkingIntegration:
    """Tests for get_effective_thinking_params on ModelCapabilities."""

    def _make_caps(self, constraint=None, **kwargs):
        defaults = {
            "provider": ProviderType.GOOGLE,
            "model_name": "test-model",
            "friendly_name": "Test",
        }
        defaults.update(kwargs)
        return ModelCapabilities(thinking_constraint=constraint, **defaults)

    def test_no_constraint_returns_none(self):
        caps = self._make_caps(constraint=None)
        assert caps.get_effective_thinking_params() is None
        assert caps.get_effective_thinking_params("high") is None

    def test_token_budget_with_explicit_mode(self):
        c = TokenBudgetThinkingConstraint(max_tokens=32768)
        caps = self._make_caps(constraint=c)
        result = caps.get_effective_thinking_params("high")
        assert result == {"thinking_budget": int(32768 * 0.67)}

    def test_token_budget_with_default_mode(self):
        c = TokenBudgetThinkingConstraint(max_tokens=32768, default_mode="low")
        caps = self._make_caps(constraint=c)
        result = caps.get_effective_thinking_params()  # no mode → uses default
        assert result == {"thinking_budget": int(32768 * 0.08)}

    def test_effort_level_with_explicit_mode(self):
        c = EffortLevelThinkingConstraint()
        caps = self._make_caps(constraint=c)
        assert caps.get_effective_thinking_params("max") == {"effort": "high"}

    def test_effort_level_with_default_mode(self):
        c = EffortLevelThinkingConstraint(default_mode="high")
        caps = self._make_caps(constraint=c)
        assert caps.get_effective_thinking_params() == {"effort": "high"}

    def test_always_on_returns_empty_dict(self):
        c = AlwaysOnThinkingConstraint()
        caps = self._make_caps(constraint=c)
        result = caps.get_effective_thinking_params("high")
        assert result == {}

    def test_always_on_default_returns_empty_dict(self):
        c = AlwaysOnThinkingConstraint()
        caps = self._make_caps(constraint=c)
        result = caps.get_effective_thinking_params()
        assert result == {}

    def test_none_mode_uses_constraint_default(self):
        """Passing None for thinking_mode should use the constraint's default."""
        c = TokenBudgetThinkingConstraint(max_tokens=10000, default_mode="high")
        caps = self._make_caps(constraint=c)
        result = caps.get_effective_thinking_params(None)
        assert result == {"thinking_budget": int(10000 * 0.67)}


# ---------------------------------------------------------------------------
# Provider MODEL_CAPABILITIES have correct constraints
# ---------------------------------------------------------------------------


class TestProviderConstraintWiring:
    """Verify that provider MODEL_CAPABILITIES include the expected constraints."""

    def test_deepseek_has_always_on(self):
        from providers.deepseek import DeepSeekProvider

        caps = DeepSeekProvider.MODEL_CAPABILITIES["deepseek-v4-pro"]
        assert isinstance(caps.thinking_constraint, AlwaysOnThinkingConstraint)

    def test_moonshot_thinking_turbo_has_always_on(self):
        from providers.moonshot import MoonshotProvider

        caps = MoonshotProvider.MODEL_CAPABILITIES["kimi-k2-thinking-turbo"]
        assert isinstance(caps.thinking_constraint, AlwaysOnThinkingConstraint)

    def test_moonshot_k26_has_always_on(self):
        from providers.moonshot import MoonshotProvider

        caps = MoonshotProvider.MODEL_CAPABILITIES["kimi-k2.6"]
        assert isinstance(caps.thinking_constraint, AlwaysOnThinkingConstraint)


# ---------------------------------------------------------------------------
# Thinking-mode extra_body merge — regression test for caller-supplied keys
# ---------------------------------------------------------------------------


class TestThinkingExtraBodyMerge:
    """Providers must merge the thinking toggle into any caller-supplied
    extra_body, not skip it via setdefault on the outer dict."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def test_moonshot_merges_thinking_into_caller_extra_body(self):
        from unittest.mock import MagicMock, patch

        from providers.moonshot import MoonshotProvider
        from providers.openai_compatible import OpenAICompatibleProvider

        provider = MoonshotProvider("test-key")
        with patch.object(
            OpenAICompatibleProvider, "generate_content", return_value=MagicMock()
        ) as super_gen:
            provider.generate_content(
                prompt="hi",
                model_name="kimi-k2.6",
                temperature=1.0,
                extra_body={"my_custom": "value"},
            )
        forwarded = super_gen.call_args.kwargs["extra_body"]
        assert forwarded == {
            "my_custom": "value",
            "thinking": {"type": "enabled"},
        }

    def test_deepseek_merges_thinking_into_caller_extra_body(self):
        from unittest.mock import MagicMock, patch

        from providers.deepseek import DeepSeekProvider
        from providers.openai_compatible import OpenAICompatibleProvider

        provider = DeepSeekProvider("test-key")
        with patch.object(
            OpenAICompatibleProvider, "generate_content", return_value=MagicMock()
        ) as super_gen:
            provider.generate_content(
                prompt="hi",
                model_name="deepseek-v4-pro",
                temperature=1.0,
                extra_body={"my_custom": "value"},
            )
        forwarded = super_gen.call_args.kwargs["extra_body"]
        assert forwarded == {
            "my_custom": "value",
            "thinking": {"type": "enabled"},
        }

    def test_caller_thinking_value_wins(self):
        """If a caller explicitly disables thinking via extra_body, the
        provider must respect that and not override with the always-on default."""
        from unittest.mock import MagicMock, patch

        from providers.deepseek import DeepSeekProvider
        from providers.openai_compatible import OpenAICompatibleProvider

        provider = DeepSeekProvider("test-key")
        with patch.object(
            OpenAICompatibleProvider, "generate_content", return_value=MagicMock()
        ) as super_gen:
            provider.generate_content(
                prompt="hi",
                model_name="deepseek-v4-pro",
                temperature=1.0,
                extra_body={"thinking": {"type": "disabled"}},
            )
        forwarded = super_gen.call_args.kwargs["extra_body"]
        assert forwarded == {"thinking": {"type": "disabled"}}
