"""Helper types for validating model thinking/reasoning parameters."""

from abc import ABC, abstractmethod
from typing import Any, Optional

__all__ = [
    "ThinkingConstraint",
    "TokenBudgetThinkingConstraint",
    "EffortLevelThinkingConstraint",
    "AlwaysOnThinkingConstraint",
]

# Valid thinking modes accepted from the tool layer.
VALID_THINKING_MODES = {"minimal", "low", "medium", "high", "max"}

# Default budget percentages of max_thinking_tokens for each mode.
DEFAULT_TOKEN_BUDGETS: dict[str, float] = {
    "minimal": 0.005,  # 0.5% — minimal thinking for fast responses
    "low": 0.08,  # 8% — light reasoning tasks
    "medium": 0.33,  # 33% — balanced reasoning (default)
    "high": 0.67,  # 67% — complex analysis
    "max": 1.0,  # 100% — full thinking budget
}

# Mapping from 5-level thinking modes to OpenAI's 3-level effort scale.
DEFAULT_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
}


class ThinkingConstraint(ABC):
    """Contract for thinking/reasoning parameter resolution.

    Concrete providers describe their reasoning behaviour by creating
    subclasses that translate a universal ``thinking_mode`` string
    (minimal/low/medium/high/max) into provider-specific API parameters.

    Mirrors :class:`TemperatureConstraint` for temperature handling.
    """

    @abstractmethod
    def resolve(self, thinking_mode: str) -> dict[str, Any]:
        """Return provider-specific API parameters for the given thinking mode.

        Returns an empty dict when thinking is always-on with no configurable
        parameters.  The caller is responsible for merging the returned dict
        into the provider's request payload.
        """

    @abstractmethod
    def get_default_mode(self) -> str:
        """Return the default thinking mode string."""

    @abstractmethod
    def get_description(self) -> str:
        """Describe the supported thinking modes for error messages."""

    @staticmethod
    def normalize_mode(thinking_mode: str) -> str:
        """Normalize a thinking mode string to a valid value.

        Returns ``"medium"`` for unrecognised inputs.
        """
        mode = thinking_mode.strip().lower()
        return mode if mode in VALID_THINKING_MODES else "medium"

    @staticmethod
    def create(
        constraint_type: str,
        max_thinking_tokens: int = 0,
    ) -> "ThinkingConstraint":
        """Factory that yields the appropriate constraint for a configuration hint.

        Args:
            constraint_type: One of ``"token_budget"``, ``"effort_level"``,
                or ``"always_on"``.
            max_thinking_tokens: Maximum thinking tokens for the model (used
                by ``token_budget`` constraints).
        """
        if constraint_type == "token_budget":
            return TokenBudgetThinkingConstraint(max_tokens=max_thinking_tokens)
        if constraint_type == "effort_level":
            return EffortLevelThinkingConstraint()
        if constraint_type == "always_on":
            return AlwaysOnThinkingConstraint()
        raise ValueError(f"Unknown thinking constraint type: {constraint_type!r}")


class TokenBudgetThinkingConstraint(ThinkingConstraint):
    """Constraint for providers that accept a token budget (Gemini, Anthropic).

    Translates thinking modes into integer token counts by applying a
    percentage of the model's ``max_thinking_tokens``.

    A ``min_tokens`` floor (default 1024) prevents degenerate budget values
    that would be rejected by provider APIs (e.g. Anthropic requires
    ``budget_tokens >= 1024``).
    """

    def __init__(
        self,
        max_tokens: int,
        budgets: Optional[dict[str, float]] = None,
        default_mode: str = "medium",
        min_tokens: int = 1024,
    ):
        self.max_tokens = max_tokens
        self.budgets = budgets or dict(DEFAULT_TOKEN_BUDGETS)
        self._default_mode = default_mode
        self.min_tokens = min_tokens

    def resolve(self, thinking_mode: str) -> dict[str, Any]:
        mode = self.normalize_mode(thinking_mode)
        fraction = self.budgets.get(mode, self.budgets["medium"])
        budget = max(self.min_tokens, int(self.max_tokens * fraction))
        return {"thinking_budget": budget}

    def get_default_mode(self) -> str:
        return self._default_mode

    def get_description(self) -> str:
        return (
            f"Token budget thinking (max {self.max_tokens} tokens). "
            f"Modes: {', '.join(sorted(self.budgets.keys()))}"
        )


class EffortLevelThinkingConstraint(ThinkingConstraint):
    """Constraint for providers that accept an effort level string (OpenAI).

    Maps the 5-level thinking modes to the provider's 3-level effort scale
    (low/medium/high).
    """

    def __init__(
        self,
        effort_map: Optional[dict[str, str]] = None,
        default_mode: str = "medium",
    ):
        self.effort_map = effort_map or dict(DEFAULT_EFFORT_MAP)
        self._default_mode = default_mode

    def resolve(self, thinking_mode: str) -> dict[str, Any]:
        mode = self.normalize_mode(thinking_mode)
        effort = self.effort_map.get(mode, "medium")
        return {"effort": effort}

    def get_default_mode(self) -> str:
        return self._default_mode

    def get_description(self) -> str:
        return f"Effort level thinking. Maps to: {', '.join(sorted(set(self.effort_map.values())))}"


class AlwaysOnThinkingConstraint(ThinkingConstraint):
    """Constraint for providers where thinking is always active (DeepSeek, Moonshot).

    Returns an empty dict because the provider's reasoning is inherent and
    requires no additional API parameters.
    """

    def resolve(self, thinking_mode: str) -> dict[str, Any]:
        return {}

    def get_default_mode(self) -> str:
        return "medium"

    def get_description(self) -> str:
        return "Always-on thinking (no configurable parameters)"
