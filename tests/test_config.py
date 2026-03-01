"""
Tests for configuration
"""

from config import (
    DEFAULT_MODEL,
    TEMPERATURE_BALANCED,
    __version__,
)


class TestConfig:
    """Test configuration values"""

    def test_version_info(self):
        """Test version information exists"""
        assert isinstance(__version__, str)

    def test_model_config(self):
        """Test model configuration"""
        # DEFAULT_MODEL is set in conftest.py for tests
        assert DEFAULT_MODEL == "gemini-2.5-flash"

    def test_temperature_defaults(self):
        """Test temperature constants"""
        assert TEMPERATURE_BALANCED == 1.0
