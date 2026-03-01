"""
Tests for Chat tool - validating SimpleTool architecture

This module contains unit tests to ensure that the Chat tool
(now using SimpleTool architecture) maintains proper functionality.
"""

import json
from unittest.mock import patch

import pytest

from tools.chat import ChatRequest, ChatTool
from tools.shared.exceptions import ToolExecutionError


class TestChatTool:
    """Test suite for ChatSimple tool"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tool = ChatTool()

    def test_tool_metadata(self):
        """Test that tool metadata matches requirements"""
        assert self.tool.get_name() == "chat"
        assert "Multi-model AI gateway" in self.tool.get_description()
        assert self.tool.get_system_prompt() is not None
        assert self.tool.get_default_temperature() > 0
        assert self.tool.get_model_category() is not None

    def test_schema_structure(self):
        """Test that schema has correct structure"""
        schema = self.tool.get_input_schema()

        # Basic schema structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Required fields
        assert "prompt" in schema["required"]

        # Properties
        properties = schema["properties"]
        assert "prompt" in properties
        assert "absolute_file_paths" in properties
        assert "images" in properties

        # working_directory_absolute_path removed (not needed for pure API passthrough)
        assert "working_directory_absolute_path" not in properties

    def test_request_model_validation(self):
        """Test that the request model validates correctly"""
        # Test valid request
        request_data = {
            "prompt": "Test prompt",
            "absolute_file_paths": ["test.txt"],
            "images": ["test.png"],
            "model": "anthropic/claude-opus-4.1",
            "temperature": 0.7,
        }

        request = ChatRequest(**request_data)
        assert request.prompt == "Test prompt"
        assert request.absolute_file_paths == ["test.txt"]
        assert request.images == ["test.png"]
        assert request.model == "anthropic/claude-opus-4.1"
        assert request.temperature == 0.7

    def test_required_fields(self):
        """Test that required fields are enforced"""
        # Missing prompt should raise validation error
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(model="anthropic/claude-opus-4.1")

    def test_model_availability(self):
        """Test that model availability works"""
        models = self.tool._get_available_models()
        assert len(models) > 0  # Should have some models
        assert isinstance(models, list)

    def test_model_field_schema(self):
        """Test that model field schema generation works correctly"""
        schema = self.tool.get_model_field_schema()

        assert schema["type"] == "string"
        assert "description" in schema

        # Description should route callers to listmodels, regardless of mode
        assert "listmodels" in schema["description"]
        if self.tool.is_effective_auto_mode():
            assert "auto mode" in schema["description"].lower()
        else:
            import config

            assert f"'{config.DEFAULT_MODEL}'" in schema["description"]

    @pytest.mark.asyncio
    async def test_prompt_preparation(self):
        """Test that prompt preparation works correctly"""
        request = ChatRequest(
            prompt="Test prompt",
            absolute_file_paths=[],
        )

        with patch.object(self.tool, "handle_prompt_file_with_fallback", return_value="Test prompt"):
            with patch.object(self.tool, "_prepare_file_content_for_prompt", return_value=("", [])):
                with patch.object(self.tool, "_validate_token_limit"):
                    prompt = await self.tool.prepare_prompt(request)

                    assert prompt == "Test prompt"

    def test_response_formatting(self):
        """Test that response formatting works correctly"""
        response = "Test response content"
        request = ChatRequest(prompt="Test")

        formatted = self.tool.format_response(response, request)

        assert formatted == response

    def test_tool_name(self):
        """Test tool name is correct"""
        assert self.tool.get_name() == "chat"

    def test_convenience_methods(self):
        """Test SimpleTool convenience methods work correctly"""
        assert self.tool.supports_custom_request_model()

        # Test that the tool fields are defined correctly
        tool_fields = self.tool.get_tool_fields()
        assert "prompt" in tool_fields
        assert "absolute_file_paths" in tool_fields
        assert "images" in tool_fields

        required_fields = self.tool.get_required_fields()
        assert "prompt" in required_fields


class TestChatRequestModel:
    """Test suite for ChatRequest model"""

    def test_field_descriptions(self):
        """Test that field descriptions are proper"""
        from tools.chat import CHAT_FIELD_DESCRIPTIONS

        # Field descriptions should exist and be descriptive
        assert len(CHAT_FIELD_DESCRIPTIONS["prompt"]) > 20
        files_desc = CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"].lower()
        assert "absolute" in files_desc
        assert "visual context" in CHAT_FIELD_DESCRIPTIONS["images"]

    def test_default_values(self):
        """Test that default values work correctly"""
        request = ChatRequest(prompt="Test")

        assert request.prompt == "Test"
        assert request.absolute_file_paths == []  # Should default to empty list
        assert request.images == []  # Should default to empty list

    def test_inheritance(self):
        """Test that ChatRequest properly inherits from ToolRequest"""
        from tools.shared.base_models import ToolRequest

        request = ChatRequest(prompt="Test")
        assert isinstance(request, ToolRequest)

        # Should have inherited fields
        assert hasattr(request, "model")
        assert hasattr(request, "temperature")
        assert hasattr(request, "thinking_mode")
        assert hasattr(request, "continuation_id")
        assert hasattr(request, "images")  # From base model too


if __name__ == "__main__":
    pytest.main([__file__])
