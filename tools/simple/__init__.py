"""
Simple tools for Vox MCP.

Simple tools follow a basic request → AI model → response pattern.
They inherit from SimpleTool which provides streamlined functionality
for tools that don't need multi-step coordination.

Available simple tools:
- chat: Multi-model AI gateway with conversation memory
"""

from .base import SimpleTool

__all__ = ["SimpleTool"]
