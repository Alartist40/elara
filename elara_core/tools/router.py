"""
ToolRouter - Executes tool calls in a sandboxed environment.
"""

import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from elara_core.tools.schema import ToolSchema
from elara_core.tools.parser import ToolCallParser, ToolCall


class ToolResult:
    """Result from a tool execution."""
    def __init__(self, name: str, output: str, success: bool, duration_ms: float):
        self.name = name
        self.output = output
        self.success = success
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "output": self.output,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


class ToolRouter:
    """
    Routes tool calls to implementations and manages execution.
    """

    def __init__(
        self,
        schema_path: Optional[str] = None,
        max_iterations: int = 3,
        timeout_seconds: int = 30,
        allowed_tools: Optional[List[str]] = None,
    ):
        self.schema = ToolSchema(schema_path)
        self.parser = ToolCallParser()
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.allowed_tools = set(allowed_tools) if allowed_tools else None

        # Built-in tool implementations
        self._implementations: Dict[str, callable] = {
            "calculator": self._tool_calculator,
            "search": self._tool_search_stub,
            "vision_analyze": self._tool_vision_stub,
        }

    def execute(self, text: str) -> List[ToolResult]:
        """
        Parse and execute all tool calls in text.

        Args:
            text: Model output that may contain tool calls.

        Returns:
            List of ToolResult objects.
        """
        calls = self.parser.parse(text)
        results = []

        for call in calls[:self.max_iterations]:
            if self.allowed_tools and call.name not in self.allowed_tools:
                results.append(ToolResult(
                    name=call.name,
                    output=f"Tool '{call.name}' is not allowed.",
                    success=False,
                    duration_ms=0.0,
                ))
                continue

            if not self.schema.validate_params(call.name, call.parameters):
                results.append(ToolResult(
                    name=call.name,
                    output=f"Invalid parameters for tool '{call.name}'.",
                    success=False,
                    duration_ms=0.0,
                ))
                continue

            result = self._execute_single(call)
            results.append(result)

        return results

    def _execute_single(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        start = time.time()

        impl = self._implementations.get(call.name)
        if impl is None:
            return ToolResult(
                name=call.name,
                output=f"No implementation for tool '{call.name}'.",
                success=False,
                duration_ms=0.0,
            )

        try:
            output = impl(**call.parameters)
            duration = (time.time() - start) * 1000
            return ToolResult(
                name=call.name, output=str(output),
                success=True, duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolResult(
                name=call.name, output=f"Error: {str(e)}",
                success=False, duration_ms=duration,
            )

    def register_tool(self, name: str, func: callable) -> None:
        """Register a custom tool implementation."""
        self._implementations[name] = func

    # === Built-in tools ===

    @staticmethod
    def _tool_calculator(expression: str, **kwargs) -> str:
        """Safe math expression evaluator."""
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters."
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _tool_search_stub(query: str, **kwargs) -> str:
        """Stub search tool (replace with actual API)."""
        return f"[Search results for: '{query}'] (stub - connect to search API)"

    @staticmethod
    def _tool_vision_stub(image_path: str, question: str = "", **kwargs) -> str:
        """Stub vision analysis (replace with actual vision model)."""
        return f"[Vision analysis of '{image_path}': {question}] (stub - connect to vision model)"
