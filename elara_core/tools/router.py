"""
Simplified ToolRouter - Only Calculator implementation.
"""

import time
import re
from typing import Dict, Any, List, Optional

class ToolResult:
    def __init__(self, name: str, output: str, success: bool, duration_ms: float):
        self.name = name
        self.output = output
        self.success = success
        self.duration_ms = duration_ms

class ToolRouter:
    """
    Routes tool calls to implementations.
    """
    def __init__(self):
        self._implementations = {
            "calculator": self._tool_calculator
        }
        # Matches digit-adjacent operators for basic math
        self.math_pattern = re.compile(r'\d\s*[\+\-\*/]\s*\d')

    def execute(self, query: str) -> Optional[ToolResult]:
        """
        Simple execution: if 'calculate' or math-like query, try calculator.
        """
        lower_query = query.lower()
        if "calculate" in lower_query or self.math_pattern.search(query):
            # Extract expression by removing 'calculate' and trimming
            expression = re.sub(r'(?i)calculate\s*', '', query).strip()
            return self._execute_single("calculator", expression)
        return None

    def _execute_single(self, name: str, expression: str) -> ToolResult:
        start = time.time()
        impl = self._implementations.get(name)
        if not impl:
            return ToolResult(name, "Tool not found", False, 0)

        try:
            output = impl(expression)
            duration = (time.time() - start) * 1000
            return ToolResult(name, output, True, duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolResult(name, str(e), False, duration)

    @staticmethod
    def _tool_calculator(expression: str) -> str:
        allowed_chars = set("0123456789+-*/().% ")
        # Clean expression
        clean_expr = "".join(c for c in expression if c in allowed_chars)
        if not clean_expr: return "Error: No valid expression"
        try:
            # Note: eval is dangerous, but we restricted chars.
            # In production, a math parser like numexpr or simpleeval is better.
            return str(eval(clean_expr, {"__builtins__": {}}, {}))
        except:
            return "Error: Invalid math expression"
