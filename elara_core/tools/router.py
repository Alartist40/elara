"""
Simplified ToolRouter - Only Calculator implementation.
"""

import time
import re
from typing import Optional

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
        """
        Safely evaluates a mathematical expression.
        Uses simpleeval if available, otherwise falls back to a safe recursive parser.
        """
        try:
            from simpleeval import simple_eval
            return str(simple_eval(expression, functions={}, names={}))
        except ImportError:
            return ToolRouter._safe_eval_fallback(expression)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def _safe_eval_fallback(expression: str) -> str:
        """
        Minimal safe math parser fallback using a simplified shunting-yard-like
        approach for basic precedence (+, -, *, /) without parentheses.
        If complex symbols like parentheses are found, it returns an error
        suggesting simpleeval installation.
        """
        import operator
        import re

        # Restricted character set: digits, basic ops, spaces
        if not re.match(r'^[0-9+\-*/%. ]+$', expression):
            return "Error: Unsupported characters or complex expressions (install simpleeval for full support)"

        try:
            # Tokenize: numbers and operators
            tokens = re.findall(r'\d+\.?\d*|[+\-*/%]', expression.replace(' ', ''))
            if not tokens: return "Error: No valid math tokens"

            # 1. Handle multiplication/division first (precedence)
            processed_tokens = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token in ['*', '/', '%']:
                    if not processed_tokens or i + 1 >= len(tokens):
                        return "Error: Invalid expression format"
                    left = float(processed_tokens.pop())
                    right = float(tokens[i+1])
                    if token == '*': res = left * right
                    elif token == '/': res = left / right
                    else: res = left % right
                    processed_tokens.append(res)
                    i += 2
                else:
                    processed_tokens.append(token)
                    i += 1

            # 2. Handle addition/subtraction
            if not processed_tokens: return "Error: Evaluation failed"

            res = float(processed_tokens[0])
            i = 1
            while i < len(processed_tokens):
                op = processed_tokens[i]
                val = float(processed_tokens[i+1])
                if op == '+': res += val
                elif op == '-': res -= val
                i += 2

            return str(res)
        except Exception as e:
            return f"Error: {e}"
