"""
ToolCallParser - Parse XML-style tool calls from model output.
"""

import re
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A parsed tool call."""
    name: str
    parameters: Dict[str, Any]
    raw_text: str


class ToolCallParser:
    """
    Parse tool calls from model text output.
    Supports XML-style: <tool>name</tool> <params>{"key": "value"}</params>
    Also supports function-call style: tool_name(param1="value")
    """

    # XML-style pattern
    XML_PATTERN = re.compile(
        r'<tool>(.*?)</tool>\s*<params>(.*?)</params>',
        re.DOTALL,
    )

    # Function-call style pattern
    FUNC_PATTERN = re.compile(
        r'(\w+)\((.*?)\)',
        re.DOTALL,
    )

    def parse(self, text: str) -> List[ToolCall]:
        """Parse all tool calls from text."""
        calls = []

        # Try XML-style first
        for match in self.XML_PATTERN.finditer(text):
            name = match.group(1).strip()
            params_str = match.group(2).strip()
            try:
                import json
                params = json.loads(params_str)
            except (json.JSONDecodeError, ValueError):
                params = {"raw": params_str}

            calls.append(ToolCall(
                name=name,
                parameters=params,
                raw_text=match.group(0),
            ))

        return calls

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(self.XML_PATTERN.search(text))

    def strip_tool_calls(self, text: str) -> str:
        """Remove tool call markup from text."""
        return self.XML_PATTERN.sub('', text).strip()
