"""
ToolSchema - Defines tool schemas and validates tool calls.
"""

import json
from pathlib import Path
from typing import Optional, Any


class ToolSchema:
    """Manages tool schema definitions loaded from JSON."""

    def __init__(self, schema_path: Optional[str] = None):
        self.tools: dict[str, dict[str, Any]] = {}
        if schema_path:
            self.load(schema_path)

    def load(self, path: str) -> None:
        """Load tool schemas from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        for tool in data.get("tools", []):
            self.tools[tool["name"]] = tool

    def get_tool(self, name: str) -> Optional[dict[str, Any]]:
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def validate_params(self, name: str, params: dict[str, Any]) -> bool:
        """Validate parameters against schema."""
        tool = self.get_tool(name)
        if not tool:
            return False
        required = tool.get("parameters", {}).get("required", [])
        return all(r in params for r in required)

    def to_prompt_string(self) -> str:
        """Format tools as a prompt string for the model."""
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.get('description', '')}")
            params = tool.get("parameters", {}).get("properties", {})
            for pname, pinfo in params.items():
                lines.append(f"    {pname} ({pinfo.get('type', 'any')}): {pinfo.get('description', '')}")
        return "\n".join(lines)
