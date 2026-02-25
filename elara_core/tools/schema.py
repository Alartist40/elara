"""
ToolSchema - Defines tool schemas and validates tool calls.
"""

import json
from pathlib import Path
from typing import Optional, Any


class ToolSchema:
    """Manages tool schema definitions loaded from JSON."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the ToolSchema and optionally load tool definitions from a JSON schema file.
        
        Parameters:
            schema_path (Optional[str]): Path to a JSON file containing tool definitions. If provided, the file is loaded and parsed into the instance's `tools` mapping; file I/O or JSON parsing errors raised by the loader will propagate.
        """
        self.tools: dict[str, dict[str, Any]] = {}
        if schema_path:
            self.load(schema_path)

    def load(self, path: str) -> None:
        """
        Load tool schemas from a JSON file into this instance's tools mapping.
        
        Parameters:
            path (str): Filesystem path to a JSON file containing a top-level "tools" array.
        
        Description:
            Parses the JSON file and stores each tool object in self.tools keyed by the tool's "name".
            Existing entries with the same name will be overwritten. File I/O or JSON parsing errors
            will propagate to the caller.
        """
        with open(path, "r") as f:
            data = json.load(f)
        for tool in data.get("tools", []):
            self.tools[tool["name"]] = tool

    def get_tool(self, name: str) -> Optional[dict[str, Any]]:
        """
        Return the tool definition for the given tool name if present.
        
        Returns:
            dict[str, Any] | None: The tool dictionary for the specified name, or None if the tool is not found.
        """
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """
        Return a list of all registered tool names.
        
        Returns:
            list[str]: Tool names currently stored in this ToolSchema.
        """
        return list(self.tools.keys())

    def validate_params(self, name: str, params: dict[str, Any]) -> bool:
        """
        Check whether the provided params include all parameters required by the tool schema.
        
        Parameters:
            name (str): Name of the tool whose schema to validate.
            params (dict[str, Any]): Mapping of parameter names to values to validate.
        
        Returns:
            bool: True if the tool exists and every required parameter name is present in `params`, False otherwise.
        """
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
