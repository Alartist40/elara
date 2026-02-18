"""
Principle data structures and YAML loader for the Constitutional Layer.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import re


@dataclass
class Principle:
    """A single constitutional principle."""
    id: str
    category: str
    scriptural_basis: str
    rule_type: str           # "block", "flag", "rewrite"
    patterns: List[str]
    context: str             # "all", "voice_output", "vision", etc.
    action: str              # "block", "add_disclaimer", "add_watermark_disclosure", etc.
    response: str            # Default response when triggered
    _compiled_patterns: List[re.Pattern] = field(
        default_factory=list, repr=False, init=False
    )

    def __post_init__(self):
        """Compile patterns into regex for fast matching."""
        self._compiled_patterns = [
            re.compile(re.escape(p), re.IGNORECASE)
            for p in self.patterns
        ]

    def matches(self, text: str) -> bool:
        """Check if text matches any of this principle's patterns."""
        text_lower = text.lower()
        for pattern in self.patterns:
            if pattern.lower() in text_lower:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "scriptural_basis": self.scriptural_basis,
            "rule_type": self.rule_type,
            "patterns": self.patterns,
            "context": self.context,
            "action": self.action,
            "response": self.response,
        }


class PrincipleLoader:
    """Load and manage constitutional principles from YAML."""

    def __init__(self, principles_path: Optional[str] = None):
        self.principles: List[Principle] = []
        self.categories: Dict[str, List[Principle]] = {}
        if principles_path:
            self.load(principles_path)

    def load(self, path: str) -> None:
        """Load principles from YAML file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Principles file not found: {path}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "principles" not in data:
            raise ValueError(f"Invalid principles file: {path}")

        self.principles = []
        self.categories = {}

        for entry in data["principles"]:
            principle = Principle(
                id=entry["id"],
                category=entry["category"],
                scriptural_basis=entry["scriptural_basis"],
                rule_type=entry["rule_type"],
                patterns=entry.get("patterns", []),
                context=entry.get("context", "all"),
                action=entry.get("action", "block"),
                response=entry.get("response", "Content blocked by safety policy."),
            )
            self.principles.append(principle)

            # Index by category
            if principle.category not in self.categories:
                self.categories[principle.category] = []
            self.categories[principle.category].append(principle)

    def get_by_category(self, category: str) -> List[Principle]:
        """Get all principles in a category."""
        return self.categories.get(category, [])

    def get_by_context(self, context: str) -> List[Principle]:
        """Get all principles applicable to a given context."""
        return [
            p for p in self.principles
            if p.context == "all" or p.context == context
        ]

    def get_by_id(self, principle_id: str) -> Optional[Principle]:
        """Get a specific principle by ID."""
        for p in self.principles:
            if p.id == principle_id:
                return p
        return None

    def __len__(self) -> int:
        return len(self.principles)

    def __iter__(self):
        return iter(self.principles)
