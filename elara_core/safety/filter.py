import re
import yaml
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

class SafetyFilter:
    """Simple rule-based filtering. No 'principles engine'."""

    def __init__(self, rules_path: str = ""):
        self.rules_path = (
            Path(rules_path) if rules_path
            else Path(__file__).parent / "rules.yaml"
        )

        if self.rules_path.exists():
            try:
                with open(self.rules_path, encoding="utf-8") as f:
                    self.rules = yaml.safe_load(f) or {"block": [], "flag": []}
            except yaml.YAMLError as exc:
                logger.error("Failed to parse safety rules from %s: %s", self.rules_path, exc)
                self.rules = {"block": [], "flag": []}
        else:
            logger.warning("Safety rules file not found at %s; all content will pass.", self.rules_path)
            self.rules = {"block": [], "flag": []}

        # Compile patterns for performance
        self.block_patterns = [
            (re.compile(rule["pattern"], re.IGNORECASE), rule["reason"])
            for rule in self.rules.get("block", [])
        ]
        self.flag_patterns = [
            (re.compile(rule["pattern"], re.IGNORECASE), rule["replacement"])
            for rule in self.rules.get("flag", [])
        ]

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Returns: (allowed, reason_or_cleaned_text)
        """
        # 1. Check original text
        for pattern, reason in self.block_patterns:
            if pattern.search(text):
                return False, f"Blocked: {reason}"

        # 2. Apply cleaning
        cleaned = text
        for pattern, replacement in self.flag_patterns:
            cleaned = pattern.sub(replacement, cleaned)

        # 3. Re-check cleaned text (safety bypass prevention)
        for pattern, reason in self.block_patterns:
            if pattern.search(cleaned):
                return False, f"Blocked: {reason}"

        return True, cleaned
