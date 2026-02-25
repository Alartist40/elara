import re
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SafetyFilter:
    """Simple rule-based filtering. No 'principles engine'."""

    def __init__(self, rules_path: str = ""):
        """
        Initialize the SafetyFilter by loading rules from a YAML file and compiling regex patterns.

        If a non-empty `rules_path` is provided it is used; otherwise the module-local "rules.yaml" is used. If the rules file is missing or cannot be parsed, the filter falls back to an empty rules set {"block": [], "flag": []} (a warning or error is logged respectively). Loaded rules are expected to contain "block" entries with "pattern" and "reason", and "flag" entries with "pattern" and "replacement". Compiles `block_patterns` as (compiled_regex, reason) and `flag_patterns` as (compiled_regex, replacement) using case-insensitive matching.

        Parameters:
            rules_path (str): Optional path to a YAML rules file; when empty the default "rules.yaml" next to this module is used.
        """
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

    def check(self, text: str) -> tuple[bool, str]:
        """
        Check text against configured block and flag rules and return whether it is allowed plus either a block reason or the cleaned text.

        Returns:
            tuple[bool, str]: A tuple (allowed, reason_or_cleaned_text) — `allowed` is True if no block patterns match the text after flag substitutions, False if a block pattern matches; when False, `reason_or_cleaned_text` is the block reason prefixed with "Blocked: "; when True, it is the cleaned text with flag substitutions applied.
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
