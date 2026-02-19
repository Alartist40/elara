import re
import yaml
from pathlib import Path
from typing import Tuple, List

class SafetyFilter:
    """Simple rule-based filtering. No 'principles engine'."""

    def __init__(self, rules_path: str = "elara_core/safety/rules.yaml"):
        self.rules_path = Path(rules_path)
        if self.rules_path.exists():
            with open(self.rules_path) as f:
                self.rules = yaml.safe_load(f)
        else:
            self.rules = {"block": [], "flag": []}

    def check(self, text: str) -> Tuple[bool, str]:
        """
        Returns: (allowed, reason_or_cleaned_text)
        """
        for rule in self.rules.get("block", []):
            if re.search(rule["pattern"], text, re.IGNORECASE):
                return False, f"Blocked: {rule['reason']}"

        # Optional: simple cleaning
        cleaned = text
        for rule in self.rules.get("flag", []):
            cleaned = re.sub(rule["pattern"], rule["replacement"], cleaned)

        return True, cleaned
