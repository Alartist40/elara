"""Simple tier selection. No ML, no complexity."""

import re
from typing import List

class TierRouter:
    def __init__(self, tier2_engine):
        self.tier2 = tier2_engine

    def select_tier(self, query: str) -> int:
        # Tier 3 triggers: tool keywords or explicit complexity
        tool_keywords = ["search", "look up", "find", "weather", "calculate"]
        if any(kw in query.lower() for kw in tool_keywords):
            return 3

        # Tier 2: has relevant documents
        if self.tier2.has_relevant_docs(query):
            return 2

        # Default: Tier 1
        return 1
