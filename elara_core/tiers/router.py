"""Simple tier selection. No ML, no complexity."""

class TierRouter:
    # Tier 3 triggers: tool keywords or explicit complexity
    TOOL_KEYWORDS = ("search", "look up", "find", "weather", "calculate")

    def __init__(self, tier2_engine):
        self.tier2 = tier2_engine

    def select_tier(self, query: str) -> int:
        # Bolt: Lowercase query once to avoid redundant allocations in keyword check
        lower_query = query.lower()
        if any(kw in lower_query for kw in self.TOOL_KEYWORDS):
            return 3

        # Tier 2: has relevant documents
        if self.tier2.has_relevant_docs(query):
            return 2

        # Default: Tier 1
        return 1
