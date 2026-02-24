"""Simple tier selection. No ML, no complexity."""

class TierRouter:
    # Tier 3 triggers: tool keywords or explicit complexity
    TOOL_KEYWORDS = ("search", "look up", "find", "weather", "calculate")

    def __init__(self, tier2_engine):
        """
        Initialize the TierRouter with the provided tier-2 engine.
        
        Parameters:
            tier2_engine: An object that exposes a `has_relevant_docs(query: str) -> bool` method used to determine whether a query should be routed to tier 2. The instance is stored as `self.tier2`.
        """
        self.tier2 = tier2_engine

    def select_tier(self, query: str) -> int:
        # Bolt: Lowercase query once to avoid redundant allocations in keyword check
        """
        Select the routing tier (1, 2, or 3) for a given user query.
        
        Parameters:
            query (str): The user query to evaluate.
        
        Returns:
            int: `3` if the query contains any tool keyword, `2` if the Tier 2 engine reports relevant documents for the query, `1` otherwise.
        """
        lower_query = query.lower()
        if any(kw in lower_query for kw in self.TOOL_KEYWORDS):
            return 3

        # Tier 2: has relevant documents
        if self.tier2.has_relevant_docs(query):
            return 2

        # Default: Tier 1
        return 1
