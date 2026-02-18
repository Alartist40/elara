"""
TierRouter - Intelligent query routing to appropriate processing tier.
"""

from typing import Dict, Any, Optional


class TierRouter:
    """
    Routes queries to the appropriate processing tier based on complexity.

    Tier 1 (95%): Simple queries → Mistral direct
    Tier 2 (4%):  Memory/retrieval queries → CLaRa + Mistral
    Tier 3 (1%):  Complex reasoning → TRM + TiDAR + Tools
    """

    def __init__(
        self,
        tier_2_complexity: float = 0.5,
        tier_3_complexity: float = 0.8,
        voice_query_boost: int = 1,
        force_tier: Optional[int] = None,
    ):
        self.tier_2_threshold = tier_2_complexity
        self.tier_3_threshold = tier_3_complexity
        self.voice_boost = voice_query_boost
        self.force_tier = force_tier

        # Heuristic keywords
        self._complex_keywords = {
            "analyze", "compare", "explain why", "step by step",
            "calculate", "solve", "debug", "implement",
        }
        self._memory_keywords = {
            "remember", "earlier", "you said", "previous",
            "what was", "recall", "last time", "history",
        }

    def select_tier(
        self,
        text: str,
        modality: str = "text",
        has_tools: bool = False,
        has_memory: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Select processing tier for a query.

        Args:
            text: Input text.
            modality: "text", "voice", "image", "multimodal".
            has_tools: Whether tools are requested.
            has_memory: Whether CLaRa store has relevant docs.
            context: Additional routing context.

        Returns:
            Tier number (1, 2, or 3).
        """
        if self.force_tier is not None:
            return self.force_tier

        text_lower = text.lower()
        score = 0.0

        # Length-based complexity
        word_count = len(text.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.1

        # Keyword matching
        for keyword in self._complex_keywords:
            if keyword in text_lower:
                score += 0.2
                break

        for keyword in self._memory_keywords:
            if keyword in text_lower:
                score += 0.25
                break

        # Tool use triggers Tier 3
        if has_tools:
            score += 0.6

        # Multimodal triggers higher tier
        if modality in ("image", "multimodal"):
            score += 0.3

        # Voice gets a boost toward Tier 2
        if modality == "voice":
            score += 0.05 * self.voice_boost

        # Memory availability triggers Tier 2
        if has_memory:
            score += 0.25

        # Route based on score
        if score >= self.tier_3_threshold:
            return 3
        elif score >= self.tier_2_threshold:
            return 2
        else:
            return 1
