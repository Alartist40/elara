"""
MetricsTracker - Performance and tier usage metrics.
"""

import time
from typing import Dict, Any, List
from collections import defaultdict


class MetricsTracker:
    """Tracks performance metrics across all tiers."""

    def __init__(self):
        self.tier_hits: Dict[int, int] = defaultdict(int)
        self.latencies: Dict[int, List[float]] = defaultdict(list)
        self.voice_conversations: int = 0
        self.blocked_requests: int = 0
        self.total_requests: int = 0
        self._start_time = time.time()

    def record_request(
        self, tier: int, latency_ms: float, voice: bool = False, blocked: bool = False
    ) -> None:
        self.total_requests += 1
        self.tier_hits[tier] += 1
        self.latencies[tier].append(latency_ms)
        if voice:
            self.voice_conversations += 1
        if blocked:
            self.blocked_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        avg_latencies = {}
        for tier, lats in self.latencies.items():
            avg_latencies[tier] = sum(lats) / len(lats) if lats else 0.0

        uptime = time.time() - self._start_time

        return {
            "total_requests": self.total_requests,
            "tier_distribution": dict(self.tier_hits),
            "avg_latency_ms": avg_latencies,
            "voice_conversations": self.voice_conversations,
            "blocked_requests": self.blocked_requests,
            "uptime_seconds": round(uptime, 1),
        }

    def reset(self) -> None:
        self.tier_hits = defaultdict(int)
        self.latencies = defaultdict(list)
        self.voice_conversations = 0
        self.blocked_requests = 0
        self.total_requests = 0
        self._start_time = time.time()
